"""Persistent stdio service wrapping Plaid-1B for babble-on.

Loads the model once, then serves generation requests over a newline-delimited
JSON protocol. ALL logging goes to stderr; stdout carries ONLY protocol JSON.

Protocol:
  startup (after the model is resident):  {"type":"ready"}
  host -> stdin (one JSON object per line):
      {"steps":256,"seq_len":256,"n_samples":1,"seed":0,"preview_every":24,
       "entropy_hex":"<optional hex bytes to seed the initial latent z1>"}
  service -> stdout:
      {"type":"step","i":N,"total":T,"text":["partial ...", ...]}   (periodic)
      {"type":"done","i":T,"total":T,"text":["final ...", ...],"elapsed":12.3}
      {"type":"error","message":"..."}

If "entropy_hex" is supplied, those TrueRNG bytes become the Gaussian initial
latent (the literal noise tensor); otherwise a PRNG is used.
"""
import json
import os
import sys
import time

import compat  # noqa: F401  (must precede lib.models — registers MPS shims)
import mup
import torch
from tokenizers import Tokenizer

import lib.models
import trng

HERE = os.path.dirname(os.path.abspath(__file__))


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


class Plaid:
    def __init__(self, weights_path, device="mps", dim=2048, n_blocks=24,
                 n_heads=32, embed_dim=16, vocab_size=32768,
                 gamma_0=-3.0, gamma_1=6.0, tokenizer_path=None):
        self.device = device
        self.embed_dim = embed_dim
        tokenizer_path = tokenizer_path or os.path.join(HERE, "misc", "owt2_tokenizer.json")

        def create(d, h):
            return {
                "noise_schedule": lib.models.NoiseSchedule().float(),
                "gamma_bounds": lib.models.GammaBounds(gamma_0, gamma_1).float(),
                "embedding_matrix": lib.models.EmbeddingMatrix(vocab_size, embed_dim).float(),
                "model": lib.models.DiffusionModel(d, embed_dim, n_blocks, h, vocab_size).float(),
            }
        mods = create(dim, n_heads)
        base, delta = create(256, 4), create(128, 2)
        for k in mods:
            mup.set_base_shapes(mods[k], base[k], delta=delta[k])
        for name, m in mods.items():
            sd = torch.load(os.path.join(weights_path, f"{name}.pt"),
                            map_location="cpu", weights_only=True)
            m.load_state_dict(sd)

        self.model = mods["model"].to(device).eval()
        self.embedding_matrix = mods["embedding_matrix"].to(device)
        self.noise_schedule = mods["noise_schedule"]      # CPU / fp64
        self.gamma_bounds = mods["gamma_bounds"]          # CPU / fp64
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

    def _decode(self, logits):
        ids = logits.argmax(dim=-1).cpu()
        return [self.tokenizer.decode(row.tolist(), skip_special_tokens=False)
                for row in ids]

    @torch.no_grad()
    def generate(self, steps, seq_len, n_samples, seed, preview_every,
                 entropy_hex=None, score_temp=0.9, initial_noise_scale=1.0):
        torch.manual_seed(seed)
        shape = (n_samples, seq_len, self.embed_dim)
        if entropy_hex:
            raw = bytes.fromhex(entropy_hex)
            z = trng.bytes_to_gaussians(raw, shape) * initial_noise_scale
        else:
            z = torch.randn(shape, dtype=torch.float64) * initial_noise_scale

        emb = self.embedding_matrix()
        gamma_0, gamma_1 = self.gamma_bounds()
        x_selfcond = torch.zeros(n_samples, seq_len, self.embed_dim, device=self.device)
        ts = torch.linspace(1.0, 0.0, steps, dtype=torch.float64)
        gamma_t = None

        for i in range(steps):
            t = ts[i:i + 1]
            s = t - 1.0 / steps
            gamma_s = gamma_0 + (gamma_1 - gamma_0) * self.noise_schedule(s).double()
            gamma_t = gamma_0 + (gamma_1 - gamma_0) * self.noise_schedule(t).double()
            a2s, a2t = torch.sigmoid(-gamma_s), torch.sigmoid(-gamma_t)
            at, st = a2t.sqrt(), torch.sigmoid(gamma_t).sqrt()

            zf = z.to(self.device, torch.float32)
            logits, x_reconst = self.model(z=zf, gamma=gamma_t.float().to(self.device),
                                           embedding_matrix=emb, bias_scale=1.0,
                                           x_selfcond=x_selfcond)
            x_selfcond = x_reconst.clone().detach()
            xr = x_reconst.cpu().double()

            eps = (z - at * xr) / st / score_temp
            xr = (z - st * eps) / at
            if ts[i] > 0:
                c = -torch.expm1(gamma_s - gamma_t)
                z = z * ((1 - c) * a2s.sqrt() / a2t.sqrt())
                z = z + c * (a2s.sqrt() * xr)
                z = z + (c * (1 - a2s)).sqrt() * torch.randn(z.shape, dtype=torch.float64)

            if preview_every and (i % preview_every == 0) and i > 0:
                yield {"type": "step", "i": i, "total": steps, "text": self._decode(logits)}

        zf = z.to(self.device, torch.float32)
        logits, _ = self.model(z=zf, gamma=gamma_t.float().to(self.device),
                               embedding_matrix=emb, bias_scale=1.0, x_selfcond=x_selfcond)
        yield {"type": "final", "text": self._decode(logits)}


def main():
    weights = os.environ.get("PLAID_WEIGHTS") or os.path.join(HERE, "..", "models", "plaid1b")
    device = os.environ.get("PLAID_DEVICE", "mps")
    log(f"sidecar: loading Plaid-1B from {weights} on {device} ...")
    t0 = time.time()
    plaid = Plaid(weights, device=device)
    log(f"sidecar: model resident in {time.time() - t0:.1f}s")
    emit({"type": "ready"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            emit({"type": "error", "message": f"bad request json: {e}"})
            continue
        try:
            t0 = time.time()
            for ev in plaid.generate(
                steps=int(req.get("steps", 256)),
                seq_len=int(req.get("seq_len", 256)),
                n_samples=int(req.get("n_samples", 1)),
                seed=int(req.get("seed", 0)),
                preview_every=int(req.get("preview_every", 24)),
                entropy_hex=req.get("entropy_hex"),
                score_temp=float(req.get("score_temp", 0.9)),
            ):
                if ev["type"] == "final":
                    emit({"type": "done", "i": int(req.get("steps", 256)),
                          "total": int(req.get("steps", 256)), "text": ev["text"],
                          "elapsed": round(time.time() - t0, 1)})
                else:
                    emit(ev)
        except Exception as e:  # noqa: BLE001 — report any generation failure to the host
            emit({"type": "error", "message": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
