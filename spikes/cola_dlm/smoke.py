"""Cola-DLM smoke test (throwaway spike): does it run on MPS, is the output a
pure function of TrueRNG bytes, and does the latent noise change *content*?

Entropy enters as each generation block's Gaussian noise eps ~ N(0, I)
(16 tokens x 16 latent dims). We route every `torch.randn` the sampler makes
through a byte tape: 4 bytes -> uniform in (0,1) -> inverse normal CDF, as
diffusion-rs/src/trng.rs does for Plaid. Decoding is greedy (temperature 0),
so the noise is the only source of variation.

    .venv/bin/python smoke.py <carrier.bin> [--device mps]
"""
import argparse
import contextlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from cola_dlm import ColaDiTModel, ColaTextVAEModel, generate_task_repaint_inference

HF = Path(__file__).resolve().parent / "hf_models"


class Tape:
    def __init__(self, data: bytes):
        self.data, self.pos, self.calls = data, 0, []

    def normals(self, n: int) -> torch.Tensor:
        need = 4 * n
        if self.pos + need > len(self.data):
            raise RuntimeError("tape exhausted")
        u32 = np.frombuffer(self.data[self.pos:self.pos + need], dtype="<u4").astype(np.float64)
        self.pos += need
        u = (u32 + 0.5) / 2.0**32
        return torch.special.ndtri(torch.from_numpy(u)).to(torch.float32)


@contextlib.contextmanager
def tape_randn(tape: Tape):
    orig = torch.randn

    def randn(*size, generator=None, device=None, dtype=None, **kw):
        if generator is not None:
            return orig(*size, generator=generator, device=device, dtype=dtype, **kw)
        shape = tuple(size[0]) if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)) else tuple(size)
        tape.calls.append(shape)
        z = tape.normals(int(np.prod(shape))).reshape(shape)
        return z.to(device=device or "cpu", dtype=dtype or torch.float32)

    torch.randn = randn
    try:
        yield
    finally:
        torch.randn = orig


def load(device):
    # The inference code casts its inputs to bfloat16 and relies on CUDA autocast
    # to meet the fp32 weights; off CUDA autocast is disabled, so run the weights
    # in bfloat16 too (the precision the model computes in on CUDA).
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    dit = ColaDiTModel.from_pretrained(str(HF / "cola_dlm/cola_dit")).to(device, dtype).eval()
    vae = ColaTextVAEModel.from_pretrained(str(HF / "cola_dlm/cola_vae")).to(device, dtype).eval()
    if dtype != torch.float32:
        # the latent is integrated in fp32 and handed to the decoder uncast
        orig_decode = vae.decode

        def decode(z, *a, **kw):
            return orig_decode(z.to(dtype), *a, **kw)
        vae.decode = decode
    return dit, vae, Tokenizer.from_file(str(HF / "tokenizer.json"))


def generate(models, device, prompt: str, raw: bytes, max_new_tokens: int):
    dit, vae, tok = models
    tape = Tape(raw)
    with torch.no_grad(), tape_randn(tape):
        r = generate_task_repaint_inference(dit=dit, vae=vae, tokenizer=tok, prompts=[{"question": prompt}],
                                            task_name="lambada", device=device, max_new_tokens=max_new_tokens,
                                            temperature=0.0, eos_token_id=100257, im_end_token_id=100265)
    return r[0]["generate"], tape


def jaccard(a: str, b: str) -> float:
    x, y = set(a.lower().split()), set(b.lower().split())
    return len(x & y) / max(len(x | y), 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("carrier")
    p.add_argument("--device", default="mps")
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--offset", type=int, default=80_000_000)
    p.add_argument("--out", default="runs/smoke.jsonl")
    args = p.parse_args()

    device = torch.device(args.device)
    t0 = time.time()
    models = load(device)
    print(f"loaded on {device} in {time.time() - t0:.1f}s", flush=True)
    data = Path(args.carrier).read_bytes()
    per = 64 * 1024
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(args.out, "w") as f:
        # determinism: the same bytes twice
        raw = data[args.offset:args.offset + per]
        a, tape = generate(models, device, "Once upon a time", raw, args.tokens)
        b, _ = generate(models, device, "Once upon a time", raw, args.tokens)
        print(f"same bytes -> identical text: {a == b}; randn calls {len(tape.calls)} {tape.calls[:3]}, "
              f"{tape.pos} bytes used", flush=True)
        for prompt in ("Once upon a time", "The"):
            texts = []
            for i in range(args.n):
                raw = data[args.offset + (i + 1) * per: args.offset + (i + 2) * per]
                t1 = time.time()
                text, tape = generate(models, device, prompt, raw, args.tokens)
                dt = time.time() - t1
                texts.append(text)
                row = {"prompt": prompt, "i": i, "text": text, "seconds": round(dt, 1), "bytes": tape.pos}
                rows.append(row)
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"[{prompt!r} #{i}] {dt:.1f}s, {tape.pos} B: {text[:160]!r}", flush=True)
            sims = [jaccard(x, y) for x, y in itertools.combinations(texts, 2)]
            print(f"{prompt!r}: mean pairwise word overlap {np.mean(sims):.2f}", flush=True)


if __name__ == "__main__":
    main()
