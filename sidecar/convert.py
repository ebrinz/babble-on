"""Convert Plaid-1B torch weights -> a single safetensors for the candle port.

Loads the four published `.pt` state dicts (via the same muP-aware build the
sidecar uses), then writes:
  - models/plaid1b/plaid1b.safetensors  : all parameters, prefixed by submodule
  - models/plaid1b/meta.json            : arch dims + the baked readout scalar
  - models/plaid1b/validation.safetensors: a fixed forward pass (z, gamma ->
    x_reconst, logits) for numerically validating the Rust/candle port.

Run:  .venv/bin/python convert.py
"""
import json
import os

import compat  # noqa: F401  (MPS shims, must precede lib.models)
import mup
import torch
from safetensors.torch import save_file

import lib.models

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "..", "models", "plaid1b")

DIM, N_BLOCKS, N_HEADS = 2048, 24, 32
EMBED_DIM, VOCAB = 16, 32768
GAMMA_0, GAMMA_1 = -3.0, 6.0


def build():
    def create(d, h):
        return {
            "noise_schedule": lib.models.NoiseSchedule().float(),
            "gamma_bounds": lib.models.GammaBounds(GAMMA_0, GAMMA_1).float(),
            "embedding_matrix": lib.models.EmbeddingMatrix(VOCAB, EMBED_DIM).float(),
            "model": lib.models.DiffusionModel(d, EMBED_DIM, N_BLOCKS, h, VOCAB).float(),
        }
    mods = create(DIM, N_HEADS)
    base, delta = create(256, 4), create(128, 2)
    for k in mods:
        mup.set_base_shapes(mods[k], base[k], delta=delta[k])
        sd = torch.load(os.path.join(WEIGHTS, f"{k}.pt"), map_location="cpu", weights_only=True)
        mods[k].load_state_dict(sd)
    return mods


def main():
    mods = build()
    model = mods["model"].eval()

    # The only inference-time muP effect: the readout output scale.
    readout_scale = float(model.output_linear.output_mult / model.output_linear.width_mult())
    print(f"readout_scale (output_mult/width_mult) = {readout_scale}")

    # Flatten every parameter into one tensor dict (contiguous fp32).
    tensors = {}
    for prefix, mod in [("model", mods["model"]), ("emb", mods["embedding_matrix"]),
                        ("sched", mods["noise_schedule"]), ("bounds", mods["gamma_bounds"])]:
        for name, p in mod.state_dict().items():
            tensors[f"{prefix}.{name}"] = p.detach().contiguous().float().cpu()
    out = os.path.join(WEIGHTS, "plaid1b.safetensors")
    save_file(tensors, out)
    print(f"wrote {out} ({len(tensors)} tensors)")

    meta = {
        "dim": DIM, "n_blocks": N_BLOCKS, "n_heads": N_HEADS,
        "embed_dim": EMBED_DIM, "vocab_size": VOCAB,
        "gamma_0": GAMMA_0, "gamma_1": GAMMA_1,
        "readout_scale": readout_scale,
    }
    with open(os.path.join(WEIGHTS, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("wrote meta.json:", meta)

    # --- Validation reference: one deterministic forward pass ---------------
    torch.manual_seed(1234)
    n, seq = 1, 16
    z = torch.randn(n, seq, EMBED_DIM)
    gamma = torch.tensor([1.5])
    emb = mods["embedding_matrix"]()
    x_selfcond = torch.zeros(n, seq, EMBED_DIM)
    with torch.no_grad():
        logits, x_reconst = model(z=z, gamma=gamma, embedding_matrix=emb,
                                  bias_scale=1.0, x_selfcond=x_selfcond)
    save_file(
        {"z": z, "gamma": gamma, "x_reconst": x_reconst.contiguous(),
         "logits": logits.contiguous()},
        os.path.join(WEIGHTS, "validation.safetensors"),
    )
    print(f"wrote validation.safetensors (z {tuple(z.shape)} -> "
          f"x_reconst {tuple(x_reconst.shape)}, logits {tuple(logits.shape)})")


if __name__ == "__main__":
    main()
