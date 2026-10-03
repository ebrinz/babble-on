"""Sidecar mode for phase 2 (the app's engine swap): speaks the newline-JSON
protocol in ``sidecar/README.md`` so the Tauri backend can spawn this instead
of the Plaid engine.

  stdin : {"steps":48,"seq_len":256,"prompt":"...","entropy_hex":"...","preview_every":1}
  stdout: {"type":"ready"} once, then per request
          {"type":"step","i":N,"total":T,"tokens":[...per-position strings...]}
          {"type":"done","i":T,"total":T,"tokens":[...],"text":"...","elapsed":12.3}
          {"type":"error","message":"..."}

``entropy_hex`` must hold at least ``budget_bytes(seq_len, steps)`` bytes;
with none, a PRNG tape is used and the reply says so (``"seed":"prng"``).
``seq_len`` must equal the model's canvas length (256 for DiffusionGemma,
16 for the development stub); any other value is refused with an ``error``
reply so the host never gets a canvas it did not ask for.
"""
from __future__ import annotations

import json
import sys
import time

from .models import load_denoiser
from .noise import EntropyTape, budget_bytes, prng_tape
from .sampler import SamplerConfig, sample_canvas


def emit(obj) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def serve(model: str = "diffusion_gemma", **model_kw) -> None:
    den = load_denoiser(model, **model_kw)
    emit({"type": "ready", "model": model})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            emit({"type": "error", "message": f"bad request json: {e}"}); continue
        try:
            steps = int(req.get("steps", 48))
            # The canvas length is the model's; a request that asks for another
            # length is refused rather than silently resized.
            canvas = int(getattr(den, "canvas_length", None) or 256)
            seq_len = int(req.get("seq_len", canvas))
            if seq_len != canvas:
                emit({"type": "error", "message": f"seq_len {seq_len} unsupported: this model's canvas is {canvas}"}); continue
            cfg = SamplerConfig(canvas_length=seq_len, vocab_size=getattr(den, "vocab_size", 262_144),
                                max_denoising_steps=steps, early_stop=bool(req.get("early_stop", True)))
            need = budget_bytes(seq_len, steps)
            if req.get("entropy_hex"):
                raw = bytes.fromhex(req["entropy_hex"])
                if len(raw) < need:
                    emit({"type": "error", "message": f"entropy_hex holds {len(raw)} bytes, need {need}"}); continue
                tape, seed = EntropyTape(raw, label="app"), "entropy"
            else:
                tape, seed = prng_tape(int(req.get("seed", 0)), need), "prng"
            if hasattr(den, "set_prompt"):
                den.set_prompt(req.get("prompt") or None)
            every = int(req.get("preview_every", 1))
            t0 = time.time()

            def on_step(tr):
                if every and tr.index % every == 0:
                    toks = den.decode_each(tr.argmax) if hasattr(den, "decode_each") else [str(i) for i in tr.argmax]
                    emit({"type": "step", "i": tr.index + 1, "total": steps, "tokens": toks,
                          "accepted": int(tr.n_accepted), "entropy": tr.mean_entropy})

            res = sample_canvas(den, tape, cfg, keep_capture=False, on_step=on_step)
            ids = res.final_canvas
            toks = den.decode_each(ids) if hasattr(den, "decode_each") else [str(i) for i in ids]
            text = den.decode(den.trim_after_eos(ids)) if hasattr(den, "decode") else "".join(toks)
            emit({"type": "done", "i": res.n_steps, "total": steps, "tokens": toks, "text": text,
                  "elapsed": round(time.time() - t0, 1), "seed": seed, "entropy_used": res.tape_consumed,
                  "seq_len": seq_len, "stopped_early": res.stopped_early})
        except Exception as e:  # noqa: BLE001 — report to the host
            emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
