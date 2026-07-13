# Sampler parity audit — candle port vs. reference (2026-07-09)

Scope: `diffusion-rs/src/sampler.rs::generate` vs. `sidecar/sidecar.py` (the
torch pipeline previously validated against upstream Plaid).

| Aspect | Reference (sidecar.py) | Candle port | Verdict |
|---|---|---|---|
| Timesteps | `ts = linspace(1,0,steps)`; `s = t − 1/steps` | `t = 1 − i/(steps−1)`; `s = t − 1/steps` | identical |
| Schedule | `γ = γ0 + (γ1−γ0)·sched(t)` | same (`g` closure) | identical |
| Self-conditioning | `x_selfcond = x_reconst` (raw model output, pre-temp) | same | identical |
| Score temp | `eps = (z − a_t·xr)/s_t/temp; xr = (z − s_t·eps)/a_t` | same | identical |
| Ancestral update | `c = −expm1(γs−γt)`; three-coef update | same | identical |
| Final decode | one forward at last γ, argmax | same | identical |
| Forward pass | — | validated numerically (`cargo run` validate: max\|Δ\| < 1e-2) | validated |

Conclusion: no sampling-fidelity gap. Text quality is bounded by Plaid-1B
itself (≈GPT-2-small likelihood). Remaining lever exercised here: more reverse
steps ("ultra" preset). RePlaid (arXiv:2605.18530) remains the upgrade path —
weights still unpublished as of 2026-07-07.

## 768-step sample

Evidence run (not a benchmark), Metal, `generate 768 96`: 96 tokens, 768 steps,
done in 151.0 s (197 ms/step).

> itching to join the PVP lobby. This will give you more options and feel very
> different from the Arcade games. It will go live in the first week of October
> and you can expect the new game to be live in about 3 months. We do have some
> other improvements in the works for console players and we will be giving you
> some information soon. So make sure you check back here as we talk to you
> about player and guild improvements soon.
>
> Here is information for
