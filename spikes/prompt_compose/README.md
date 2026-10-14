# Spike: the device writes the brief (2026-10-14)

Throwaway spike. Question: can TrueRNG bytes author the *content* of
DiffusionGemma's text, not just its wording, by composing the prompt?

`compose.py` turns 16 bytes into four slot choices (subject, setting, mood,
form; 32 entries each, ordered roughly from ordinary to strange) through the
harness's bytes → uniform → index contract, builds a prompt such as "Write a
recipe about a fisherman, inside a single raindrop. The mood is feverish.
Keep it under 200 words.", and generates through the harness sampler on the
MLX backend with the next 99 328 bytes as the tape. Bytes are real TrueRNG
bytes from the 2026-10-12 capture (`ordinary.bin`, offset 40 000 000).

    ../../harness/.venv/bin/python compose.py ../coherence/runs/2026-10-12/ordinary.bin runs/2026-10-14.jsonl --n 12 --offset 40000000

Tests: `../../harness/.venv/bin/python -m pytest tests -q`.

## Result

12 of 12 passages follow every drawn slot: a recipe for fishing inside a
raindrop, a classified ad for a desert clockmaker, birdsong translated under
a frozen lake, a police report on a lighthouse keeper in a café, a spell
for a cartographer, a eulogy for a violin. Prose quality is the instruct
model's usual. Mean pairwise word overlap is 0.08, against 0.3–0.5 for the
fixed-prompt runs of 2026-10-12, so the bytes now decide what the text is
about. Cost: 67 s per passage on the M1 Pro (22.5 denoising steps on
average, against ~14 for the fixed prompt), because unusual briefs leave the
model less certain for longer.

Because the lists are ordered, a byte stream skewed high pulls every slot
toward the strange end, so a coherence excursion would show as a legible
drift in the brief. That coupling is designed in, not discovered.
