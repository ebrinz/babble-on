# Coherence Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A pre-registered A/B experiment that generates DiffusionGemma text from anomaly-bank bytes, ordinary TrueRNG bytes, a CSPRNG and a biased RNG, scores each text, and reports whether anomaly bytes measurably change the output, with a positive control and a null calibration.

**Architecture:** Six small Python scripts under `spikes/coherence/`, each a pure function from input files to output files under `runs/<run-id>/`: capture → classify → build → generate → score → analyze. Generation reuses `spikes/diffusiongemma/entropy.py` (the recording entropy pool that makes DiffusionGemma output a pure function of its bytes). No app code changes.

**Tech Stack:** Python 3.12 via `uv`, mlx-vlm 0.7.6 (DiffusionGemma 26B-A4B 4-bit, already in the HF cache), numpy, pyserial, torch + transformers (GPT-2 scorer), pytest.

**Spec:** `docs/superpowers/specs/2026-10-12-coherence-spike-design.md`

## Global Constraints

- Nothing under `src-tauri/`, `src/`, or `diffusion-rs/` is modified.
- All new code lives in `spikes/coherence/`; its venv is `spikes/coherence/.venv` (git-ignored by the repo's `.venv` rule). Run every Python command from `spikes/coherence/` as `.venv/bin/python …`; run tests as `.venv/bin/python -m pytest tests -q`.
- Trial size `TRIAL_BYTES = 5120`; threshold `Z95 = 1.959_963_98`; a trial is anomalous when `abs(sigma) >= Z95` (matches `Band::of` in `src-tauri/src/stats.rs`).
- Sample reservoir `SAMPLE_BYTES = 51_200`; conditions in order `("anomaly", "ordinary", "csprng", "biased")`; 100 samples per condition; round-robin layout.
- Biased RNG: `x = (x * 1103515245 + 12345) & 0xFFFFFFFF`, byte `((x >> 16) & 0xFF) & 0x3F`, seed `0x12345678`; first eight bytes `31071d142c091307`.
- Generation: prompt `"Write a short passage of prose."` through the chat template, `temperature=0.0`, `max_tokens=256`, model-default denoising steps. Failed samples are recorded with `"error"` set and never replaced by software noise.
- Primary metric `gpt2_nll` (mean per-token NLL under `gpt2`, nats). Secondary `steps = (consumed - 4) // 1024 - 1` and `distinct_ratio`.
- Permutation test: 20 000 shuffles, two-sided, seed 0. Decision rule exactly as the spec: positive control `p < 0.001` required; null calibration expected `p > 0.05`; primary at `α = 0.01`; a hit is "candidate effect".
- Run artefacts: `*.bin` are git-ignored (repo rule) and the spike's own `.gitignore` drops `batch.jsonl`, logs and the smoke run; commit `classify.json`, `manifest.json`, `scored.jsonl`, `report.md`.
- All shell steps, including `git add`/`git commit`, run from `spikes/coherence/` (git resolves the relative paths).
- Commit after every task. Final task appends the entry to `docs/experiments/LOG.md` in the same commit as the results (repo `CLAUDE.md` rule).

## Review Focus

1. **Capture cut short by a USB dropout.** The file must keep every byte written so far and the run must still be usable if the anomaly pool clears the minimum. Pinned by `test_capture_to_keeps_partial_file_on_dropout` (Task 2) and `test_check_pools_names_short_pools` (Task 1).
2. **Trailing partial trial in the capture.** Must be dropped, never scored as a short trial. Pinned by `test_classify_drops_trailing_partial_trial` (Task 1).
3. **A condition pool runs out while building the batch.** Must fail with the condition named, never pad or wrap. Pinned by `test_build_raises_when_pool_exhausted` (Task 3).
4. **A sample's reservoir is exhausted or generation throws.** The sample is recorded with `error` set, `text` empty, and excluded from scoring and analysis, with the count reported. Pinned by `test_run_sample_records_exhausted_reservoir_as_error` (Task 4), `test_score_line_errors_and_empty_text_get_null_nll` (Task 5), `test_groups_excludes_null_and_counts` (Task 6).
5. **Null calibration split.** CSPRNG samples sit at global indices 2, 6, 10, … (all even), so the split must use rank within the condition, not the global index. Pinned by `test_groups_splits_csprng_by_rank_parity` (Task 6).

---

### Task 1: Scaffold and the coherence-walk classifier

**Files:**
- Create: `spikes/coherence/requirements.txt`
- Create: `spikes/coherence/walk.py`
- Create: `spikes/coherence/classify.py`
- Create: `spikes/coherence/tests/test_walk.py`
- Create: `spikes/coherence/tests/test_classify.py`

**Interfaces:**
- Produces: `walk.trial_z(trial: bytes) -> float`, `walk.classify(data: bytes, trial_bytes: int = 5120) -> tuple[bytes, bytes, list[float]]` returning `(anomaly, ordinary, sigmas)`; constants `walk.Z95`, `walk.TRIAL_BYTES`; `classify.check_pools(anomaly_len: int, ordinary_len: int, min_bytes: int) -> list[str]`.
- CLI: `classify.py <capture.bin> <outdir> [--min-bytes 5120000]` writes `anomaly.bin`, `ordinary.bin`, `classify.json`; exit 1 if a pool is short.

- [ ] **Step 1: Create the venv**

```bash
cd spikes/coherence
cat > requirements.txt <<'REQ'
mlx-vlm==0.7.6
numpy
pyserial
torch
transformers
pytest
REQ
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
mkdir -p tests runs
cat > .gitignore <<'IGN'
runs/smoke/
runs/**/batch.jsonl
runs/**/*.log
IGN
```

- [ ] **Step 2: Write the failing tests**

`tests/test_walk.py`:

```python
import pytest

from walk import TRIAL_BYTES, Z95, classify, trial_z


def test_trial_z_all_zero_bytes():
    # 256 bytes = 2048 bits, zero ones: z = (0 - 1024) / sqrt(512)
    assert trial_z(bytes(256)) == pytest.approx(-1024 / 512 ** 0.5)


def test_trial_z_balanced_bytes_is_zero():
    assert trial_z(bytes([0b10101010]) * 256) == 0.0


def test_classify_all_zero_stream_is_all_anomalous():
    data = bytes(TRIAL_BYTES * 4)
    anomaly, ordinary, sigmas = classify(data)
    assert anomaly == data and ordinary == b""
    assert len(sigmas) == 4 and all(abs(s) >= Z95 for s in sigmas)


def test_classify_balanced_stream_is_all_ordinary():
    data = bytes([0b10101010]) * (TRIAL_BYTES * 3)
    anomaly, ordinary, sigmas = classify(data)
    assert anomaly == b"" and ordinary == data
    assert sigmas == [0.0, 0.0, 0.0]


def test_classify_drops_trailing_partial_trial():
    data = bytes([0b10101010]) * (TRIAL_BYTES + 100)
    anomaly, ordinary, sigmas = classify(data)
    assert len(anomaly) + len(ordinary) == TRIAL_BYTES
    assert len(sigmas) == 1


def test_classify_keeps_byte_order_within_each_pool():
    balanced = bytes([0b10101010]) * TRIAL_BYTES
    ones = bytes([0xFF]) * TRIAL_BYTES
    anomaly, ordinary, sigmas = classify(balanced + ones)
    assert ordinary == balanced and anomaly == ones
    assert sigmas[0] == 0.0 and sigmas[1] >= Z95


def test_classify_honours_custom_trial_size():
    _, _, sigmas = classify(bytes(1000), trial_bytes=100)
    assert len(sigmas) == 10
```

`tests/test_classify.py`:

```python
from classify import check_pools


def test_check_pools_names_short_pools():
    assert check_pools(10, 10, 100) == ["anomaly", "ordinary"]
    assert check_pools(100, 10, 100) == ["ordinary"]
    assert check_pools(100, 100, 100) == []
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests -q`
Expected: errors, `ModuleNotFoundError: No module named 'walk'`

- [ ] **Step 4: Write `walk.py`**

```python
"""Offline port of the app's coherence walk (src-tauri/src/stats.rs).

The app finalises a trial every 100 ms with at least 2048 bits; offline there
is no clock, so a trial is a fixed TRIAL_BYTES (what 100 ms delivers at the
TrueRNG's ~50 KB/s). z-score, walk, sigma and the 95% threshold are verbatim.
"""
import numpy as np

Z95 = 1.959_963_98
TRIAL_BYTES = 5120


def trial_z(trial: bytes) -> float:
    """Monobit z for one trial: mean n/2, variance n/4."""
    n = len(trial) * 8
    ones = int(np.unpackbits(np.frombuffer(trial, dtype=np.uint8)).sum())
    return (ones - n / 2) / (n / 4) ** 0.5


def classify(data: bytes, trial_bytes: int = TRIAL_BYTES):
    """Split `data` into (anomaly, ordinary, sigmas).

    Trial i's bytes are anomalous when |sigma| >= Z95 after trial i, where
    sigma = C_k / sqrt(k) and C_k is the running sum of z-scores. A trailing
    partial trial is dropped.
    """
    anomaly, ordinary, sigmas = bytearray(), bytearray(), []
    cum = 0.0
    for start in range(0, len(data) - trial_bytes + 1, trial_bytes):
        trial = data[start : start + trial_bytes]
        cum += trial_z(trial)
        k = len(sigmas) + 1
        sigma = cum / k ** 0.5
        sigmas.append(sigma)
        (anomaly if abs(sigma) >= Z95 else ordinary).extend(trial)
    return bytes(anomaly), bytes(ordinary), sigmas
```

- [ ] **Step 5: Write `classify.py`**

```python
"""Split a raw TrueRNG capture into anomaly and ordinary byte pools."""
import argparse
import json
import sys
from pathlib import Path

from walk import TRIAL_BYTES, Z95, classify


def check_pools(anomaly_len: int, ordinary_len: int, min_bytes: int) -> list[str]:
    """Names of the pools shorter than `min_bytes` (empty list means OK)."""
    pools = (("anomaly", anomaly_len), ("ordinary", ordinary_len))
    return [name for name, length in pools if length < min_bytes]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("capture")
    p.add_argument("outdir")
    p.add_argument("--min-bytes", type=int, default=100 * 51_200)
    args = p.parse_args()

    data = Path(args.capture).read_bytes()
    anomaly, ordinary, sigmas = classify(data)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "anomaly.bin").write_bytes(anomaly)
    (out / "ordinary.bin").write_bytes(ordinary)

    n_anom = sum(abs(s) >= Z95 for s in sigmas)
    summary = {
        "capture_bytes": len(data),
        "trial_bytes": TRIAL_BYTES,
        "trials": len(sigmas),
        "anomalous_trials": n_anom,
        "anomalous_fraction": n_anom / len(sigmas) if sigmas else 0.0,
        "anomaly_bytes": len(anomaly),
        "ordinary_bytes": len(ordinary),
        "final_sigma": sigmas[-1] if sigmas else 0.0,
        "sigma_trace": [round(s, 3) for s in sigmas],
    }
    (out / "classify.json").write_text(json.dumps(summary, indent=1))
    print(f"{len(sigmas)} trials, {n_anom} anomalous ({summary['anomalous_fraction']:.1%}), "
          f"final sigma {summary['final_sigma']:+.2f}")
    print(f"anomaly {len(anomaly)} B, ordinary {len(ordinary)} B")

    short = check_pools(len(anomaly), len(ordinary), args.min_bytes)
    if short:
        print(f"ERROR: pool(s) below {args.min_bytes} bytes: {', '.join(short)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests -q`
Expected: `8 passed`

- [ ] **Step 7: Commit**

```bash
git add .
git commit -m "spike(coherence): scaffold + offline port of the coherence walk classifier"
```

---

### Task 2: TrueRNG capture that survives a dropout

**Files:**
- Create: `spikes/coherence/capture.py`
- Create: `spikes/coherence/tests/test_capture.py`

**Interfaces:**
- Produces: `capture.capture_to(read: Callable[[int], bytes], path, total: int, chunk: int = 65536) -> int` (bytes written; stops early on `OSError`, keeping the partial file).
- CLI: `capture.py <out.bin> [--mib 128] [--device /dev/cu.usbmodem112401]`; exit 0 only if the full total was captured.

- [ ] **Step 1: Write the failing tests**

`tests/test_capture.py`:

```python
from capture import capture_to


def test_capture_to_writes_requested_total(tmp_path):
    out = tmp_path / "c.bin"
    assert capture_to(lambda n: b"\x01" * n, out, 1000, chunk=300) == 1000
    assert out.stat().st_size == 1000


def test_capture_to_keeps_partial_file_on_dropout(tmp_path, capsys):
    calls = []

    def read(n):
        if len(calls) == 2:
            raise OSError("Device not configured")
        calls.append(n)
        return b"\x02" * n

    out = tmp_path / "c.bin"
    assert capture_to(read, out, 1000, chunk=300) == 600
    assert out.stat().st_size == 600
    assert "600 bytes" in capsys.readouterr().err
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_capture.py -q`
Expected: `ModuleNotFoundError: No module named 'capture'`

- [ ] **Step 3: Write `capture.py`**

```python
"""Record raw TrueRNG bytes to a file, incrementally, surviving a USB dropout."""
import argparse
import sys
import time

DEVICE = "/dev/cu.usbmodem112401"
CHUNK = 65536


def capture_to(read, path, total: int, chunk: int = CHUNK) -> int:
    """Write `total` bytes from `read(n)` to `path` in `chunk` pieces.

    On OSError (pyserial's SerialException is one) keep what was written,
    report, and return the byte count.
    """
    written = 0
    with open(path, "wb") as f:
        while written < total:
            n = min(chunk, total - written)
            try:
                buf = read(n)
            except OSError as e:
                print(f"capture stopped after {written} bytes: {e}", file=sys.stderr)
                break
            f.write(buf)
            f.flush()
            written += len(buf)
    return written


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--mib", type=int, default=128)
    p.add_argument("--device", default=DEVICE)
    args = p.parse_args()

    import serial

    ser = serial.Serial(args.device, 9600, timeout=3)
    ser.dtr = True  # TrueRNG streams once DTR is asserted
    ser.reset_input_buffer()

    def read(n):
        buf = bytearray()
        while len(buf) < n:
            c = ser.read(n - len(buf))
            if not c:
                raise OSError("TrueRNG read timed out (no data)")
            buf += c
        return bytes(buf)

    total = args.mib * 1024 * 1024
    t0 = time.perf_counter()
    got = capture_to(read, args.out, total)
    dt = time.perf_counter() - t0
    print(f"{got} of {total} bytes in {dt:.0f}s ({got / dt / 1000:.0f} KB/s)")
    sys.exit(0 if got == total else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests -q`
Expected: `10 passed`

- [ ] **Step 5: Commit**

```bash
git add capture.py tests/test_capture.py
git commit -m "spike(coherence): incremental TrueRNG capture"
```

---

### Task 3: Batch builder with the biased RNG and round-robin layout

**Files:**
- Create: `spikes/coherence/batch_build.py`
- Create: `spikes/coherence/tests/test_batch_build.py`

**Interfaces:**
- Produces: `batch_build.SAMPLE_BYTES = 51_200`, `batch_build.CONDITIONS = ("anomaly", "ordinary", "csprng", "biased")`, `batch_build.PROMPT`, `batch_build.bad_rng(n: int, seed: int = 0x12345678) -> bytes`, `batch_build.build(pools: dict[str, bytes], per_condition: int, sample_bytes: int = SAMPLE_BYTES) -> tuple[bytes, list[dict]]` where each dict is `{"index": int, "condition": str}`.
- CLI: `batch_build.py <anomaly.bin> <ordinary.bin> <outdir> [--per-condition 100]` writes `batch.bin` and `manifest.json` (`sample_bytes`, `prompt`, `temperature`, `max_tokens`, `samples`).

- [ ] **Step 1: Write the failing tests**

`tests/test_batch_build.py`:

```python
import pytest

from batch_build import CONDITIONS, bad_rng, build


def test_bad_rng_matches_the_apps_lcg():
    assert bad_rng(8) == bytes.fromhex("31071d142c091307")


def test_bad_rng_only_uses_six_bits():
    assert all(b < 64 for b in bad_rng(10_000))


def test_build_lays_samples_out_round_robin():
    pools = {"anomaly": b"A" * 8, "ordinary": b"B" * 8, "csprng": b"C" * 8, "biased": b"D" * 8}
    batch, samples = build(pools, per_condition=2, sample_bytes=4)
    assert batch == b"AAAABBBBCCCCDDDD" * 2
    assert [s["condition"] for s in samples] == list(CONDITIONS) * 2
    assert [s["index"] for s in samples] == list(range(8))


def test_build_consumes_each_pool_sequentially():
    pools = {"anomaly": b"12345678", "ordinary": b"B" * 8, "csprng": b"C" * 8, "biased": b"D" * 8}
    batch, _ = build(pools, per_condition=2, sample_bytes=4)
    assert batch[0:4] == b"1234" and batch[16:20] == b"5678"


def test_build_raises_when_pool_exhausted():
    pools = {"anomaly": b"A" * 4, "ordinary": b"B" * 8, "csprng": b"C" * 8, "biased": b"D" * 8}
    with pytest.raises(ValueError, match="anomaly pool exhausted at sample 1"):
        build(pools, per_condition=2, sample_bytes=4)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_batch_build.py -q`
Expected: `ModuleNotFoundError: No module named 'batch_build'`

- [ ] **Step 3: Write `batch_build.py`**

```python
"""Lay out the four conditions' entropy into one round-robin batch file.

Each sample owns a fixed SAMPLE_BYTES reservoir (worst case for 48 denoising
steps is 50 180 bytes). The generator reads only batch.bin + sample_bytes;
condition labels live in manifest.json.
"""
import argparse
import json
import os
from pathlib import Path

SAMPLE_BYTES = 51_200
CONDITIONS = ("anomaly", "ordinary", "csprng", "biased")
PROMPT = "Write a short passage of prose."


def bad_rng(n: int, seed: int = 0x12345678) -> bytes:
    """The app's deliberately biased source (src-tauri/src/source.rs run_software)."""
    x = seed
    out = bytearray(n)
    for i in range(n):
        x = (x * 1103515245 + 12345) & 0xFFFFFFFF
        out[i] = ((x >> 16) & 0xFF) & 0x3F
    return bytes(out)


def build(pools: dict, per_condition: int, sample_bytes: int = SAMPLE_BYTES):
    """Round-robin A, B, C, D, A, B, … Each pool is consumed front to back."""
    offsets = {c: 0 for c in CONDITIONS}
    out = bytearray()
    samples = []
    for i in range(per_condition):
        for c in CONDITIONS:
            start = offsets[c]
            chunk = pools[c][start : start + sample_bytes]
            if len(chunk) < sample_bytes:
                raise ValueError(
                    f"{c} pool exhausted at sample {i} (need {sample_bytes}, have {len(chunk)})"
                )
            offsets[c] = start + sample_bytes
            out += chunk
            samples.append({"index": len(samples), "condition": c})
    return bytes(out), samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("anomaly")
    p.add_argument("ordinary")
    p.add_argument("outdir")
    p.add_argument("--per-condition", type=int, default=100)
    args = p.parse_args()

    n = args.per_condition * SAMPLE_BYTES
    pools = {
        "anomaly": Path(args.anomaly).read_bytes(),
        "ordinary": Path(args.ordinary).read_bytes(),
        "csprng": os.urandom(n),
        "biased": bad_rng(n),
    }
    batch, samples = build(pools, args.per_condition)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "batch.bin").write_bytes(batch)
    manifest = {
        "sample_bytes": SAMPLE_BYTES,
        "prompt": PROMPT,
        "temperature": 0.0,
        "max_tokens": 256,
        "samples": samples,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(samples)} samples, {len(batch)} bytes -> {out / 'batch.bin'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests -q`
Expected: `15 passed`

- [ ] **Step 5: Commit**

```bash
git add batch_build.py tests/test_batch_build.py
git commit -m "spike(coherence): batch builder with biased-RNG control and round-robin layout"
```

---

### Task 4: Resumable generator over the entropy pool

**Files:**
- Create: `spikes/coherence/generate.py`
- Create: `spikes/coherence/tests/test_generate.py`
- Read only: `spikes/diffusiongemma/entropy.py` (`EntropyPool(replay=bytes)`, `entropy_driven(pool)`, `pool.consumed`, `pool.take(n)`), `spikes/diffusiongemma/sweep.py` (how `load`, `apply_chat_template`, `generate` are called)

**Interfaces:**
- Consumes: `manifest.json` and `batch.bin` from Task 3.
- Produces: `generate.slice_for(index: int, sample_bytes: int, data: bytes) -> bytes`, `generate.done_indices(jsonl_path) -> set[int]`, `generate.run_sample(slice_bytes: bytes, generate_text: Callable[[EntropyPool], str]) -> dict` with keys `text`, `consumed`, `elapsed_s`, `error`.
- CLI: `generate.py <batch.bin> <manifest.json> <out.jsonl> [--limit N]`; appends one JSON line per sample (`index` + the `run_sample` keys), skipping indices already in the file.

- [ ] **Step 1: Write the failing tests**

`tests/test_generate.py`:

```python
import json

import pytest

from generate import done_indices, run_sample, slice_for


def test_slice_for_returns_the_samples_reservoir():
    data = b"A" * 4 + b"B" * 4 + b"C" * 4
    assert slice_for(1, 4, data) == b"BBBB"


def test_slice_for_rejects_short_slice():
    with pytest.raises(ValueError, match="sample 3"):
        slice_for(3, 4, b"A" * 12)


def test_done_indices_reads_existing_lines(tmp_path):
    f = tmp_path / "out.jsonl"
    f.write_text(json.dumps({"index": 0}) + "\n" + json.dumps({"index": 2}) + "\n")
    assert done_indices(f) == {0, 2}


def test_done_indices_missing_file_is_empty(tmp_path):
    assert done_indices(tmp_path / "nope.jsonl") == set()


def test_run_sample_records_text_and_bytes_consumed():
    def fake_generate(pool):
        pool.take(10)
        return "hello"

    r = run_sample(b"\x00" * 64, fake_generate)
    assert r["text"] == "hello" and r["error"] is None
    assert r["consumed"] == 4 + 10  # 4-byte sampler seed + what fake_generate took
    assert r["elapsed_s"] >= 0


def test_run_sample_records_exhausted_reservoir_as_error():
    def fake_generate(pool):
        pool.take(100)
        return "never"

    r = run_sample(b"\x00" * 64, fake_generate)
    assert r["text"] == "" and "exhausted" in r["error"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_generate.py -q`
Expected: `ModuleNotFoundError: No module named 'generate'`

- [ ] **Step 3: Write `generate.py`**

```python
"""Generate one DiffusionGemma text per sample, each a pure function of its
reservoir bytes, appending to a JSONL file so a crash costs only a restart."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "diffusiongemma"))
from entropy import EntropyPool, entropy_driven  # noqa: E402

MODEL = "mlx-community/diffusiongemma-26B-A4B-it-4bit"


def slice_for(index: int, sample_bytes: int, data: bytes) -> bytes:
    s = data[index * sample_bytes : (index + 1) * sample_bytes]
    if len(s) != sample_bytes:
        raise ValueError(f"sample {index}: batch file holds {len(s)} bytes, need {sample_bytes}")
    return s


def done_indices(jsonl_path) -> set:
    path = Path(jsonl_path)
    if not path.exists():
        return set()
    return {json.loads(line)["index"] for line in path.read_text().splitlines() if line.strip()}


def run_sample(slice_bytes: bytes, generate_text) -> dict:
    """Run `generate_text(pool)` with every random draw coming from `slice_bytes`.

    An exhausted reservoir (or any RuntimeError from the model) is recorded as
    an error; the sample is never silently re-run on software noise.
    """
    pool = EntropyPool(replay=slice_bytes)
    t0 = time.perf_counter()
    try:
        with entropy_driven(pool):
            text = generate_text(pool)
        error = None
    except RuntimeError as e:
        text, error = "", str(e)
    return {
        "text": text,
        "consumed": len(pool.consumed),
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "error": error,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("batch")
    p.add_argument("manifest")
    p.add_argument("out")
    p.add_argument("--limit", type=int, default=None, help="stop after N new samples")
    args = p.parse_args()

    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    manifest = json.loads(Path(args.manifest).read_text())
    data = Path(args.batch).read_bytes()
    done = done_indices(args.out)
    todo = [s for s in manifest["samples"] if s["index"] not in done]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"{len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        return

    model, processor = load(MODEL)
    config = load_config(MODEL)
    prompt = apply_chat_template(processor, config, manifest["prompt"], num_images=0)

    def generate_text(_pool):
        return generate(
            model, processor, prompt,
            max_tokens=manifest["max_tokens"], temperature=manifest["temperature"],
        ).text

    with open(args.out, "a") as out:
        for s in todo:
            r = run_sample(slice_for(s["index"], manifest["sample_bytes"], data), generate_text)
            out.write(json.dumps({"index": s["index"], **r}) + "\n")
            out.flush()
            status = f"ERROR {r['error']}" if r["error"] else f"{r['consumed']} B, {r['elapsed_s']}s"
            print(f"sample {s['index']}: {status}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests -q`
Expected: `21 passed` (importing `entropy` pulls in mlx-vlm, so the first run takes a few seconds)

- [ ] **Step 5: Smoke the real model on two samples**

```bash
mkdir -p runs/smoke
head -c 102400 /dev/urandom > runs/smoke/anomaly.bin
head -c 102400 /dev/urandom > runs/smoke/ordinary.bin
.venv/bin/python batch_build.py runs/smoke/anomaly.bin runs/smoke/ordinary.bin runs/smoke --per-condition 2
.venv/bin/python generate.py runs/smoke/batch.bin runs/smoke/manifest.json runs/smoke/batch.jsonl --limit 2
.venv/bin/python generate.py runs/smoke/batch.bin runs/smoke/manifest.json runs/smoke/batch.jsonl --limit 1
wc -l runs/smoke/batch.jsonl
```

Expected: first run prints `0 done, 2 to go` then two `sample N: … B, …s` lines; second run prints `2 done, 1 to go` and adds one line; `wc -l` reports 3. Each `consumed` is between 11 000 and 51 200.

- [ ] **Step 6: Commit**

```bash
git add generate.py tests/test_generate.py
git commit -m "spike(coherence): resumable DiffusionGemma generator over per-sample entropy reservoirs"
```

---

### Task 5: Scorer (GPT-2 log-perplexity, distinct ratio, steps)

**Files:**
- Create: `spikes/coherence/score.py`
- Create: `spikes/coherence/tests/test_score.py`

**Interfaces:**
- Consumes: `batch.jsonl` lines from Task 4.
- Produces: `score.distinct_ratio(text: str) -> float`, `score.steps_from_consumed(consumed: int) -> int`, `score.score_line(line: dict, nll_fn: Callable[[str], float]) -> dict` (adds `gpt2_nll` or `None`, `distinct_ratio`, `steps`), `score.Gpt2Scorer` callable.
- CLI: `score.py <batch.jsonl> <scored.jsonl> [--limit N]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_score.py`:

```python
import pytest

from score import distinct_ratio, score_line, steps_from_consumed


def test_distinct_ratio():
    assert distinct_ratio("a b a c") == pytest.approx(0.75)
    assert distinct_ratio("") == 0.0


def test_steps_from_consumed_matches_spike_measurement():
    # 4-byte seed + initial canvas + 13 re-noised canvases, 1024 B each
    assert steps_from_consumed(14340) == 13


def test_score_line_adds_metrics():
    line = {"index": 0, "text": "the cat sat", "consumed": 14340, "error": None}
    out = score_line(line, lambda text: 2.5)
    assert out["gpt2_nll"] == 2.5
    assert out["distinct_ratio"] == pytest.approx(1.0)
    assert out["steps"] == 13
    assert out["index"] == 0 and out["text"] == "the cat sat"


def test_score_line_errors_and_empty_text_get_null_nll():
    failed = {"index": 1, "text": "", "consumed": 51200, "error": "replay bytes exhausted"}
    assert score_line(failed, lambda text: 1.0)["gpt2_nll"] is None
    blank = {"index": 2, "text": "   \n", "consumed": 2052, "error": None}
    assert score_line(blank, lambda text: 1.0)["gpt2_nll"] is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_score.py -q`
Expected: `ModuleNotFoundError: No module named 'score'`

- [ ] **Step 3: Write `score.py`**

```python
"""Score generated texts: GPT-2 log-perplexity (primary), distinct-word ratio
and denoising steps (secondary)."""
import argparse
import json
from pathlib import Path


def distinct_ratio(text: str) -> float:
    words = text.split()
    return len(set(words)) / len(words) if words else 0.0


def steps_from_consumed(consumed: int) -> int:
    """Denoising steps the confidence sampler ran: 4-byte seed, then one 1024-byte
    canvas initially and one per step."""
    return (consumed - 4) // 1024 - 1


def score_line(line: dict, nll_fn) -> dict:
    text = line.get("text") or ""
    usable = line.get("error") is None and text.strip() != ""
    return {
        **line,
        "gpt2_nll": float(nll_fn(text)) if usable else None,
        "distinct_ratio": distinct_ratio(text),
        "steps": steps_from_consumed(line["consumed"]) if line.get("consumed") else None,
    }


class Gpt2Scorer:
    """Mean per-token negative log-likelihood (nats) under gpt2 (124M)."""

    def __init__(self):
        import torch
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast

        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tok = GPT2TokenizerFast.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device).eval()

    def __call__(self, text: str) -> float:
        ids = self.tok(text, return_tensors="pt", truncation=True, max_length=1024).input_ids
        ids = ids.to(self.device)
        with self.torch.no_grad():
            return float(self.model(ids, labels=ids).loss)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("batch_jsonl")
    p.add_argument("out")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    lines = [json.loads(l) for l in Path(args.batch_jsonl).read_text().splitlines() if l.strip()]
    if args.limit is not None:
        lines = lines[: args.limit]
    scorer = Gpt2Scorer()
    with open(args.out, "w") as out:
        for line in lines:
            scored = score_line(line, scorer)
            out.write(json.dumps(scored) + "\n")
    n_null = sum(s is None for s in (json.loads(l)["gpt2_nll"] for l in open(args.out)))
    print(f"scored {len(lines)} samples on {scorer.device}; {n_null} without gpt2_nll")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests -q`
Expected: `25 passed`

- [ ] **Step 5: Smoke the GPT-2 scorer on the Task 4 smoke run**

```bash
.venv/bin/python score.py runs/smoke/batch.jsonl runs/smoke/scored.jsonl
.venv/bin/python -c "import json; [print(j['index'], j['gpt2_nll'], j['steps']) for j in map(json.loads, open('runs/smoke/scored.jsonl'))]"
```

Expected: first run downloads `gpt2` (~500 MB) once; prints `scored 3 samples on mps; 0 without gpt2_nll`; each `gpt2_nll` is a float between 2 and 7 and each `steps` is between 10 and 48.

- [ ] **Step 6: Commit**

```bash
git add score.py tests/test_score.py
git commit -m "spike(coherence): GPT-2 log-perplexity scorer with secondary metrics"
```

---

### Task 6: Analyzer with the pre-registered decision rule

**Files:**
- Create: `spikes/coherence/analyze.py`
- Create: `spikes/coherence/tests/test_analyze.py`

**Interfaces:**
- Consumes: `scored.jsonl` (Task 5) and `manifest.json` (Task 3).
- Produces: `analyze.cohen_d(a, b) -> float`, `analyze.perm_test(a, b, n_perm=20000, seed=0) -> float`, `analyze.groups(scored: list[dict], manifest: dict, metric: str) -> tuple[dict[str, np.ndarray], int]`, `analyze.compare(scored, manifest) -> list[dict]`, `analyze.verdict(results: list[dict]) -> str`, `analyze.report(results, verdict_line, excluded) -> str`.
- CLI: `analyze.py <scored.jsonl> <manifest.json> <report.md>`.

- [ ] **Step 1: Write the failing tests**

`tests/test_analyze.py`:

```python
import numpy as np
import pytest

from analyze import cohen_d, compare, groups, perm_test, verdict


def test_cohen_d_known_value():
    assert cohen_d(np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0])) == pytest.approx(-1.0)


def test_perm_test_detects_a_shift():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 1, 50), rng.normal(1, 1, 50)
    assert perm_test(a, b, n_perm=2000) < 0.001


def test_perm_test_is_quiet_on_same_distribution():
    rng = np.random.default_rng(2)
    a, b = rng.normal(0, 1, 50), rng.normal(0, 1, 50)
    assert perm_test(a, b, n_perm=2000) > 0.05


def _manifest(n_per):
    conds = ("anomaly", "ordinary", "csprng", "biased")
    return {"samples": [{"index": i, "condition": conds[i % 4]} for i in range(4 * n_per)]}


def test_groups_splits_csprng_by_rank_parity():
    m = _manifest(4)
    scored = [{"index": i, "gpt2_nll": float(i)} for i in range(16)]
    g, excluded = groups(scored, m, "gpt2_nll")
    assert excluded == 0
    # csprng sits at global indices 2, 6, 10, 14: split by rank, not global parity
    assert g["csprng_even"].tolist() == [2.0, 10.0]
    assert g["csprng_odd"].tolist() == [6.0, 14.0]


def test_groups_excludes_null_and_counts():
    m = _manifest(2)
    scored = [{"index": i, "gpt2_nll": None if i == 0 else 1.0} for i in range(8)]
    g, excluded = groups(scored, m, "gpt2_nll")
    assert excluded == 1 and len(g["anomaly"]) == 1


def _results(pc, nc, pr):
    rows = {("positive_control", pc), ("null_calibration", nc), ("primary", pr), ("anomaly_vs_csprng", 0.5)}
    return [{"comparison": c, "metric": "gpt2_nll", "p": p} for c, p in rows]


def test_verdict_inconclusive_when_positive_control_fails():
    assert verdict(_results(pc=0.2, nc=0.5, pr=0.001)).startswith("INCONCLUSIVE")


def test_verdict_candidate_effect():
    v = verdict(_results(pc=0.0001, nc=0.5, pr=0.004))
    assert v.startswith("CANDIDATE EFFECT") and "suspicion" not in v


def test_verdict_no_difference_with_null_warning():
    v = verdict(_results(pc=0.0001, nc=0.01, pr=0.4))
    assert v.startswith("NO DETECTABLE DIFFERENCE") and "suspicion" in v


def test_compare_covers_every_comparison_and_metric():
    m = _manifest(10)
    rng = np.random.default_rng(3)
    scored = [{"index": i, "gpt2_nll": float(rng.normal()), "steps": 12, "distinct_ratio": 0.7} for i in range(40)]
    results = compare(scored, m, n_perm=200)
    assert len(results) == 4 * 3
    assert {r["comparison"] for r in results} == {"positive_control", "null_calibration", "primary", "anomaly_vs_csprng"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_analyze.py -q`
Expected: `ModuleNotFoundError: No module named 'analyze'`

- [ ] **Step 3: Write `analyze.py`**

```python
"""Pre-registered comparisons (see the spec) over scored.jsonl, with a
permutation test, Cohen's d, and the decision rule."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ("gpt2_nll", "steps", "distinct_ratio")
COMPARISONS = (
    ("positive_control", "biased", "csprng"),
    ("null_calibration", "csprng_even", "csprng_odd"),
    ("primary", "anomaly", "ordinary"),
    ("anomaly_vs_csprng", "anomaly", "csprng"),
)
N_PERM = 20_000


def cohen_d(a, b) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def perm_test(a, b, n_perm: int = N_PERM, seed: int = 0) -> float:
    """Two-sided permutation p-value for the difference of means."""
    if len(a) == 0 or len(b) == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    observed = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    na = len(a)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        if abs(pooled[:na].mean() - pooled[na:].mean()) >= observed:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def groups(scored, manifest, metric):
    """Metric values per condition (plus csprng split by rank parity) and the
    number of samples excluded for a null metric."""
    condition = {s["index"]: s["condition"] for s in manifest["samples"]}
    g = defaultdict(list)
    excluded = 0
    for line in sorted(scored, key=lambda l: l["index"]):
        v = line.get(metric)
        if v is None:
            excluded += 1
            continue
        g[condition[line["index"]]].append(float(v))
    cs = g.get("csprng", [])
    g["csprng_even"], g["csprng_odd"] = cs[0::2], cs[1::2]
    return {k: np.array(v) for k, v in g.items()}, excluded


def compare(scored, manifest, n_perm: int = N_PERM):
    results = []
    for metric in METRICS:
        g, _ = groups(scored, manifest, metric)
        empty = np.array([])
        for name, x, y in COMPARISONS:
            a, b = g.get(x, empty), g.get(y, empty)
            results.append({
                "comparison": name, "metric": metric, "n1": len(a), "n2": len(b),
                "mean1": float(a.mean()) if len(a) else float("nan"),
                "mean2": float(b.mean()) if len(b) else float("nan"),
                "d": cohen_d(a, b), "p": perm_test(a, b, n_perm=n_perm),
            })
    return results


def verdict(results) -> str:
    p = {(r["comparison"], r["metric"]): r["p"] for r in results}
    pc, nc, pr = p[("positive_control", "gpt2_nll")], p[("null_calibration", "gpt2_nll")], p[("primary", "gpt2_nll")]
    if pc >= 0.001:
        return (f"INCONCLUSIVE: positive control failed (biased vs csprng gpt2_nll p={pc:.3f}); "
                "the metric cannot detect gross bias")
    note = "" if nc > 0.05 else f" (null calibration FAILED, p={nc:.3f}; treat with suspicion)"
    if pr < 0.01:
        return (f"CANDIDATE EFFECT: anomaly vs ordinary gpt2_nll p={pr:.4f}; "
                f"needs replication on a second capture{note}")
    return f"NO DETECTABLE DIFFERENCE: anomaly vs ordinary gpt2_nll p={pr:.3f}{note}"


def report(results, verdict_line: str, excluded: int) -> str:
    lines = [
        "# Coherence spike report", "",
        f"Samples excluded for a missing primary metric: {excluded}", "",
        "| comparison | metric | n1 | n2 | mean1 | mean2 | d | p |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['comparison']} | {r['metric']} | {r['n1']} | {r['n2']} | "
                     f"{r['mean1']:.4f} | {r['mean2']:.4f} | {r['d']:+.3f} | {r['p']:.4f} |")
    lines += ["", f"**Verdict:** {verdict_line}", ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scored")
    p.add_argument("manifest")
    p.add_argument("out")
    args = p.parse_args()
    scored = [json.loads(l) for l in Path(args.scored).read_text().splitlines() if l.strip()]
    manifest = json.loads(Path(args.manifest).read_text())
    results = compare(scored, manifest)
    _, excluded = groups(scored, manifest, "gpt2_nll")
    text = report(results, verdict(results), excluded)
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests -q`
Expected: `34 passed`

- [ ] **Step 5: Run the analyzer on the smoke run**

```bash
.venv/bin/python analyze.py runs/smoke/scored.jsonl runs/smoke/manifest.json runs/smoke/report.md
```

Expected: prints a 12-row table and a verdict line (`INCONCLUSIVE …` is expected with 3 samples; the point is that it runs end to end).

- [ ] **Step 6: Commit**

```bash
git add analyze.py tests/test_analyze.py
git commit -m "spike(coherence): permutation-test analyzer with the pre-registered decision rule"
```

---

### Task 7: Run the experiment, write it up, log it

**Files:**
- Create: `spikes/coherence/README.md`
- Create: `spikes/coherence/runs/2026-10-12/{classify.json,manifest.json,scored.jsonl,report.md}` (the `.bin` and `batch.jsonl` files stay local)
- Modify: `docs/experiments/LOG.md` (append one entry at the bottom; `../../docs/experiments/LOG.md` from the spike dir)

**Interfaces:**
- Consumes: every CLI from Tasks 1–6.

- [ ] **Step 1: Capture (about 45 minutes, unattended; TrueRNG must be plugged in)**

```bash
cd spikes/coherence
ls /dev/cu.usbmodem*          # confirm the device path; pass --device if it differs
mkdir -p runs/2026-10-12
.venv/bin/python capture.py runs/2026-10-12/capture.bin --mib 128
```

Expected: `134217728 of 134217728 bytes in ~2700s (~50 KB/s)`, exit 0. If the device drops out, the partial file is kept; continue to Step 2 and let `classify.py` decide whether it is enough.

- [ ] **Step 2: Classify**

```bash
.venv/bin/python classify.py runs/2026-10-12/capture.bin runs/2026-10-12
```

Expected: `~26214 trials, N anomalous (x%), final sigma ±y.yy` and both pools reported; exit 0. Record the anomalous fraction and final sigma for the write-up. If it exits 1 with a short pool, capture more (`--mib 64` to a second file, `cat` them together) and re-run.

- [ ] **Step 3: Build the batch**

```bash
.venv/bin/python batch_build.py runs/2026-10-12/anomaly.bin runs/2026-10-12/ordinary.bin runs/2026-10-12 --per-condition 100
```

Expected: `400 samples, 20480000 bytes -> runs/2026-10-12/batch.bin`

- [ ] **Step 4: Generate (about 3 hours, unattended; resumable)**

```bash
nohup .venv/bin/python generate.py runs/2026-10-12/batch.bin runs/2026-10-12/manifest.json runs/2026-10-12/batch.jsonl > runs/2026-10-12/generate.log 2>&1 &
```

Expected: `generate.log` gains one `sample N: … B, …s` line every 20–30 s; `wc -l runs/2026-10-12/batch.jsonl` reaches 400. If the process dies, re-run the same command; it skips finished indices. Count lines containing `ERROR` for the write-up.

- [ ] **Step 5: Score and analyse**

```bash
.venv/bin/python score.py runs/2026-10-12/batch.jsonl runs/2026-10-12/scored.jsonl
.venv/bin/python analyze.py runs/2026-10-12/scored.jsonl runs/2026-10-12/manifest.json runs/2026-10-12/report.md
```

Expected: `scored 400 samples on mps; K without gpt2_nll`, then the 12-row table and one verdict line.

- [ ] **Step 6: Write `README.md`**

```markdown
# Coherence spike

Does seeding DiffusionGemma from anomaly-bank bytes (TrueRNG bytes captured
while the coherence walk is outside its 95% envelope) measurably change the
text, compared with ordinary TrueRNG bytes? Pre-registered design:
`docs/superpowers/specs/2026-10-12-coherence-spike-design.md`.

## Pipeline

    .venv/bin/python capture.py runs/<id>/capture.bin --mib 128
    .venv/bin/python classify.py runs/<id>/capture.bin runs/<id>
    .venv/bin/python batch_build.py runs/<id>/anomaly.bin runs/<id>/ordinary.bin runs/<id>
    .venv/bin/python generate.py runs/<id>/batch.bin runs/<id>/manifest.json runs/<id>/batch.jsonl
    .venv/bin/python score.py runs/<id>/batch.jsonl runs/<id>/scored.jsonl
    .venv/bin/python analyze.py runs/<id>/scored.jsonl runs/<id>/manifest.json runs/<id>/report.md

Setup: `uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt`.
Tests: `.venv/bin/python -m pytest tests -q`.

## Run 2026-10-12

<fill from classify.json, generate.log and report.md: capture size and
anomalous fraction, final sigma, samples failed, the comparisons table, the
verdict line>
```

Replace the angle-bracket block with the actual numbers and the table from `report.md` before committing.

- [ ] **Step 7: Append the experiment-log entry**

Append to `docs/experiments/LOG.md`, filling every bracketed value from the run:

```markdown

---

## 2026-10-12 — Coherence spike: do anomaly-bank bytes measurably change DiffusionGemma's text?

**Question:** Is text generated from out-of-band TrueRNG bytes measurably
different from text generated from in-band bytes, and is the measurement
sensitive enough that a null result means something?

**Done:** Pre-registered A/B (spec
`docs/superpowers/specs/2026-10-12-coherence-spike-design.md`). One [128] MiB
TrueRNG capture split offline by a port of the app's coherence walk
([x.x]% of trials anomalous, final σ [±y.yy]); 100 samples per condition
(anomaly, ordinary, CSPRNG, biased LCG), round-robin, DiffusionGemma at
temperature 0 with the "short passage of prose" prompt, every random draw
from the sample's reservoir. Primary metric GPT-2 log-perplexity; secondary
denoising steps and distinct-word ratio; permutation test, 20 000 shuffles.
Code in `spikes/coherence/`, artefacts in `spikes/coherence/runs/2026-10-12/`.
[N] samples failed and were excluded.

**Result:** Positive control (biased vs CSPRNG) p=[…], d=[…]. Null
calibration p=[…]. Primary (anomaly vs ordinary) p=[…], d=[…]; anomaly vs
CSPRNG p=[…]. Secondary metrics: [one line]. Verdict line from the analyzer:
"[…]".

**Decision:** [What the verdict implies for the app's premise, and whether a
replication capture is warranted.]
```

- [ ] **Step 8: Commit the results and the log entry together**

```bash
git add README.md runs/2026-10-12/classify.json runs/2026-10-12/manifest.json runs/2026-10-12/scored.jsonl runs/2026-10-12/report.md ../../docs/experiments/LOG.md
git status --short   # confirm no .bin, batch.jsonl or .log is staged
git commit -m "spike(coherence): run 2026-10-12 results + experiment log entry"
```
