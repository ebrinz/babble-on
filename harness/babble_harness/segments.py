"""Turn a labelled recording into fixed-size seeds per condition.

Conditions (the label on each ``EntropyTape``):
  in_band   bytes that arrived while the walk was inside the 95 % envelope
  out_band  bytes that arrived while it was outside — what the app's bank keeps
  prng      PCG64 control
  remote    a recording made from a remote QRNG (ANU), labelled the same way
            but reported under its own condition name because the point of
            that arm is locality, not band
"""
from __future__ import annotations

from dataclasses import dataclass

from .coherence import LabelledFrame, label_frames
from .noise import EntropyTape, prng_tape
from .recording import Frame, RecordingHeader


@dataclass
class Run:
    band_group: str  # "in_band" | "out_band"
    start: int
    end: int
    t_start_ns: int
    t_end_ns: int
    peak_sigma: float
    bands: set[str]

    @property
    def length(self) -> int:
        return self.end - self.start


def group_of(band: str) -> str:
    return "in_band" if band == "in-band" else "out_band"


def contiguous_runs(labelled: list[LabelledFrame]) -> list[Run]:
    """Merge adjacent frames with the same band group into byte runs."""
    runs: list[Run] = []
    for f in labelled:
        g = group_of(f.band)
        if runs and runs[-1].band_group == g and runs[-1].end == f.offset:
            r = runs[-1]
            r.end = f.offset + f.length
            r.t_end_ns = f.t_ns
            if abs(f.sigma) > abs(r.peak_sigma):
                r.peak_sigma = f.sigma
            r.bands.add(f.band)
        else:
            runs.append(Run(g, f.offset, f.offset + f.length, f.t_ns, f.t_ns, f.sigma, {f.band}))
    return runs


def cut_seeds(
    stream: bytes,
    runs: list[Run],
    seed_bytes: int,
    group: str,
    label: str | None = None,
    max_seeds: int | None = None,
    allow_concat: bool = False,
) -> list[EntropyTape]:
    """Cut ``seed_bytes``-sized tapes from the runs of one band group.

    By default a seed never spans two runs, so each seed is one uninterrupted
    stretch of same-condition bytes with one provenance record. With
    ``allow_concat`` leftover tails of consecutive same-group runs are joined
    (more seeds, weaker provenance — each seed then lists every run it drew
    from).
    """
    label = label or group
    tapes: list[EntropyTape] = []
    pending = bytearray()
    pending_runs: list[Run] = []

    def emit(data: bytes, srcs: list[Run]) -> None:
        tapes.append(EntropyTape(
            data=bytes(data), label=label,
            meta={
                "group": group,
                "runs": [{"start": r.start, "end": r.end, "t_start_ns": r.t_start_ns,
                          "t_end_ns": r.t_end_ns, "peak_sigma": r.peak_sigma,
                          "bands": sorted(r.bands)} for r in srcs],
            },
        ))

    for r in runs:
        if r.band_group != group:
            if not allow_concat:
                pending.clear(); pending_runs.clear()
            continue
        pos = r.start
        while r.end - pos >= seed_bytes and (max_seeds is None or len(tapes) < max_seeds):
            emit(stream[pos:pos + seed_bytes], [r])
            pos += seed_bytes
        tail = stream[pos:r.end]
        if allow_concat and tail:
            pending += tail
            pending_runs.append(r)
            while len(pending) >= seed_bytes and (max_seeds is None or len(tapes) < max_seeds):
                emit(bytes(pending[:seed_bytes]), list(pending_runs))
                del pending[:seed_bytes]
                pending_runs = pending_runs[-1:]
        elif not allow_concat:
            pending.clear(); pending_runs.clear()
        if max_seeds is not None and len(tapes) >= max_seeds:
            break
    return tapes


def seeds_from_recording(
    header: RecordingHeader,
    frames: list[Frame],
    seed_bytes: int,
    max_per_group: int | None = None,
    allow_concat: bool = False,
    label_prefix: str = "",
) -> dict[str, list[EntropyTape]]:
    labelled, _walk = label_frames(frames, header)
    stream = b"".join(f.data for f in frames)
    runs = contiguous_runs(labelled)
    return {
        g: cut_seeds(stream, runs, seed_bytes, g, label=label_prefix + g,
                     max_seeds=max_per_group, allow_concat=allow_concat)
        for g in ("in_band", "out_band")
    }


def prng_seeds(n: int, seed_bytes: int, base_seed: int = 0) -> list[EntropyTape]:
    return [prng_tape(base_seed + i, seed_bytes) for i in range(n)]
