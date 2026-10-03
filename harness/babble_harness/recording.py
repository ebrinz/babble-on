"""``.bbrec`` stream recordings (mirror of ``bbrec/src/recording.rs``).

    magic "BBREC001" | u32 LE header length | header JSON |
    frames: u64 LE t_ns | u32 LE len | bytes
"""
from __future__ import annotations

import json
import struct
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import BinaryIO, Iterator

MAGIC = b"BBREC001"
MAX_FRAME = 64 * 1024 * 1024  # a corrupt length field beyond this ends the file


@dataclass
class RecordingHeader:
    source: str
    started_at_ms: int
    trial_interval_ms: int = 100
    trial_min_bits: int = 2048
    notes: str = ""
    # coherence-walk state when recording started (0 = fresh walk / old file)
    walk_cum: float = 0.0
    walk_k: int = 0
    trial_ones: int = 0
    trial_bits: int = 0
    since_last_trial_ns: int = 0
    extra: dict = field(default_factory=dict)  # header keys this version does not know

    @classmethod
    def from_json(cls, d: dict) -> "RecordingHeader":
        known = {f.name for f in fields(cls)} - {"extra"}
        return cls(**{k: v for k, v in d.items() if k in known}, extra={k: v for k, v in d.items() if k not in known})

    def to_json(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "extra"}
        d.update(self.extra)
        return d


@dataclass
class Frame:
    t_ns: int
    data: bytes


class RecordingWriter:
    def __init__(self, f: BinaryIO, header: RecordingHeader):
        self.f = f
        h = json.dumps(header.to_json(), separators=(",", ":")).encode()
        f.write(MAGIC)
        f.write(struct.pack("<I", len(h)))
        f.write(h)
        self.frames = 0
        self.bytes = 0

    @classmethod
    def create(cls, path: str | Path, header: RecordingHeader) -> "RecordingWriter":
        return cls(open(path, "wb"), header)

    def frame(self, t_ns: int, data: bytes) -> None:
        if not data:
            return
        self.f.write(struct.pack("<QI", t_ns, len(data)))
        self.f.write(data)
        self.frames += 1
        self.bytes += len(data)

    def close(self) -> None:
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class RecordingReader:
    def __init__(self, f: BinaryIO):
        self.f = f
        if f.read(8) != MAGIC:
            raise ValueError("not a .bbrec file (bad magic)")
        raw_len = f.read(4)
        if len(raw_len) < 4:
            raise ValueError("truncated .bbrec header")
        (hlen,) = struct.unpack("<I", raw_len)
        raw = f.read(hlen)
        if len(raw) < hlen:
            raise ValueError("truncated .bbrec header")
        try:
            self.header = RecordingHeader.from_json(json.loads(raw))
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"bad .bbrec header: {e}") from e

    @classmethod
    def open(cls, path: str | Path) -> "RecordingReader":
        return cls(open(path, "rb"))

    def __iter__(self) -> Iterator[Frame]:
        while True:
            head = self.f.read(12)
            if len(head) < 12:
                return  # clean end, or a truncated tail: treat as end
            t_ns, n = struct.unpack("<QI", head)
            if n > MAX_FRAME:
                return  # corrupt length: treat as the end
            data = self.f.read(n)
            if len(data) < n:
                return
            yield Frame(t_ns, data)

    def frames(self) -> list[Frame]:
        return list(self)

    def close(self) -> None:
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_recording(path: str | Path) -> tuple[RecordingHeader, list[Frame]]:
    with RecordingReader.open(path) as r:
        return r.header, r.frames()
