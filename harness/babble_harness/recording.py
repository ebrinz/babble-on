"""``.bbrec`` stream recordings (mirror of ``bbrec/src/recording.rs``).

    magic "BBREC001" | u32 LE header length | header JSON |
    frames: u64 LE t_ns | u32 LE len | bytes
"""
from __future__ import annotations

import json
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

MAGIC = b"BBREC001"


@dataclass
class RecordingHeader:
    source: str
    started_at_ms: int
    trial_interval_ms: int = 100
    trial_min_bits: int = 2048
    notes: str = ""


@dataclass
class Frame:
    t_ns: int
    data: bytes


class RecordingWriter:
    def __init__(self, f: BinaryIO, header: RecordingHeader):
        self.f = f
        h = json.dumps(asdict(header), separators=(",", ":")).encode()
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
        (hlen,) = struct.unpack("<I", f.read(4))
        self.header = RecordingHeader(**json.loads(f.read(hlen)))

    @classmethod
    def open(cls, path: str | Path) -> "RecordingReader":
        return cls(open(path, "rb"))

    def __iter__(self) -> Iterator[Frame]:
        while True:
            head = self.f.read(12)
            if len(head) < 12:
                return  # clean end, or a truncated tail: treat as end
            t_ns, n = struct.unpack("<QI", head)
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
