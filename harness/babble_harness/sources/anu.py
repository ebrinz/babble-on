"""ANU QRNG bulk fetch -> ``.bbrec`` (the *remote* control arm).

Uses the key-based API (https://api.quantumnumbers.anu.edu.au). The free tier
is 100 requests/month; the paid tier is per-request, so fetch once in bulk and
keep the file. Each request is one frame, timestamped at receipt — the walk
then runs over arrival order, which is the only clock a remote source has.

    ANU_API_KEY=... python -m babble_harness.cli fetch-anu --bytes 2000000 out.bbrec
"""
from __future__ import annotations

import json
import os
import time
import urllib.request

from ..recording import RecordingHeader, RecordingWriter

ENDPOINT = "https://api.quantumnumbers.anu.edu.au"
MAX_LENGTH = 1024  # values per request
BLOCK_SIZE = 10  # hex16 block size: 1024 blocks × 10 bytes ≈ 10 KiB / request


def fetch_block(api_key: str, length: int = MAX_LENGTH, block_size: int = BLOCK_SIZE, timeout: float = 30.0) -> bytes:
    url = f"{ENDPOINT}?length={length}&type=hex16&size={block_size}"
    req = urllib.request.Request(url, headers={"x-api-key": api_key})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    if not body.get("success", False):
        raise RuntimeError(f"ANU API error: {body}")
    return b"".join(bytes.fromhex(h) for h in body["data"])


def fetch_to_recording(path: str, nbytes: int, api_key: str | None = None,
                       pause_s: float = 0.0, log=print) -> int:
    api_key = api_key or os.environ.get("ANU_API_KEY")
    if not api_key:
        raise SystemExit("set ANU_API_KEY (https://quantumnumbers.anu.edu.au)")
    header = RecordingHeader(source="anu", started_at_ms=int(time.time() * 1000),
                             notes=f"ANU QRNG bulk fetch, {nbytes} bytes requested")
    t0 = time.monotonic_ns()
    got = 0
    with RecordingWriter.create(path, header) as w:
        while got < nbytes:
            data = fetch_block(api_key)
            w.frame(time.monotonic_ns() - t0, data)
            got += len(data)
            log(f"anu: {got}/{nbytes} bytes ({w.frames} requests)")
            if pause_s:
                time.sleep(pause_s)
    return got
