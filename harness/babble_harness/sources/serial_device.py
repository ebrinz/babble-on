"""Record a TrueRNG-style USB-CDC serial device straight to ``.bbrec``
without the app (for long corpus-building runs on a headless box).

    python -m babble_harness.cli record-serial /dev/cu.usbmodem212201 out.bbrec --seconds 3600
"""
from __future__ import annotations

import time

from ..recording import RecordingHeader, RecordingWriter


def record_serial(path: str, device: str, seconds: float, baud: int = 9600,
                  tick_s: float = 0.05, log=print) -> int:
    import serial  # pyserial, optional dependency

    ser = serial.Serial(device, baud, timeout=0)
    ser.dtr = True  # TrueRNG streams once DTR is asserted
    ser.reset_input_buffer()
    header = RecordingHeader(source=device, started_at_ms=int(time.time() * 1000),
                             notes=f"direct serial capture @ {baud} baud, tick {tick_s}s")
    t0 = time.monotonic_ns()
    total = 0
    try:
        with RecordingWriter.create(path, header) as w:
            while (time.monotonic_ns() - t0) < seconds * 1e9:
                time.sleep(tick_s)
                chunk = ser.read(ser.in_waiting or 1)
                if chunk:
                    w.frame(time.monotonic_ns() - t0, chunk)
                    total += len(chunk)
            log(f"serial: {total} bytes in {w.frames} frames")
    finally:
        ser.close()
    return total
