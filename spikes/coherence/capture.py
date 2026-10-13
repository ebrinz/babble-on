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
