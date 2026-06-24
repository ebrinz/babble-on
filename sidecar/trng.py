"""TrueRNG -> Gaussian noise source for the babble-on diffusion spike.

Reads raw bytes from a TrueRNG (USB-CDC serial) device, or from a captured
file, and converts them to standard-normal samples via inverse-CDF transform
(uint32 -> uniform -> ndtri). Used to seed the diffusion initial latent z1 with
true hardware entropy in place of a PRNG, so the generated passage is a direct
function of physical randomness.
"""
import numpy as np
import torch


def read_trng_bytes(n, path="/dev/cu.usbmodem212201", baud=9600):
    """Read exactly `n` raw bytes from the TrueRNG (asserts DTR to start it)."""
    import serial
    ser = serial.Serial(path, baud, timeout=3)
    try:
        ser.dtr = True            # TrueRNG streams once DTR is asserted
        ser.reset_input_buffer()
        buf = bytearray()
        while len(buf) < n:
            chunk = ser.read(n - len(buf))
            if not chunk:
                raise IOError("TrueRNG read timed out (no data)")
            buf += chunk
        return bytes(buf)
    finally:
        ser.close()


def bytes_to_gaussians(raw, shape, dtype=torch.float64):
    """Map raw bytes -> N(0,1) tensor of `shape` (4 bytes per sample)."""
    n = int(np.prod(shape))
    need = n * 4
    if len(raw) < need:
        raise ValueError(f"need {need} bytes for {n} gaussians, got {len(raw)}")
    u32 = np.frombuffer(raw[:need], dtype="<u4").astype(np.float64)
    u = (u32 + 0.5) / 2.0 ** 32     # uniform in (0,1), endpoints avoided
    g = torch.from_numpy(u).to(dtype)
    return torch.special.ndtri(g).reshape(shape)   # inverse standard-normal CDF
