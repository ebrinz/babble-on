"""Route every random draw in mlx-vlm's DiffusionGemma loop through TrueRNG.

mlx-vlm draws randomness in three places: the initial canvas, a re-noise of the
unaccepted positions on every step (both via `_diffusion_initialize_canvas`),
and the categorical sampler when temperature > 0 (`mx.random`). `entropy_driven`
replaces the canvas function with one that consumes bytes from an EntropyPool
and seeds `mx.random` from the pool, so the output is a pure function of the
bytes the pool hands out.
"""
import importlib
from contextlib import contextmanager

import mlx.core as mx
import numpy as np

# `mlx_vlm.generate` the attribute is the generate() function, which shadows the subpackage.
dg = importlib.import_module("mlx_vlm.generate.diffusion")

DEVICE = "/dev/cu.usbmodem112401"


class TrueRNG:
    """Raw bytes from a TrueRNG (USB-CDC serial). Streams once DTR is asserted."""

    def __init__(self, path=DEVICE, baud=9600):
        import serial

        self.ser = serial.Serial(path, baud, timeout=3)
        self.ser.dtr = True
        self.ser.reset_input_buffer()

    def read(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self.ser.read(n - len(buf))
            if not chunk:
                raise IOError("TrueRNG read timed out (no data)")
            buf += chunk
        return bytes(buf)


class EntropyPool:
    """Hands out bytes in order and records everything consumed, so a run can
    be replayed later from `pool.consumed` via `EntropyPool(replay=...)`."""

    def __init__(self, source=None, replay=None):
        self.source = source
        self.replay = replay
        self.pos = 0
        self.consumed = bytearray()

    def take(self, n):
        if self.replay is not None:
            if self.pos + n > len(self.replay):
                raise RuntimeError("replay bytes exhausted")
            out = self.replay[self.pos : self.pos + n]
            self.pos += n
        else:
            out = self.source.read(n)
        self.consumed += out
        return bytes(out)

    def u32(self):
        return int(np.frombuffer(self.take(4), dtype="<u4")[0])

    def tokens(self, shape, vocab_size):
        """4 bytes per token -> id uniform over the vocab (u32 * V >> 32)."""
        n = int(np.prod(shape))
        u32 = np.frombuffer(self.take(n * 4), dtype="<u4").astype(np.uint64)
        ids = (u32 * np.uint64(vocab_size)) >> np.uint64(32)
        return ids.astype(np.int32).reshape(shape)


@contextmanager
def entropy_driven(pool):
    """Within the block, the initial canvas, every per-step re-noise, and the
    sampler PRNG seed all come from `pool`."""

    def init_canvas(batch_size, canvas_length, vocab_size, dtype):
        return mx.array(pool.tokens((batch_size, canvas_length), vocab_size)).astype(dtype)

    orig = dg._diffusion_initialize_canvas
    dg._diffusion_initialize_canvas = init_canvas
    mx.random.seed(pool.u32())
    try:
        yield
    finally:
        dg._diffusion_initialize_canvas = orig
