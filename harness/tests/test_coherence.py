import numpy as np

from babble_harness.coherence import Walk, band_of, duty_cycle, label_frames, popcount
from babble_harness.recording import Frame, RecordingHeader
from babble_harness.segments import contiguous_runs, cut_seeds, prng_seeds, seeds_from_recording


def test_band_classification_matches_rust():
    assert band_of(0.0) == "in-band"
    assert band_of(1.5) == "in-band"
    assert band_of(2.0) == "95%"
    assert band_of(-2.0) == "95%"
    assert band_of(2.7) == "99%"
    assert band_of(-3.5) == "99.9%"


def test_popcount():
    assert popcount(b"") == 0
    assert popcount(b"\xff\x00\x0f") == 12


def test_trial_requires_interval_and_bits():
    w = Walk()
    w.push(bytes(300))  # 2400 bits of zeros, strongly biased
    assert w.tick(50_000_000) is None  # interval not elapsed
    w2 = Walk()
    w2.push(bytes(10))
    assert w2.tick(200_000_000) is None  # not enough bits
    t = w.tick(100_000_000)
    assert t is not None and t.k == 1
    assert t.z < -40 and t.band == "99.9%"


def frames_from(stream: bytes, chunk: int = 512, dt_ns: int = 100_000_000) -> list[Frame]:
    return [Frame((i + 1) * dt_ns, stream[i * chunk:(i + 1) * chunk]) for i in range(len(stream) // chunk)]


def test_uniform_stream_stays_in_band_biased_escapes():
    rng = np.random.default_rng(3)
    good = frames_from(rng.bytes(512 * 200))
    lab, walk = label_frames(good)
    assert walk.k == 200
    dc = duty_cycle(lab)
    assert dc["in-band"] > 0.8, dc  # a healthy stream sits inside most of the time

    # 60 % ones per bit on average -> the walk bolts out within a few trials
    biased = bytes(np.where(rng.random(512 * 30) < 0.6, 0xFF, 0x00).astype(np.uint8))
    lab_b, walk_b = label_frames(frames_from(biased))
    assert walk_b.band == "99.9%"
    assert lab_b[-1].band == "99.9%"
    assert lab_b[0].offset == 0 and lab_b[1].offset == 512


def test_runs_and_seed_cutting_respect_run_boundaries():
    rng = np.random.default_rng(5)
    stream = rng.bytes(512 * 40)
    frames = frames_from(stream)
    labelled, _ = label_frames(frames)
    # Force a known labelling: frames 0-9 in, 10-19 out, 20-39 in.
    for i, f in enumerate(labelled):
        f.band = "95%" if 10 <= i < 20 else "in-band"
        f.sigma = 2.2 if 10 <= i < 20 else 0.3
    runs = contiguous_runs(labelled)
    assert [(r.band_group, r.start, r.end) for r in runs] == [
        ("in_band", 0, 5120), ("out_band", 5120, 10240), ("in_band", 10240, 20480)]
    assert runs[1].peak_sigma == 2.2 and runs[1].bands == {"95%"}

    out = cut_seeds(stream, runs, 2048, "out_band")
    assert len(out) == 2  # 5120 // 2048, tail of 1024 dropped
    assert out[0].data == stream[5120:7168] and out[0].label == "out_band"
    assert out[0].meta["runs"][0]["peak_sigma"] == 2.2
    inn = cut_seeds(stream, runs, 4096, "in_band")
    assert len(inn) == 1 + 2  # run 1: 5120//4096=1 ; run 3: 10240//4096=2
    inn_cat = cut_seeds(stream, runs, 4096, "in_band", allow_concat=True)
    assert len(inn_cat) == 3  # tails 1024 + 2048 = 3072 < 4096: no extra seed
    assert len(cut_seeds(stream, runs, 1024, "in_band", max_seeds=2)) == 2


def test_seeds_from_recording_end_to_end():
    rng = np.random.default_rng(9)
    frames = frames_from(rng.bytes(512 * 50))
    seeds = seeds_from_recording(RecordingHeader("sim", 0), frames, seed_bytes=1024, max_per_group=3)
    assert set(seeds) == {"in_band", "out_band"}
    assert all(len(t.data) == 1024 for g in seeds.values() for t in g)
    assert len(seeds["in_band"]) == 3


def test_prng_seeds():
    s = prng_seeds(3, 16, base_seed=100)
    assert len({t.data for t in s}) == 3 and all(t.label == "prng" for t in s)
