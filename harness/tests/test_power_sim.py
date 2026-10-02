import json

from babble_harness.cli import main
from babble_harness.power import mw_power, probe_power, seeds_needed, stream_cost
from babble_harness.recording import read_recording


def test_mw_power_increases_with_effect_and_n():
    assert mw_power(0.0, 20, sims=200) < 0.15
    assert mw_power(1.0, 30, sims=200) > 0.9
    assert mw_power(0.5, 10, sims=200) < mw_power(0.5, 60, sims=200)


def test_probe_power_separable():
    assert probe_power(3.0, 15, dim=16, sims=8, n_perm=20) > 0.8
    assert probe_power(0.0, 15, dim=16, sims=8, n_perm=20) < 0.5


def test_seeds_needed_and_cost():
    n, curve = seeds_needed(mw_power, 1.2, grid=(6, 10, 20), sims=100)
    assert n in (6, 10, 20) and curve[n] >= 0.8
    c = stream_cost(10)
    assert c["seed_bytes"] == 99_328 and abs(c["stream_mib"] - 10 * 99_328 / 0.05 / 2**20) < 1e-6


def test_record_sim_and_label(tmp_path, capsys):
    rec = tmp_path / "sim.bbrec"
    assert main(["record-sim", str(rec), "--seconds", "20", "--rate", "20000", "--bias", "0.56", "--bias-from", "10"]) == 0
    header, frames = read_recording(rec)
    assert header.source.startswith("bad") and len(frames) == 400 and len(frames[0].data) == 1000
    assert main(["label", str(rec), "--canvas", "16", "--steps", "4"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["trials"] == 200 and out["final_band"] == "99.9%"
    assert out["duty_cycle"]["in-band"] > 0.3 and out["seeds_available"]["out_band"] > 0


def test_power_command_quick(capsys):
    assert main(["power", "--effects", "1.2", "--quick", "--dim", "8"]) == 0
    out = capsys.readouterr().out
    assert "| 1.2 |" in out and "MiB" in out
