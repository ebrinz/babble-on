import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compose import BYTES_PER_PROMPT, SLOTS, compose  # noqa: E402


def test_lists_are_large_enough_and_unique():
    for name, items in SLOTS.items():
        assert len(items) >= 32, name
        assert len(set(items)) == len(items), name


def test_same_bytes_same_prompt_and_different_bytes_differ():
    rng = np.random.default_rng(0)
    a = rng.bytes(BYTES_PER_PROMPT)
    b = rng.bytes(BYTES_PER_PROMPT)
    assert compose(a) == compose(a)
    assert compose(a)[0] != compose(b)[0]


def test_low_and_high_bytes_pick_the_ends_of_each_list():
    _, lo = compose(bytes(BYTES_PER_PROMPT))
    _, hi = compose(bytes([255]) * BYTES_PER_PROMPT)
    for name, items in SLOTS.items():
        assert lo[name] == items[0] and hi[name] == items[-1]


def test_prompt_mentions_every_choice():
    prompt, picks = compose(np.random.default_rng(1).bytes(BYTES_PER_PROMPT))
    assert all(v in prompt for v in picks.values())


def test_short_input_is_refused():
    with pytest.raises(ValueError):
        compose(bytes(BYTES_PER_PROMPT - 1))
