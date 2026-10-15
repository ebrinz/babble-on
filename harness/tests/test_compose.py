import numpy as np
import pytest

from babble_harness.compose import BYTES_PER_BRIEF, SLOTS, compose


def test_slots_are_four_lists_of_32_unique_entries_in_order():
    assert list(SLOTS) == ["subject", "setting", "mood", "form"]
    for name, items in SLOTS.items():
        assert len(items) == 32, name
        assert len(set(items)) == 32, name
    assert BYTES_PER_BRIEF == 16


def test_same_bytes_same_brief_and_different_bytes_differ():
    rng = np.random.default_rng(0)
    a, b = rng.bytes(16), rng.bytes(16)
    assert compose(a) == compose(a)
    assert compose(a)[0] != compose(b)[0]


def test_extreme_bytes_pick_the_ends_of_every_list():
    _, lo = compose(bytes(16))
    _, hi = compose(bytes([255]) * 16)
    for name, items in SLOTS.items():
        assert lo[name] == items[0]
        assert hi[name] == items[-1]


def test_prompt_has_the_fixed_shape_and_every_pick():
    prompt, picks = compose(bytes(16))
    assert prompt == (f"Write {picks['form']} about {picks['subject']}, {picks['setting']}. "
                      f"The mood is {picks['mood']}. Keep it under 200 words.")


def test_only_the_first_16_bytes_matter_and_short_input_is_refused():
    raw = np.random.default_rng(1).bytes(40)
    assert compose(raw) == compose(raw[:16])
    with pytest.raises(ValueError, match="16"):
        compose(raw[:15])
