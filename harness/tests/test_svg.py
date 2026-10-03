import xml.etree.ElementTree as ET

from babble_harness import svg
from babble_harness.svg import nice_ticks


def parse(s: str):
    return ET.fromstring(s)


def test_wrap():
    from babble_harness.svg import wrap
    assert wrap("", 10) == [] and wrap("one two", 10) == ["one two"]
    assert wrap("alpha beta gamma delta", 11) == ["alpha beta", "gamma delta"]


def test_nice_ticks():
    assert nice_ticks(0, 1) == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert nice_ticks(0, 47)[-1] >= 47 and nice_ticks(0, 47)[0] == 0
    assert nice_ticks(3, 3) == [3.0, 3.2, 3.4, 3.6, 3.8, 4.0]


def test_figures_are_well_formed_and_transparent():
    figs = {
        "header": svg.header("babble-on", "sub"),
        "line": svg.line_chart({"in_band": [3, 2, 1], "out_band": [3, 2.5, 2]}, "t", bands={"in_band": ([2.9, 1.9, 0.9], [3.1, 2.1, 1.1])}),
        "cols": svg.columns(["0.1", "0.5"], [1, 4], "t", marker=(0.3, "obs"), cat_positions=[0.1, 0.5]),
        "heat": svg.heatmap([[0, 1], [2, None]], "t", commit_order=[[0, 1], [2, 0]]),
        "meter": svg.band_meter({"in-band": 0.9, "95%": 0.1}, "t"),
        "empty_heat": svg.heatmap([], "t"),
        "all_none_heat": svg.heatmap([[None, None]], "t", empty_note="nothing"),
        "collide": svg.line_chart({"a": [1, 1], "b": [1, 1], "c": [1, 1.5]}, "t"),
        "static_line": svg.line_chart({"a": [1, float("nan"), 2]}, "t", animate=False),
    }
    for name, s in figs.items():
        root = parse(s)
        assert root.tag.endswith("svg"), name
        assert "fill=\"#000" not in s and "fill=\"#fff" not in s and "fill=\"white\"" not in s and "fill=\"black\"" not in s, name
        assert 'role="img"' in s and "aria-label" in s
        # no opaque background rect covering the canvas
        assert "<rect x=\"0\" y=\"0\" width=\"" not in s, name
    assert "prefers-reduced-motion" in figs["line"]
    assert figs["line"].count("<path") == 2 and "animation: draw" in figs["line"]
    assert "animation: draw" not in figs["static_line"]
    assert figs["heat"].count("<rect") >= 3 + len(svg.BLUE_RAMP)  # 3 cells + ramp legend
    assert svg.CONDITION_COLORS["out_band"] in figs["line"] and svg.CONDITION_COLORS["in_band"] in figs["line"]
    assert "no cells to draw" in figs["all_none_heat"] and "nothing" in figs["all_none_heat"]
    # coinciding end points: one end label drawn, the legend still names all three
    for name in "abc":  # legend names every series exactly once
        assert figs["collide"].count(f'class="lg">{name}</text>') == 1
    assert figs["collide"].count('class="lg">a 1<') == 1 and 'class="lg">b 1<' not in figs["collide"]
    assert 'class="lg">c 1.5<' in figs["collide"]


def test_quotes_in_titles_stay_well_formed():
    s = svg.header('run "a"', 'dev "b"')
    parse(s)
    assert "&quot;" in s
    parse(svg.line_chart({"x": [1, 2]}, 'say "hi"'))


def test_condition_colors_are_fixed_and_mid_tone():
    # the validated dual-surface set: never cycle, never recolour on filter
    assert list(svg.CONDITION_COLORS) == ["in_band", "out_band", "prng", "remote"]
    assert svg.color_for("in_band", 3) == "#199e70" and svg.color_for("unknown", 1) == svg.SLOTS[1]
