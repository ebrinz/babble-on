"""Dependency-free SVG figures for reports that render on light *and* dark
pages (GitHub, editors): transparent backgrounds, every colour chosen to pass
contrast on both #ffffff and #0d1117.

Palette (validated with the dataviz six-check validator on both surfaces,
adjacent pairlist; all-pairs caps at three series, so faceted forms use the
first three slots):

  condition slots  in_band #199e70  out_band #c98500  prng #3987e5  remote #d95926
  sequential ramp  blue 250..700 (heatmaps)
  status (bands)   in-band #0ca30c  95% #fab219  99% #ec835a  99.9% #d03b3b
  ink              muted #898781 (3.6:1 light / 5.3:1 dark), strong #7f8c88 (3.5 / 5.4),
                   accent teal #1f8a7a, accent gold #c98500

Animations are CSS (`@keyframes`) so `prefers-reduced-motion` disables them;
they run once and only where motion carries meaning (lines drawing on in
step order, heatmap cells committing in step order).
"""
from __future__ import annotations

import math
from xml.sax.saxutils import escape

INK = "#898781"
INK_STRONG = "#7f8c88"
TEAL = "#1f8a7a"
GOLD = "#c98500"
GRID = "#898781"

CONDITION_COLORS = {"in_band": "#199e70", "out_band": "#c98500", "prng": "#3987e5", "remote": "#d95926"}
SLOTS = ["#199e70", "#c98500", "#3987e5", "#d95926"]
STATUS = {"in-band": "#0ca30c", "95%": "#fab219", "99%": "#ec835a", "99.9%": "#d03b3b"}
BLUE_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

SANS = "system-ui, -apple-system, 'Segoe UI', sans-serif"
MONO = "ui-monospace, 'SF Mono', Menlo, Consolas, monospace"

STYLE = f"""
  .t {{ font: 600 15px {SANS}; fill: {INK_STRONG}; }}
  .s {{ font: 400 12px {SANS}; fill: {INK}; }}
  .ax {{ font: 400 11px {SANS}; fill: {INK}; font-variant-numeric: tabular-nums; }}
  .lg {{ font: 400 12px {SANS}; fill: {INK}; }}
  .grid {{ stroke: {GRID}; stroke-opacity: 0.22; stroke-width: 1; }}
  .axis {{ stroke: {GRID}; stroke-opacity: 0.45; stroke-width: 1; }}
  .line {{ fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }}
  @keyframes draw {{ to {{ stroke-dashoffset: 0; }} }}
  @keyframes appear {{ to {{ opacity: 1; }} }}
  @media (prefers-reduced-motion: reduce) {{
    .anim {{ animation: none !important; stroke-dashoffset: 0 !important; opacity: 1 !important; }}
  }}
"""


def color_for(name: str, i: int) -> str:
    return CONDITION_COLORS.get(name, SLOTS[i % len(SLOTS)])


def _doc(w: int, h: int, body: str, label: str, extra_style: str = "") -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="{escape(label)}">\n<style>{STYLE}{extra_style}</style>\n{body}\n</svg>\n')


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if not math.isfinite(lo) or not math.isfinite(hi):
        return [0.0, 1.0]
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / max(n, 1)
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if (hi - lo) / step <= n + 0.5:
            break
    start = math.floor(lo / step) * step
    end = math.ceil(hi / step - 1e-9) * step
    k = int(round((end - start) / step))
    return [round(start + i * step, 10) for i in range(k + 1)]


def wrap(text: str, width: int) -> list[str]:
    """Greedy word wrap for subtitles (SVG text does not wrap by itself)."""
    if not text:
        return []
    words, lines, cur = text.split(), [], ""
    for wd in words:
        if cur and len(cur) + 1 + len(wd) > width:
            lines.append(cur)
            cur = wd
        else:
            cur = f"{cur} {wd}" if cur else wd
    lines.append(cur)
    return lines


def fmt(v: float) -> str:
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if float(v).is_integer():
        return f"{int(v)}"
    return f"{v:.3g}"


# ---- header ------------------------------------------------------------------

def header(title: str, subtitle: str = "", w: int = 800, h: int = 150, tiles: int = 36, seed: int = 1) -> str:
    """Title banner: a row of canvas tiles that 'commit' from gold to teal in a
    pseudo-random order (the crystallisation the experiment studies), under
    the title. Transparent; mid-tone inks only."""
    rng = _lcg(seed)
    tw = (w - 80) / tiles
    cells = []
    order = sorted(range(tiles), key=lambda _: rng())
    for k, i in enumerate(order):
        x = 40 + i * tw
        delay = 0.4 + 2.4 * k / tiles
        cells.append(f'<rect x="{x:.1f}" y="{h - 34}" width="{tw - 3:.1f}" height="10" rx="2" fill="{GOLD}" opacity="0.55">'
                     f'<animate attributeName="fill" from="{GOLD}" to="{TEAL}" begin="{delay:.2f}s" dur="0.6s" fill="freeze"/>'
                     f'<animate attributeName="opacity" from="0.55" to="0.9" begin="{delay:.2f}s" dur="0.6s" fill="freeze"/></rect>')
    env = []
    for z, col, op in ((1.96, GOLD, 0.35), (2.576, "#ec835a", 0.22)):
        pts_u, pts_d = [], []
        for i in range(0, 41):
            x = 40 + (w - 80) * i / 40
            y = 22 * z * math.sqrt(i / 40)
            pts_u.append(f"{x:.1f},{h / 2 - 8 - y:.1f}")
            pts_d.append(f"{x:.1f},{h / 2 - 8 + y:.1f}")
        for pts in (pts_u, pts_d):
            env.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{col}" stroke-opacity="{op}" stroke-width="1"/>')
    body = "\n".join(env) + "\n" + "\n".join(cells)
    body += (f'\n<text x="{w / 2}" y="{h / 2 - 2}" text-anchor="middle" style="font: 600 30px {MONO}; fill: {TEAL}">{escape(title)}</text>')
    if subtitle:
        body += (f'\n<text x="{w / 2}" y="{h / 2 + 24}" text-anchor="middle" style="font: 400 13px {MONO}; fill: {INK}; letter-spacing: 2px">'
                 f'{escape(subtitle)}</text>')
    return _doc(w, h, body, f"{title} — {subtitle}")


def _lcg(seed: int):
    state = [seed * 2654435761 % 2**32 or 1]

    def nxt() -> float:
        state[0] = (1103515245 * state[0] + 12345) % 2**31
        return state[0] / 2**31
    return nxt


# ---- line chart ----------------------------------------------------------------

def line_chart(series: dict[str, list[float]], title: str, subtitle: str = "", x_label: str = "step",
               y_label: str = "", bands: dict[str, tuple[list[float], list[float]]] | None = None,
               w: int = 760, h: int = 320, animate: bool = True, y_min: float | None = None) -> str:
    """Multi-series line chart; `series[name]` is y per x index (x = 0..n-1).
    `bands[name] = (lo, hi)` draws a 10 % wash. Legend always; end labels
    direct. Lines draw on in x order when animated."""
    L, R, T, B = 56, 110, 62, 44
    pw, ph = w - L - R, h - T - B
    n = max((len(v) for v in series.values()), default=1)
    ys = [y for v in series.values() for y in v if y is not None and math.isfinite(y)]
    if bands:
        ys += [y for lo, hi in bands.values() for y in lo + hi if math.isfinite(y)]
    lo, hi = (min(ys), max(ys)) if ys else (0.0, 1.0)
    if y_min is not None:
        lo = min(lo, y_min)
    ticks = nice_ticks(lo, hi)
    lo, hi = ticks[0], ticks[-1]
    sx = lambda i: L + (pw * i / max(n - 1, 1))  # noqa: E731
    sy = lambda v: T + ph - (ph * (v - lo) / (hi - lo if hi > lo else 1))  # noqa: E731
    out = [f'<text x="{L}" y="22" class="t">{escape(title)}</text>']
    if subtitle:
        out.append(f'<text x="{L}" y="37" class="s">{escape(subtitle)}</text>')
    for t in ticks:
        y = sy(t)
        out.append(f'<line x1="{L}" x2="{L + pw}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        out.append(f'<text x="{L - 8}" y="{y + 4:.1f}" text-anchor="end" class="ax">{fmt(t)}</text>')
    xt = nice_ticks(0, n - 1, 6)
    for t in xt:
        if 0 <= t <= n - 1:
            out.append(f'<text x="{sx(t):.1f}" y="{T + ph + 16}" text-anchor="middle" class="ax">{fmt(t)}</text>')
    out.append(f'<line x1="{L}" x2="{L + pw}" y1="{T + ph}" y2="{T + ph}" class="axis"/>')
    out.append(f'<text x="{L + pw / 2}" y="{h - 8}" text-anchor="middle" class="s">{escape(x_label)}</text>')
    if y_label:
        out.append(f'<text transform="translate(14 {T + ph / 2}) rotate(-90)" text-anchor="middle" class="s">{escape(y_label)}</text>')
    legend_y = T - 10
    placed: list[float] = []  # y of end labels already drawn (avoid stacking collisions)
    for i, (name, vals) in enumerate(series.items()):
        col = color_for(name, i)
        if bands and name in bands:
            blo, bhi = bands[name]
            pts = [f"{sx(j):.1f},{sy(v):.1f}" for j, v in enumerate(bhi) if math.isfinite(v)]
            pts += [f"{sx(j):.1f},{sy(v):.1f}" for j, v in reversed(list(enumerate(blo))) if math.isfinite(v)]
            if pts:
                out.append(f'<polygon points="{" ".join(pts)}" fill="{col}" fill-opacity="0.10"/>')
        pts = [(sx(j), sy(v)) for j, v in enumerate(vals) if v is not None and math.isfinite(v)]
        if not pts:
            continue
        d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        length = sum(math.dist(pts[k], pts[k + 1]) for k in range(len(pts) - 1)) + 1
        anim = (f' class="line anim" stroke-dasharray="{length:.0f}" stroke-dashoffset="{length:.0f}" '
                f'style="animation: draw 1.4s ease-out {0.15 * i:.2f}s forwards"') if animate else ' class="line"'
        out.append(f'<path d="{d}" stroke="{col}"{anim}/>')
        ex, ey = pts[-1]
        out.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="4" fill="{col}"/>')
        # direct end label only when it does not collide with one already placed;
        # the legend carries identity otherwise (never stack colliding labels)
        if all(abs(ey - py) >= 13 for py in placed):
            out.append(f'<text x="{ex + 8:.1f}" y="{ey + 4:.1f}" class="lg">{escape(name)} {fmt(vals[-1])}</text>')
            placed.append(ey)
        # legend row under the title block
        lx = L + i * 96
        out.append(f'<line x1="{lx}" x2="{lx + 14}" y1="{legend_y - 4}" y2="{legend_y - 4}" stroke="{col}" stroke-width="2"/>')
        out.append(f'<text x="{lx + 18}" y="{legend_y}" class="lg">{escape(name)}</text>')
    return _doc(w, h, "\n".join(out), title)


# ---- columns / histogram ---------------------------------------------------------

def columns(categories: list[str], values: list[float], title: str, subtitle: str = "", color: str = SLOTS[2],
            marker: tuple[float, str] | None = None, x_label: str = "", y_label: str = "count",
            w: int = 760, h: int = 300, cat_positions: list[float] | None = None) -> str:
    """Single-series columns (4 px rounded caps, ≤ 24 px thick, 2 px gaps).
    `marker=(x_value, label)` draws a vertical reference (e.g. observed accuracy)
    when `cat_positions` gives each category's numeric x."""
    L, R, T, B = 56, 24, 44, 48
    pw, ph = w - L - R, h - T - B
    n = max(len(values), 1)
    hi = max(values) if values else 1.0
    ticks = nice_ticks(0, hi, 4)
    hi = ticks[-1] or 1.0
    slot = pw / n
    bw = min(24.0, slot - 2)
    out = [f'<text x="{L}" y="22" class="t">{escape(title)}</text>']
    if subtitle:
        out.append(f'<text x="{L}" y="37" class="s">{escape(subtitle)}</text>')
    for t in ticks:
        y = T + ph - ph * t / hi
        out.append(f'<line x1="{L}" x2="{L + pw}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        out.append(f'<text x="{L - 8}" y="{y + 4:.1f}" text-anchor="end" class="ax">{fmt(t)}</text>')
    for i, (c, v) in enumerate(zip(categories, values)):
        x = L + slot * i + (slot - bw) / 2
        bh = ph * v / hi
        y = T + ph - bh
        r = min(4.0, bh / 2)
        if bh > 0:
            d = (f"M{x:.1f},{T + ph} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} H{x + bw - r:.1f} "
                 f"Q{x + bw:.1f},{y:.1f} {x + bw:.1f},{y + r:.1f} V{T + ph} Z")
            out.append(f'<path d="{d}" fill="{color}" class="anim" opacity="0" style="animation: appear 0.5s ease-out {0.02 * i:.2f}s forwards"/>')
        step = max(1, n // 10)
        if i % step == 0:
            out.append(f'<text x="{x + bw / 2:.1f}" y="{T + ph + 16}" text-anchor="middle" class="ax">{escape(str(c))}</text>')
    out.append(f'<line x1="{L}" x2="{L + pw}" y1="{T + ph}" y2="{T + ph}" class="axis"/>')
    if marker and cat_positions:
        xv, label = marker
        xs = cat_positions
        if len(xs) >= 2 and xs[-1] != xs[0]:
            frac = (xv - xs[0]) / (xs[-1] - xs[0])
            mx = L + slot / 2 + frac * (slot * (n - 1))
            out.append(f'<line x1="{mx:.1f}" x2="{mx:.1f}" y1="{T}" y2="{T + ph}" stroke="{GOLD}" stroke-width="2"/>')
            out.append(f'<text x="{mx + 6:.1f}" y="{T + 12}" class="lg">{escape(label)}</text>')
    if x_label:
        out.append(f'<text x="{L + pw / 2}" y="{h - 8}" text-anchor="middle" class="s">{escape(x_label)}</text>')
    if y_label:
        out.append(f'<text transform="translate(14 {T + ph / 2}) rotate(-90)" text-anchor="middle" class="s">{escape(y_label)}</text>')
    return _doc(w, h, "\n".join(out), title)


# ---- heatmap ---------------------------------------------------------------------

def heatmap(matrix: list[list[float]], title: str, subtitle: str = "", x_label: str = "step",
            y_label: str = "position", w: int = 760, cell: int | None = None, animate: bool = True,
            vmin: float | None = None, vmax: float | None = None, legend: tuple[str, str] = ("low", "high"),
            commit_order: list[list[int]] | None = None, empty_note: str = "every value is undefined") -> str:
    """Sequential one-hue heatmap. With `commit_order[r][c]` (an integer step
    index per cell) cells appear in that order — the canvas crystallising."""
    rows, cols = len(matrix), max((len(r) for r in matrix), default=0)
    if rows == 0 or cols == 0 or not any(v is not None and math.isfinite(v) for r in matrix for v in r):
        body = (f'<text x="20" y="24" class="t">{escape(title)}</text>'
                f'<text x="20" y="44" class="s">{escape(subtitle or "")}</text>'
                f'<text x="20" y="66" class="s">no cells to draw — {escape(empty_note)}</text>')
        return _doc(w, 80, body, title)
    sub_lines = wrap(subtitle, 100)
    L, R, T, B = 56, 24, 50 + 14 * max(len(sub_lines) - 1, 0), 40
    pw = w - L - R
    cs = cell or max(2, min(14, int(pw / cols)))
    rs = max(2, min(cs, int(360 / rows)))
    h = T + rs * rows + B
    vals = [v for r in matrix for v in r if v is not None and math.isfinite(v)]
    lo = vmin if vmin is not None else (min(vals) if vals else 0.0)
    hi = vmax if vmax is not None else (max(vals) if vals else 1.0)
    out = [f'<text x="{L}" y="22" class="t">{escape(title)}</text>']
    for k, line in enumerate(sub_lines):
        out.append(f'<text x="{L}" y="{37 + 14 * k}" class="s">{escape(line)}</text>')
    maxo = max((o for r in (commit_order or []) for o in r), default=0) or 1
    for i, row in enumerate(matrix):
        for j, v in enumerate(row):
            if v is None or not math.isfinite(v):
                continue
            k = 0 if hi <= lo else int(round((v - lo) / (hi - lo) * (len(BLUE_RAMP) - 1)))
            col = BLUE_RAMP[max(0, min(len(BLUE_RAMP) - 1, k))]
            x, y = L + j * cs, T + i * rs
            anim = ""
            if animate and commit_order is not None:
                delay = 1.8 * commit_order[i][j] / maxo
                anim = f' class="anim" opacity="0" style="animation: appear 0.3s ease-out {delay:.2f}s forwards"'
            out.append(f'<rect x="{x}" y="{y}" width="{max(cs - 1, 1)}" height="{max(rs - 1, 1)}" fill="{col}"{anim}/>')
    for t in nice_ticks(0, cols - 1, 6):
        if 0 <= t <= cols - 1:
            out.append(f'<text x="{L + t * cs + cs / 2:.1f}" y="{T + rs * rows + 16}" text-anchor="middle" class="ax">{fmt(t)}</text>')
    for t in nice_ticks(0, rows - 1, 5):
        if 0 <= t <= rows - 1:
            out.append(f'<text x="{L - 8}" y="{T + t * rs + rs / 2 + 4:.1f}" text-anchor="end" class="ax">{fmt(t)}</text>')
    out.append(f'<text x="{L + pw / 2}" y="{h - 6}" text-anchor="middle" class="s">{escape(x_label)}</text>')
    out.append(f'<text transform="translate(14 {T + rs * rows / 2}) rotate(-90)" text-anchor="middle" class="s">{escape(y_label)}</text>')
    # ramp legend, right end of the title row
    lx = L + pw - 136
    for k, c in enumerate(BLUE_RAMP):
        out.append(f'<rect x="{lx + k * 10}" y="{14}" width="10" height="8" fill="{c}"/>')
    out.append(f'<text x="{lx - 6}" y="{22}" text-anchor="end" class="ax">{escape(legend[0])}</text>')
    out.append(f'<text x="{lx + 106}" y="{22}" class="ax">{escape(legend[1])}</text>')
    return _doc(w, h, "\n".join(out), title)


# ---- stacked meter -------------------------------------------------------------

def band_meter(fractions: dict[str, float], title: str, subtitle: str = "", w: int = 760, h: int = 110) -> str:
    """Duty-cycle meter: share of bytes per coherence band (status palette,
    labelled, 2 px gaps)."""
    L, R = 24, 24
    pw = w - L - R
    y, bh = 48, 18
    out = [f'<text x="{L}" y="22" class="t">{escape(title)}</text>']
    if subtitle:
        out.append(f'<text x="{L}" y="37" class="s">{escape(subtitle)}</text>')
    x = L
    order = ["in-band", "95%", "99%", "99.9%"]
    for i, name in enumerate(order):
        f = fractions.get(name, 0.0)
        wdt = pw * f
        if wdt > 0:
            out.append(f'<rect x="{x:.1f}" y="{y}" width="{max(wdt - 2, 0.5):.1f}" height="{bh}" rx="3" fill="{STATUS[name]}"/>')
        lx = L + i * 150
        out.append(f'<rect x="{lx}" y="{y + bh + 14}" width="10" height="10" rx="2" fill="{STATUS[name]}"/>')
        out.append(f'<text x="{lx + 15}" y="{y + bh + 23}" class="lg">{escape(name)} {100 * f:.1f}%</text>')
        x += wdt
    return _doc(w, h, "\n".join(out), title)
