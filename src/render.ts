const TURQUOISE = "#6cf0d0";
const GOLD = "#f2c14e";
const ORANGE = "#ff9d3c";
const RED = "#ff4d4d";

export function bandColor(band: string): string {
  switch (band) {
    case "95%": return GOLD;
    case "99%": return ORANGE;
    case "99.9%": return RED;
    default: return TURQUOISE;
  }
}
export function verdictColor(v: string): string {
  switch (v) {
    case "pass": return TURQUOISE;
    case "warn": return GOLD;
    case "fail": return RED;
    default: return "#888";
  }
}
export function envelopePoints(maxK: number, z: number): [number, number][] {
  const pts: [number, number][] = [];
  for (let k = 1; k <= maxK; k++) pts.push([k, z * Math.sqrt(k)]);
  return pts;
}

// --- Crystallization (streaming diffusion) helpers --------------------------
// Diffusion refines every token position in place across steps, so we diff
// position-by-position and track a per-position "heat" = steps since it last
// changed. Just-changed tokens are hot (gold); long-settled ones cool to
// turquoise — the passage visibly boils, then crystallizes.

export function crystallize(
  prev: string[],
  next: string[],
  heat: number[],
): { changed: boolean[]; heat: number[] } {
  const changed = next.map((tok, i) => prev[i] !== tok);
  const newHeat = next.map((_, i) => (changed[i] ? 0 : (heat[i] ?? 99) + 1));
  return { changed, heat: newHeat };
}

function hexToRgb(h: string): [number, number, number] {
  const n = parseInt(h.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function lerpColor(a: string, b: string, t: number): string {
  const [ar, ag, ab] = hexToRgb(a);
  const [br, bg, bb] = hexToRgb(b);
  const m = (x: number, y: number) => Math.round(x + (y - x) * t);
  return `rgb(${m(ar, br)},${m(ag, bg)},${m(ab, bb)})`;
}

/// heat 0 (just changed) = gold; cools to turquoise by heat >= 6.
export function heatColor(h: number): string {
  return lerpColor(GOLD, TURQUOISE, Math.min(h, 6) / 6);
}

// --- Canvas painters (not unit-tested; verified manually in Task 8) ---
type Dto = any;

export function drawCoherence(ctx: CanvasRenderingContext2D, dto: Dto, w: number, h: number) {
  ctx.clearRect(0, 0, w, h);
  const walk: [number, number][] = dto.walk || [];

  // X-axis is anchored at the session origin: trial k=0 maps to x=0, so the
  // significance envelopes converge to a point at the left (the "horseshoe"
  // vertex) and the walk grows out of it. The walk retains enough history to
  // reach back to the vertex, so it fills the panel without floating right.
  const kLast = walk.length ? walk[walk.length - 1][0] : Math.max(50, dto.trial_count || 50);
  const maxK = Math.max(50, kLast);
  const sx = (k: number) => (k / maxK) * w;

  // Y-axis centered on the mean (C = 0); scaled so the 99.9% envelope just fits.
  const yMax = Math.max(4, 3.29052673 * Math.sqrt(maxK) * 1.08);
  const sy = (c: number) => h / 2 - (c / yMax) * (h / 2);

  // Significance envelopes ±z·√k drawn from the vertex (k=0) out to maxK.
  const STEPS = 96;
  for (const [z, color] of [[1.95996398, GOLD], [2.5758293, ORANGE], [3.29052673, RED]] as [number, string][]) {
    for (const sign of [1, -1]) {
      ctx.strokeStyle = color; ctx.globalAlpha = 0.5; ctx.lineWidth = 1; ctx.beginPath();
      for (let i = 0; i <= STEPS; i++) {
        const k = (maxK * i) / STEPS;
        const x = (i / STEPS) * w, y = sy(sign * z * Math.sqrt(k));
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
  }
  ctx.globalAlpha = 1;

  // Mean reference line at C = 0 — the expected value of the walk under pure
  // randomness. A healthy stream wanders symmetrically around it.
  ctx.save();
  ctx.strokeStyle = "#5a6b66"; ctx.lineWidth = 1; ctx.setLineDash([5, 5]);
  ctx.beginPath(); ctx.moveTo(0, sy(0)); ctx.lineTo(w, sy(0)); ctx.stroke();
  ctx.restore();
  ctx.fillStyle = "#5a6b66"; ctx.font = "11px ui-monospace, monospace";
  ctx.fillText("mean 0", 4, sy(0) - 4);

  // The cumulative-deviation walk, growing from the vertex, colored by band.
  if (walk.length >= 2) {
    ctx.strokeStyle = bandColor(dto.coherence_band); ctx.lineWidth = 1.5; ctx.beginPath();
    walk.forEach(([k, c], i) => { const x = sx(k), y = sy(c); i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); });
    ctx.stroke();
  }
}

export function drawHistogram(ctx: CanvasRenderingContext2D, dto: Dto, w: number, h: number) {
  ctx.clearRect(0, 0, w, h);
  const hist: number[] = dto.histogram;
  const max = Math.max(1, ...hist);
  const bw = w / 256;
  ctx.fillStyle = TURQUOISE;
  hist.forEach((c, i) => { const bh = (c / max) * h; ctx.fillRect(i * bw, h - bh, Math.max(1, bw - 0.5), bh); });
}

export function drawBitstream(ctx: CanvasRenderingContext2D, dto: Dto, w: number, h: number) {
  ctx.clearRect(0, 0, w, h);
  const bytes: number[] = dto.recent;
  const cols = Math.floor(w / 6);
  bytes.slice(-cols * 8).forEach((b, i) => {
    const x = (i % cols) * 6, y = Math.floor(i / cols) * 6;
    ctx.fillStyle = `hsl(${(b / 255) * 360}, 70%, 55%)`;
    ctx.fillRect(x, y, 5, 5);
  });
}
