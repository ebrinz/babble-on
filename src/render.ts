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

// --- Canvas painters (not unit-tested; verified manually in Task 8) ---
type Dto = any;

export function drawCoherence(ctx: CanvasRenderingContext2D, dto: Dto, w: number, h: number) {
  ctx.clearRect(0, 0, w, h);
  const walk: [number, number][] = dto.walk;
  const maxK = Math.max(50, dto.trial_count || 50);
  const yMax = 3.29052673 * Math.sqrt(maxK) * 1.2 || 10;
  const sx = (k: number) => (k / maxK) * w;
  const sy = (c: number) => h / 2 - (c / yMax) * (h / 2);
  // envelopes
  for (const [z, color] of [[1.95996398, GOLD], [2.5758293, ORANGE], [3.29052673, RED]] as [number,string][]) {
    for (const sign of [1, -1]) {
      ctx.strokeStyle = color; ctx.globalAlpha = 0.5; ctx.beginPath();
      envelopePoints(maxK, z).forEach(([k, c], i) => {
        const x = sx(k), y = sy(sign * c);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.stroke();
    }
  }
  ctx.globalAlpha = 1;
  // the walk
  ctx.strokeStyle = bandColor(dto.coherence_band); ctx.lineWidth = 1.5; ctx.beginPath();
  walk.forEach(([k, c], i) => { const x = sx(k), y = sy(c); i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); });
  ctx.stroke();
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
