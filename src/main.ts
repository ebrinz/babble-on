import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import { drawCoherence, drawHistogram, drawBitstream, verdictColor, bandColor } from "./render";
import "./style.css";

function canvas(id: string): [CanvasRenderingContext2D, number, number] {
  const el = document.getElementById(id) as HTMLCanvasElement;
  const w = el.clientWidth, h = el.clientHeight;
  if (el.width !== w || el.height !== h) { el.width = w; el.height = h; }
  return [el.getContext("2d")!, w, h];
}

let paused = false;

// Wall-clock anchor for the anomaly log: the backend timestamps excursions in
// seconds since its session start, so we peg that origin to a wall time. Re-peg
// when the stats reset (total_bytes drops back).
let sessionStartWall = Date.now();
let lastTotal = -1;

function fmtClock(atSecs: number): string {
  return new Date(sessionStartWall + atSecs * 1000).toLocaleTimeString([], { hour12: false });
}
function fmtElapsed(s: number): string {
  const m = Math.floor(s / 60), sec = Math.floor(s % 60);
  return `+${m}:${sec.toString().padStart(2, "0")}`;
}

function renderAnomalies(anomalies: any[]) {
  const el = document.getElementById("anomalies")!;
  if (!anomalies.length) {
    el.innerHTML = `<div class="empty">no band excursions yet — the walk is in-band</div>`;
    return;
  }
  el.innerHTML = anomalies.map((a) => {
    const up = a.peak_sigma >= 0;
    const col = bandColor(a.band);
    const dir = `${up ? "▲" : "▼"} ${up ? "+" : "−"}${Math.abs(a.peak_sigma).toFixed(2)}σ`;
    const dur = a.ongoing
      ? `<span class="live">● live ${a.duration_secs.toFixed(1)}s</span>`
      : `${a.duration_secs.toFixed(1)}s`;
    return `<div class="anom">` +
      `<span class="t" title="${fmtElapsed(a.at_secs)}">${fmtClock(a.at_secs)}</span>` +
      `<span class="dir" style="color:${col}">${dir}</span>` +
      `<span class="band" style="color:${col}">${a.band}</span>` +
      `<span class="dur">${dur}</span></div>`;
  }).join("");
}

listen<any>("snapshot", (e) => {
  const dto = e.payload;
  // Re-peg the wall-clock origin on first frame and whenever stats reset.
  if (lastTotal < 0 || dto.total_bytes < lastTotal) sessionStartWall = Date.now();
  lastTotal = dto.total_bytes;
  document.getElementById("label")!.textContent = dto.label;
  document.getElementById("status")!.textContent = dto.status;
  document.getElementById("rate")!.textContent =
    `${(dto.throughput_bps / 1024).toFixed(0)} KiB/s · ${(dto.total_bytes / 1048576).toFixed(1)} MiB`;

  const [cc, cw, ch] = canvas("coherence"); drawCoherence(cc, dto, cw, ch);
  const [hc, hw, hh] = canvas("histogram"); drawHistogram(hc, dto, hw, hh);
  const [bc, bw, bh] = canvas("bitstream"); drawBitstream(bc, dto, bw, bh);

  const rows = [
    ["Shannon", dto.shannon], ["Min-entropy", dto.min_entropy], ["Monobit", dto.monobit],
    ["Chi-square", dto.chi_square], ["Serial corr", dto.serial_corr],
  ] as [string, any][];
  document.getElementById("metrics")!.innerHTML = rows.map(([name, m]) =>
    `<li style="color:${verdictColor(m.verdict)}">${name}: ${m.value == null ? "…" : m.value.toFixed(4)} <b>${m.verdict.toUpperCase()}</b></li>`
  ).join("");

  renderAnomalies(dto.anomalies || []);
});

document.getElementById("source")!.addEventListener("change", (ev) =>
  invoke("start_source", { kind: (ev.target as HTMLSelectElement).value }));
document.getElementById("reset")!.addEventListener("click", () => invoke("reset"));
document.getElementById("pause")!.addEventListener("click", () => {
  paused = !paused; invoke("set_paused", { paused });
  document.getElementById("pause")!.textContent = paused ? "resume" : "pause";
});
