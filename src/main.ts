import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import { drawCoherence, drawHistogram, drawBitstream, verdictColor, bandColor, crystallize, heatColor, seedStamp, recLabel } from "./render";
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

  const cap = dto.bank_capacity || 0;
  const pct = cap ? (dto.bank_fill / cap) * 100 : 0;
  (document.getElementById("bank-fill") as HTMLElement).style.width = `${pct.toFixed(1)}%`;
  document.getElementById("bank-label")!.textContent =
    `bank ${(dto.bank_fill / 1024).toFixed(1)}/${(cap / 1024).toFixed(0)} KiB`;
  renderRecording(dto.recording ?? null, dto.recording_bytes ?? 0);
});

document.getElementById("source")!.addEventListener("change", (ev) =>
  invoke("start_source", { kind: (ev.target as HTMLSelectElement).value }));
document.getElementById("reset")!.addEventListener("click", () => invoke("reset"));
document.getElementById("pause")!.addEventListener("click", () => {
  paused = !paused; invoke("set_paused", { paused });
  document.getElementById("pause")!.textContent = paused ? "resume" : "pause";
});

// --- Stream recording (.bbrec for the harness) ---------------------------
const recBtn = document.getElementById("record") as HTMLButtonElement;
let recording: string | null = null;
recBtn.addEventListener("click", async () => {
  if (recording) {
    await invoke("stop_recording");
    return;
  }
  try {
    await invoke<string>("start_recording", {});
  } catch (err) {
    document.getElementById("status")!.textContent = `error: ${err}`;
  }
});
function renderRecording(path: string | null, bytes: number) {
  recording = path;
  recBtn.classList.toggle("rec", !!path);
  recBtn.textContent = path ? recLabel(bytes) : "● record";
  recBtn.title = path ? `recording to ${path} — click to stop` : "write the raw stream to a .bbrec recording for the experiment harness";
}

// --- Diffusion pane ---------------------------------------------------------
const MODES: Record<string, { steps: number; seqLen: number }> = {
  fast: { steps: 48, seqLen: 96 },
  balanced: { steps: 192, seqLen: 256 },
  quality: { steps: 384, seqLen: 256 },
  ultra: { steps: 768, seqLen: 256 },
};

const genBtn = document.getElementById("gen-btn") as HTMLButtonElement;
const genStatus = document.getElementById("gen-status")!;
const genOutput = document.getElementById("gen-output")!;
const promptInput = document.getElementById("prompt") as HTMLInputElement;
const tempInput = document.getElementById("gen-temp") as HTMLInputElement;
const noiseInput = document.getElementById("gen-noise") as HTMLInputElement;
const ddimInput = document.getElementById("gen-ddim") as HTMLInputElement;
const tempVal = document.getElementById("gen-temp-val")!;
const noiseVal = document.getElementById("gen-noise-val")!;
tempInput.addEventListener("input", () => (tempVal.textContent = (+tempInput.value).toFixed(2)));
noiseInput.addEventListener("input", () => (noiseVal.textContent = (+noiseInput.value).toFixed(2)));

const escape = (s: string) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

let crystalTokens: string[] = [];
let crystalHeat: number[] = [];

// Render the passage as per-position spans: changed-this-step tokens flash gold,
// and each token is colored by its heat (steps since it last changed) so the
// text visibly boils then crystallizes.
function renderCrystal(tokens: string[], flash: boolean) {
  const { changed, heat } = crystallize(crystalTokens, tokens, crystalHeat);
  crystalTokens = tokens;
  crystalHeat = heat;
  genOutput.innerHTML = tokens
    .map((tok, i) => {
      const cls = flash && changed[i] ? "tok flash" : "tok";
      return `<span class="${cls}" style="color:${heatColor(heat[i])}">${escape(tok)}</span>`;
    })
    .join("");
}

listen<any>("diffusion", (e) => {
  const m = e.payload;
  switch (m.type) {
    case "loading":
      genStatus.textContent = "loading model… (first run only)";
      break;
    case "step":
      genStatus.textContent = `crystallizing… step ${m.i}/${m.total}`;
      renderCrystal(m.tokens, true);
      break;
    case "done":
      genStatus.textContent = `done in ${m.elapsed}s`;
      genOutput.classList.remove("boiling");
      renderCrystal(m.tokens, false);
      genBtn.disabled = false;
      break;
    case "error":
      genStatus.textContent = `error: ${m.message}`;
      genOutput.classList.remove("boiling");
      genBtn.disabled = false;
      break;
    case "seeded":
      document.getElementById("gen-seed")!.textContent =
        seedStamp(m.bank_fraction, m.tags || [], fmtClock);
      break;
  }
});

document.getElementById("export-seed")!.addEventListener("click", async () => {
  genStatus.textContent = "exporting seed (spends the bank)…";
  try {
    const path = await invoke<string>("export_seed", {});
    genStatus.textContent = `seed exported → ${path}`;
  } catch (err) {
    genStatus.textContent = `error: ${err}`;
  }
});

genBtn.addEventListener("click", () => {
  const mode = (document.getElementById("gen-mode") as HTMLSelectElement).value;
  const { steps, seqLen } = MODES[mode] || MODES.balanced;
  const prompt = promptInput.value.trim() || undefined;
  genBtn.disabled = true;
  genStatus.textContent = "seeding from entropy…";
  genOutput.classList.add("boiling");
  genOutput.innerHTML = "";
  document.getElementById("gen-seed")!.textContent = "";
  crystalTokens = [];
  crystalHeat = [];
  invoke("generate", {
    steps, seqLen, nSamples: 1, prompt,
    temperature: +tempInput.value,
    noiseScale: +noiseInput.value,
    ddim: ddimInput.checked,
  }).catch((err) => {
    genStatus.textContent = `error: ${err}`;
    genOutput.classList.remove("boiling");
    genBtn.disabled = false;
  });
});
