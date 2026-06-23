import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import { drawCoherence, drawHistogram, drawBitstream, verdictColor } from "./render";
import "./style.css";

function canvas(id: string): [CanvasRenderingContext2D, number, number] {
  const el = document.getElementById(id) as HTMLCanvasElement;
  const w = el.clientWidth, h = el.clientHeight;
  if (el.width !== w || el.height !== h) { el.width = w; el.height = h; }
  return [el.getContext("2d")!, w, h];
}

let paused = false;

listen<any>("snapshot", (e) => {
  const dto = e.payload;
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
});

document.getElementById("source")!.addEventListener("change", (ev) =>
  invoke("start_source", { kind: (ev.target as HTMLSelectElement).value }));
document.getElementById("reset")!.addEventListener("click", () => invoke("reset"));
document.getElementById("pause")!.addEventListener("click", () => {
  paused = !paused; invoke("set_paused", { paused });
  document.getElementById("pause")!.textContent = paused ? "resume" : "pause";
});
