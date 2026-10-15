// Headless check of the webview UI with the Tauri bridge mocked.
// Serves the built `dist/`, injects a fake `window.__TAURI_INTERNALS__` so
// `invoke`/`listen` from @tauri-apps/api work without Tauri, then drives the
// page: snapshot rendering, engine presets, record button states, the
// generate call's arguments, and the seeded/step/done message handling.
import { chromium } from "playwright";
import http from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";

const dist = path.resolve("dist");
const types = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml" };
const server = http.createServer(async (req, res) => {
  const file = path.join(dist, req.url === "/" ? "index.html" : req.url.split("?")[0]);
  try {
    res.writeHead(200, { "content-type": types[path.extname(file)] || "application/octet-stream" });
    res.end(await readFile(file));
  } catch {
    res.writeHead(404); res.end();
  }
});
await new Promise((r) => server.listen(0, r));
const url = `http://127.0.0.1:${server.address().port}/`;

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1100, height: 900 } });
page.on("pageerror", (e) => { console.error("page error:", e.message); process.exitCode = 1; });

await page.addInitScript(() => {
  const callbacks = new Map();
  const listeners = {};
  let next = 1;
  window.__calls = [];
  window.__TAURI_INTERNALS__ = {
    transformCallback(cb) { const id = next++; callbacks.set(id, cb); return id; },
    async invoke(cmd, args) {
      window.__calls.push({ cmd, args });
      if (cmd === "plugin:event|listen") { listeners[args.event] = args.handler; return next++; }
      if (cmd === "start_recording") return "/data/recordings/stream-1.bbrec";
      if (cmd === "export_seed") return "/data/exports/seed-1.seed.bin";
      return null;
    },
  };
  window.__emit = (event, payload) => { const h = listeners[event]; if (h) callbacks.get(h)({ event, payload, id: 0 }); };
});
await page.goto(url);
await page.waitForFunction(() => window.__calls.some((c) => c.cmd === "plugin:event|listen" && c.args.event === "snapshot"));

const snapshot = (over = {}) => page.evaluate((o) => window.__emit("snapshot", Object.assign({
  label: "simulator · xoshiro256**", status: "streaming", throughput_bps: 51200, total_bytes: 1048576,
  shannon: { value: 7.99, quality: 1, verdict: "pass" }, min_entropy: { value: 7.9, quality: 1, verdict: "pass" },
  monobit: { value: 0.5, quality: 1, verdict: "pass" }, chi_square: { value: 250, quality: 1, verdict: "pass" },
  serial_corr: { value: 0.001, quality: 1, verdict: "pass" }, histogram: new Array(256).fill(10), hist_expected: 10,
  recent: [1, 2, 3], coherence_sigma: 0.3, coherence_band: "in-band", trial_count: 50,
  walk: [[1, 0.5], [2, -0.2], [3, 0.4]], anomalies: [], bank_fill: 4096, bank_capacity: 524288,
  recording: null, recording_bytes: 0,
}, o)), over);

const assert = (cond, msg) => { if (!cond) { console.error("FAIL:", msg); process.exitCode = 1; } else console.log("ok:", msg); };
const text = (sel) => page.locator(sel).innerText();

await snapshot();
assert((await text("#label")).includes("simulator"), "snapshot renders the source label");
assert((await text("#bank-label")).includes("4.0/512 KiB"), "bank meter shows fill/capacity in KiB");
assert((await text("#metrics")).includes("PASS"), "metrics list renders verdicts");
assert((await text("#record")) === "● record", "record button idle");

// engine presets
const modes = async () => page.$$eval("#gen-mode option", (os) => os.map((o) => o.textContent));
assert((await modes()).some((m) => m.includes("192 steps")), "plaid presets by default");
await page.selectOption("#gen-engine", "gemma");
assert((await modes()).some((m) => m.includes("48 steps")) && !(await modes()).some((m) => m.includes("192")), "gemma presets after switching engine");
assert((await page.$eval(".gen-params", (e) => e.style.opacity)) === "0.4", "plaid-only knobs dimmed for gemma");

// record button
await page.click("#record");
assert((await page.evaluate(() => window.__calls.some((c) => c.cmd === "start_recording"))), "record click invokes start_recording");
await snapshot({ recording: "/data/recordings/stream-1.bbrec", recording_bytes: 1.5 * 1048576 });
assert((await text("#record")) === "■ rec 1.5 MiB", "record button shows running size");
await page.click("#record");
assert((await page.evaluate(() => window.__calls.some((c) => c.cmd === "stop_recording"))), "second click invokes stop_recording");
await snapshot();
assert((await text("#record")) === "● record", "record button returns to idle");

// generate with the gemma engine
await page.evaluate(() => window.__emit("diffusion", { type: "brief", bytes: 16, prompt: "x", picks: { subject: "stale", setting: "s", mood: "m", form: "f" } }));
await page.fill("#prompt", "The sea at night");
await page.click("#gen-btn");
assert((await text("#gen-brief")) === "", "clicking generate clears a stale brief line");
const gen = await page.evaluate(() => window.__calls.find((c) => c.cmd === "generate"));
assert(gen && gen.args.engine === "gemma" && gen.args.steps === 32 && gen.args.seqLen === 256 && gen.args.prompt === "The sea at night",
  `generate args: ${JSON.stringify(gen && gen.args)}`);
assert(await page.$eval("#gen-btn", (b) => b.disabled), "generate button disabled while running");
await page.evaluate(() => window.__emit("diffusion", { type: "loading" }));
assert((await text("#gen-status")).includes("sidecar"), "loading status names the sidecar for gemma");
await page.evaluate(() => window.__emit("diffusion", { type: "seeded", bank_fraction: 0.72, banked_bytes: 71528, tags: [{ at_secs: 10, peak_sigma: 3.2, band: "99%" }] }));
assert((await text("#gen-seed")).startsWith("seed: 72% anomaly bank"), "seed stamp rendered");
await page.evaluate(() => window.__emit("diffusion", { type: "brief", bytes: 16, prompt: "Write a letter about a dog, on a farm. The mood is calm. Keep it under 200 words.", picks: { subject: "a dog", setting: "on a farm", mood: "calm", form: "a letter" } }));
assert((await text("#gen-brief")) === "a letter · a dog · on a farm · calm", "brief line rendered from the brief event");
assert((await page.$$eval("#gen-brief .pick.from-bank", (t) => t.length)) === 4, "72% of 99344 bytes covers all four picks as bank-drawn");
await page.evaluate(() => window.__emit("diffusion", { type: "step", i: 3, total: 32, tokens: ["The", " sea", " at"] }));
assert((await text("#gen-status")).includes("step 3/32"), "step status");
assert((await page.$$eval("#gen-output .tok", (t) => t.length)) === 3, "tokens rendered as spans");
await page.evaluate(() => window.__emit("diffusion", { type: "step", i: 4, total: 32, tokens: ["<|channel>", "thought", "\n", "<channel|>", "The", " sea", "<eos>"] }));
assert((await page.$$eval("#gen-output .tok", (t) => t.length)) === 7 && (await text("#gen-output")) === "The sea", "live view blanks scaffolding in place");
await page.evaluate(() => window.__emit("diffusion", { type: "done", i: 12, total: 32, tokens: ["The", " sea", " at", " night"], elapsed: 4.2, stopped_early: true }));
assert((await text("#gen-status")).includes("stopped early at step 12/32"), "done status reports early stop");
assert((await text("#gen-brief")) === "a letter · a dog · on a farm · calm", "done with no brief leaves the line alone");
await page.evaluate(() => window.__emit("diffusion", { type: "done", i: 5, total: 5, tokens: ["x"], elapsed: 1, seed: "entropy", brief: { picks: { subject: "a cup of tea", setting: "in a small town", mood: "warm", form: "a postcard" }, prompt: "…" } }));
assert((await text("#gen-brief")) === "a letter · a dog · on a farm · calm", "a brief already shown is not replaced by done");
assert(!(await text("#gen-status")).includes("⚠"), "no warning when the seed came from the stream");
await page.evaluate(() => window.__emit("diffusion", { type: "done", i: 5, total: 5, tokens: ["x"], elapsed: 1, seed: "prng" }));
assert((await text("#gen-status")).includes("seeded from prng"), "a non-entropy seed is flagged");
assert(!(await page.$eval("#gen-btn", (b) => b.disabled)), "generate re-enabled after done");
await page.evaluate(() => window.__emit("diffusion", { type: "done", i: 5, total: 5, tokens: ["<|channel>", "thought", "\n", "<channel|>", "The", " sea", "<eos>", "<eos>"], elapsed: 1, seed: "entropy" }));
assert((await page.$$eval("#gen-output .tok", (t) => t.length)) === 2 && (await text("#gen-output")) === "The sea", "final render hides the channel header and eos padding");

// a done carrying a brief renders it when the brief event was missed
await page.click("#gen-btn"); // new generation, line cleared
await page.evaluate(() => window.__emit("diffusion", { type: "done", i: 5, total: 5, tokens: ["x"], elapsed: 1, seed: "entropy", brief: { picks: { subject: "a cup of tea", setting: "in a small town", mood: "warm", form: "a postcard" }, prompt: "…" } }));
assert((await text("#gen-brief")) === "a postcard · a cup of tea · in a small town · warm", "done renders the brief when the brief event was missed");

// export seed
await page.click("#export-seed");
await page.waitForFunction(() => document.getElementById("gen-status").textContent.includes("seed exported"));
assert((await text("#gen-status")).includes("seed-1.seed.bin"), "export seed reports the path");

await browser.close();
server.close();
console.log(process.exitCode ? "UI TEST FAILED" : "UI TEST PASSED");
