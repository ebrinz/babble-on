import { describe, it, expect } from "vitest";
import { envelopePoints, bandColor, verdictColor, crystallize, heatColor, seedStamp, recLabel, briefLine, finalTokens, liveTokens } from "./render";

describe("render helpers", () => {
  it("envelope is the parabola z*sqrt(k)", () => {
    const pts = envelopePoints(100, 1.95996398);
    const [k, c] = pts[pts.length - 1];
    expect(k).toBe(100);
    expect(c).toBeCloseTo(1.95996398 * Math.sqrt(100), 6);
  });
  it("maps bands and verdicts to palette colors", () => {
    expect(bandColor("99.9%")).toBe("#ff4d4d");
    expect(bandColor("in-band")).toBe("#6cf0d0");
    expect(verdictColor("pass")).toBe("#6cf0d0");
    expect(verdictColor("fail")).toBe("#ff4d4d");
  });
  it("crystallize flags changed positions and resets/increments heat", () => {
    const { changed, heat } = crystallize(["a", "b", "c"], ["a", "x", "c"], [3, 3, 3]);
    expect(changed).toEqual([false, true, false]);
    expect(heat).toEqual([4, 0, 4]);
  });
  it("heat color is gold when just changed, turquoise once settled", () => {
    expect(heatColor(0)).toBe("rgb(242,193,78)"); // GOLD
    expect(heatColor(6)).toBe("rgb(108,240,208)"); // TURQUOISE
  });
  it("seed stamp: live stream when nothing came from the bank", () => {
    expect(seedStamp(0, [], (s) => `T${s}`)).toBe("seed: live stream");
  });
  it("seed stamp: bank fraction with signed sigma per anomaly", () => {
    const fmt = (s: number) => `T${s}`;
    const tags = [
      { at_secs: 10, peak_sigma: 3.21, band: "99%" },
      { at_secs: 40, peak_sigma: -2.84, band: "95%" },
    ];
    expect(seedStamp(0.72, tags, fmt)).toBe(
      "seed: 72% anomaly bank — +3.2σ @ T10, −2.8σ @ T40",
    );
  });
  it("record label switches units at 1 MiB", () => {
    expect(recLabel(0)).toBe("■ rec 0 KiB");
    expect(recLabel(512 * 1024)).toBe("■ rec 512 KiB");
    expect(recLabel(1.5 * 1048576)).toBe("■ rec 1.5 MiB");
  });
  it("briefLine orders form · subject · setting · mood and marks bank-drawn slots", () => {
    const picks = { subject: "a dog", setting: "on a farm", mood: "calm", form: "a letter" };
    const none = briefLine(picks, 0);
    expect(none.text).toBe("a letter · a dog · on a farm · calm");
    expect(none.parts.map((p) => p.bank)).toEqual([false, false, false, false]);
    // 8 banked bytes cover subject and setting (slots 0 and 1 in list order)
    const half = briefLine(picks, 8);
    expect(half.parts.map((p) => [p.slot, p.bank])).toEqual([
      ["form", false], ["subject", true], ["setting", true], ["mood", false],
    ]);
    expect(briefLine(picks, 16).parts.every((p) => p.bank)).toBe(true);
    expect(briefLine(picks, 99_344).parts.every((p) => p.bank)).toBe(true);
    // 7 bytes: subject's 4 are covered, setting's are not
    expect(briefLine(picks, 7).parts.map((p) => p.bank)).toEqual([false, true, false, false]);
  });
  it("finalTokens drops the thought-channel header and everything from the first <eos>", () => {
    expect(finalTokens(["<|channel>", "thought", "\n", "<channel|>", "The", " sea", "<eos>", "<eos>"]))
      .toEqual(["The", " sea"]);
    expect(finalTokens(["The", " sea", " at", " night"])).toEqual(["The", " sea", " at", " night"]);
    // a close marker with no opener before it is ordinary text
    expect(finalTokens(["x", "<channel|>", "y"])).toEqual(["x", "<channel|>", "y"]);
    // markers glued to text in one token still count
    expect(finalTokens(["<|channel>thought", "\n", "<channel|>In", " the"])).toEqual(["In", " the"]);
  });
  it("liveTokens blanks the channel header and special markers but keeps every position", () => {
    const toks = ["<|channel>", "thought", "\n", "<channel|>", "The", " sea", "<eos>", "<pad>", "<turn|>", " <3"];
    expect(liveTokens(toks)).toEqual(["", "", "", "", "The", " sea", "", "", "", " <3"]);
    // a close marker with no opener is ordinary text; eos is still blanked
    expect(liveTokens(["x", "<channel|>", "<eos>"])).toEqual(["x", "", ""]);
    // markers glued to text keep the text
    expect(liveTokens(["<channel|>In", "now.<eos>"])).toEqual(["In", "now."]);
  });
});
