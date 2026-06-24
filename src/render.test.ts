import { describe, it, expect } from "vitest";
import { envelopePoints, bandColor, verdictColor, crystallize, heatColor } from "./render";

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
});
