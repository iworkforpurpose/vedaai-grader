import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { contrastRatio, parseColour } from "./colour";

const CSS = readFileSync(join(__dirname, "../app/globals.css"), "utf8");

/**
 * Read one token's declared value out of the stylesheet.
 *
 * Reading the real file rather than a copy is the whole point: a contrast table
 * maintained by hand beside the palette is a table that goes stale the first time
 * someone nudges a colour.
 */
function token(name: string): string {
  const match = CSS.match(new RegExp(`\\${name}\\s*:\\s*([^;]+);`));
  if (!match?.[1]) throw new Error(`token ${name} is not defined`);
  return match[1].trim();
}

describe("colour maths", () => {
  it("agrees with known sRGB conversions", () => {
    expect(contrastRatio("#000000", "#ffffff")).toBeCloseTo(21, 5);
    expect(contrastRatio("#ffffff", "#ffffff")).toBeCloseTo(1, 5);
  });

  it("reads oklch and hex as the same colour", () => {
    const fromHex = parseColour("#ff5623");
    const fromOklch = parseColour("oklch(67.8% 0.213 36.1)");
    expect(fromOklch.r).toBeCloseTo(fromHex.r, 1);
    expect(fromOklch.g).toBeCloseTo(fromHex.g, 1);
    expect(fromOklch.b).toBeCloseTo(fromHex.b, 1);
  });
});

/**
 * Every pair here carries text below 18.66px or below 700 weight, which is what
 * WCAG calls normal — so the floor is 4.5:1 rather than 3:1.
 *
 * Five of them were below it when this was written, the primary button's own
 * white-on-orange label among them.
 */
describe("text contrast", () => {
  const AA = 4.5;

  it("white label on the primary button", () => {
    expect(contrastRatio("#ffffff", token("--accent-ink"))).toBeGreaterThanOrEqual(AA);
  });

  it("accent text on a white surface", () => {
    expect(contrastRatio(token("--accent-ink"), "#ffffff")).toBeGreaterThanOrEqual(AA);
  });

  it("the breadcrumb on a white surface", () => {
    expect(contrastRatio(token("--muted"), "#ffffff")).toBeGreaterThanOrEqual(AA);
  });

  it.each([
    ["answered", "--status-answered", "--status-answered-soft"],
    ["unanswered", "--status-unanswered", "--status-unanswered-soft"],
    ["review", "--status-review", "--status-review-soft"],
    ["not required", "--status-not-required", "--status-not-required-soft"],
    ["missing", "--status-missing", "--status-missing-soft"],
    ["neutral", "--status-neutral", "--status-neutral-soft"],
  ])("the %s chip", (_name, ink, fill) => {
    expect(contrastRatio(token(ink), token(fill))).toBeGreaterThanOrEqual(AA);
  });

  it("body text on the page ground", () => {
    expect(contrastRatio(token("--body-solid"), token("--page"))).toBeGreaterThanOrEqual(AA);
  });
});
