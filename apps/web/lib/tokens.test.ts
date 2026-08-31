import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const WEB = join(__dirname, "..");

/**
 * Every custom property the app reads must be one the stylesheet defines.
 *
 * `var(--missing)` is not an error anywhere in the platform: the declaration is
 * simply dropped, so a square corner or an uncoloured chip is the only symptom.
 * Seven such properties shipped before this test existed, across the review model
 * and the debug route, and three components that nothing rendered were where most
 * of them hid — a dropped declaration in code that never runs looks exactly like
 * code that works.
 *
 * A `var()` carrying a fallback is exempt. That is the documented way to read a
 * property set from somewhere the stylesheet cannot see — an inline style, or a
 * parent component — and `--stagger` genuinely does that, from `QuestionCard`.
 */
function walk(dir: string, out: string[] = []): string[] {
  for (const entry of readdirSync(dir)) {
    if (entry === "node_modules" || entry === ".next" || entry.startsWith(".")) continue;
    // This file's own prose names properties that do not exist, on purpose.
    if (entry === "tokens.test.ts") continue;
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) walk(full, out);
    else if (/\.(tsx?|css)$/.test(entry)) out.push(full);
  }
  return out;
}

/**
 * Properties the stylesheet is right not to define.
 *
 * `next/font` generates a hashed family name at build time and puts it on `<html>`
 * as a custom property. The stylesheet reads it and must not define it — defining
 * it would override the real font with a name that resolves to nothing.
 */
const SET_ELSEWHERE = new Set([
  "--font-display",
  "--font-sans",
  "--font-mono-face",
]);

const CSS = readFileSync(join(WEB, "app/globals.css"), "utf8");

const defined = new Set(
  [...CSS.matchAll(/(--[a-z0-9-]+)\s*:/gi)].flatMap((m) =>
    m[1] ? [m[1].toLowerCase()] : [],
  ),
);

/** `var(--x)` with no comma before the closing paren, so no fallback. */
const USE = /var\(\s*(--[a-z0-9-]+)\s*\)/gi;

describe("custom properties", () => {
  it("are all defined before they are read", () => {
    const missing: string[] = [];

    for (const file of [
      ...walk(join(WEB, "components")),
      ...walk(join(WEB, "app")),
      ...walk(join(WEB, "lib")),
    ]) {
      const text = readFileSync(file, "utf8");
      for (const match of text.matchAll(USE)) {
        const name = match[1]?.toLowerCase();
        if (name && !defined.has(name) && !SET_ELSEWHERE.has(name)) {
          missing.push(`${file.replace(`${WEB}/`, "")}: ${name}`);
        }
      }
    }

    expect(missing).toEqual([]);
  });
});
