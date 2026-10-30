import type { CSSProperties } from "react";

/**
 * The per-item accents are chosen to read on the dark ground, and on the light
 * one several of them are not readable at all: `#00ff88` on white measures
 * 1.34:1 where WCAG AA asks for 4.5:1.
 *
 * The palette is not changing. Both values ride on the element as custom
 * properties and the theme decides which is painted, so the dark theme keeps
 * exactly the colour it had and the light theme gets the same hue carried down
 * to where it can be read.
 */

function srgbToLinear(channel: number): number {
  return channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
}

function luminance(hex: string): number {
  const h = hex.replace("#", "");
  const [r, g, b] = [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16) / 255);
  return 0.2126 * srgbToLinear(r) + 0.7152 * srgbToLinear(g) + 0.0722 * srgbToLinear(b);
}

function contrast(a: string, b: string): number {
  const [x, y] = [luminance(a), luminance(b)];
  return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05);
}

/**
 * The darkest ground an accent is asked to sit on in the light theme. Solving
 * against white would pass and still leave the tinted cards failing.
 */
const LIGHT_GROUND = "#f0f1f1";

/** The lightest of the dark theme's grounds, for the same reason. */
const DARK_GROUND = "#0a1a0f";

/**
 * Measured against the grounds each accent actually appears on. Every value is
 * the same hue and saturation, moved down in lightness only as far as it takes
 * to clear 4.5:1 — the before and after ratios are in `E2-contrast.md`.
 */
const ON_LIGHT: Record<string, string> = {
  "#00ff88": "#008245",
  "#00e5ff": "#007c8a",
  "#a78bfa": "#774cf7",
  "#ffd700": "#857000",
  "#ff9500": "#a25f00",
  "#ff6b6b": "#de0000",
  "#16a34a": "#11823b",
  "#ea580c": "#cd4d0b",
  "#5865f2": "#505df1",
  "#1d9bf0": "#0c75bc",
  "#e8eaed": "#697486",
  "#2563eb": "#1d51c4",
};

/**
 * Two of the accents are dark enough that they fail on the dark ground as well:
 * `#2563eb` measures 3.48:1 there. Every other accent — the whole neon set
 * included — clears 4.5:1 untouched, so this map has exactly the two entries
 * that need it and nothing else moves.
 */
const ON_DARK: Record<string, string> = {
  "#2563eb": "#457aee",
  "#5865f2": "#6672f3",
};

function darkenToAA(hex: string): string {
  const h = hex.replace("#", "");
  const rgb = [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
  for (let step = 0; step <= 100; step++) {
    const scale = 1 - step / 100;
    const candidate =
      "#" +
      rgb
        .map((c) => Math.round(c * scale).toString(16).padStart(2, "0"))
        .join("");
    if (contrast(candidate, LIGHT_GROUND) >= 4.5) return candidate;
  }
  return "#000000";
}

/** The same accent, dark enough to read on the light theme's grounds. */
export function accentOnLight(accent: string): string {
  const key = accent.toLowerCase();
  return ON_LIGHT[key] ?? darkenToAA(key);
}

function lightenToAA(hex: string): string {
  const h = hex.replace("#", "");
  const rgb = [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
  for (let step = 0; step <= 100; step++) {
    const t = step / 100;
    const candidate =
      "#" +
      rgb
        .map((c) => Math.round(c + (255 - c) * t).toString(16).padStart(2, "0"))
        .join("");
    if (contrast(candidate, DARK_GROUND) >= 4.5) return candidate;
  }
  return "#ffffff";
}

/** The same accent, light enough to read on the dark theme's grounds. */
export function accentOnDark(accent: string): string {
  const key = accent.toLowerCase();
  if (ON_DARK[key]) return ON_DARK[key];
  return contrast(key, DARK_GROUND) >= 4.5 ? accent : lightenToAA(key);
}

/**
 * Paint text in an accent, in whichever theme is showing.
 *
 * This only puts the two candidates on the element. The rule that picks between
 * them lives in `globals.css` and is matched against the element itself, which
 * is the part that has to be right: a custom property is resolved where it is
 * *declared*, so a `--accent-text` computed on `:root` — where neither
 * candidate is set — collapses to `currentColor`, and every accent silently
 * becomes whatever colour it inherited.
 */
export function accentTextStyle(accent: string, rest: CSSProperties = {}): CSSProperties {
  return {
    ...rest,
    ["--accent-on-dark" as string]: accentOnDark(accent),
    ["--accent-on-light" as string]: accentOnLight(accent),
  };
}
