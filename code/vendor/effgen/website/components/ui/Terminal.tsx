"use client";

import { CSSProperties, useState } from "react";
import { FiCheck, FiCopy } from "react-icons/fi";

/** One run of text and the attributes the command printed it with. */
export type AnsiSpan = [attributes: string, text: string];

interface TerminalProps {
  /** The command that produced the output, without a leading `$`. */
  command?: string;
  /** Captured stdout, verbatim. Never retyped, never trimmed for effect. */
  output: string;
  /**
   * The same capture with its colour intact, one array of spans per line, as
   * `scripts/gen_capture_data.py` reads it out of the escape codes. Pass this
   * only where the colour is the subject — the four named themes. Everywhere
   * else the structural colouring below reads better in both site themes.
   */
  spans?: AnsiSpan[][];
  /** Ground the spans are drawn on. `light` is for the light terminal theme. */
  ground?: "dark" | "light";
  /** Window title. Defaults to the command's first word. */
  title?: string;
  /** Cap the height and scroll inside the frame. Rows of text, roughly. */
  maxLines?: number;
  /** How many lines of `output` to render. The rest stays in the copy button. */
  visibleLines?: number;
  className?: string;
}

// The eight ANSI colours and their bright pair, resolved once per ground so
// both are legible where they are drawn. These are the colours the terminal
// itself would have chosen; nothing is substituted for a different hue.
const ANSI_COLORS: Record<"dark" | "light", Record<string, string>> = {
  dark: {
    black: "#5b6b62", red: "#ff6b6b", green: "#00ff88", yellow: "#ffd166",
    blue: "#7aa2ff", magenta: "#e59bff", cyan: "#5fe3e0", white: "#d7e0da",
    "bright-black": "#8b9a92", "bright-red": "#ff9b9b", "bright-green": "#7dffbe",
    "bright-yellow": "#ffe49a", "bright-blue": "#a9c3ff", "bright-magenta": "#f2c4ff",
    "bright-cyan": "#9df3f1", "bright-white": "#ffffff",
  },
  light: {
    black: "#4b5563", red: "#b91c1c", green: "#047857", yellow: "#a16207",
    blue: "#1d4ed8", magenta: "#a21caf", cyan: "#0e7490", white: "#374151",
    "bright-black": "#6b7280", "bright-red": "#dc2626", "bright-green": "#059669",
    "bright-yellow": "#b45309", "bright-blue": "#2563eb", "bright-magenta": "#c026d3",
    "bright-cyan": "#0891b2", "bright-white": "#111827",
  },
};

function spanStyle(attributes: string, ground: "dark" | "light"): CSSProperties {
  const [flags, color] = attributes.split(":");
  const style: CSSProperties = {};
  if (flags.includes("b")) style.fontWeight = 700;
  if (flags.includes("i")) style.fontStyle = "italic";
  if (flags.includes("u")) style.textDecoration = "underline";
  // ANSI dim, but not below the point where the text stops being readable:
  // 0.62 composited the light ground's blue down to 2.98:1.
  if (flags.includes("d")) style.opacity = 0.9;
  if (color) style.color = ANSI_COLORS[ground][color];
  if (flags.includes("r")) {
    style.background = style.color ?? (ground === "dark" ? "#d7e0da" : "#111827");
    style.color = ground === "dark" ? "#040f0a" : "#ffffff";
  }
  return style;
}

// A captured terminal carries SGR escape sequences and the odd carriage return
// from a progress line. Neither survives being pasted into HTML, so they come
// out here rather than in the capture — the text itself stays byte-for-byte what
// the command printed.
// The escape character is part of the pattern. Without it the expression
// also matches ordinary bracketed text, and a Python sample's `list[str]`
// comes out of a capture as `tr]`.
const ANSI = /\x1b\[[0-9;?]*[A-Za-z]/g;

function clean(text: string): string {
  return text
    .replace(ANSI, "")
    // `\r\n` is a line ending, not a rewrite. Only a bare `\r` overwrites what
    // was on the line, which is how a progress line redraws itself.
    .replace(/\r\n/g, "\n")
    .split("\n")
    .map((line) => line.split("\r").pop() ?? line)
    .join("\n")
    .replace(/\s+$/, "");
}

// A captured session is not flat text: a diff has added and removed lines, a
// hunk header, and a result line at the end. Rendering all of it in one colour
// throws that structure away and leaves the reader to find it again. These are
// the shapes `effgen` actually prints, coloured from the palette the rest of
// the site uses — the text itself is untouched.
function lineClass(line: string, fixed: "dark" | "light" | null): string {
  // Every returned string is written out in full below rather than assembled
  // here: Tailwind scans the source for whole class names, so a class built at
  // runtime is purged from the stylesheet and silently does nothing.
  const pick = (light: string, dark: string, both: string) =>
    fixed === "dark" ? dark : fixed === "light" ? light : both;
  if (/^(---|\+\+\+) /.test(line)) return pick("text-gray-600", "text-gray-400", "text-gray-600 dark:text-gray-400");
  if (/^@@/.test(line)) return pick("text-violet-600", "text-violet-400", "text-violet-600 dark:text-violet-400");
  if (/^\+/.test(line)) return pick("text-green-700", "text-green-400", "text-green-700 dark:text-green-400");
  if (/^-/.test(line)) return pick("text-red-600", "text-red-400", "text-red-600 dark:text-red-400");
  if (/^\s*[✓✔]/.test(line)) return pick("text-green-700 font-semibold", "text-green-400 font-semibold", "text-green-700 dark:text-green-400 font-semibold");
  if (/^\s*[⚠✗✖]/.test(line)) return pick("text-orange-700", "text-orange-400", "text-orange-700 dark:text-orange-400");
  if (/^(new file|edit|deleted) /.test(line)) return pick("text-cyan-700 font-semibold", "text-cyan-400 font-semibold", "text-cyan-700 dark:text-cyan-400 font-semibold");
  if (/^[A-Za-z][^:]{0,40}:\s/.test(line) && !line.startsWith("http")) {
    return pick("text-gray-700", "text-gray-300", "text-gray-700 dark:text-gray-300");
  }
  return pick("text-gray-700", "text-gray-400", "text-gray-700 dark:text-gray-400");
}

/**
 * Real captured output from a real run, in a frame that looks like a terminal.
 *
 * The rule this component exists to enforce: **terminal output ships as text.**
 * Not a screenshot of text — text is selectable, searchable, copyable, readable
 * by a screen reader, legible at any zoom, and diffable when the command's
 * output changes. An image of a terminal is none of those.
 *
 * The output passed in is what the command printed. If a frame here disagrees
 * with what the command does today, the capture is stale and gets retaken; it is
 * never edited to agree.
 */
export default function Terminal({
  command,
  output,
  spans,
  ground = "dark",
  title,
  maxLines,
  visibleLines,
  className = "",
}: TerminalProps) {
  const [copied, setCopied] = useState(false);
  const text = clean(output);
  const lines = text.split("\n");
  const shown = visibleLines ? lines.slice(0, visibleLines) : lines;
  const label = title ?? command?.trim().split(/\s+/)[0] ?? "terminal";

  const copy = async () => {
    await navigator.clipboard.writeText(command ? `${command}\n${text}` : text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // `spans` means the frame paints a captured terminal's own ground, which does
  // not change with the site theme. Inside such a frame a `dark:` variant is
  // decided by the wrong thing — on a light frame in the dark theme it would
  // put light text on a light ground — so the chrome is told the ground.
  const fixedGround = spans ? ground : null;
  const chromeText =
    fixedGround === "dark"
      ? "text-gray-400"
      : fixedGround === "light"
        ? "text-gray-600"
        : "text-gray-600 dark:text-gray-400";

  return (
    <figure
      className={`rounded-2xl overflow-hidden border ${
        spans && ground === "dark"
          ? "border-green-500/25 bg-[#040f0a]"
          : spans
            ? "border-gray-300 bg-[#fbfaf7]"
            : "border-gray-200 dark:border-green-500/20 bg-gray-50 dark:bg-[#040f0a]"
      } ${className}`}
    >
      <div
        className={`flex items-center gap-2 px-4 py-2.5 border-b ${
          spans && ground === "dark"
            ? "border-green-500/15 bg-[#071a10]"
            : spans
              ? "border-gray-300 bg-[#f0eee9]"
              : "border-gray-200 dark:border-green-500/15 bg-gray-100 dark:bg-[#071a10]"
        }`}
      >
        <span className="flex gap-1.5" aria-hidden="true">
          <span className="w-3 h-3 rounded-full bg-red-400/70" />
          <span className="w-3 h-3 rounded-full bg-yellow-400/70" />
          <span className="w-3 h-3 rounded-full bg-green-400/70" />
        </span>
        <span className={`ml-2 text-xs font-mono ${chromeText} truncate`}>
          {label}
        </span>
        <button
          type="button"
          onClick={copy}
          className={`ml-auto flex items-center gap-1.5 text-xs font-mono ${chromeText} hover:text-green-600 dark:hover:text-green-400 transition-colors`}
        >
          {copied ? <FiCheck size={13} /> : <FiCopy size={13} />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>

      <div
        className="overflow-auto focus-visible:outline focus-visible:outline-2 focus-visible:outline-green-500 focus-visible:-outline-offset-2"
        style={maxLines ? { maxHeight: `${maxLines * 1.55}em` } : undefined}
        tabIndex={0}
        role="group"
        aria-label={`Captured output of ${label}`}
      >
        <pre
          className={`px-4 py-4 text-xs sm:text-[13px] font-mono leading-relaxed ${
            spans && ground === "dark"
              ? "text-[#d7e0da]"
              : spans
                ? "text-[#1f2937]"
                : "text-gray-800 dark:text-gray-300"
          }`}
        >
          {command && (
            <span
              className={`block font-semibold ${
                spans && ground === "light"
                  ? "text-green-700"
                  : spans
                    ? "text-green-400"
                    : "text-green-700 dark:text-green-400"
              }`}
            >
              <span className="select-none opacity-60">$ </span>
              {command}
            </span>
          )}
          {command && <span className="block h-2" aria-hidden="true" />}

          {spans
            ? spans.map((line, i) => (
                <span key={i} className="block">
                  {line.length === 0
                    ? "\u00a0"
                    : line.map(([attributes, run], j) => (
                        <span key={j} style={spanStyle(attributes, ground)}>
                          {run}
                        </span>
                      ))}
                </span>
              ))
            : shown.map((line, i) => (
                <span key={i} className={`block ${lineClass(line, fixedGround)}`}>
                  {line || "\u00a0"}
                </span>
              ))}
        </pre>
      </div>
    </figure>
  );
}
