"use client";

import { useState } from "react";
import { FiCheck, FiCopy } from "react-icons/fi";
import { highlightCode } from "@/components/syntaxHighlight";
import { accentTextStyle } from "@/components/accentText";

interface CodeSampleProps {
  /** The sample, exactly as it was run. */
  code: string;
  language?: "python" | "bash";
  /** Accent for the language chip and the hairline, from the section's palette. */
  accent?: string;
  /** What the sample printed when it was run. Verbatim, never edited to agree. */
  output?: string;
  /** Label over the output pane. */
  outputLabel?: string;
  className?: string;
}

/**
 * A code sample and, next to it, what that sample actually printed.
 *
 * The chrome is the one `QuickStart` already used for a code block — the
 * language chip, the copy button, the hairline in the section's accent — pulled
 * out here because the landing page now shows a sample in several places and
 * they should not drift apart.
 *
 * The `output` pane is the point of it. A sample with no output asks the reader
 * to take the claim on trust; a sample with the run's real stdout under it does
 * not. Nothing in an `output` is written by hand: it is pasted from the
 * transcript of the run that produced it.
 */
export default function CodeSample({
  code,
  language = "python",
  accent = "#00ff88",
  output,
  outputLabel = "output",
  className = "",
}: CodeSampleProps) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div
      className={`relative rounded-xl bg-gray-50 dark:bg-[#0a1a0f] border border-gray-300 dark:border-gray-800 overflow-hidden ${className}`}
    >
      <div
        className="absolute top-0 left-0 right-0 h-px"
        style={{ background: `linear-gradient(90deg, transparent, ${accent}60, transparent)` }}
      />

      <div className="flex items-center justify-between px-4 py-2.5 border-b border-gray-200 dark:border-gray-700/50">
        <span
          className="text-xs font-mono font-semibold px-2 py-0.5 rounded"
          style={accentTextStyle(accent, { background: `${accent}15`, border: `1px solid ${accent}30` })}
        >
          {language}
        </span>
        <button
          type="button"
          onClick={copy}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gray-200 dark:bg-gray-800 hover:bg-gray-300 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-all text-xs font-medium border border-gray-300 dark:border-gray-700"
        >
          {copied ? (
            <><FiCheck className="text-green-400" size={12} /> Copied</>
          ) : (
            <><FiCopy size={12} /> Copy</>
          )}
        </button>
      </div>

      <pre
        className="p-4 text-[13px] font-mono text-gray-800 dark:text-gray-200 overflow-x-auto leading-relaxed focus-visible:outline focus-visible:outline-2 focus-visible:outline-green-500 focus-visible:-outline-offset-2"
        tabIndex={0}
        role="group"
        aria-label={`${language} code`}
      >
        <code
          className="syntax-code"
          dangerouslySetInnerHTML={{ __html: highlightCode(code, language) }}
        />
      </pre>

      {output && (
        <div className="border-t border-gray-200 dark:border-gray-800 bg-white dark:bg-black/40">
          <div className="px-4 pt-3 text-[10px] font-mono uppercase tracking-widest text-gray-600 dark:text-gray-400">
            {outputLabel}
          </div>
          <pre className="px-4 pb-4 pt-1.5 text-[13px] font-mono text-gray-700 dark:text-gray-300 overflow-x-auto leading-relaxed whitespace-pre-wrap">
            {output}
          </pre>
        </div>
      )}
    </div>
  );
}
