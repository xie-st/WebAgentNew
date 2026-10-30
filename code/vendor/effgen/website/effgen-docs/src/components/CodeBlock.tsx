import React, { ReactNode, useMemo, useState } from 'react';
import { Copy, Check, MoreVertical } from 'lucide-react';
import { tokenize, type SynToken } from '@shared/syntaxHighlight';
import './CodeBlock.css';

/** Split the token stream into one array per line, keeping every character. */
function toLines(tokens: SynToken[]): SynToken[][] {
  const lines: SynToken[][] = [[]];
  for (const token of tokens) {
    const parts = token.text.split('\n');
    parts.forEach((part, i) => {
      if (i > 0) lines.push([]);
      if (part) lines[lines.length - 1].push({ cls: token.cls, text: part });
    });
  }
  return lines;
}

/** `"2,5-7"` -> the line numbers it names. */
function parseHighlight(spec: string | undefined): Set<number> {
  const out = new Set<number>();
  if (!spec) return out;
  for (const part of spec.split(',')) {
    const range = part.trim().split('-').map((n) => Number.parseInt(n, 10));
    if (range.length === 2 && Number.isFinite(range[0]) && Number.isFinite(range[1])) {
      for (let n = range[0]; n <= range[1]; n++) out.add(n);
    } else if (Number.isFinite(range[0])) {
      out.add(range[0]);
    }
  }
  return out;
}

export interface CodeBlockProps {
  /** The sample, exactly as it was run. */
  code: string;
  language?: string;
  /** The file this belongs in, when the reader is meant to save it. */
  filename?: string;
  showLineNumbers?: boolean;
  /** Lines to mark — `"3"`, `"3,7"` or `"3-9"`, counted from `startLine`. */
  highlight?: string;
  /** The number the first line carries. Set it on a block that continues one above. */
  startLine?: number;
  /**
   * This block carries on from the block immediately above it: it is not a
   * standalone file and will not run on its own.
   */
  continues?: boolean;
  /** A line under the block — where the sample came from, what it printed. */
  caption?: ReactNode;
}

export default function CodeBlock({
  code,
  language = 'python',
  filename,
  showLineNumbers = true,
  highlight,
  startLine = 1,
  continues = false,
  caption,
}: CodeBlockProps) {
  const [copied, setCopied] = useState(false);
  const lines = useMemo(() => toLines(tokenize(code, language)), [code, language]);
  const marked = useMemo(() => parseHighlight(highlight), [highlight]);

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // A browser that refuses the clipboard leaves the button as it was,
      // which is what happened: the text was not copied.
    }
  };

  return (
    <figure className={`code-block ${continues ? 'continues' : ''}`}>
      <div className="code-header">
        <div className="code-header-left">
          {continues ? (
            <span className="code-continues" title="Carries on from the block above">
              <MoreVertical size={14} aria-hidden="true" />
              <span>continues</span>
            </span>
          ) : (
            <span className="language-badge">{language}</span>
          )}
          {filename && <span className="filename">{filename}</span>}
        </div>
        <button
          type="button"
          className={`copy-button ${copied ? 'copied' : ''}`}
          onClick={copyToClipboard}
          aria-label={filename ? `Copy the contents of ${filename}` : 'Copy this code'}
        >
          {copied ? (
            <>
              <Check size={14} aria-hidden="true" />
              <span>Copied</span>
            </>
          ) : (
            <>
              <Copy size={14} aria-hidden="true" />
              <span>Copy</span>
            </>
          )}
        </button>
      </div>

      {/* The one scroll container. A wide line scrolls here and only here, so it
          never makes the page itself scroll sideways; `tabIndex` is what lets a
          keyboard reach the scroll. */}
      <div
        className="code-content"
        tabIndex={0}
        role="group"
        aria-label={`${language} code${filename ? `, ${filename}` : ''}`}
      >
        <pre className="code-lines"><code className={`language-${language}`}>
          {lines.map((tokens, i) => {
            const number = startLine + i;
            return (
              <span
                key={i}
                className={`code-line ${marked.has(number) ? 'marked' : ''}`}
                data-line={number}
              >
                {showLineNumbers && (
                  <span className="code-line-number" aria-hidden="true">
                    {number}
                  </span>
                )}
                <span className="code-line-text">
                  {tokens.map((token, j) =>
                    token.cls ? (
                      <span key={j} className={token.cls}>
                        {token.text}
                      </span>
                    ) : (
                      <React.Fragment key={j}>{token.text}</React.Fragment>
                    ),
                  )}
                  {/* Keeps an empty line the height of a full one. */}
                  {tokens.length === 0 && '\u200b'}
                </span>
              </span>
            );
          })}
        </code></pre>
      </div>

      {caption && <figcaption className="code-caption">{caption}</figcaption>}
    </figure>
  );
}

export interface CodeTab extends Omit<CodeBlockProps, 'continues' | 'caption'> {
  /** What the tab is called — `Python`, `CLI`, `curl`. */
  label: string;
}

/**
 * The same thing done several ways, one tab each.
 *
 * For a task a reader can carry out from Python or from the command line, not
 * for several different steps — those are separate blocks in the order they are
 * done.
 */
export function CodeTabs({ tabs, caption }: { tabs: CodeTab[]; caption?: ReactNode }) {
  const [active, setActive] = useState(0);
  const current = tabs[active] ?? tabs[0];
  if (!current) return null;

  return (
    <div className="code-tabs">
      <div className="code-tablist" role="tablist" aria-label="Ways to do this">
        {tabs.map((tab, i) => (
          <button
            key={tab.label}
            type="button"
            role="tab"
            id={`code-tab-${tab.label.replace(/\W+/g, '-')}`}
            aria-selected={i === active}
            aria-controls={`code-panel-${tab.label.replace(/\W+/g, '-')}`}
            tabIndex={i === active ? 0 : -1}
            className={`code-tab ${i === active ? 'active' : ''}`}
            onClick={() => setActive(i)}
            onKeyDown={(event) => {
              if (event.key === 'ArrowRight') setActive((i + 1) % tabs.length);
              if (event.key === 'ArrowLeft') setActive((i - 1 + tabs.length) % tabs.length);
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div
        role="tabpanel"
        id={`code-panel-${current.label.replace(/\W+/g, '-')}`}
        aria-labelledby={`code-tab-${current.label.replace(/\W+/g, '-')}`}
      >
        <CodeBlock {...current} caption={caption} />
      </div>
    </div>
  );
}
