import { useState, type ReactNode } from 'react'
import { Check, Copy } from 'lucide-react'
import './Terminal.css'

interface TerminalProps {
  /** The command that produced the output, without a leading `$`. */
  command?: string
  /** Captured stdout, verbatim. Never retyped, never trimmed for effect. */
  output: string
  /** Window title. Defaults to the command's first word. */
  title?: string
  /** Cap the height and scroll inside the frame. Rows of text, roughly. */
  maxLines?: number
  /** Shown under the frame — the version, or the host a capture was taken on. */
  caption?: ReactNode
}

// A captured terminal carries SGR escape sequences and the occasional carriage
// return from a progress line. Neither survives being pasted into HTML, so they
// come out here rather than in the capture — the text itself stays byte-for-byte
// what the command printed.
const ANSI = new RegExp(String.fromCharCode(27) + '\\[[0-9;?]*[A-Za-z]', 'g')
const OVERWRITTEN = new RegExp('.*' + String.fromCharCode(13) + '(?!\\n)')

function clean(text: string): string {
  return text
    .replace(ANSI, '')
    .split('\n')
    .map((line) => line.replace(OVERWRITTEN, ''))
    .join('\n')
    .replace(/\s+$/, '')
}

/**
 * Real captured output from a real run, in a frame that looks like a terminal.
 *
 * The rule this component exists to enforce: **terminal output ships as text.**
 * Not a screenshot of text — text is selectable, searchable, copyable, readable
 * by a screen reader, legible at any zoom, and diffable when the command's
 * output changes. An image of a terminal is none of those.
 *
 * `CodeBlock` is for code a reader is meant to run. This is for what came back.
 */
export default function Terminal({
  command,
  output,
  title,
  maxLines,
  caption,
}: TerminalProps) {
  const [copied, setCopied] = useState(false)
  const text = clean(output)
  const label = title ?? command?.trim().split(/\s+/)[0] ?? 'terminal'

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(command ? `${command}\n${text}` : text)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      // A denied clipboard permission does not deserve an error state; the text
      // is selectable either way.
    }
  }

  return (
    <figure className="terminal-frame">
      <div className="terminal-header">
        <span className="terminal-lights" aria-hidden="true">
          <span />
          <span />
          <span />
        </span>
        <span className="terminal-title">{label}</span>
        <button
          type="button"
          className={`terminal-copy ${copied ? 'copied' : ''}`}
          onClick={copy}
          aria-label="Copy terminal output"
        >
          {copied ? <Check size={14} /> : <Copy size={14} />}
          <span>{copied ? 'Copied!' : 'Copy'}</span>
        </button>
      </div>

      {/* One scroll container, and one only — the same arrangement `CodeBlock`
          uses. A wide line scrolls inside this box rather than moving the page,
          and `tabIndex` is what lets a keyboard reach that scroll. */}
      <div
        className="terminal-body"
        style={maxLines ? { maxHeight: `${maxLines * 1.55}em` } : undefined}
        tabIndex={0}
        role="group"
        aria-label={`Captured output of ${label}`}
      >
        <pre>
          {command && (
            <span className="terminal-command">
              <span className="terminal-prompt" aria-hidden="true">
                ${' '}
              </span>
              {command}
            </span>
          )}
          {text}
        </pre>
      </div>

      {caption && <figcaption className="terminal-caption">{caption}</figcaption>}
    </figure>
  )
}
