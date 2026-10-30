"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { FiCheck, FiCopy, FiPause, FiPlay, FiSkipForward } from "react-icons/fi";
import { useReducedMotion } from "@/components/useReducedMotion";

interface TerminalReplayProps {
  /** The command that produced the capture, without a leading `$`. */
  command: string;
  /** The capture, verbatim. Never retyped, never trimmed for effect. */
  output: string;
  /** Window title. Defaults to the command's first word. */
  title?: string;
  /** Height of the frame, in rows. The replay scrolls inside it. */
  rows?: number;
  /** Milliseconds between lines. */
  interval?: number;
  /** Read out under the frame. Say what produced it and what it shows. */
  caption?: React.ReactNode;
  className?: string;
}

// The same structural colouring `Terminal` uses, so a replayed capture and a
// static one look like the same thing. The shapes are what `effgen` prints; the
// text is untouched.
function lineClass(line: string): string {
  if (/^(---|\+\+\+) /.test(line)) return "text-gray-400";
  if (/^@@/.test(line)) return "text-violet-400";
  if (/^\+/.test(line)) return "text-green-400";
  if (/^-/.test(line)) return "text-red-400";
  if (/^\s*[✓✔]/.test(line)) return "text-green-400 font-semibold";
  if (/^\s*[⚠✗✖]/.test(line)) return "text-orange-400";
  if (/^(new file|edit|deleted) /.test(line)) return "text-cyan-400 font-semibold";
  if (/^[A-Za-z][^:]{0,40}:\s/.test(line) && !line.startsWith("http")) {
    return "text-gray-300";
  }
  return "text-gray-400";
}

/**
 * A captured session played back the way it arrived, one line at a time.
 *
 * It is a reveal, not a re-enactment: the transcript is already in the page and
 * the replay only controls how much of it is showing, so the text is selectable
 * and copyable at any point and the whole capture is in the HTML whether the
 * replay ever runs or not.
 *
 * **Under reduced motion it does not run at all.** The frame opens on the
 * finished transcript — which is the state the replay was heading for — and the
 * transport control is replaced by nothing, because there is nothing to
 * transport. That is the designed resting state, not a paused animation.
 *
 * The playing frame is hidden from assistive technology and a complete copy of
 * the transcript sits beside it for screen readers, so the session is read once,
 * whole, instead of being announced a line at a time.
 */
export default function TerminalReplay({
  command,
  output,
  title,
  rows = 22,
  interval = 130,
  caption,
  className = "",
}: TerminalReplayProps) {
  const reduced = useReducedMotion();
  const lines = output.replace(/\s+$/, "").split("\n");
  const label = title ?? command.trim().split(/\s+/)[0];

  // The frame opens on the finished transcript, and the replay rewinds it when
  // it actually starts. That ordering is deliberate: it puts the whole session
  // in the prerendered HTML, so the frame is readable with JavaScript off and
  // there is never an empty box waiting for a script to fill it.
  const [shown, setShown] = useState(lines.length);
  const [playing, setPlaying] = useState(false);
  const [copied, setCopied] = useState(false);
  const [started, setStarted] = useState(false);
  const frameRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const finished = shown >= lines.length;

  // Start once the frame is on screen, so a visitor who lands further down the
  // page does not arrive at a replay that has already finished off-screen.
  useEffect(() => {
    if (reduced || started) return;
    const node = frameRef.current;
    if (!node) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          setStarted(true);
          setShown(0);
          setPlaying(true);
          observer.disconnect();
        }
      },
      { threshold: 0.25 },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [reduced, started]);

  useEffect(() => {
    if (!playing || reduced) return;
    if (finished) {
      setPlaying(false);
      return;
    }
    const timer = window.setTimeout(() => setShown((count) => count + 1), interval);
    return () => window.clearTimeout(timer);
  }, [playing, reduced, finished, shown, interval]);

  // Keep the newest line in view while the replay runs, without moving the page.
  useEffect(() => {
    if (!playing || reduced) return;
    const node = scrollRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [shown, playing, reduced]);

  const copy = useCallback(async () => {
    await navigator.clipboard.writeText(`${command}\n${output}`);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 2000);
  }, [command, output]);

  const visible = reduced ? lines : lines.slice(0, shown);

  return (
    <figure className={className}>
      <div
        ref={frameRef}
        className="rounded-2xl overflow-hidden border border-gray-200 dark:border-green-500/20 bg-gray-900 dark:bg-[#040f0a] shadow-sm"
      >
        <div className="flex items-center gap-2 px-4 py-2.5 border-b border-gray-700 dark:border-green-500/15 bg-gray-800 dark:bg-[#071a10]">
          <span className="flex gap-1.5" aria-hidden="true">
            <span className="w-3 h-3 rounded-full bg-red-400/70" />
            <span className="w-3 h-3 rounded-full bg-yellow-400/70" />
            <span className="w-3 h-3 rounded-full bg-green-400/70" />
          </span>
          <span className="ml-2 text-xs font-mono text-gray-400 truncate">{label}</span>

          <span className="ml-auto flex items-center gap-3">
            {!reduced && (
              <>
                <span className="text-[10px] font-mono text-gray-400 tabular-nums">
                  {Math.min(shown, lines.length)}/{lines.length}
                </span>
                <span className="sr-only" role="status">
                  {finished ? "Replay finished." : playing ? "Replaying." : "Paused."}
                </span>
                <button
                  type="button"
                  onClick={() => {
                    if (finished) setShown(0);
                    setStarted(true);
                    setPlaying((was) => (finished ? true : !was));
                  }}
                  className="flex items-center gap-1.5 text-xs font-mono text-gray-400 hover:text-green-400 transition-colors"
                >
                  {finished ? <FiPlay size={13} /> : playing ? <FiPause size={13} /> : <FiPlay size={13} />}
                  {finished ? "Replay" : playing ? "Pause" : "Play"}
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setPlaying(false);
                    setShown(lines.length);
                  }}
                  className="flex items-center gap-1.5 text-xs font-mono text-gray-400 hover:text-green-400 transition-colors"
                >
                  <FiSkipForward size={13} />
                  Show all
                </button>
              </>
            )}
            <button
              type="button"
              onClick={copy}
              className="flex items-center gap-1.5 text-xs font-mono text-gray-400 hover:text-green-400 transition-colors"
            >
              {copied ? <FiCheck size={13} /> : <FiCopy size={13} />}
              {copied ? "Copied" : "Copy"}
            </button>
          </span>
        </div>

        <div
          ref={scrollRef}
          className="overflow-auto"
          style={{ height: `${rows * 1.55}em` }}
          aria-hidden="true"
        >
          <pre className="px-4 py-4 text-xs sm:text-[13px] font-mono leading-relaxed text-gray-300">
            <span className="block text-green-400 font-semibold">
              <span className="select-none opacity-60">$ </span>
              {command}
            </span>
            <span className="block h-2" />
            {visible.map((line, i) => (
              <span key={i} className={`block ${lineClass(line)}`}>
                {line || " "}
              </span>
            ))}
          </pre>
        </div>
      </div>

      {/* The whole session, once, for anything that reads the page aloud. */}
      <pre className="sr-only">
        {`$ ${command}\n${output}`}
      </pre>

      {caption && (
        <figcaption className="mt-3 text-sm text-gray-600 dark:text-gray-400">
          {caption}
          <code className="block mt-1.5 text-xs font-mono text-gray-600 dark:text-gray-400 break-all">
            $ {command}
          </code>
        </figcaption>
      )}
    </figure>
  );
}
