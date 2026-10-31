"use client";

import { ReactNode } from "react";
import { withBasePath } from "@/components/basePath";

interface FigureProps {
  /** Site-absolute path to a vendored image, e.g. `/screens/dashboard-dark.png`. */
  src: string;
  /** What the image shows, for someone who cannot see it. Never decorative here. */
  alt: string;
  /** Shown under the frame. Say what produced it. */
  caption: ReactNode;
  /** The exact command this was captured from, rendered in mono under the caption. */
  command?: string;
  /** Intrinsic size, so the page does not reflow while the image decodes. */
  width: number;
  height: number;
  className?: string;
  /**
   * Extra classes on the frame around the image. A capture of a whole scrolling
   * page is thousands of pixels tall; bounding the frame and letting it scroll
   * keeps the figure the size of a figure. Never used to crop what a caption
   * claims is there.
   */
  frameClassName?: string;
}

/**
 * A captured screenshot, framed and captioned.
 *
 * Every image on a product page goes through here, and every one of them has to
 * say where it came from — the caption and the command are required, not
 * optional decoration, because a screenshot with no provenance is
 * indistinguishable from a mock-up. Nothing here is retouched.
 *
 * `width` and `height` are required for the same reason: the browser reserves
 * the space before the bytes arrive, so nothing below the image jumps.
 *
 * Terminal output does **not** belong here — it belongs in `Terminal`, as text.
 * Use this for the browser surfaces: the dashboard, the playground, the model
 * browser, the topology graph.
 */
export default function Figure({
  src,
  alt,
  caption,
  command,
  width,
  height,
  className = "",
  frameClassName = "",
}: FigureProps) {
  return (
    <figure className={className}>
      {/* A caller that caps the frame's height (`frameClassName`) turns it into a
          scroll container, and a scroll container has to be reachable by a
          keyboard. A frame that does not scroll stays out of the tab order. */}
      <div
        className={
          "rounded-2xl overflow-hidden border border-gray-200 dark:border-green-500/20 " +
          "focus-visible:outline focus-visible:outline-2 focus-visible:outline-green-500 " +
          `focus-visible:-outline-offset-2 bg-gray-50 dark:bg-[#040f0a] ${frameClassName}`
        }
        {...(frameClassName.includes("overflow-") ? { tabIndex: 0, role: "group", "aria-label": alt } : {})}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={withBasePath(src)}
          alt={alt}
          width={width}
          height={height}
          loading="lazy"
          decoding="async"
          className="block w-full h-auto"
        />
      </div>
      <figcaption className="mt-3 text-sm text-gray-600 dark:text-gray-400">
        {caption}
        {command && (
          <code className="block mt-1.5 text-xs font-mono text-gray-600 dark:text-gray-400">
            {command}
          </code>
        )}
      </figcaption>
    </figure>
  );
}
