import type { ReactNode } from 'react'
import './Figure.css'

interface FigureProps {
  /** Path to a vendored image, relative to the docs base (`import.meta.env.BASE_URL`). */
  src: string
  /**
   * Which root a site-absolute `src` is resolved against.
   *
   * `docs` (the default) is the documentation bundle's own base. `site` is the
   * site the bundle is served from, one level up — the captures of the web
   * surfaces live there, beside the docs rather than inside them, so both
   * halves of the site show the same file.
   */
  root?: 'docs' | 'site'
  /** What the image shows, for someone who cannot see it. Never decorative here. */
  alt: string
  /** Shown under the frame. Say what produced it. */
  caption: ReactNode
  /** The exact command this was captured from, rendered in mono under the caption. */
  command?: string
  /** Intrinsic size, so the page does not reflow while the image decodes. */
  width: number
  height: number
}

/**
 * A captured screenshot, framed and captioned.
 *
 * The caption is required, not optional decoration: a screenshot with no
 * provenance is indistinguishable from a mock-up, and this site does not ship
 * mock-ups. Nothing here is retouched.
 *
 * `width` and `height` are required for the same reason — the browser reserves
 * the space before the bytes arrive, so nothing below the image jumps.
 *
 * Terminal output does **not** belong here; it belongs in `Terminal`, as text.
 * Use this for the browser surfaces: the dashboard, the playground, the model
 * browser, the topology graph.
 */
export default function Figure({
  src,
  root = 'docs',
  alt,
  caption,
  command,
  width,
  height,
}: FigureProps) {
  // The docs bundle is served from a base that changes between the Netlify site
  // (/docs/) and the GitHub Pages project page (/<repo>/docs/). Vite records the
  // resolved base here, so a site-absolute path is built rather than assumed.
  const base = import.meta.env.BASE_URL.replace(/\/$/, '')
  // The site root is that base with its last segment — `docs` — removed, which
  // is `` on the Netlify site and `/<repo>` on a project page.
  const siteRoot = base.replace(/\/docs$/, '')
  const prefix = root === 'site' ? siteRoot : base
  const href = src.startsWith('/') ? `${prefix}${src}` : src

  return (
    <figure className="doc-figure">
      <div className="doc-figure-frame">
        <img
          src={href}
          alt={alt}
          width={width}
          height={height}
          loading="lazy"
          decoding="async"
        />
      </div>
      <figcaption className="doc-figure-caption">
        {caption}
        {command && <code className="doc-figure-command">{command}</code>}
      </figcaption>
    </figure>
  )
}
