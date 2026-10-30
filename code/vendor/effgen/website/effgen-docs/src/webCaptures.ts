// The screenshots of effGen's web surfaces, read out of the manifest that
// `scripts/gen_web_captures.py` writes at the repository root.
//
// The landing site reads the same file through `components/webCaptures.ts`, so
// a picture on `/dashboard` and the same picture on `/docs/dashboard` are the
// same bytes with the same alt text and the same provenance line. A figure
// either resolves to a capture in the manifest or it does not exist, which is
// what stops a drawing of a surface from standing in for the surface.

import raw from '@data/captures.web.json'

export interface WebCapture {
  slug: string
  /** `dark` or `light` — the theme the browser was in. */
  theme: string
  /** `dashboard` or `playground`. */
  surface: string
  /** Site-absolute path under the landing site's `public/`. */
  src: string
  width: number
  height: number
  bytes: number
  sha256: string
  /** What was on screen and how it got there. */
  produced_by: string
  /** What the image shows, for someone who cannot see it. */
  alt: string
}

interface WebCaptureSet {
  generated_at: string
  captures: Record<string, WebCapture>
}

const set = raw as unknown as WebCaptureSet

/**
 * One captured surface, by slug and theme.
 *
 * Throws rather than rendering a broken image: a missing capture is a build
 * failure, not a gap on the page.
 */
export function webCapture(slug: string, theme: 'dark' | 'light' = 'dark'): WebCapture {
  const capture = set.captures[`${slug}-${theme}`]
  if (!capture) {
    throw new Error(
      `No web capture named "${slug}-${theme}". Add it to SHOTS in ` +
        'scripts/gen_web_captures.py and re-run the generator.',
    )
  }
  return capture
}

/**
 * The props `Figure` needs, taken from a capture.
 *
 * `root="site"` because the images live beside the documentation bundle rather
 * than inside it: the docs are served from `<site>/docs/` and the captures from
 * `<site>/captures/`, so the path is resolved against the site root and not
 * against the bundle's own base.
 */
export function figureOf(capture: WebCapture): {
  src: string
  alt: string
  width: number
  height: number
  root: 'site'
} {
  return {
    src: capture.src,
    alt: capture.alt,
    width: capture.width,
    height: capture.height,
    root: 'site',
  }
}
