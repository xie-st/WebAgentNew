// The screenshots of effGen's web surfaces the product pages show, read out of
// the manifest `scripts/gen_web_captures.py` writes.
//
// Every image on `/dashboard` and `/models` comes through here, and every one
// carries the surface it shows, the theme it was taken in, what produced it,
// its dimensions and the sha256 of the bytes that ship. A figure either
// resolves to one of these or it does not exist, which is what stops a drawing
// of a surface from standing in for the surface.

import raw from "@/data/captures.web.json";

export interface WebCapture {
  slug: string;
  /** `dark` or `light` — the theme the browser was in. */
  theme: string;
  /** `dashboard` or `playground`. */
  surface: string;
  /** Site-absolute path under `public/`. */
  src: string;
  width: number;
  height: number;
  bytes: number;
  sha256: string;
  /** What was on screen and how it got there. */
  produced_by: string;
  /** What the image shows, for someone who cannot see it. */
  alt: string;
}

interface WebCaptureSet {
  generated_at: string;
  captures: Record<string, WebCapture>;
}

const set = raw as unknown as WebCaptureSet;

/**
 * One captured surface, by slug and theme.
 *
 * Throws rather than rendering a broken image: a missing capture is a build
 * failure, not a gap on the page.
 */
export function webCapture(slug: string, theme: "dark" | "light" = "dark"): WebCapture {
  const capture = set.captures[`${slug}-${theme}`];
  if (!capture) {
    throw new Error(
      `No web capture named "${slug}-${theme}". Add it to SHOTS in ` +
        "scripts/gen_web_captures.py and re-run the generator.",
    );
  }
  return capture;
}

/**
 * The props `Figure` needs, taken from a capture.
 *
 * The alt text and the dimensions travel with the image rather than being
 * written at the call site, so an image and its description cannot drift apart
 * and a page cannot forget to reserve the space the picture will take.
 */
export function figureOf(capture: WebCapture): {
  src: string;
  alt: string;
  width: number;
  height: number;
} {
  return {
    src: capture.src,
    alt: capture.alt,
    width: capture.width,
    height: capture.height,
  };
}
