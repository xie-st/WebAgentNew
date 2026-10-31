// Where the site is served from.
//
// Next rewrites the paths it owns — `next/link` hrefs and `next/image` sources —
// with the configured basePath, but it cannot rewrite a path written into a raw
// `<a>`, an `<img>`, or a metadata field, because those are plain strings. On a
// site served from the domain root the two are indistinguishable; under a path
// prefix the untouched ones resolve above the site and 404.
//
// Anything that is not a `next/link` or a `next/image` builds its URL here.

export const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

/**
 * Prefix a site-absolute path (`/favicon.svg`) with the base the site is served
 * from.
 *
 * Anything else is returned unchanged, so this is safe to apply to a mixed list
 * of links: an external URL, a bare fragment and a relative path all pass
 * through untouched.
 */
export function withBasePath(path: string): string {
  if (!path.startsWith("/")) return path;
  return `${BASE_PATH}${path}`;
}
