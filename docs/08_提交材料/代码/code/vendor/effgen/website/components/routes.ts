// Where a link to a page that does not exist yet goes in the meantime.
//
// The navbar, the footer, the release band, the launch modal and the 404 page
// all point at the six product routes and at the migration guide. Those pages
// are being written. Until each one lands, a visitor who
// follows the link should reach something that answers the question rather
// than a 404, so each one resolves to the framework's own documentation for
// that topic — which was re-run against 1.0.0 and is the source those pages
// will be written from.
//
// It deliberately does **not** fall back to the documentation site's current
// pages for these topics. Those still describe 0.3.x and carry known-wrong
// model ids and a tool-call shape 1.0.0 does not accept, so sending a reader
// there would be worse than sending them nowhere.
//
// **This file is how a page goes live.** When `/agents` is built, delete its
// line here; nothing else has to change, and the check below stops reporting
// it. When the last line is gone, delete the file and the `resolveRoute` calls
// with it.

const FRAMEWORK_DOCS = "https://github.com/ctrl-gaurav/effGen/blob/main";

/** Site route → where it points until the page exists. Every URL was checked. */
const PENDING: Record<string, string> = {
  "/docs/migration": `${FRAMEWORK_DOCS}/docs/migration.md`,
};

export interface ResolvedRoute {
  href: string;
  /** True while the link leaves the site, so the caller can mark it as such. */
  external: boolean;
}

/**
 * Resolve a link target.
 *
 * A route that exists is returned unchanged and stays internal. One that does
 * not yet exist is swapped for the framework documentation that covers it, and
 * reported as external so the caller can open it in a new tab and say so to a
 * screen reader.
 */
export function resolveRoute(path: string): ResolvedRoute {
  const pending = PENDING[path];
  return pending ? { href: pending, external: true } : { href: path, external: false };
}

/** Whether this route is still standing in for a page that has not been built. */
export function isPending(path: string): boolean {
  return path in PENDING;
}
