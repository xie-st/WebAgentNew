// Links from the documentation out to the main site.
//
// The documentation is a separate app mounted at `/docs`, so a router `<Link>`
// cannot reach `/examples` or `/dashboard` on the landing site — those are
// different pages served by the other half. This resolves a site-absolute path
// against whatever root the bundle was built for, the same way `Figure`'s
// `root="site"` does, so the link works under a project page as well as at the
// domain root.

/** A site-absolute path (`/examples`) as an href from a documentation page. */
export function siteHref(path: string): string {
  const base = import.meta.env.BASE_URL.replace(/\/$/, '')
  const root = base.replace(/\/docs$/, '')
  return `${root}${path}`
}
