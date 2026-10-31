import { useEffect } from 'react'

// What a search result and a shared link say about one documentation page.
//
// The documentation is a single-page app: the browser is handed one HTML file
// for all 72 routes, so without this every page would share the title and the
// description written into `index.html`, and a crawler that renders the app
// would still find one canonical address for all of them. That is the same
// defect as having no metadata at all — a result list where 72 rows read
// identically.
//
// So each page sets its own as it renders. The title, the description and the
// canonical address are the ones `nav.ts` already holds, which is what keeps
// them in step with the sidebar entry, the breadcrumb and the heading.

/** The address the documentation is served from, and the only one a canonical names. */
const SITE_URL = 'https://effgen.org'

/**
 * `/quickstart` → `https://effgen.org/docs/quickstart`.
 *
 * Built from the site address rather than from `window.location`, so a page
 * opened from the built export on disk, from a preview host or from a project
 * page still points a crawler at the published address.
 */
function canonicalFor(pathname: string): string {
  const path = pathname.replace(/\/$/, '')
  return `${SITE_URL}/docs${path === '' ? '' : path}`
}

/** Set `<meta name="...">`, creating the tag the first time it is needed. */
function setNamed(name: string, content: string) {
  let el = document.head.querySelector<HTMLMetaElement>(`meta[name="${name}"]`)
  if (!el) {
    el = document.createElement('meta')
    el.setAttribute('name', name)
    document.head.appendChild(el)
  }
  el.setAttribute('content', content)
}

/** Set `<meta property="...">` — Open Graph uses `property`, not `name`. */
function setProperty(property: string, content: string) {
  let el = document.head.querySelector<HTMLMetaElement>(`meta[property="${property}"]`)
  if (!el) {
    el = document.createElement('meta')
    el.setAttribute('property', property)
    document.head.appendChild(el)
  }
  el.setAttribute('content', content)
}

export interface DocumentMeta {
  /** The page's own name, without the site name — this appends it. */
  title: string
  /** One sentence saying what is on the page. */
  description: string
  /** Route under `/docs`, with a leading slash. */
  pathname: string
}

export function useDocumentMeta({ title, description, pathname }: DocumentMeta) {
  useEffect(() => {
    // Short enough that the page's own name survives in a result list: a
    // title runs past what is shown at about sixty characters, and the two
    // longest page names here leave only a little room after them.
    const full = `${title} · effGen docs`
    const url = canonicalFor(pathname)

    document.title = full
    setNamed('description', description)
    setProperty('og:title', full)
    setProperty('og:description', description)
    setProperty('og:url', url)
    setNamed('twitter:title', full)
    setNamed('twitter:description', description)

    let link = document.head.querySelector<HTMLLinkElement>('link[rel="canonical"]')
    if (!link) {
      link = document.createElement('link')
      link.setAttribute('rel', 'canonical')
      document.head.appendChild(link)
    }
    link.setAttribute('href', url)
  }, [title, description, pathname])
}
