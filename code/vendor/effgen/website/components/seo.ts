import type { Metadata } from "next";

// What a search result and a shared link say about a page.
//
// Every route on the site declares its own title and description here rather
// than inheriting one, because a set of pages that all describe themselves the
// same way is a set of pages a reader cannot tell apart in a result list. The
// canonical address, the Open Graph card and the Twitter card are derived from
// the same three fields, so they cannot drift from each other.
//
// Lengths are what a result list will show: a title over about 60 characters and
// a description over about 160 are cut off mid-sentence. `scripts/check_meta.py`
// reads the built export and reports any that would be.

import { version } from "./siteData";

/** The address the site is served from, and the only one a canonical names. */
export const SITE_URL = "https://effgen.org";
export const SITE_NAME = "effGen";

/**
 * What every route's title ends with, and what the root layout's title template
 * appends. A page passes the part before it and nothing else, so the suffix is
 * written once and the version in it moves with the release.
 */
export const TITLE_SUFFIX = `${SITE_NAME} ${version}`;

/** The share card a route uses when it has no card of its own. */
const DEFAULT_CARD = "home";

export interface PageMeta {
  /** Route path, with a leading slash. `/` for the landing page. */
  path: string;
  /** The page's own name. The site name and version are appended for it. */
  title: string;
  /** One or two sentences saying what is on the page. */
  description: string;
  /** Basename of the card in `public/og/`. Falls back to the site's own. */
  card?: string;
  /** Set on a page that should not be indexed — the 404. */
  noindex?: boolean;
}

/** `/agents` → `https://effgen.org/agents/`, matching the export's own paths. */
export function canonicalUrl(path: string): string {
  if (path === "/") return `${SITE_URL}/`;
  return `${SITE_URL}${path.replace(/\/$/, "")}/`;
}

/**
 * The metadata for one route: title, description, canonical, and the two share
 * cards, all derived from the same fields.
 *
 * The card URLs are absolute because Open Graph is read by a crawler that has no
 * page to resolve a relative path against.
 */
export function pageMetadata({
  path,
  title,
  description,
  card = DEFAULT_CARD,
  noindex = false,
}: PageMeta): Metadata {
  const url = canonicalUrl(path);
  const image = `${SITE_URL}/og/${card}.png`;
  // A crawler reading the card has no title template to apply, so the card
  // carries the same full string a browser tab ends up showing.
  const fullTitle = `${title} · ${TITLE_SUFFIX}`;

  return {
    title,
    description,
    alternates: { canonical: url },
    ...(noindex ? { robots: { index: false, follow: true } } : {}),
    openGraph: {
      type: "website",
      siteName: SITE_NAME,
      url,
      title: fullTitle,
      description,
      images: [{ url: image, width: 1200, height: 630, alt: `${title} — ${SITE_NAME}` }],
    },
    twitter: {
      card: "summary_large_image",
      title: fullTitle,
      description,
      images: [image],
    },
  };
}
