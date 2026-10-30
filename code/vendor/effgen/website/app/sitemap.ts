import type { MetadataRoute } from "next";
import { SITE_URL, canonicalUrl } from "@/components/seo";
import { examplesData } from "./examples/[id]/examplesData";
import docsRoutes from "@/data/docsRoutes.json";

// Every address on the site, in one file, for a crawler that would otherwise
// have to find the documentation by following links into a single-page app.
//
// The three lists are read rather than typed: the landing routes from the
// navigation this file declares, the example detail pages from the module the
// route generates its static params from, and the 72 documentation pages from
// `data/docsRoutes.json`, which `scripts/gen_docs_routes.mjs` copies out of the
// documentation's own table of contents before this builds.
//
// The 404 is deliberately absent: it is the one page that should not be found.

/** The landing routes, most important first. */
const LANDING = [
  "/",
  "/agents",
  "/models",
  "/cli",
  "/code",
  "/dashboard",
  "/production",
  "/examples",
  "/changelog",
  "/leaderboard",
  "/community",
];

export const dynamic = "force-static";

export default function sitemap(): MetadataRoute.Sitemap {
  const lastModified = new Date();

  const landing = LANDING.map((path) => ({
    url: canonicalUrl(path),
    lastModified,
    changeFrequency: "monthly" as const,
    priority: path === "/" ? 1 : 0.8,
  }));

  const examples = Object.keys(examplesData).map((id) => ({
    url: canonicalUrl(`/examples/${id}`),
    lastModified,
    changeFrequency: "monthly" as const,
    priority: 0.6,
  }));

  const docs = docsRoutes.pages.map((page) => ({
    url: `${SITE_URL}${docsRoutes.base}${page.path}`,
    lastModified,
    changeFrequency: "monthly" as const,
    priority: 0.7,
  }));

  return [...landing, ...examples, ...docs];
}
