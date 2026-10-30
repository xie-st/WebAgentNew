import type { MetadataRoute } from "next";
import { SITE_URL } from "@/components/seo";

// Nothing on this site is private, so everything is crawlable. What this is
// really for is naming the sitemap: without it a crawler finds the eleven
// landing pages by following links and never reaches the 72 documentation
// pages, which are routes of a single-page app rather than files it can walk.

export const dynamic = "force-static";

export default function robots(): MetadataRoute.Robots {
  return {
    rules: [{ userAgent: "*", allow: "/" }],
    sitemap: `${SITE_URL}/sitemap.xml`,
    host: SITE_URL,
  };
}
