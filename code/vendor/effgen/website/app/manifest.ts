import type { MetadataRoute } from "next";
import { withBasePath } from "@/components/basePath";
import { SITE_NAME } from "@/components/seo";
import { siteData } from "@/components/siteData";

// What a browser stores when someone installs the site to a home screen: the
// name it goes under, the icons it draws, and the colours it opens in. The icons
// are the ones `scripts/gen_icons.py` renders from `public/favicon.svg`, so the
// installed icon is the mark in the tab.

export const dynamic = "force-static";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: `${SITE_NAME} — agents built for small language models`,
    short_name: SITE_NAME,
    description:
      `A framework for building agents on small language models: ${siteData.tools.count} tools, ` +
      `${siteData.presets.count} presets, retrieval, evaluation, a coding agent and an ` +
      `OpenAI-compatible server.`,
    start_url: withBasePath("/"),
    scope: withBasePath("/"),
    display: "standalone",
    // The dark ground the site opens in, so the splash does not flash white.
    background_color: "#020c08",
    theme_color: "#020c08",
    icons: [
      { src: withBasePath("/icons/icon-192.png"), sizes: "192x192", type: "image/png" },
      { src: withBasePath("/icons/icon-512.png"), sizes: "512x512", type: "image/png" },
      {
        src: withBasePath("/icons/icon-maskable-512.png"),
        sizes: "512x512",
        type: "image/png",
        purpose: "maskable",
      },
    ],
  };
}
