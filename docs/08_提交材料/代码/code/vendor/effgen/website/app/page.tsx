import type { Metadata } from "next";
import HomeView from "./HomeView";
import { siteData } from "@/components/siteData";
import { SITE_NAME, pageMetadata } from "@/components/seo";

// The page is a client tree, so the route is a thin server component around it
// and owns the title, the description and the share card — the same shape the
// six product routes use.
export const metadata: Metadata = pageMetadata({
  path: "/",
  card: "home",
  title: `${SITE_NAME} — agents built for small language models`,
  description:
    "Build agents on small language models, on your own hardware or any OpenAI-compatible " +
    `server. ${siteData.tools.count} tools, ${siteData.presets.count} presets, RAG, ` +
    "evaluation and a coding agent.",
});

export default function Page() {
  return <HomeView />;
}
