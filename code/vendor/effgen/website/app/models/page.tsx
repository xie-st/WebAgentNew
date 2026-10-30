import type { Metadata } from "next";
import ModelsView from "./ModelsView";
import { siteData } from "@/components/siteData";
import { pageMetadata } from "@/components/seo";

// The view is a client component, so the route is a thin server component
// around it and owns the page's title and description.
export const metadata: Metadata = pageMetadata({
  path: "/models",
  card: "models",
  title: "Any model, anywhere",
  description:
    `${siteData.models.adapter_count} provider adapters, ${siteData.models.with_catalog_count} ` +
    `with a catalog of ${siteData.models.models} models and their prices, ` +
    `${siteData.models.local_engines.length} local engines, and one base_url for any ` +
    "OpenAI-protocol server.",
});

export default function Page() {
  return <ModelsView />;
}
