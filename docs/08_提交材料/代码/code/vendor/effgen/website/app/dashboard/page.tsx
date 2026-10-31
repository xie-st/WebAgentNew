import type { Metadata } from "next";
import DashboardView from "./DashboardView";
import { siteData } from "@/components/siteData";
import { pageMetadata } from "@/components/seo";

// The view is a client component, so the route is a thin server component
// around it and owns the page's title and description.
export const metadata: Metadata = pageMetadata({
  path: "/dashboard",
  card: "dashboard",
  title: "The dashboard and the playground",
  description:
    `effgen serve carries its own web surfaces: ${siteData.web.dashboard.panels.length} panels ` +
    "of live traffic, cost, latency and topology, a model browser and an in-browser playground. " +
    "No CDN.",
});

export default function Page() {
  return <DashboardView />;
}
