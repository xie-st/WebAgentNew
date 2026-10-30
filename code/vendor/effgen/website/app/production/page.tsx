import type { Metadata } from "next";
import ProductionView from "./ProductionView";
import { siteData } from "@/components/siteData";
import { pageMetadata } from "@/components/seo";

// The view is a client component, so the route is a thin server component
// around it and owns the page's title and description.
export const metadata: Metadata = pageMetadata({
  path: "/production",
  card: "production",
  title: "Running it for real",
  description:
    "An OpenAI-compatible server that is never unauthenticated by default: OIDC or a key, " +
    `${siteData.production.rbac_roles.length} roles with spend caps, rate limits, Prometheus ` +
    "metrics, traces and SLOs.",
});

export default function Page() {
  return <ProductionView />;
}
