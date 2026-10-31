import type { Metadata } from "next";
import AgentsView from "./AgentsView";
import { siteData } from "@/components/siteData";
import { pageMetadata } from "@/components/seo";

// The view is a client component, so the route is a thin server component
// around it and owns the page's title and description.
export const metadata: Metadata = pageMetadata({
  path: "/agents",
  card: "agents",
  title: "The agent library",
  description:
    "Agent(AgentConfig(...)).run(task) returns the answer and every tool call behind it: " +
    `${siteData.presets.count} presets, ${siteData.tools.count} tools, middleware, ` +
    "sessions, teams and resumable workflows.",
});

export default function Page() {
  return <AgentsView />;
}
