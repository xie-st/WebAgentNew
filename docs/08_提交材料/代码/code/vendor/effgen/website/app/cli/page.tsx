import type { Metadata } from "next";
import CliView from "./CliView";
import { siteData } from "@/components/siteData";
import { pageMetadata } from "@/components/seo";

// The view is a client component, so the route is a thin server component
// around it and owns the page's title and description.
export const metadata: Metadata = pageMetadata({
  path: "/cli",
  card: "cli",
  title: "The effgen command line",
  description:
    `${siteData.cli.command_count} commands and ${siteData.cli.subcommand_count} sub-commands: ` +
    "run an agent, race models, serve an OpenAI-compatible API, watch what it costs, and " +
    "render a run as a report.",
});

export default function Page() {
  return <CliView />;
}
