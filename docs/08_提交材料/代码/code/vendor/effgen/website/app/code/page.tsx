import type { Metadata } from "next";
import CodeView from "./CodeView";
import { pageMetadata } from "@/components/seo";

// The view is a client component, so the route is a thin server component
// around it and owns the page's title and description — the same shape the
// changelog and example detail routes use.
export const metadata: Metadata = pageMetadata({
  path: "/code",
  card: "code",
  title: "effgen code — the coding agent",
  description:
    "A coding agent for the terminal that shows every change as a unified diff before it " +
    "touches disk. Four permission modes, a review mode and a git allow-list.",
});

export default function Page() {
  return <CodeView />;
}
