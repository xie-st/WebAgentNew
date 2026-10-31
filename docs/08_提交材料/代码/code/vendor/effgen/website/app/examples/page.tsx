import type { Metadata } from "next";
import ExamplesView from "./ExamplesView";
import { examples } from "./[id]/examplesData";
import { pageMetadata } from "@/components/seo";

// The view is a client component, so the route is a thin server component
// around it and owns the page's title and description. The count comes from the
// same module the cards render from, so the two cannot disagree.
export const metadata: Metadata = pageMetadata({
  path: "/examples",
  card: "examples",
  title: "Examples",
  description:
    `${examples.length} complete agent programs with the output they actually produced — the ` +
    "code, the model each ran on, what it cost, and the script in the repository it came from.",
});

export default function Page() {
  return <ExamplesView />;
}
