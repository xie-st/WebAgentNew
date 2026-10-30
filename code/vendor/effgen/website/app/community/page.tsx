import type { Metadata } from "next";
import CommunityView from "./CommunityView";
import { pageMetadata } from "@/components/seo";

// The view is a client component, so the route is a thin server component
// around it and owns the page's title and description.
export const metadata: Metadata = pageMetadata({
  path: "/community",
  card: "community",
  title: "Community",
  description:
    "The repository, the issue tracker and the releases, the four kinds of contribution the " +
    "project takes, the ground rules, and how to cite the work.",
});

export default function Page() {
  return <CommunityView />;
}
