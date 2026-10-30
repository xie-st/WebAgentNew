import type { Metadata } from "next";
import LeaderboardView from "./LeaderboardView";
import { pageMetadata } from "@/components/seo";

// The view is a client component, so the route is a thin server component
// around it and owns the page's title and description.
export const metadata: Metadata = pageMetadata({
  path: "/leaderboard",
  card: "leaderboard",
  title: "Benchmark leaderboard",
  description:
    "10 small language models across 13 benchmarks and 5 frameworks — calculator, math " +
    "reasoning, agentic, memory and retrieval results from the effGen paper.",
});

export default function Page() {
  return <LeaderboardView />;
}
