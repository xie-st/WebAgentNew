import type { Metadata } from "next";
import { examplesData } from "./examplesData";
import ExampleDetail from "./ExampleDetail";
import { pageMetadata } from "@/components/seo";

// A static export needs to know every id this route renders. The view is a
// client component, which cannot declare that, so the route is a thin server
// component around it and reads the ids from the shared data module.
export function generateStaticParams() {
  return Object.keys(examplesData).map((id) => ({ id }));
}

// Six examples, six different titles and descriptions: each one is taken from
// the example's own name and one-line summary, so a result list distinguishes
// them and neither can drift from what the page renders.
export async function generateMetadata({
  params,
}: {
  params: Promise<{ id: string }>;
}): Promise<Metadata> {
  const { id } = await params;
  const example = examplesData[id];
  if (!example) {
    return pageMetadata({
      path: `/examples/${id}`,
      card: "examples",
      title: "Example not found",
      description: "That example does not exist. The examples index lists the ones that do.",
      noindex: true,
    });
  }
  return pageMetadata({
    path: `/examples/${id}`,
    card: "examples",
    title: `${example.title} — example`,
    description:
      `${example.subtitle} ` +
      `${example.tools.length} tool${example.tools.length === 1 ? "" : "s"}, ` +
      `run on ${example.run.model}.`,
  });
}

export default async function Page({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <ExampleDetail id={id} />;
}
