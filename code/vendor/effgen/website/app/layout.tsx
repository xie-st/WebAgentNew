import type { Metadata } from "next";
import "./globals.css";
import { ThemeProvider } from "@/components/ThemeProvider";
import LaunchModal from "@/components/LaunchModal";
import MotionPreference from "@/components/MotionPreference";
import { withBasePath } from "@/components/basePath";
import { siteData } from "@/components/siteData";
import { SITE_NAME, SITE_URL, TITLE_SUFFIX, canonicalUrl } from "@/components/seo";
import { structuredData } from "@/components/structuredData";

// The site-wide defaults. Each route overrides the title and the description
// with its own in `components/seo.ts`; what is set here is what every page
// shares — the base every relative URL resolves against, the icons, the web
// manifest and the theme colour.
//
// The description is what a search result and a shared link show for the landing
// page, so its figures come from data/effgen.json like every other count on the
// site rather than being typed in and left behind by the next release.
export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: {
    default: `${SITE_NAME} — agents built for small language models`,
    template: `%s · ${TITLE_SUFFIX}`,
  },
  description:
    `Build agents on small language models and run them on your own hardware, or on any ` +
    `OpenAI-compatible server. ${siteData.tools.count} tools, ${siteData.presets.count} presets, ` +
    `RAG, evaluation and a coding agent.`,
  applicationName: SITE_NAME,
  alternates: { canonical: canonicalUrl("/") },
  manifest: withBasePath("/manifest.webmanifest"),
  icons: {
    icon: [
      { url: withBasePath("/favicon.svg"), type: "image/svg+xml" },
      { url: withBasePath("/icons/favicon-32.png"), sizes: "32x32", type: "image/png" },
      { url: withBasePath("/icons/favicon-16.png"), sizes: "16x16", type: "image/png" },
    ],
    shortcut: [withBasePath("/favicon.ico")],
    apple: [{ url: withBasePath("/icons/apple-touch-icon.png"), sizes: "180x180" }],
  },
  openGraph: {
    type: "website",
    siteName: SITE_NAME,
    url: canonicalUrl("/"),
  },
  twitter: { card: "summary_large_image" },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning className="overflow-x-hidden">
      <head>
        {/* The colour a mobile browser paints its chrome with. Both themes are
            declared, so the bar matches whichever one the reader is in. */}
        <meta name="theme-color" media="(prefers-color-scheme: light)" content="#ffffff" />
        <meta name="theme-color" media="(prefers-color-scheme: dark)" content="#020c08" />
        {/* What the project is, in the vocabulary a search engine reads: the
            package, its licence, where it is developed and where it installs
            from. Every field is read from the same generated data the pages
            are, so it cannot describe a release the site is not on. */}
        <script
          type="application/ld+json"
          // The content is built from generated data, not from anything a
          // visitor supplies, and JSON-LD has to be injected as text.
          dangerouslySetInnerHTML={{ __html: JSON.stringify(structuredData) }}
        />
      </head>
      <body className="antialiased overflow-x-hidden" suppressHydrationWarning>
        <a href="#main" className="skip-link">
          Skip to content
        </a>
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          enableSystem
          disableTransitionOnChange
        >
          <MotionPreference>
            {children}
            <LaunchModal />
          </MotionPreference>
        </ThemeProvider>
      </body>
    </html>
  );
}
