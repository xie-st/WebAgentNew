import type { NextConfig } from "next";

// Where the site is served from. A GitHub Pages project page lives under
// /<repo>, so every asset and route needs that prefix; the Netlify build and a
// server-rendered build set neither and behave as before.
const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

// The site ships as a static export — Netlify publishes `out/`, and the export
// is what makes "the site opens with the network off" checkable. A host that
// runs `next start` opts out with NEXT_STATIC_EXPORT=0.
//
// `next dev` never exports, and it has a server, so it takes the rewrite path
// below instead. Without that, `/docs` has nothing serving it in development:
// the documentation is a built bundle sitting in `public/docs/`, and Next does
// not serve a directory index out of `public/`, so the Docs link answered 404
// on a developer's machine while working on the deployed site.
const isDev = process.env.NODE_ENV === "development";
const isStaticExport = process.env.NEXT_STATIC_EXPORT !== "0" && !isDev;

const nextConfig: NextConfig = {
  // A static host serves /path/ as /path/index.html. Development keeps the same
  // rule so an address behaves the same way in both places.
  trailingSlash: true,
  // The export has no server to optimise images on request.
  images: { unoptimized: true },

  ...(isStaticExport
    ? { output: "export" as const }
    : {
        // Rewrites need a server, so they are declared only for that build.
        // The docs SPA is emitted into public/docs and served from there; this
        // hands its client-side routes back to its own index.html. The static
        // export gets the same behaviour from the `_redirects` fallback instead.
        async rewrites() {
          return [
            { source: "/docs", destination: "/docs/index.html" },
            { source: "/docs/", destination: "/docs/index.html" },
            // Assets are served as themselves; every other docs address is the app.
            {
              source: "/docs/:path((?!assets/).*)",
              destination: "/docs/index.html",
            },
          ];
        },
      }),

  ...(basePath ? { basePath, assetPrefix: basePath } : {}),
};

export default nextConfig;
