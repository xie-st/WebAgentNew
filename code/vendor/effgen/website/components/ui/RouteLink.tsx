"use client";

import Link from "next/link";
import { CSSProperties, ReactNode } from "react";
import { withBasePath } from "@/components/basePath";
import { resolveRoute } from "@/components/routes";

interface RouteLinkProps {
  /** The route as the finished site will have it, e.g. `/models`. */
  to: string;
  className?: string;
  style?: CSSProperties;
  onClick?: () => void;
  "aria-label"?: string;
  /** `"page"` on the link to the address the reader is already on. */
  "aria-current"?: "page";
  children: ReactNode;
}

/**
 * A link to a page on this site, which knows what to do when that page has not
 * been built yet.
 *
 * Three cases, decided by `routes.ts`:
 *
 *  - the route exists → a normal `next/link`;
 *  - it is a documentation route → a plain anchor, because the documentation is
 *    a separate application and `next/link` cannot route into it;
 *  - it does not exist yet → the framework's own documentation for that topic,
 *    opened in a new tab and announced as leaving the site.
 *
 * The point of the third case is that a visitor who clicks "effgen code" gets
 * something that describes `effgen code`. When the page lands, one line comes
 * out of `routes.ts` and this link becomes internal with no other change.
 */
export default function RouteLink({
  to,
  className,
  style,
  onClick,
  "aria-label": ariaLabel,
  "aria-current": ariaCurrent,
  children,
}: RouteLinkProps) {
  const { href, external } = resolveRoute(to);

  if (external) {
    return (
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className={className}
        style={style}
        onClick={onClick}
        aria-label={ariaLabel}
        aria-current={ariaCurrent}
      >
        {children}
        <span className="sr-only"> (opens the framework documentation on GitHub)</span>
      </a>
    );
  }

  if (href.startsWith("/docs")) {
    return (
      <a
        href={withBasePath(href)}
        className={className}
        style={style}
        onClick={onClick}
        aria-label={ariaLabel}
        aria-current={ariaCurrent}
      >
        {children}
      </a>
    );
  }

  return (
    <Link
      href={href}
      className={className}
      style={style}
      onClick={onClick}
      aria-label={ariaLabel}
      aria-current={ariaCurrent}
    >
      {children}
    </Link>
  );
}
