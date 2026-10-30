import React, { ReactNode, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { ChevronRight, ChevronLeft, Link2 } from 'lucide-react';
import { Link, useLocation } from 'react-router-dom';
import { groupFor, neighboursOf, pageFor } from '../nav';
import { useDocumentMeta } from '../hooks/useDocumentMeta';
import { slugify } from './slugify';
import './DocPage.css';

interface Breadcrumb {
  label: string;
  path?: string;
}

/** One heading found in the rendered page, and where it sits in the outline. */
export interface Outline {
  id: string;
  text: string;
  level: 2 | 3;
}

// What a page says about itself when it is not in the navigation and has no
// subtitle of its own to fall back on.
const DEFAULT_DESCRIPTION =
  'Documentation for effGen: every behaviour, flag and parameter, with code you can copy and run.';

// How far below the top of the viewport a heading should land when it is
// scrolled to, so the fixed header does not cover it.
const SCROLL_OFFSET = 88;

function scrollToHeading(id: string) {
  const el = document.getElementById(id);
  if (!el) return;
  const top = el.getBoundingClientRect().top + window.scrollY - SCROLL_OFFSET;
  window.scrollTo({ top, behavior: 'auto' });
}

/**
 * Give every heading in the page an id and a link to itself, and report the
 * outline.
 *
 * Doing this over the rendered headings rather than asking each page to declare
 * them means every page gets anchors — the ones written years ago as well as the
 * ones written next — and a page author cannot forget. A heading that already
 * carries an `id` keeps it, so an address someone has already linked to stays
 * valid.
 */
/** Whether two measured outlines describe the same headings, in the same order. */
function sameOutline(a: Outline[], b: Outline[]): boolean {
  return (
    a.length === b.length &&
    a.every((h, i) => h.id === b[i].id && h.text === b[i].text && h.level === b[i].level)
  );
}

function useOutline(root: React.RefObject<HTMLDivElement | null>, pathname: string): Outline[] {
  const [outline, setOutline] = useState<Outline[]>([]);

  useLayoutEffect(() => {
    const el = root.current;
    if (!el) return;
    const seen = new Set<string>();
    const found: Outline[] = [];

    el.querySelectorAll<HTMLHeadingElement>('h2, h3').forEach((heading) => {
      const text = (heading.dataset.headingText ?? heading.textContent ?? '').trim();
      if (!text) return;
      heading.dataset.headingText = text;

      let id = heading.id;
      if (!id) {
        const base = slugify(text) || 'section';
        id = base;
        let n = 2;
        while (seen.has(id)) id = `${base}-${n++}`;
        heading.id = id;
      }
      seen.add(id);

      if (!heading.dataset.anchored) {
        const anchor = document.createElement('a');
        anchor.className = 'heading-anchor';
        anchor.href = `#${id}`;
        anchor.textContent = '#';
        anchor.setAttribute('aria-label', `Link to “${text}”`);
        heading.appendChild(anchor);
        heading.dataset.anchored = 'yes';
      }

      found.push({ id, text, level: heading.tagName === 'H2' ? 2 : 3 });
    });

    // The outline is measured from the headings this render produced, which is
    // what `useLayoutEffect` is for; returning the previous array when nothing
    // moved keeps the measurement from re-rendering the page that produced it.
    // eslint-disable-next-line react-hooks/set-state-in-effect -- reading back the rendered DOM
    setOutline((prev) => (sameOutline(prev, found) ? prev : found));
  }, [root, pathname]);

  return outline;
}

interface DocPageProps {
  /**
   * Overrides the title the navigation gives this route.
   *
   * A page in the navigation should not pass one: the title belongs to
   * `nav.ts`, so the heading, the sidebar entry, the breadcrumb, the
   * previous/next pair and every "see also" that points here say the same
   * thing. It is here for a page that has no route of its own — the 404.
   */
  title?: string;
  /** The one sentence that says what the page is for. */
  subtitle?: ReactNode;
  icon?: ReactNode;
  /** Overrides the trail derived from the navigation. */
  breadcrumbs?: Breadcrumb[];
  /** Set false on a page that is one long list rather than a set of sections. */
  toc?: boolean;
  children: ReactNode;
}

export default function DocPage({
  title: titleOverride,
  subtitle,
  icon,
  breadcrumbs,
  toc = true,
  children,
}: DocPageProps) {
  const { pathname, hash } = useLocation();
  const contentRef = useRef<HTMLDivElement>(null);
  const outline = useOutline(contentRef, pathname);

  const entry = pageFor(pathname);
  const title = titleOverride ?? entry?.title ?? 'Documentation';
  const group = groupFor(pathname);
  const trail: Breadcrumb[] =
    breadcrumbs ?? (group ? [{ label: group.title }, { label: title }] : [{ label: title }]);

  const { prev, next } = neighboursOf(pathname);

  // A new page starts at the top; an address with a fragment starts at that
  // section. The router changes the URL without either happening on its own.
  useEffect(() => {
    if (hash) {
      const id = decodeURIComponent(hash.slice(1));
      // One frame late, so the outline pass has assigned the ids.
      requestAnimationFrame(() => scrollToHeading(id));
    } else {
      window.scrollTo({ top: 0, behavior: 'auto' });
    }
  }, [pathname, hash]);

  // The title, the description and the canonical address, all from the entry
  // the sidebar and the breadcrumb read, so a result list can tell the 72
  // documentation pages apart. A page outside the navigation — the 404 — has no
  // entry, so it falls back to its own subtitle when that is plain text and to
  // the documentation's own description otherwise.
  useDocumentMeta({
    title,
    description:
      entry?.lede ??
      (typeof subtitle === 'string' ? subtitle : DEFAULT_DESCRIPTION),
    pathname,
  });

  return (
    <div className="doc-page">
      <div className={`doc-shell ${toc && outline.length > 1 ? 'with-toc' : ''}`}>
        <div className="doc-main">
          <nav className="breadcrumbs" aria-label="Breadcrumb">
            <Link to="/introduction" className="breadcrumb-item">
              Docs
            </Link>
            {trail.map((crumb, index) => (
              <React.Fragment key={index}>
                <ChevronRight size={14} className="breadcrumb-separator" aria-hidden="true" />
                {crumb.path ? (
                  <Link to={crumb.path} className="breadcrumb-item">
                    {crumb.label}
                  </Link>
                ) : (
                  <span className="breadcrumb-item current">{crumb.label}</span>
                )}
              </React.Fragment>
            ))}
          </nav>

          <header className="doc-header">
            {icon && (
              <div className="doc-icon" aria-hidden="true">
                {icon}
              </div>
            )}
            <h1 className="doc-title">{title}</h1>
            {subtitle && <p className="doc-subtitle">{subtitle}</p>}
          </header>

          <div className="doc-content" ref={contentRef}>
            {children}
          </div>

          {(prev || next) && (
            <nav className="doc-pager" aria-label="Previous and next page">
              {prev ? (
                <Link to={prev.path} className="doc-pager-link prev">
                  <ChevronLeft size={16} aria-hidden="true" />
                  <span>
                    <span className="doc-pager-label">Previous</span>
                    <span className="doc-pager-title">{prev.title}</span>
                  </span>
                </Link>
              ) : (
                <span />
              )}
              {next && (
                <Link to={next.path} className="doc-pager-link next">
                  <span>
                    <span className="doc-pager-label">Next</span>
                    <span className="doc-pager-title">{next.title}</span>
                  </span>
                  <ChevronRight size={16} aria-hidden="true" />
                </Link>
              )}
            </nav>
          )}
        </div>

        {toc && outline.length > 1 && <DocToc outline={outline} />}
      </div>
    </div>
  );
}

/**
 * The on-page table of contents, with the section you are reading marked.
 *
 * It is drawn only where there is room for it beside the text — below that width
 * the page is the same page it has always been, and the headings are still
 * reachable by their own anchors.
 */
function DocToc({ outline }: { outline: Outline[] }) {
  const [active, setActive] = useState(outline[0]?.id ?? '');

  useEffect(() => {
    const ids = outline.map((o) => o.id);
    const update = () => {
      let current = ids[0] ?? '';
      for (const id of ids) {
        const el = document.getElementById(id);
        if (!el) continue;
        if (el.getBoundingClientRect().top - SCROLL_OFFSET <= 1) current = id;
        else break;
      }
      // At the very bottom the last section may never reach the line, so the
      // last heading wins once the page cannot scroll any further.
      if (window.innerHeight + window.scrollY >= document.body.scrollHeight - 2) {
        current = ids[ids.length - 1] ?? current;
      }
      setActive(current);
    };
    update();
    window.addEventListener('scroll', update, { passive: true });
    window.addEventListener('resize', update);
    return () => {
      window.removeEventListener('scroll', update);
      window.removeEventListener('resize', update);
    };
  }, [outline]);

  return (
    <aside className="doc-toc" aria-label="On this page">
      <div className="doc-toc-inner">
        <p className="doc-toc-title">On this page</p>
        <ul className="doc-toc-list">
          {outline.map((item) => (
            <li key={item.id} className={`doc-toc-item level-${item.level}`}>
              <a
                href={`#${item.id}`}
                className={item.id === active ? 'active' : ''}
                aria-current={item.id === active ? 'true' : undefined}
                onClick={(event) => {
                  event.preventDefault();
                  history.replaceState(null, '', `#${item.id}`);
                  scrollToHeading(item.id);
                }}
              >
                {item.text}
              </a>
            </li>
          ))}
        </ul>
      </div>
    </aside>
  );
}

/* ============================================================
   Content primitives
   ============================================================ */

const CALLOUT_ICONS = {
  info: '💡',
  note: '📝',
  tip: '💡',
  warning: '⚠️',
  danger: '⛔',
  success: '✅',
  error: '❌',
} as const;

export type CalloutType = keyof typeof CALLOUT_ICONS;

/**
 * An aside beside the main line of the page.
 *
 * `note` adds context, `tip` offers a shortcut, `warning` says something will
 * cost you time, and `danger` says something will cost you data or money. The
 * older `info` / `success` / `error` names still render, and are what the pages
 * written before this file was extended pass.
 */
export function InfoBox({
  type = 'info',
  title,
  children,
}: {
  type?: CalloutType;
  title?: string;
  children: ReactNode;
}) {
  return (
    <div className={`info-box ${type}`}>
      {title && (
        <div className="info-box-title">
          <span aria-hidden="true">{CALLOUT_ICONS[type]}</span>
          <span>{title}</span>
        </div>
      )}
      <div className="info-box-content">{children}</div>
    </div>
  );
}

/** The doc-facing name for the same component. */
export const Callout = InfoBox;

export function ApiTable({
  headers,
  rows,
  caption,
}: {
  headers: string[];
  rows: (string | ReactNode)[][];
  /** Say where the rows came from — a `--help` output, a signature, a docstring. */
  caption?: ReactNode;
}) {
  return (
    <div
      className="api-table-container"
      tabIndex={0}
      role="group"
      aria-label={typeof caption === 'string' ? `Table: ${caption}` : `Table: ${headers.filter(Boolean).join(', ')}`}
    >
      <table className="api-table">
        <thead>
          <tr>
            {headers.map((header, i) => (
              <th key={i} scope="col">
                {header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {row.map((cell, j) => (
                <td key={j}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {caption && <p className="api-table-caption">{caption}</p>}
    </div>
  );
}

export function QuickLinks({
  links,
}: {
  links: { icon: string; title: string; description: string; path: string }[];
}) {
  return (
    <div className="quick-links">
      {links.map((link, i) => (
        <Link key={i} to={link.path} className="quick-link-card">
          <div className="quick-link-icon" aria-hidden="true">
            {link.icon}
          </div>
          <div className="quick-link-title">{link.title}</div>
          <div className="quick-link-desc">{link.description}</div>
        </Link>
      ))}
    </div>
  );
}

export function FeatureList({
  features,
}: {
  features: { icon: string; title: string; description: ReactNode }[];
}) {
  return (
    <div className="feature-list">
      {features.map((feature, i) => (
        <div key={i} className="feature-item">
          <span className="feature-icon" aria-hidden="true">
            {feature.icon}
          </span>
          <div className="feature-content">
            <strong>{feature.title}</strong>
            <span>{feature.description}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

/**
 * A numbered sequence the reader is meant to carry out in order.
 *
 * Use it for something that has to happen in order — install, then configure,
 * then run. A list of things that are merely related is a list, not steps.
 */
export function Steps({ children }: { children: ReactNode }) {
  const items = React.Children.toArray(children);
  return (
    <ol className="doc-steps">
      {items.map((item, i) => (
        <li key={i} className="doc-step">
          <span className="doc-step-number" aria-hidden="true">
            {i + 1}
          </span>
          <div className="doc-step-body">{item}</div>
        </li>
      ))}
    </ol>
  );
}

export function Step({ title, children }: { title: string; children: ReactNode }) {
  return (
    <>
      <p className="doc-step-title">{title}</p>
      {children}
    </>
  );
}

/**
 * Render a navigation lede.
 *
 * A lede in `nav.ts` is plain text, and the ones naming an identifier write it
 * between backticks the way the rest of the project does. Rendered as-is the
 * backticks would show, so they become `<code>` here — in the "see also" cards
 * and anywhere else a lede is displayed.
 */
export function Lede({ text }: { text: string }) {
  return (
    <>
      {text.split(/`([^`]+)`/).map((part, i) =>
        i % 2 === 1 ? <code key={i}>{part}</code> : <React.Fragment key={i}>{part}</React.Fragment>,
      )}
    </>
  );
}

/**
 * The three pages a reader most often wants next.
 *
 * Takes routes, not titles: the title and the one-line description come from
 * `nav.ts`, so a page that is renamed is renamed everywhere that points at it.
 */
export function SeeAlso({ paths }: { paths: string[] }) {
  const entries = paths.map((path) => pageFor(path)).filter((page) => page !== undefined);
  if (entries.length === 0) return null;
  return (
    <section className="see-also" aria-labelledby="see-also-heading">
      <h2 id="see-also-heading" className="see-also-heading">
        See also
      </h2>
      <ul className="see-also-list">
        {entries.map((page) => (
          <li key={page.path}>
            <Link to={page.path} className="see-also-link">
              <Link2 size={15} aria-hidden="true" />
              <span>
                <strong>{page.title}</strong>
                <span className="see-also-lede">
                  <Lede text={page.lede} />
                </span>
              </span>
            </Link>
          </li>
        ))}
      </ul>
    </section>
  );
}
