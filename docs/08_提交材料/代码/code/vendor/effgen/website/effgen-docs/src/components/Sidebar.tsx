import React, { useMemo, useRef, useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import {
  Activity, AlertTriangle, AppWindow, ArrowRightLeft, BarChart3, Bell, Blocks,
  Book, BookOpen, Bot, Boxes, Braces, Brain, Bug, CheckCircle, ChefHat,
  ChevronDown, ChevronRight, Cloud, Code, Cog, Cpu, Database, DollarSign,
  FileCode, FileText, Files, FlaskConical, FolderOpen, FolderPlus, Gauge,
  GitBranch, GitCompare, Globe, GraduationCap, Hammer, HardDrive, Hash, History,
  HelpCircle, Image, Layers, LayoutDashboard, LayoutGrid, Library, LineChart,
  ListChecks, Lock, MessageSquare, Minimize2, Network, Notebook, Palette,
  PenLine, Play, Plug, Puzzle, Radar, Rocket, Route, Save, ScrollText, Search,
  Server, ServerCog, Shield, ShieldCheck, SlidersHorizontal, Terminal,
  UserCheck, Users, Workflow, Wrench, X, Zap, Download,
} from 'lucide-react';
import './Sidebar.css';
import { NAV, groupFor } from '../nav';
import { HEADINGS } from '../searchIndex.generated';
import { usePyPIVersion } from '../hooks/usePyPIVersion';
import { useFocusTrap } from '../hooks/useFocusTrap';

/**
 * One icon per route.
 *
 * Kept here rather than in `nav.ts` so that file stays free of components and
 * can be read by the build scripts. A route with no entry falls back to a hash,
 * which is what a new page looks like until it is given one.
 */
const ICONS: Record<string, React.ReactNode> = {
  '/introduction': <Book size={18} />,
  '/installation': <Download size={18} />,
  '/quickstart': <Rocket size={18} />,
  '/first-project': <FolderPlus size={18} />,
  '/configuration': <Cog size={18} />,
  '/migration': <ArrowRightLeft size={18} />,
  '/faq': <HelpCircle size={18} />,
  '/releases': <Activity size={18} />,

  '/agents': <Bot size={18} />,
  '/presets': <Boxes size={18} />,
  '/models': <Brain size={18} />,
  '/providers': <Server size={18} />,
  '/openai-compatible': <Plug size={18} />,
  '/local-models': <Cpu size={18} />,
  '/catalog': <Library size={18} />,
  '/routing': <Route size={18} />,
  '/tool-calling': <Puzzle size={18} />,
  '/generation': <SlidersHorizontal size={18} />,

  '/tools': <Wrench size={18} />,
  '/tools/gallery': <LayoutGrid size={18} />,
  '/native-provider-tools': <Plug size={18} />,
  '/custom-tools': <Hammer size={18} />,
  '/execution': <Zap size={18} />,
  '/protocols': <Globe size={18} />,

  '/memory': <Database size={18} />,
  '/sessions': <MessageSquare size={18} />,
  '/compaction': <Minimize2 size={18} />,
  '/rag': <BookOpen size={18} />,
  '/multimodal': <Image size={18} />,

  '/prompts': <FileText size={18} />,
  '/prompts/gallery': <Files size={18} />,
  '/prompts/authoring': <PenLine size={18} />,

  '/multi-agent': <Users size={18} />,
  '/workflows': <Workflow size={18} />,
  '/checkpointing': <Save size={18} />,
  '/middleware': <Layers size={18} />,
  '/domains': <Blocks size={18} />,

  '/guardrails': <Shield size={18} />,
  '/human-loop': <UserCheck size={18} />,
  '/security': <Lock size={18} />,
  '/reliability': <ShieldCheck size={18} />,
  '/errors': <AlertTriangle size={18} />,

  '/api-server': <ServerCog size={18} />,
  '/openai-api': <Network size={18} />,
  '/clients': <Code size={18} />,
  '/deployment': <Cloud size={18} />,
  '/hardware': <HardDrive size={18} />,

  '/observability': <Radar size={18} />,
  '/metrics': <BarChart3 size={18} />,
  '/tracing': <GitBranch size={18} />,
  '/slos': <Bell size={18} />,
  '/loadtest': <Gauge size={18} />,
  '/cost': <DollarSign size={18} />,

  '/evaluation': <CheckCircle size={18} />,
  '/compare': <GitCompare size={18} />,

  '/cli': <Terminal size={18} />,
  '/cli/run': <Play size={18} />,
  '/cli/code': <FileCode size={18} />,
  '/cli/top': <LineChart size={18} />,
  '/cli/reports': <ScrollText size={18} />,
  '/cli/history': <History size={18} />,
  '/cli/appearance': <Palette size={18} />,
  '/cli/batch': <ListChecks size={18} />,

  '/dashboard': <LayoutDashboard size={18} />,
  '/playground': <FlaskConical size={18} />,
  '/jupyter': <Notebook size={18} />,
  '/vscode': <AppWindow size={18} />,
  '/debug': <Bug size={18} />,

  '/api-reference': <Braces size={18} />,
  '/tutorials': <GraduationCap size={18} />,
  '/cookbook': <ChefHat size={18} />,
  '/examples': <FolderOpen size={18} />,
};

// The documentation is served under the landing site, at <site>/docs/. Vite's
// BASE_URL is that prefix, so trimming the trailing "docs/" gives the way back
// to the site root under a plain domain and under a project-page path alike.
const SITE_ROOT = import.meta.env.BASE_URL.replace(/docs\/?$/, '') || '/';

interface SearchHit {
  path: string;
  title: string;
  /** The heading matched, when the match was inside the page rather than its title. */
  heading?: { text: string; id: string };
}

/**
 * Search page titles and the headings inside them.
 *
 * The headings come from `searchIndex.generated.ts`, which is written from the
 * page sources at build time — so a section can be found without the page it
 * lives on having been loaded. A page whose title matches is offered whole; a
 * page whose headings match is offered as those sections.
 */
function search(query: string, limit = 24): SearchHit[] {
  // Hyphens, slashes and dots are word separators here, on both sides of the
  // comparison — so "rate limit" finds "Rate-Limit Stores", and "tool calling"
  // finds "Tool-calling". Without it a reader has to guess the punctuation.
  const fold = (text: string) => text.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
  const needle = fold(query);
  if (needle.length < 2) return [];
  const hits: SearchHit[] = [];

  for (const group of NAV) {
    for (const page of group.pages) {
      if (fold(page.title).includes(needle) || fold(page.path).includes(needle)) {
        hits.push({ path: page.path, title: page.title });
      }
      for (const heading of HEADINGS[page.path] ?? []) {
        if (fold(heading.text).includes(needle)) {
          hits.push({ path: page.path, title: page.title, heading });
        }
      }
    }
  }
  return hits.slice(0, limit);
}

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function Sidebar({ isOpen, onClose }: SidebarProps) {
  const location = useLocation();
  const [query, setQuery] = useState('');
  const pypiVersion = usePyPIVersion();
  const asideRef = useRef<HTMLElement>(null);

  // The group holding the page you are on is open; the rest stay shut until you
  // open them. With seventy-two entries, opening everything at once would put
  // the page you are reading several screens down its own navigation.
  const currentGroup = groupFor(location.pathname)?.id;
  const [expanded, setExpanded] = useState<string[]>(currentGroup ? [currentGroup] : []);

  // Opening the group happens while rendering the navigation for a new address,
  // not afterwards in an effect, so the panel never paints with the group you
  // just navigated into still shut. `openedFor` remembers which address was
  // answered, so closing the group by hand keeps it closed until you move to a
  // page in a different group.
  const [openedFor, setOpenedFor] = useState(currentGroup);
  if (currentGroup && currentGroup !== openedFor) {
    setOpenedFor(currentGroup);
    setExpanded((prev) => (prev.includes(currentGroup) ? prev : [...prev, currentGroup]));
  }

  // On a narrow screen this panel is a modal menu over the page, so it answers
  // a keyboard the way the landing site's mobile menu does: Tab stays inside,
  // Escape closes it, and focus goes back to the button that opened it. On a
  // wide screen `isOpen` is never set — the panel is simply part of the page —
  // and the trap stays off.
  useFocusTrap(asideRef, isOpen, onClose);

  const toggleGroup = (id: string) =>
    setExpanded((prev) => (prev.includes(id) ? prev.filter((t) => t !== id) : [...prev, id]));

  const hits = useMemo(() => search(query), [query]);
  const searching = query.trim().length >= 2;

  const leave = () => {
    setQuery('');
    onClose();
  };

  return (
    <>
      <div className={`sidebar-overlay ${isOpen ? 'active' : ''}`} onClick={onClose} />
      <aside id="docs-sidebar" ref={asideRef} className={`sidebar ${isOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <a href={SITE_ROOT} className="sidebar-logo" onClick={onClose}>
            <div className="logo-icon">
              <svg width="32" height="32" viewBox="0 0 40 40" fill="none" aria-hidden="true">
                <path d="M20 2L36 11V29L20 38L4 29V11L20 2Z" stroke="url(#logoGrad)" strokeWidth="2.5" strokeLinejoin="round"/>
                <circle cx="20" cy="20" r="5" fill="url(#logoGrad)"/>
                <defs>
                  <linearGradient id="logoGrad" x1="4" y1="2" x2="36" y2="38">
                    <stop offset="0%" stopColor="#00ff88"/>
                    <stop offset="50%" stopColor="#00c96e"/>
                    <stop offset="100%" stopColor="#00e5ff"/>
                  </linearGradient>
                </defs>
              </svg>
            </div>
            <span className="logo-text">effGen</span>
            <span className="version-badge">v{pypiVersion.loading ? '…' : pypiVersion.version}</span>
          </a>
          <button className="sidebar-close" onClick={onClose} aria-label="Close the menu">
            <X size={20} />
          </button>
        </div>

        <div className="sidebar-search">
          <Search size={16} className="search-icon" aria-hidden="true" />
          <input
            type="search"
            placeholder="Search pages and sections…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Escape' && query) {
                e.stopPropagation();
                setQuery('');
              }
            }}
            className="search-input"
            aria-label="Search the documentation"
            aria-describedby="sidebar-search-status"
          />
          {query && (
            <button className="search-clear" onClick={() => setQuery('')} aria-label="Clear the search">
              <X size={14} />
            </button>
          )}
        </div>

        <p id="sidebar-search-status" className="sr-only" role="status">
          {searching ? `${hits.length} result${hits.length === 1 ? '' : 's'} for ${query}` : ''}
        </p>

        {searching ? (
          <nav className="sidebar-nav" aria-label="Search results">
            {hits.length === 0 ? (
              <p className="search-empty">
                Nothing matches “{query}”. Search looks across page titles and the
                headings inside them.
              </p>
            ) : (
              <ul className="search-results">
                {hits.map((hit, i) => (
                  <li key={`${hit.path}-${hit.heading?.id ?? 'page'}-${i}`}>
                    <NavLink
                      to={hit.heading ? `${hit.path}#${hit.heading.id}` : hit.path}
                      className="search-result"
                      onClick={leave}
                    >
                      <span className="search-result-icon" aria-hidden="true">
                        {hit.heading ? <Hash size={15} /> : (ICONS[hit.path] ?? <Hash size={15} />)}
                      </span>
                      <span className="search-result-text">
                        <span className="search-result-title">
                          {hit.heading ? hit.heading.text : hit.title}
                        </span>
                        {hit.heading && <span className="search-result-page">{hit.title}</span>}
                      </span>
                    </NavLink>
                  </li>
                ))}
              </ul>
            )}
          </nav>
        ) : (
          <nav className="sidebar-nav" aria-label="Documentation">
            {NAV.map((group) => {
              const open = expanded.includes(group.id);
              return (
                <div key={group.id} className="nav-section">
                  <button
                    className="nav-section-title"
                    onClick={() => toggleGroup(group.id)}
                    aria-expanded={open}
                    aria-controls={`nav-group-${group.id}`}
                  >
                    <span>{group.title}</span>
                    {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                  </button>
                  {open && (
                    <ul className="nav-list" id={`nav-group-${group.id}`}>
                      {group.pages.map((page) => (
                        <li key={page.path}>
                          <NavLink
                            to={page.path}
                            end
                            className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
                            onClick={leave}
                          >
                            <span className="nav-icon" aria-hidden="true">
                              {ICONS[page.path] ?? <Hash size={18} />}
                            </span>
                            <span className="nav-label">{page.title}</span>
                          </NavLink>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              );
            })}
          </nav>
        )}

        <div className="sidebar-footer">
          <a
            href="https://github.com/ctrl-gaurav/effGen"
            target="_blank"
            rel="noopener noreferrer"
            className="github-link"
            aria-label="effGen on GitHub"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
              <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
            </svg>
            <span>GitHub</span>
          </a>
        </div>
      </aside>
    </>
  );
}
