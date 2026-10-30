import React, { Suspense, lazy } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider } from './context/ThemeContext';
import Layout from './components/Layout';
import { NAV, REDIRECTS } from './nav';
import StubPage from './pages/StubPage';
import Introduction from './pages/Introduction';
import Installation from './pages/Installation';
import QuickStart from './pages/QuickStart';
import FirstProject from './pages/FirstProject';
import Configuration from './pages/Configuration';
import Migration from './pages/Migration';
import FAQ from './pages/FAQ';
import Releases from './pages/Releases';
import Agents from './pages/Agents';
import Presets from './pages/Presets';
import Models from './pages/Models';
import Providers from './pages/Providers';
import OpenAICompatible from './pages/OpenAICompatible';
import LocalModels from './pages/LocalModels';
import Catalog from './pages/Catalog';
import Routing from './pages/Routing';
import ToolCalling from './pages/ToolCalling';
import Generation from './pages/Generation';
import Tools from './pages/Tools';
import ToolGallery from './pages/ToolGallery';
import CustomTools from './pages/CustomTools';
import NativeProviderTools from './pages/NativeProviderTools';
import Execution from './pages/Execution';
import Protocols from './pages/Protocols';
import Memory from './pages/Memory';
import Sessions from './pages/Sessions';
import Compaction from './pages/Compaction';
import RAG from './pages/RAG';
import Multimodal from './pages/Multimodal';
import Prompts from './pages/Prompts';
import PromptGallery from './pages/PromptGallery';
import PromptAuthoring from './pages/PromptAuthoring';
import MultiAgent from './pages/MultiAgent';
import Workflows from './pages/Workflows';
import Checkpointing from './pages/Checkpointing';
import Middleware from './pages/Middleware';
import Domains from './pages/Domains';
import Guardrails from './pages/Guardrails';
import HumanLoop from './pages/HumanLoop';
import Security from './pages/Security';
import Reliability from './pages/Reliability';
import Errors from './pages/Errors';
import APIServer from './pages/APIServer';
import OpenAIAPI from './pages/OpenAIAPI';
import Clients from './pages/Clients';
import Deployment from './pages/Deployment';
import Hardware from './pages/Hardware';
import Observability from './pages/Observability';
import Metrics from './pages/Metrics';
import Tracing from './pages/Tracing';
import SLOs from './pages/SLOs';
import LoadTest from './pages/LoadTest';
import Cost from './pages/Cost';
import Evaluation from './pages/Evaluation';
import Compare from './pages/Compare';
import Cli from './pages/Cli';
import CliRun from './pages/CliRun';
import CliCode from './pages/CliCode';
import CliTop from './pages/CliTop';
import CliReports from './pages/CliReports';
import CliHistory from './pages/CliHistory';
import CliAppearance from './pages/CliAppearance';
import CliBatch from './pages/CliBatch';
import Dashboard from './pages/Dashboard';
import Playground from './pages/Playground';
import Jupyter from './pages/Jupyter';
import VSCode from './pages/VSCode';
import Debug from './pages/Debug';
// The API reference renders all 223 public names out of a 412 kB data file.
// Bundled with everything else it is the single largest thing on the site, and
// every one of the other 71 routes downloads it without using a byte. Behind a
// lazy boundary it is fetched only when someone opens that page.
const APIReference = lazy(() => import('./pages/APIReference'));
import Tutorials from './pages/Tutorials';
import Cookbook from './pages/Cookbook';
import Examples from './pages/Examples';
import NotFound from './pages/NotFound';
import './styles/globals.css';

/**
 * The routes that have a page written for them, and which component renders it.
 *
 * `nav.ts` holds the list of routes; this holds the components. A route in the
 * navigation with no entry here renders `StubPage`, which says so on the page
 * rather than leaving a reader on something that looks finished — so the
 * navigation is complete and walkable from now on, and a page that has not been
 * written is impossible to miss.
 *
 * Writing a page is therefore two lines: import it, and add it here. Nothing
 * else changes, and the sidebar, the breadcrumb, the previous/next pair and the
 * search index all pick it up from `nav.ts` as they already did.
 *
 * The tuple shape is read by `scripts/gen-search-index.mjs`, which maps a route
 * to the file whose headings belong to it.
 */
const PAGE_COMPONENTS: Array<[string, React.ReactElement]> = [
  ['/introduction', <Introduction />],
  ['/installation', <Installation />],
  ['/quickstart', <QuickStart />],
  ['/first-project', <FirstProject />],
  ['/configuration', <Configuration />],
  ['/migration', <Migration />],
  ['/faq', <FAQ />],
  ['/releases', <Releases />],
  ['/agents', <Agents />],
  ['/presets', <Presets />],
  ['/models', <Models />],
  ['/providers', <Providers />],
  ['/openai-compatible', <OpenAICompatible />],
  ['/local-models', <LocalModels />],
  ['/catalog', <Catalog />],
  ['/routing', <Routing />],
  ['/tool-calling', <ToolCalling />],
  ['/generation', <Generation />],
  ['/tools', <Tools />],
  ['/tools/gallery', <ToolGallery />],
  ['/custom-tools', <CustomTools />],
  ['/native-provider-tools', <NativeProviderTools />],
  ['/execution', <Execution />],
  ['/protocols', <Protocols />],
  ['/memory', <Memory />],
  ['/sessions', <Sessions />],
  ['/compaction', <Compaction />],
  ['/rag', <RAG />],
  ['/multimodal', <Multimodal />],
  ['/prompts', <Prompts />],
  ['/prompts/gallery', <PromptGallery />],
  ['/prompts/authoring', <PromptAuthoring />],
  ['/multi-agent', <MultiAgent />],
  ['/workflows', <Workflows />],
  ['/checkpointing', <Checkpointing />],
  ['/middleware', <Middleware />],
  ['/domains', <Domains />],
  ['/guardrails', <Guardrails />],
  ['/human-loop', <HumanLoop />],
  ['/security', <Security />],
  ['/reliability', <Reliability />],
  ['/errors', <Errors />],
  ['/api-server', <APIServer />],
  ['/openai-api', <OpenAIAPI />],
  ['/clients', <Clients />],
  ['/deployment', <Deployment />],
  ['/hardware', <Hardware />],
  ['/observability', <Observability />],
  ['/metrics', <Metrics />],
  ['/tracing', <Tracing />],
  ['/slos', <SLOs />],
  ['/loadtest', <LoadTest />],
  ['/cost', <Cost />],
  ['/evaluation', <Evaluation />],
  ['/compare', <Compare />],
  ['/cli', <Cli />],
  ['/cli/run', <CliRun />],
  ['/cli/code', <CliCode />],
  ['/cli/top', <CliTop />],
  ['/cli/reports', <CliReports />],
  ['/cli/history', <CliHistory />],
  ['/cli/appearance', <CliAppearance />],
  ['/cli/batch', <CliBatch />],
  ['/dashboard', <Dashboard />],
  ['/playground', <Playground />],
  ['/jupyter', <Jupyter />],
  ['/vscode', <VSCode />],
  ['/debug', <Debug />],
  [
    '/api-reference',
    <Suspense fallback={<div className="doc-loading">Loading the API reference…</div>}>
      <APIReference />
    </Suspense>,
  ],
  ['/tutorials', <Tutorials />],
  ['/cookbook', <Cookbook />],
  ['/examples', <Examples />],
];

const WRITTEN = new Map(PAGE_COMPONENTS);

function App() {
  return (
    <ThemeProvider>
      <BrowserRouter basename={import.meta.env.BASE_URL}>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Navigate to="/introduction" replace />} />

            {/* Every route in the navigation, in navigation order. */}
            {NAV.flatMap((group) =>
              group.pages.map((page) => (
                <Route
                  key={page.path}
                  path={page.path.slice(1)}
                  element={WRITTEN.get(page.path) ?? <StubPage path={page.path} />}
                />
              )),
            )}

            {/* Routes that were retired into the pages above. A bookmark, an
                older release's README or a search result still points at these,
                so each one lands on the page that took its topic over. */}
            {Object.entries(REDIRECTS).map(([from, to]) => (
              <Route key={from} path={from.slice(1)} element={<Navigate to={to} replace />} />
            ))}

            {/* Anything else. The site is a single-page app behind a catch-all
                redirect, so an unknown /docs/* address reaches the router rather
                than the host's 404 — which means the router has to answer it. */}
            <Route path="*" element={<NotFound />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ThemeProvider>
  );
}

export default App;
