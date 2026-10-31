import { useEffect, useId, useRef, useState } from 'react';
import mermaid from 'mermaid';
import { useTheme } from '../context/theme';
import './MermaidDiagram.css';

interface MermaidDiagramProps {
  chart: string;
  title?: string;
  /**
   * What the diagram shows, in a sentence, for a reader who cannot see it.
   * A diagram is content, so this is not optional in new work; it falls back to
   * the title on the pages written before it existed.
   */
  description?: string;
}

// Both palettes are the documentation's own tokens, written out as literals
// because mermaid resolves these at render time and cannot read a CSS variable.
// They are the same two sets the rest of the page uses, so a diagram sits in the
// page rather than on it — in either colour mode.
const THEME_VARIABLES = {
  dark: {
    primaryColor: '#00c96e',
    primaryTextColor: '#e6efe9',
    primaryBorderColor: '#00ff88',
    lineColor: '#8fa89a',
    secondaryColor: '#1a221e',
    tertiaryColor: '#0f1512',
    background: '#0a0f0d',
    mainBkg: '#1a221e',
    secondBkg: '#0f1512',
    textColor: '#e6efe9',
    nodeBorder: '#00ff88',
    nodeTextColor: '#e6efe9',
    clusterBkg: '#151c18',
    clusterBorder: '#1e2e26',
    titleColor: '#e6efe9',
    edgeLabelBackground: '#1a221e',
    labelColor: '#e6efe9',
    actorBkg: '#1a221e',
    actorBorder: '#00ff88',
    actorTextColor: '#e6efe9',
    signalColor: '#8fa89a',
    signalTextColor: '#e6efe9',
    labelBoxBkgColor: '#1a221e',
    labelBoxBorderColor: '#00ff88',
    labelTextColor: '#e6efe9',
    loopTextColor: '#e6efe9',
    noteBkgColor: '#151c18',
    noteBorderColor: '#2d4437',
    noteTextColor: '#e6efe9',
    altBackground: '#0f1512',
  },
  light: {
    primaryColor: '#e8f5ee',
    primaryTextColor: '#1a1a1a',
    primaryBorderColor: '#00894a',
    lineColor: '#555555',
    secondaryColor: '#f7f8f8',
    tertiaryColor: '#f0f1f1',
    background: '#ffffff',
    mainBkg: '#ffffff',
    secondBkg: '#f7f8f8',
    textColor: '#1a1a1a',
    nodeBorder: '#00894a',
    nodeTextColor: '#1a1a1a',
    clusterBkg: '#f0f1f1',
    clusterBorder: '#e2e2e2',
    titleColor: '#1a1a1a',
    edgeLabelBackground: '#ffffff',
    labelColor: '#1a1a1a',
    actorBkg: '#ffffff',
    actorBorder: '#00894a',
    actorTextColor: '#1a1a1a',
    signalColor: '#555555',
    signalTextColor: '#1a1a1a',
    labelBoxBkgColor: '#ffffff',
    labelBoxBorderColor: '#00894a',
    labelTextColor: '#1a1a1a',
    loopTextColor: '#1a1a1a',
    noteBkgColor: '#f0f1f1',
    noteBorderColor: '#d5d5d5',
    noteTextColor: '#1a1a1a',
    altBackground: '#f7f8f8',
  },
} as const;

export default function MermaidDiagram({ chart, title, description }: MermaidDiagramProps) {
  const { theme } = useTheme();
  const [svg, setSvg] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  // Stable across renders and unique on the page, so two diagrams never collide
  // over an element id and a re-render does not orphan the last one's nodes.
  const reactId = useId();
  const domId = `mermaid${reactId.replace(/[^a-zA-Z0-9]/g, '')}`;
  const alive = useRef(true);

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  useEffect(() => {
    const dark = theme === 'dark';
    mermaid.initialize({
      startOnLoad: false,
      // Diagram text is written by this repository, never by a visitor, but the
      // strict level is what keeps a future page from being able to put markup
      // into a label.
      securityLevel: 'strict',
      theme: dark ? 'dark' : 'base',
      themeVariables: dark ? THEME_VARIABLES.dark : THEME_VARIABLES.light,
      flowchart: { curve: 'basis', padding: 20, useMaxWidth: false },
      sequence: { useMaxWidth: false },
      fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    });

    let cancelled = false;
    (async () => {
      try {
        const { svg: rendered } = await mermaid.render(domId, chart);
        if (!cancelled && alive.current) {
          setSvg(rendered);
          setError(null);
        }
      } catch (err) {
        if (cancelled || !alive.current) return;
        setSvg('');
        setError(err instanceof Error ? err.message : String(err));
      }
    })();

    return () => {
      cancelled = true;
      // mermaid leaves the element it measured in behind when a render is
      // abandoned; without this a theme switch grows one orphan per diagram.
      document.getElementById(`d${domId}`)?.remove();
    };
  }, [chart, theme, domId]);

  const label = description ?? title;

  return (
    <figure className="mermaid-container">
      {title && <figcaption className="mermaid-title">{title}</figcaption>}
      {error ? (
        <p className="mermaid-error">
          This diagram could not be drawn: {error}
        </p>
      ) : (
        // The one scroll container: a wide diagram scrolls here, never taking
        // the page sideways with it. Focusable so a keyboard can reach it.
        <div
          className="mermaid-scroll"
          tabIndex={0}
          role="img"
          aria-label={label ?? 'Diagram'}
          dangerouslySetInnerHTML={{ __html: svg }}
        />
      )}
    </figure>
  );
}
