import { useMemo, useState } from 'react';
import { BookOpen, Search } from 'lucide-react';
import { Link } from 'react-router-dom';
import { Callout, CodeBlock, DocPage, ParamTable, SeeAlso } from '../components/docs';
import type { Param } from '../components/docs';
import { siteData, version } from '../siteData';
import './PromptGallery.css';

const prompts = siteData.prompts;

const VARIANT_LABEL: Record<string, string> = {
  zero_shot: 'zero-shot',
  few_shot: 'few-shot',
  cot: 'chain of thought',
  structured: 'structured',
  tool: 'tool-augmented',
};

const CHECK_LABEL: Record<string, string> = {
  json: 'output checked as JSON',
  regex: 'output checked against a pattern',
  function: 'output checked by a function',
  none: 'rendering checked, output not',
};

const DOMAIN_NOTE: Record<string, string> = {
  legal: 'Every legal template carries the same disclaimer verbatim: this output is for informational purposes only and does not constitute legal advice.',
  medical: 'Every medical template carries a disclaimer and is written for clinicians rather than for patients.',
};

export default function PromptGallery() {
  const [query, setQuery] = useState('');
  const [domain, setDomain] = useState('all');

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return prompts.items.filter((prompt) => {
      if (domain !== 'all' && prompt.domain !== domain) return false;
      if (!needle) return true;
      return (
        prompt.name.toLowerCase().includes(needle) ||
        prompt.description.toLowerCase().includes(needle) ||
        prompt.variant.includes(needle) ||
        prompt.tags.some((tag) => tag.toLowerCase().includes(needle))
      );
    });
  }, [query, domain]);

  const byDomain = prompts.domains
    .map((name) => [name, visible.filter((p) => p.domain === name)] as const)
    .filter(([, items]) => items.length > 0);

  return (
    <DocPage
      subtitle="Every template in the library, by domain, with its variables."
      icon={<BookOpen size={48} />}
      toc={false}
    >
      <p>
        All {prompts.library} templates effGen {version} ships, grouped by domain, with the inputs
        each one takes and how its output is checked. Every row is read from the installed package,
        so a template the library gains appears here without the page being edited.
      </p>

      <Callout type="info" title="Rendering one">
        <p>
          <code>registry.get(name).render(**inputs)</code> in Python, or{' '}
          <code>effgen prompts render &lt;name&gt;</code> from a shell — both are on{' '}
          <Link to="/prompts">Prompt library</Link>. Omitting the inputs renders the template's
          own worked example.
        </p>
      </Callout>

      <div className="prompt-gallery-controls">
        <label className="prompt-gallery-search">
          <Search size={16} aria-hidden="true" />
          <span className="sr-only">Search the prompt gallery</span>
          <input
            type="search"
            value={query}
            placeholder={`Search ${prompts.library} templates by name, variant or tag`}
            onChange={(event) => setQuery(event.target.value)}
          />
        </label>

        <div className="prompt-gallery-filters" role="group" aria-label="Filter by domain">
          <button
            type="button"
            className={domain === 'all' ? 'is-active' : undefined}
            aria-pressed={domain === 'all'}
            onClick={() => setDomain('all')}
          >
            All <span>{prompts.library}</span>
          </button>
          {prompts.domains.map((name) => (
            <button
              key={name}
              type="button"
              className={domain === name ? 'is-active' : undefined}
              aria-pressed={domain === name}
              onClick={() => setDomain(name)}
            >
              {name} <span>{prompts.domain_counts[name]}</span>
            </button>
          ))}
        </div>
      </div>

      <p className="prompt-gallery-count" aria-live="polite">
        {visible.length === prompts.library
          ? `Showing all ${prompts.library} templates.`
          : `Showing ${visible.length} of ${prompts.library} templates.`}
      </p>

      {visible.length === 0 && (
        <p className="prompt-gallery-empty">
          No template matches that. Clear the search, or pick a different domain.
        </p>
      )}

      {byDomain.map(([name, items]) => (
        <section key={name} id={`domain-${name}`} className="prompt-domain">
          <h2>{name}</h2>
          {DOMAIN_NOTE[name] && <p className="prompt-domain-note">{DOMAIN_NOTE[name]}</p>}

          {items.map((prompt) => {
            const params: Param[] = prompt.variables.map((variable) => ({
              name: variable.name,
              type: variable.type,
              required: variable.required,
              description: variable.enum
                ? `${variable.description} One of: ${variable.enum.join(', ')}.`
                : variable.description,
            }));

            return (
              <article key={prompt.name} className="prompt-card" id={`prompt-${prompt.name}`}>
                <header className="prompt-card-head">
                  <h3>
                    <code>{prompt.name}</code>
                  </h3>
                  <div className="prompt-card-badges">
                    <span className="prompt-badge prompt-badge-variant">
                      {VARIANT_LABEL[prompt.variant] ?? prompt.variant}
                    </span>
                    <span className="prompt-badge">{CHECK_LABEL[prompt.check] ?? prompt.check}</span>
                  </div>
                </header>

                <p className="prompt-card-description">{prompt.description}</p>

                {params.length > 0 && (
                  <ParamTable
                    nameLabel="Variable"
                    params={params}
                    caption={
                      <>
                        {params.length} variable{params.length === 1 ? '' : 's'}, from the
                        template's own <code>input_schema</code>. Its worked example fills{' '}
                        {prompt.fixture_keys.length} of them.
                      </>
                    }
                  />
                )}
              </article>
            );
          })}
        </section>
      ))}

      <h2>Rendering any of them</h2>

      <CodeBlock
        filename="render_any.py"
        code={`from effgen.prompts.library import registry

prompt = registry.get("coding.docstring_fill.v1")

print(prompt.render_fixture())          # the template's own worked example
print(prompt.render(code="def add(a, b):\\n    return a + b", style="google"))`}
        caption={
          <>
            <code>render_fixture()</code> is the shortest way to see what a template produces —
            it is what <code>effgen prompts render</code> prints when you give it no input file.
          </>
        }
      />

      <SeeAlso paths={['/prompts', '/prompts/authoring', '/domains']} />
    </DocPage>
  );
}
