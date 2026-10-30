import type { ReactNode } from 'react'
import './ParamTable.css'

export interface Param {
  /** The flag or parameter exactly as written — `--session-id`, `raise_on_error`. */
  name: string
  /** Its type, as the signature or `--help` gives it. */
  type?: string
  /** The default, as a literal. Omit when there is none. */
  default?: string
  /** Whether the caller has to supply it. */
  required?: boolean
  /** What it does. One sentence, matching the help text rather than paraphrasing. */
  description: ReactNode
}

interface ParamTableProps {
  params: Param[]
  /** Heading for the first column. `Flag` on a CLI page, `Parameter` in the API. */
  nameLabel?: string
  /** A caption naming the command or signature these rows came from. */
  caption?: ReactNode
}

/**
 * The options table every command and every signature in the documentation is
 * described with.
 *
 * The contract that matters is not visual: **the rows have to match `--help` or
 * the docstring exactly** — same spelling, same type, same default, same
 * meaning. A table that quietly paraphrases is worse than no table, because a
 * reader will copy from it and then be wrong.
 *
 * `ApiTable` in `DocPage.tsx` stays for free-form tables. This one is for
 * options, and its shape is what makes a missing default or an unstated type
 * visible while the page is being written.
 *
 * It scrolls inside its own container, so a wide table never makes the page
 * scroll sideways.
 */
export default function ParamTable({
  params,
  nameLabel = 'Option',
  caption,
}: ParamTableProps) {
  const showDefaults = params.some((p) => p.default !== undefined)

  return (
    <div className="param-table-wrap">
      <div className="param-table-scroll" tabIndex={0} role="group" aria-label={`${nameLabel} table`}>
        <table className="param-table">
          <thead>
            <tr>
              <th scope="col">{nameLabel}</th>
              <th scope="col">Type</th>
              {showDefaults && <th scope="col">Default</th>}
              <th scope="col">Description</th>
            </tr>
          </thead>
          <tbody>
            {params.map((param) => (
              <tr key={param.name}>
                <td>
                  <code className="param-name">{param.name}</code>
                  {param.required && <span className="param-required">required</span>}
                </td>
                <td className="param-mono">{param.type ?? '—'}</td>
                {showDefaults && (
                  <td className="param-mono">{param.default ?? '—'}</td>
                )}
                <td>{param.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {caption && <p className="param-table-caption">{caption}</p>}
    </div>
  )
}
