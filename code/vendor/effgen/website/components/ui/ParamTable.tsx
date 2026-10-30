"use client";

import { ReactNode } from "react";

export interface Param {
  /** The flag or parameter exactly as it is written, e.g. `--session-id` or `raise_on_error`. */
  name: string;
  /** Its type, as the signature or `--help` gives it. */
  type?: string;
  /** The default, as a literal. Omit when there is none. */
  default?: string;
  /** Whether the caller has to supply it. */
  required?: boolean;
  /** What it does. One sentence, matching the help text rather than paraphrasing it. */
  description: ReactNode;
}

interface ParamTableProps {
  params: Param[];
  /** Column heading for the first column. `Flag` on a CLI page, `Parameter` in the API. */
  nameLabel?: string;
  /** A caption naming the command or signature these came from. */
  caption?: ReactNode;
  className?: string;
}

/**
 * The options table every command and every signature on the site is documented
 * with.
 *
 * The contract that matters is not visual: **the rows have to match `--help` or
 * the docstring exactly** — same flag spelling, same type, same default, same
 * meaning. A table that quietly paraphrases is worse than no table, because a
 * reader will copy from it.
 *
 * It scrolls inside its own container, so a wide table never makes the page
 * scroll sideways.
 */
export default function ParamTable({
  params,
  nameLabel = "Option",
  caption,
  className = "",
}: ParamTableProps) {
  const showDefaults = params.some((p) => p.default !== undefined);

  return (
    <div className={className}>
      <div
        className="overflow-x-auto rounded-2xl border border-gray-200 dark:border-green-500/15 focus-visible:outline focus-visible:outline-2 focus-visible:outline-green-500 focus-visible:-outline-offset-2"
        tabIndex={0}
        role="group"
        aria-label={nameLabel ? `${nameLabel} table` : 'Options table'}
      >
        <table className="w-full text-left text-sm border-collapse">
          <thead>
            <tr className="bg-gray-50 dark:bg-[#071a10]">
              <th scope="col" className="px-4 py-3 font-semibold text-gray-700 dark:text-gray-300 whitespace-nowrap">
                {nameLabel}
              </th>
              <th scope="col" className="px-4 py-3 font-semibold text-gray-700 dark:text-gray-300 whitespace-nowrap">
                Type
              </th>
              {showDefaults && (
                <th scope="col" className="px-4 py-3 font-semibold text-gray-700 dark:text-gray-300 whitespace-nowrap">
                  Default
                </th>
              )}
              <th scope="col" className="px-4 py-3 font-semibold text-gray-700 dark:text-gray-300">
                Description
              </th>
            </tr>
          </thead>
          <tbody>
            {params.map((param) => (
              <tr
                key={param.name}
                className="border-t border-gray-200 dark:border-green-500/10 align-top"
              >
                <td className="px-4 py-3 whitespace-nowrap">
                  <code className="font-mono text-[13px] text-green-700 dark:text-green-400">
                    {param.name}
                  </code>
                  {param.required && (
                    <span className="ml-2 text-[10px] uppercase tracking-wide font-semibold text-orange-700 dark:text-orange-400">
                      required
                    </span>
                  )}
                </td>
                <td className="px-4 py-3 whitespace-nowrap font-mono text-[13px] text-gray-600 dark:text-gray-400">
                  {param.type ?? "—"}
                </td>
                {showDefaults && (
                  <td className="px-4 py-3 whitespace-nowrap font-mono text-[13px] text-gray-600 dark:text-gray-400">
                    {param.default ?? "—"}
                  </td>
                )}
                <td className="px-4 py-3 text-gray-600 dark:text-gray-400">
                  {param.description}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {caption && (
        <p className="mt-2 text-xs text-gray-600 dark:text-gray-400">{caption}</p>
      )}
    </div>
  );
}
