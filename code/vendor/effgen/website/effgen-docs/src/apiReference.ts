// The public surface of the `effgen` package, generated from the installed
// framework by `scripts/gen_api_reference.py` at the repository root.
//
// Nothing here is written by hand. Every name, signature, default and docstring
// on `/api-reference` is read out of `effgen.__all__` at generation time, so the
// page cannot describe a release other than the one it was generated from.
//
//   export PATH=/path/to/your/effgen/env/bin:$PATH   # or activate it however you do
//   python scripts/gen_api_reference.py            # regenerate
//   python scripts/gen_api_reference.py --check    # fail if it is stale

import raw from './data/apiReference.json'

/** What kind of object a name is bound to. */
export type ApiKind =
  | 'class'
  | 'dataclass'
  | 'enum'
  | 'exception'
  | 'function'
  | 'alias'
  | 'module'
  | 'value'

export interface ApiParam {
  name: string
  type: string | null
  /** The default as a Python literal, or null when the caller must supply one. */
  default: string | null
  required: boolean
  keyword_only: boolean
  /** From the docstring's `Args:` or `Attributes:` section; empty when undocumented. */
  description: string
}

export interface ApiMember {
  name: string
  kind: 'method' | 'classmethod' | 'staticmethod' | 'property'
  signature: string
  summary: string
  is_async: boolean
  /** The base class it is defined on, when it is not this class. */
  inherited_from: string | null
}

export interface ApiName {
  name: string
  kind: ApiKind
  /** Where it is defined — not where you import it from, which is always `effgen`. */
  module: string
  area: string
  summary: string
  signature: string | null
  params: ApiParam[]
  returns: { type: string | null; description: string } | null
  raises: { name: string; description: string }[]
  bases: string[]
  members: ApiMember[]
  /** Enum members, or the arms of a union alias. */
  values: { name: string; value: string }[]
  is_async?: boolean
}

export interface ApiArea {
  id: string
  title: string
  blurb: string
  count: number
}

export interface ApiReference {
  derived_at: string
  version: string
  public_names: number
  kind_counts: Record<string, number>
  areas: ApiArea[]
  names: ApiName[]
  /** Names people look for that the top level does not export, and where they live. */
  not_exported: { name: string; module: string; kind: ApiKind; what: string }[]
}

export const apiReference = raw as unknown as ApiReference

/** The 223 names `effgen.__all__` carries, alphabetically. */
export const apiNames = apiReference.names

/** The id used in the URL fragment for one name — case-sensitive, so `Tool` and `tool` differ. */
export function anchorFor(name: string): string {
  return `api-${name}`
}
