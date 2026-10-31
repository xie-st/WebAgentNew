#!/usr/bin/env python3
"""Report effGen's headline counts (tools, prompt templates, presets, providers,
catalogued models) straight from the live package surfaces.

These numbers drift as the framework grows, so documentation and badges should be
populated from this script rather than hand-maintained. It is the single source of
truth for "how many X does effGen ship".

    python scripts/gen_counts.py            # human-readable table
    python scripts/gen_counts.py --json     # machine-readable JSON

The model count reflects the bundled offline catalog snapshot (run
``effgen models refresh`` to update it); every other count is computed by
introspecting the installed package, so it is always exact.
"""

from __future__ import annotations

import argparse
import json
import sys


def collect() -> dict:
    counts: dict[str, object] = {}

    # Builtin tools — the registry lazily discovers them on first access.
    from effgen.tools import get_registry

    counts["tools"] = len(get_registry().list_tools())

    # Two distinct prompt surfaces:
    #  - the default TemplateManager (generic, reusable templates)
    #  - the domain prompt library (curated, evaluated prompts; `effgen prompts list`)
    from effgen.prompts import create_default_template_manager

    counts["prompt_templates"] = len(create_default_template_manager().list_templates())

    from effgen.prompts.library import registry as prompt_library

    counts["prompt_library"] = len(prompt_library)

    # Ready-to-use agent presets.
    from effgen import list_presets

    counts["presets"] = len(list_presets())

    # Providers and their catalogued models (offline snapshot).
    from effgen.models import registry as model_registry

    providers = sorted(model_registry.list_providers())
    per_provider = {p: len(model_registry.list_models(p)) for p in providers}
    counts["providers"] = len(providers)
    counts["models"] = sum(per_provider.values())
    counts["models_by_provider"] = per_provider

    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="emit JSON")
    args = ap.parse_args()

    counts = collect()

    if args.json:
        print(json.dumps(counts, indent=2, sort_keys=True))
        return 0

    print("effGen package counts")
    print("=" * 40)
    print(f"  Tools             {counts['tools']:>5}")
    print(f"  Prompt templates  {counts['prompt_templates']:>5}")
    print(f"  Prompt library    {counts['prompt_library']:>5}")
    print(f"  Presets           {counts['presets']:>5}")
    print(f"  Providers         {counts['providers']:>5}")
    print(f"  Models (catalog)  {counts['models']:>5}")
    print("-" * 40)
    print("  Models by provider:")
    for prov, n in sorted(counts["models_by_provider"].items()):
        print(f"    {prov:<12} {n:>5}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
