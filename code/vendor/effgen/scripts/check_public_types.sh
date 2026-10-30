#!/usr/bin/env bash
# Type-check effGen's public API surface.
#
# effGen ships `py.typed`, so every user's mypy/pyright checks their code against
# our annotations. This script runs mypy over the modules that define the public
# surface in the project's permissive-but-real config, with
# `--follow-imports=silent` so we validate *those modules' own signatures*
# without drowning in transitive-internal noise.
#
# Policy: this lane is **allowed to warn** — internal type imprecision is
# reported (so drift is visible) but does not gate. What must never break is the
# public surface itself: that `effgen.__all__` fully resolves and `py.typed`
# ships is enforced deterministically by tests/unit/test_typed_surface.py, which
# *is* a hard gate.
#
# Usage: scripts/check_public_types.sh   (exit 0 unless mypy itself crashes)
set -uo pipefail
cd "$(dirname "$0")/.."

PUBLIC_MODULES=(
  effgen/__init__.py
  effgen/core/agent.py
  effgen/models/model_loader.py
  effgen/models/base.py
  effgen/tools/base_tool.py
  effgen/tools/__init__.py
  effgen/presets/registry.py
)

echo "==> mypy (public surface, permissive, allowed-to-warn)"
python -m mypy --follow-imports=silent --ignore-missing-imports "${PUBLIC_MODULES[@]}"
status=$?

# mypy exit codes: 0 = clean, 1 = type errors found (allowed-to-warn here),
# >=2 = mypy crashed / bad invocation (a real failure we want to surface).
if [ "$status" -ge 2 ]; then
  echo "ERROR: mypy failed to run (exit $status)" >&2
  exit "$status"
fi
if [ "$status" -eq 1 ]; then
  echo "NOTE: mypy reported type warnings on internal signatures (non-gating)."
fi
echo "==> public-surface type check completed (advisory)"
exit 0
