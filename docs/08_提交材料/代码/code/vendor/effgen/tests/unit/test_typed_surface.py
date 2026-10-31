"""The shipped typed surface must be accurate.

effGen ships ``effgen/py.typed``, which tells every user's mypy/pyright to type
-check their code against effGen's annotations. That is only safe if the public
surface is *not* broken: every name in ``effgen.__all__`` must resolve, and the
public callables must have introspectable signatures (no dangling lazy targets
or malformed stubs). These checks are deterministic and run in plain CI without
needing a type checker installed, complementing the advisory mypy lane
(``scripts/check_public_types.sh``).
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest

import effgen


def test_py_typed_marker_ships():
    marker = Path(effgen.__file__).parent / "py.typed"
    assert marker.exists(), (
        "effgen/py.typed is missing — either ship it (and keep the public "
        "surface type-accurate) or remove the typed-package claim entirely."
    )


def test_public_all_names_resolve():
    """Every advertised name imports — no broken lazy targets / dead stubs."""
    # The public surface includes the local engines, which import torch at module
    # scope; on an install without it those names resolve to a typed "optional
    # component is not installed" error rather than an object.
    pytest.importorskip("torch")

    missing: list[str] = []
    for name in effgen.__all__:
        try:
            getattr(effgen, name)
        except Exception as exc:  # noqa: BLE001 - collect all, report together
            missing.append(f"{name}: {type(exc).__name__}: {exc}")
    assert not missing, "Unresolvable names in effgen.__all__:\n  " + "\n  ".join(missing)


def test_public_callables_have_introspectable_signatures():
    """Public functions/classes expose a signature (annotations load cleanly)."""
    # The public surface includes the local engines, which import torch at module
    # scope; on an install without it those names resolve to a typed "optional
    # component is not installed" error rather than an object.
    pytest.importorskip("torch")

    bad: list[str] = []
    for name in effgen.__all__:
        obj = getattr(effgen, name)
        # Exception/Warning classes inherit BaseException's C-level __init__ and
        # legitimately have no introspectable signature — skip them.
        if inspect.isclass(obj) and issubclass(obj, BaseException):
            continue
        if inspect.isfunction(obj) or inspect.isclass(obj):
            try:
                inspect.signature(obj)
            except (ValueError, TypeError):
                # Some builtins/C-extensions legitimately lack a signature;
                # only flag pure-Python effGen objects.
                mod = getattr(obj, "__module__", "") or ""
                if mod.startswith("effgen"):
                    bad.append(name)
    assert not bad, "Public objects without an introspectable signature: " + ", ".join(bad)


def test_key_public_modules_import_clean():
    """The modules backing the public surface import without error."""
    for mod in (
        "effgen.core.agent",
        "effgen.models.model_loader",
        "effgen.models.base",
        "effgen.tools.base_tool",
        "effgen.presets.registry",
    ):
        importlib.import_module(mod)
