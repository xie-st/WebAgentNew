"""Browser backends. Imports are lazy so optional dependencies stay optional."""

from __future__ import annotations

from typing import Any

__all__ = ["create_backend", "available_backends"]


def available_backends() -> dict[str, bool]:
    """Importability per backend, without launching a browser."""

    import importlib.util

    return {
        "playwright": importlib.util.find_spec("playwright") is not None,
        "browser-use": importlib.util.find_spec("browser_use") is not None,
    }


def create_backend(name: str, **kwargs: Any) -> Any:
    if name == "playwright":
        from .playwright import PlaywrightBackend

        return PlaywrightBackend(**kwargs)
    if name == "browser-use":
        from .browser_use import BrowserUseBackend

        return BrowserUseBackend(**kwargs)
    raise ValueError(f"unknown browser backend {name!r}")
