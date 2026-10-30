"""
PromptRegistry — singleton registry with auto-discovery of domain packages.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import pkgutil
import sys
from collections.abc import Iterator
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema import exceptions as jsonschema_exceptions

from .base import LibraryPrompt

logger = logging.getLogger(__name__)

#: Environment variable naming one or more directories of user prompt files
#: (``os.pathsep``-separated) discovered alongside the built-in library.
USER_PROMPTS_ENV = "EFFGEN_PROMPTS_DIR"

VALID_VARIANTS = {"zero_shot", "cot", "few_shot", "tool", "structured"}


class PromptRegistry:
    """Registry for LibraryPrompt instances.

    Provides register/get/search and auto-discovers domain packages under
    effgen.prompts.library.domains on first access.
    """

    def __init__(self) -> None:
        self._prompts: dict[str, LibraryPrompt] = {}
        self._discovered = False

    def register(self, prompt: LibraryPrompt) -> None:
        """Add *prompt* to the registry (an existing prompt of the same name is replaced)."""
        self._validate_prompt(prompt)
        if prompt.name in self._prompts:
            logger.warning("Overwriting prompt %s in registry", prompt.name)
        self._prompts[prompt.name] = prompt

    @staticmethod
    def _validate_prompt(prompt: LibraryPrompt) -> None:
        """Validate prompt metadata before adding it to the registry."""
        if not isinstance(prompt, LibraryPrompt):
            raise TypeError("PromptRegistry.register() expects a LibraryPrompt")

        if not prompt.name or not isinstance(prompt.name, str):
            raise ValueError("LibraryPrompt.name must be a non-empty string")
        if not prompt.domain or not isinstance(prompt.domain, str):
            raise ValueError("LibraryPrompt.domain must be a non-empty string")
        if prompt.variant not in VALID_VARIANTS:
            allowed = ", ".join(sorted(VALID_VARIANTS))
            raise ValueError(
                f"LibraryPrompt.variant must be one of: {allowed}; "
                f"got {prompt.variant!r}"
            )
        if not callable(prompt.template):
            raise TypeError("LibraryPrompt.template must be callable")
        if not isinstance(prompt.input_schema, dict):
            raise TypeError("LibraryPrompt.input_schema must be a JSON Schema dict")
        if not isinstance(prompt.fixture, dict):
            raise TypeError("LibraryPrompt.fixture must be a dict")
        if not isinstance(prompt.tags, list):
            raise TypeError("LibraryPrompt.tags must be a list")

        try:
            Draft202012Validator.check_schema(prompt.input_schema)
        except jsonschema_exceptions.SchemaError as exc:
            raise ValueError(
                f"Invalid input_schema for prompt {prompt.name!r}: {exc.message}"
            ) from exc

        try:
            Draft202012Validator(prompt.input_schema).validate(prompt.fixture)
        except jsonschema_exceptions.ValidationError as exc:
            raise ValueError(
                f"Fixture for prompt {prompt.name!r} does not match input_schema: "
                f"{exc.message}"
            ) from exc

    def get(self, name: str) -> LibraryPrompt:
        """Return the prompt *name* (raises ``KeyError`` when absent)."""
        self._ensure_discovered()
        if name not in self._prompts:
            raise KeyError(f"Prompt '{name}' not found in registry")
        return self._prompts[name]

    def search(
        self,
        domain: str | None = None,
        variant: str | None = None,
        tags: list[str] | None = None,
    ) -> list[LibraryPrompt]:
        """Return prompts filtered by domain/variant/tags, sorted by name."""
        self._ensure_discovered()
        results: list[LibraryPrompt] = []
        for p in self._prompts.values():
            if domain and p.domain != domain:
                continue
            if variant and p.variant != variant:
                continue
            if tags and not all(t in p.tags for t in tags):
                continue
            results.append(p)
        return sorted(results, key=lambda p: p.name)

    def all(self) -> list[LibraryPrompt]:
        """Return every registered prompt, sorted by name."""
        self._ensure_discovered()
        return sorted(self._prompts.values(), key=lambda p: p.name)

    def domains(self) -> list[str]:
        """Return the sorted set of domains that have prompts."""
        self._ensure_discovered()
        return sorted({p.domain for p in self._prompts.values()})

    def __len__(self) -> int:
        self._ensure_discovered()
        return len(self._prompts)

    def __iter__(self) -> Iterator[LibraryPrompt]:
        self._ensure_discovered()
        return iter(sorted(self._prompts.values(), key=lambda p: p.name))

    # ------------------------------------------------------------------
    # Auto-discovery
    # ------------------------------------------------------------------

    def _ensure_discovered(self) -> None:
        if not self._discovered:
            self._discovered = True
            self._discover_domains()
            self._discover_user_dirs()

    def _discover_domains(self) -> None:
        """Walk effgen.prompts.library.domains and import every sub-module."""
        try:
            from effgen.prompts.library import domains as _domains_pkg
        except ImportError:
            logger.debug("No domains package found; registry stays empty")
            return

        domains_path = getattr(_domains_pkg, "__path__", [])
        domains_prefix = getattr(_domains_pkg, "__name__", "effgen.prompts.library.domains") + "."

        for _finder, modname, _ispkg in pkgutil.walk_packages(
            path=domains_path,
            prefix=domains_prefix,
            onerror=lambda n: logger.warning("Error walking %s", n),
        ):
            try:
                importlib.import_module(modname)
                logger.debug("Discovered prompt module: %s", modname)
            except Exception as exc:
                logger.warning("Failed to import %s: %s", modname, exc)

    def _discover_user_dirs(self) -> None:
        """Load user templates from directories named by ``EFFGEN_PROMPTS_DIR``.

        The variable holds one or more directories (``os.pathsep``-separated).
        Each ``*.py`` file in a directory (excluding names starting with ``_``)
        is imported; a file registers templates by calling
        ``registry.register(...)`` and/or by exposing ``LibraryPrompt`` values
        (a module-level ``PROMPTS`` list or top-level ``LibraryPrompt``
        attributes), which are registered when their name is not already taken.
        """
        raw = os.environ.get(USER_PROMPTS_ENV, "").strip()
        if not raw:
            return
        for part in raw.split(os.pathsep):
            entry = part.strip()
            if entry:
                self._load_user_dir(Path(entry).expanduser())

    def _load_user_dir(self, directory: Path) -> None:
        if not directory.is_dir():
            logger.warning(
                "%s entry is not a directory: %s", USER_PROMPTS_ENV, directory
            )
            return
        for py_file in sorted(directory.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            self._load_user_file(py_file)

    def _load_user_file(self, path: Path) -> None:
        modname = f"effgen_user_prompts_{path.stem}"
        try:
            spec = importlib.util.spec_from_file_location(modname, path)
            if spec is None or spec.loader is None:
                logger.warning("Could not load prompt file %s", path)
                return
            module = importlib.util.module_from_spec(spec)
            sys.modules[modname] = module
            spec.loader.exec_module(module)
        except Exception as exc:
            logger.warning("Failed to load prompt file %s: %s", path, exc)
            return
        # A file may register directly, or just declare LibraryPrompt values.
        candidates: list[LibraryPrompt] = []
        declared = getattr(module, "PROMPTS", None)
        if isinstance(declared, list | tuple):
            candidates.extend(p for p in declared if isinstance(p, LibraryPrompt))
        for value in vars(module).values():
            if isinstance(value, LibraryPrompt):
                candidates.append(value)
        for prompt in candidates:
            if prompt.name not in self._prompts:
                try:
                    self.register(prompt)
                except Exception as exc:
                    logger.warning(
                        "Skipping prompt %r from %s: %s",
                        getattr(prompt, "name", "?"),
                        path,
                        exc,
                    )


registry = PromptRegistry()
