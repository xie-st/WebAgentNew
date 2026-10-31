"""Loading a sentence-transformers model without a working network route.

The embedding model behind RAG retrieval, semantic chunking, the vector store
and semantic eval scoring is downloaded from the model hub on first use and then
served from the local cache. When the hub cannot be reached, ``huggingface_hub``
closes its HTTP client after the first failed request, and a load makes several,
so the request that follows raises ``RuntimeError: Cannot send a request, as the
client has been closed`` instead of reading the copy already on disk.

:func:`load_sentence_transformer` retries the load against the local cache in
that case, so a machine with a warm cache and no network still works, and reports
a model that is genuinely absent by name with the way to obtain it.

The package also has optional native dependencies that can be installed but
unloadable — a decoder whose shared libraries are missing, for example. Importing
it then raises something other than ``ImportError``, which would escape every
caller's "the package is absent" branch as a stack trace from a library the user
never asked for. :func:`import_sentence_transformers` reports that state as an
``ImportError`` naming the dependency that failed and what it needs, so a caller
takes its documented absent-package path.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

INSTALL_HINT = (
    "sentence-transformers is not installed. Install with: pip install sentence-transformers"
)

#: Native dependencies that are commonly present but unloadable, and what each
#: one needs. Matched against the text of the import failure.
_DEPENDENCY_HINTS = (
    (
        "torchcodec",
        (
            "torchcodec requires FFmpeg's shared libraries (libavutil, libavcodec). "
            "Install FFmpeg, or remove torchcodec if audio and video decoding are not needed."
        ),
    ),
)


def _unusable_message(exc: BaseException) -> str:
    """Explain an installed sentence-transformers that will not import."""
    detail = " ".join(f"{type(exc).__name__}: {exc}".split())
    if len(detail) > 200:
        detail = detail[:199] + "…"
    lowered = detail.lower()
    hint = next((text for marker, text in _DEPENDENCY_HINTS if marker in lowered), "")
    return (
        f"sentence-transformers is installed but could not be imported ({detail}). "
        "One of its optional native dependencies is present but unusable on this "
        f"machine.{' ' + hint if hint else ''}"
    )


def import_sentence_transformers(*names: str) -> tuple[Any, ...]:
    """Import ``sentence_transformers`` and return the named attributes.

    Args:
        *names: Attribute or submodule names to read off the package, e.g.
            ``"SentenceTransformer"``, ``"CrossEncoder"``, ``"util"``.

    Returns:
        The requested objects, in the order they were named.

    Raises:
        ImportError: The package is not installed, or it is installed and cannot
            be imported because one of its own dependencies is unusable. The
            message says which of the two it is, so a caller reporting "install
            sentence-transformers" never hides a broken dependency.
    """
    try:
        # A plain import statement, so the ordinary import machinery is what
        # decides: a caller simulating an absent package sees its simulation.
        import sentence_transformers as package
    except ImportError as exc:
        if getattr(exc, "name", None) == "sentence_transformers":
            raise ImportError(INSTALL_HINT) from exc
        # The package is there; something it imports is not.
        raise ImportError(_unusable_message(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - reported as ImportError below
        raise ImportError(_unusable_message(exc)) from exc

    try:
        return tuple(
            getattr(package, name)
            if hasattr(package, name)
            else importlib.import_module(f"sentence_transformers.{name}")
            for name in names
        )
    except Exception as exc:  # noqa: BLE001 - reported as ImportError below
        raise ImportError(_unusable_message(exc)) from exc

# Substrings that identify a failure to reach the model hub, as opposed to a bad
# model id or a corrupt download.
_OFFLINE_MARKERS = (
    "client has been closed",
    "name resolution",
    "temporary failure",
    "connection error",
    "connectionerror",
    "max retries exceeded",
    "network is unreachable",
    "failed to establish a new connection",
    "couldn't connect to",
    "connection aborted",
    "timed out",
    "offlinemodeisenabled",
)


def looks_offline(exc: BaseException) -> bool:
    """True when *exc*, or anything it chains to, reports an unreachable host."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = f"{type(current).__name__}: {current}".lower()
        if any(marker in text for marker in _OFFLINE_MARKERS):
            return True
        current = current.__cause__ or current.__context__
    return False


def load_sentence_transformer(model_name: str, **kwargs: Any) -> Any:
    """Return a loaded ``SentenceTransformer``, reading the cache when offline.

    Args:
        model_name: Model id or path passed to ``SentenceTransformer``.
        **kwargs: Forwarded to the constructor unchanged.

    Returns:
        The loaded model.

    Raises:
        ImportError: ``sentence-transformers`` is not installed, or is installed
            and cannot be imported because one of its dependencies is unusable.
        RuntimeError: The hub is unreachable and the model is not cached.
    """
    (SentenceTransformer,) = import_sentence_transformers("SentenceTransformer")

    try:
        return SentenceTransformer(model_name, **kwargs)
    except Exception as exc:
        if not looks_offline(exc):
            raise
        logger.info(
            "Model hub unreachable while loading '%s'; reading the local cache instead.",
            model_name,
        )
        try:
            return SentenceTransformer(model_name, local_files_only=True, **kwargs)
        except Exception as cached_exc:
            raise RuntimeError(
                f"Could not load the embedding model '{model_name}': the model hub is "
                f"unreachable and no copy is in the local cache. Download it once with a "
                f"network connection, or point HF_HOME at a cache that already has it."
            ) from cached_exc
