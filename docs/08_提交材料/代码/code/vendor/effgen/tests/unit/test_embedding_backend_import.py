"""How an unusable sentence-transformers install is reported.

The package carries optional native dependencies — a media decoder, for example —
that can be installed while the shared libraries they load are absent. Importing
it then raises something other than ``ImportError``, which would escape the
"package is not installed" branch every caller has and surface as a stack trace
from a library the caller never asked for.

These pin the two halves of the contract: a genuinely absent package still says
so and names the install command, and an installed-but-unimportable one is
reported as an ``ImportError`` that names the failing dependency and what it
needs, so the callers with a documented degraded path take it.
"""

from __future__ import annotations

import builtins
import importlib
import sys

import pytest

from effgen.utils.embedding_backend import (
    INSTALL_HINT,
    import_sentence_transformers,
    load_sentence_transformer,
)

REAL_IMPORT = builtins.__import__


def _failing_import(exc: BaseException) -> object:
    """Replace ``__import__`` so importing the package raises *exc*.

    The guard goes through the ordinary import machinery rather than
    ``importlib.import_module``, so a caller that simulates the package this way
    — as the eval scorer's own tests do — still sees its simulation.
    """

    def _hook(name: str, *args: object, **kwargs: object) -> object:
        if name == "sentence_transformers":
            raise exc
        return REAL_IMPORT(name, *args, **kwargs)

    return _hook


@pytest.fixture
def absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """sentence-transformers is not installed at all."""
    monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)
    missing = ImportError("No module named 'sentence_transformers'")
    missing.name = "sentence_transformers"
    monkeypatch.setattr(builtins, "__import__", _failing_import(missing))


@pytest.fixture
def unloadable_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    """The package is installed; importing it raises from a native dependency."""
    monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)
    monkeypatch.setattr(
        builtins,
        "__import__",
        _failing_import(
            RuntimeError(
                "Could not load libtorchcodec. Likely causes: 1. FFmpeg is not "
                "properly installed in your environment."
            )
        ),
    )


def test_an_absent_package_names_the_install_command(absent: None) -> None:
    with pytest.raises(ImportError) as caught:
        import_sentence_transformers("SentenceTransformer")
    assert str(caught.value) == INSTALL_HINT


def test_an_unloadable_dependency_is_an_import_error(unloadable_dependency: None) -> None:
    """Not a RuntimeError: callers branch on ImportError to degrade."""
    with pytest.raises(ImportError) as caught:
        import_sentence_transformers("SentenceTransformer")
    message = str(caught.value)
    assert "installed but could not be imported" in message
    assert "libtorchcodec" in message
    assert "FFmpeg" in message
    assert "is not installed. Install with" not in message


def test_the_model_loader_reports_it_the_same_way(unloadable_dependency: None) -> None:
    with pytest.raises(ImportError) as caught:
        load_sentence_transformer("all-MiniLM-L6-v2")
    assert "installed but could not be imported" in str(caught.value)


def test_the_detail_is_flattened_to_one_bounded_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A decoder's failure quotes several tracebacks; the message stays readable."""
    monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)
    monkeypatch.setattr(
        builtins,
        "__import__",
        _failing_import(OSError("line one\n" + "very long detail " * 40)),
    )
    with pytest.raises(ImportError) as caught:
        import_sentence_transformers("SentenceTransformer")
    message = str(caught.value)
    assert "\n" not in message
    assert len(message) < 400


def test_semantic_scoring_falls_back_instead_of_raising(
    unloadable_dependency: None,
) -> None:
    """The eval scorer's documented fallback fires, and flags that it did."""
    from effgen.eval.evaluator import _score_semantic_similarity

    score, used_fallback = _score_semantic_similarity("the answer", "the answer")
    assert used_fallback is True
    assert 0.0 <= score <= 1.0


def test_the_reranker_stays_a_pass_through(unloadable_dependency: None) -> None:
    from effgen.rag.reranker import CrossEncoderReranker

    reranker = CrossEncoderReranker()
    assert reranker._load() is None
    assert reranker._available is False


def test_a_working_install_still_imports() -> None:
    """The guard adds no behavior where the package is usable."""
    if importlib.util.find_spec("sentence_transformers") is None:
        pytest.skip("sentence-transformers is not installed in this environment")
    (transformer,) = import_sentence_transformers("SentenceTransformer")
    assert isinstance(transformer, type)
    assert transformer.__name__ == "SentenceTransformer"


def test_a_dependency_of_the_package_going_missing_is_not_read_as_the_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ImportError from *inside* the package must not say "not installed".

    Telling someone to install a package they already have sends them the wrong
    way, so the two are told apart by which module the failure names.
    """
    monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)
    inner = ImportError("No module named 'some_inner_dependency'")
    inner.name = "some_inner_dependency"
    monkeypatch.setattr(builtins, "__import__", _failing_import(inner))
    with pytest.raises(ImportError) as caught:
        import_sentence_transformers("SentenceTransformer")
    message = str(caught.value)
    assert "some_inner_dependency" in message
    assert message != INSTALL_HINT


def test_builtins_are_untouched() -> None:
    """The guard never patches the import system for the rest of the process."""
    assert sys.modules["builtins"].__import__ is builtins.__import__
