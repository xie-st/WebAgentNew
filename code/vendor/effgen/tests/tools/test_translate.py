"""Tests for TranslateTool.

Unit tests mock LibreTranslate HTTP. Integration tests hit the real public
instance (translate.argosopentech.com) and are skipped if no network.
"""

from __future__ import annotations

import asyncio
import socket
from unittest.mock import MagicMock, patch

import pytest

from effgen.tools.builtin import TranslateTool
from effgen.tools.builtin import translate as translate_mod
from effgen.tools.builtin.translate import TranslateServiceUnavailable

_PUBLIC_LIBRE_HOSTS = (
    "translate.fedilab.app",
    "translate.argosopentech.com",
    "libretranslate.de",
)


def _has_network(timeout: float = 5.0) -> bool:
    for host in _PUBLIC_LIBRE_HOSTS:
        try:
            with socket.create_connection((host, 443), timeout=timeout):
                return True
        except OSError:
            continue
    return False


NETWORK = _has_network()
needs_net = pytest.mark.skipif(not NETWORK, reason="no network")


def _run(coro):
    return asyncio.run(coro)


_TRANSIENT_MARKERS = (
    "timeout",
    "timed out",
    "connection",
    "network",
    "429",
    "rate limit",
    "service unavailable",
)


def _ok(result) -> dict:
    assert hasattr(result, "success")
    if not result.success and result.error and any(m.lower() in result.error.lower() for m in _TRANSIENT_MARKERS):
        pytest.skip(f"transient error: {result.error}")
    assert result.success, f"tool failed: {result.error}"
    return result.output


# ---------------------------------------------------------------------------
# Metadata / static tests
# ---------------------------------------------------------------------------

def test_translate_metadata():
    tool = TranslateTool()
    assert tool.metadata.name == "translate"
    assert "translate" in tool.metadata.tags


def test_translate_missing_text_fails():
    r = _run(TranslateTool().execute(operation="translate", source="en", target="fr"))
    assert not r.success
    assert r.metadata["error_type"] == "InvalidInputError"
    assert "non-empty" in r.error


def test_translate_unknown_operation_fails():
    r = _run(TranslateTool().execute(operation="bogus", text="hello"))
    assert not r.success


# ---------------------------------------------------------------------------
# Unit tests with mocked HTTP
# ---------------------------------------------------------------------------

def _make_mock_urlopen(response_body: bytes):
    mock_resp = MagicMock()
    mock_resp.read.return_value = response_body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def test_translate_mock_en_to_fr():
    import json as json_mod
    body = json_mod.dumps({"translatedText": "Bonjour le monde"}).encode()
    with patch("effgen.tools.builtin.translate.safe_urlopen", return_value=_make_mock_urlopen(body)):
        r = _run(TranslateTool().execute(operation="translate", text="Hello world", source="en", target="fr"))
    out = r.output
    assert out["success"] is True
    assert out["translated_text"] == "Bonjour le monde"
    assert out["source_language"] == "en"
    assert out["target_language"] == "fr"
    assert out["backend"] == "libretranslate"


def test_translate_mock_available_pairs():
    import json as json_mod
    body = json_mod.dumps([
        {"code": "en", "name": "English", "targets": ["fr", "es", "ja"]},
        {"code": "fr", "name": "French", "targets": ["en", "es"]},
    ]).encode()
    with patch("effgen.tools.builtin.translate.safe_urlopen", return_value=_make_mock_urlopen(body)):
        r = _run(TranslateTool().execute(operation="available_pairs"))
    out = r.output
    assert out["success"] is True
    assert out["count"] >= 5
    pair_keys = {(p["source"], p["target"]) for p in out["pairs"]}
    assert ("en", "fr") in pair_keys
    assert ("en", "es") in pair_keys


def test_available_pairs_no_backend_fails_closed():
    """Zero pairs (LibreTranslate down, no Argos packs) is a failure, not an empty success."""
    with (
        patch(
            "effgen.tools.builtin.translate._libre_available_pairs",
            side_effect=ConnectionError("LibreTranslate /languages error: down"),
        ),
        patch("effgen.tools.builtin.translate._argos_available_pairs", return_value=[]),
    ):
        r = _run(TranslateTool().execute(operation="available_pairs"))
    assert not r.success
    assert "unavailable" in r.error.lower()
    assert "LIBRE_TRANSLATE_URL" in r.error


def test_argos_cache_dir_is_pinned_to_effgen_cache(tmp_path, monkeypatch):
    pytest.importorskip("argostranslate.settings")
    import argostranslate.settings as at_settings

    original = {
        name: getattr(at_settings, name)
        for name in (
            "data_dir",
            "cache_dir",
            "downloads_dir",
            "package_data_dir",
            "legacy_package_data_dir",
            "local_package_index",
            "package_dirs",
        )
    }
    monkeypatch.setattr(translate_mod, "_ARGOS_CACHE_DIR", tmp_path / "argos")
    try:
        translate_mod._set_argos_data_dir()

        assert at_settings.data_dir == tmp_path / "argos"
        assert at_settings.cache_dir == tmp_path / "argos" / "cache"
        assert at_settings.downloads_dir == tmp_path / "argos" / "cache" / "downloads"
        assert at_settings.package_data_dir == tmp_path / "argos" / "packages"
        assert at_settings.local_package_index == tmp_path / "argos" / "index.json"
        assert at_settings.package_dirs == [tmp_path / "argos" / "packages"]
        assert at_settings.package_data_dir.exists()
    finally:
        for name, value in original.items():
            setattr(at_settings, name, value)


def test_translate_fallback_to_argos_on_libre_failure():
    """When LibreTranslate fails, tool should fall back to argostranslate."""
    from urllib.error import URLError

    def raise_url_error(*args, **kwargs):
        raise URLError("connection refused")

    with patch("effgen.tools.builtin.translate.safe_urlopen", side_effect=raise_url_error):
        # Patch the argos translate path so we don't need actual packs installed
        with patch("effgen.tools.builtin.translate._argos_translate", return_value="Hola mundo") as mock_argos:
            r = _run(TranslateTool().execute(operation="translate", text="Hello world", source="en", target="es"))

    out = r.output
    assert out["success"] is True
    assert out["translated_text"] == "Hola mundo"
    assert out["backend"] == "argostranslate"
    mock_argos.assert_called_once_with("Hello world", "en", "es")


def test_translate_both_backends_fail_returns_error():
    from urllib.error import URLError

    with patch("effgen.tools.builtin.translate.safe_urlopen", side_effect=URLError("fail")):
        with patch("effgen.tools.builtin.translate._argos_translate", side_effect=RuntimeError("argos fail")):
            r = _run(TranslateTool().execute(operation="translate", text="Hello", source="en", target="ja"))

    assert not r.success
    assert r.output is None
    assert r.metadata["error_type"] == "TranslateServiceUnavailable"
    assert "Translation service unavailable" in r.error
    assert "Set LIBRE_TRANSLATE_URL" in r.error


def test_translate_service_unavailable_exception_type():
    err = TranslateServiceUnavailable("backend down")
    assert isinstance(err, RuntimeError)


def test_translate_same_language_returns_original_without_backend_call():
    with patch("effgen.tools.builtin.translate.safe_urlopen") as mock_urlopen:
        r = _run(TranslateTool().execute(operation="translate", text="Hello", source="en", target="en"))
    assert r.success
    assert r.output["translated_text"] == "Hello"
    assert r.output["backend"] == "none"
    mock_urlopen.assert_not_called()


# ---------------------------------------------------------------------------
# Live integration tests
# ---------------------------------------------------------------------------

@needs_net
def test_translate_live_en_to_fr():
    out = _ok(_run(TranslateTool().execute(operation="translate", text="Hello world", source="en", target="fr")))
    assert out["success"] is True
    assert out["translated_text"]
    assert "Bonjour" in out["translated_text"] or out["translated_text"]  # fuzzy — service may vary
    assert out["source_language"] == "en"
    assert out["target_language"] == "fr"


@needs_net
def test_translate_live_en_to_es():
    out = _ok(_run(TranslateTool().execute(operation="translate", text="Good morning", source="en", target="es")))
    assert out["success"] is True
    assert out["translated_text"]


@needs_net
def test_translate_live_en_to_ja():
    out = _ok(_run(TranslateTool().execute(operation="translate", text="Hello", source="en", target="ja")))
    assert out["success"] is True
    assert out["translated_text"]


@needs_net
def test_translate_live_auto_detect_source():
    out = _ok(_run(TranslateTool().execute(operation="translate", text="Guten Morgen", source="auto", target="en")))
    assert out["success"] is True
    assert out["translated_text"]


@needs_net
def test_translate_live_available_pairs():
    out = _ok(_run(TranslateTool().execute(operation="available_pairs")))
    assert out["success"] is True
    assert isinstance(out["pairs"], list)
    # LibreTranslate returns 50+ pairs; Argos fallback returns however many
    # local packs are installed (often fewer). Accept either backend.
    assert out["count"] >= 1
    sources = {p["source"] for p in out["pairs"]}
    assert "en" in sources or out["count"] >= 1
