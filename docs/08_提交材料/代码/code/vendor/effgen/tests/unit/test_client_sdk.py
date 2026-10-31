"""Unit tests for the native ``effgen.client.EffGenClient`` contract.

Covers request/response shaping that must match the effGen server:
- A list of tool *names* is expanded to the server's OpenAI tool-spec shape,
  while a full tool dict is passed through unchanged.
- Server error envelopes (``{"error": {"message": ...}}``) surface as a clean
  human message on ``str(exc)``, with ``status_code``/``payload`` still present.
- ``embed()`` defaults to a real embedding model id.

These exercise pure client helpers and error mapping; no live server is needed.
"""
from __future__ import annotations

import inspect

import pytest

from effgen.client import EffGenClient
from effgen.client.client import _tools_to_payload
from effgen.client.exceptions import (
    EffGenAPIError,
    EffGenAuthError,
    EffGenServerError,
)


class TestToolsPayload:
    def test_name_expanded_to_spec(self):
        assert _tools_to_payload(["calculator"]) == [
            {"type": "function", "function": {"name": "calculator"}}
        ]

    def test_dict_passed_through(self):
        spec = {"type": "function", "function": {"name": "calculator"}}
        assert _tools_to_payload([spec]) == [spec]

    def test_mixed(self):
        spec = {"type": "function", "function": {"name": "search"}}
        out = _tools_to_payload(["calculator", spec])
        assert out[0]["function"]["name"] == "calculator"
        assert out[1] is spec

    def test_invalid_reference_rejected(self):
        with pytest.raises(EffGenAPIError):
            _tools_to_payload([123])  # type: ignore[list-item]


class TestErrorMessageParsing:
    """The server's own message is extracted from every envelope shape it uses.

    Each error then closes with what the caller can do about that status, so the
    assertions pin the extracted text as the opening and check the follow-on
    separately rather than fixing the whole string.
    """

    def _client(self):
        return EffGenClient(base_url="http://localhost:9", api_key="k")

    def test_openai_envelope_message(self):
        c = self._client()
        payload = {"error": {"message": "Invalid API key", "type": "invalid_request_error",
                             "code": "invalid_api_key"}}
        with pytest.raises(EffGenAuthError) as ei:
            c._raise_for_status(401, payload)
        assert str(ei.value).startswith("Invalid API key.")
        assert "API key the client was built with" in str(ei.value)
        assert ei.value.status_code == 401
        assert ei.value.payload is payload

    def test_error_as_plain_string(self):
        c = self._client()
        with pytest.raises(EffGenAPIError) as ei:
            c._raise_for_status(400, {"error": "bad request"})
        assert str(ei.value).startswith("bad request.")
        assert "server logs" in str(ei.value)

    def test_top_level_message_fallback(self):
        c = self._client()
        with pytest.raises(EffGenServerError) as ei:
            c._raise_for_status(500, {"message": "boom"})
        assert str(ei.value).startswith("boom.")
        assert "retry if the status is a 5xx" in str(ei.value)

    def test_status_only_when_no_body(self):
        c = self._client()
        with pytest.raises(EffGenAPIError) as ei:
            c._raise_for_status(404, None)
        assert "404" in str(ei.value)


class TestEmbedDefault:
    def test_embed_default_model_id(self):
        assert inspect.signature(EffGenClient.embed).parameters["model"].default == (
            "text-embedding-3-small"
        )
        assert inspect.signature(EffGenClient.aembed).parameters["model"].default == (
            "text-embedding-3-small"
        )
