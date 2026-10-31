"""OpenAI native web_search grounding: sources survive an uncited search.

``generate_with_native_tools`` previously only populated ``grounding_chunks``
from ``url_citation`` annotations on the message text. A ``web_search_call``
that ran but whose URLs the model never referenced inline contributed
nothing, so ``response.sources``/``.citations`` came back empty even though a
search actually happened. These tests pin the fix offline: the adapter asks
for ``web_search_call.action.sources`` whenever a web-search tool is attached,
and folds those URLs in as recall-oriented grounding (``cited: False``) that
widens ``.sources`` without manufacturing a ``Citation`` the model never made.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from effgen.models.openai_adapter import OpenAIAdapter


def _make_adapter(model_name: str = "gpt-5-nano") -> OpenAIAdapter:
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
        adapter = OpenAIAdapter(model_name=model_name)
    adapter._is_loaded = True
    return adapter


def _capture_params(adapter: OpenAIAdapter, native_tool_specs: list[dict], response):
    captured: dict = {}

    def fake_create(**params):
        captured.update(params)
        return response

    adapter.client = MagicMock()
    adapter.client.responses.create.side_effect = fake_create
    result = adapter.generate_with_native_tools("hi", native_tool_specs)
    return captured, result


def _search_call(id_: str, query: str, urls: list[str]):
    sources = [SimpleNamespace(type="url", url=u) for u in urls]
    action = SimpleNamespace(type="search", query=query, sources=sources)
    return SimpleNamespace(type="web_search_call", id=id_, action=action, status="completed")


def _message(text: str, cited_urls: list[str] = ()):
    annotations = [
        SimpleNamespace(type="url_citation", url=u, title=None) for u in cited_urls
    ]
    content = SimpleNamespace(type="output_text", text=text, annotations=annotations)
    return SimpleNamespace(type="message", content=[content])


class TestIncludeRequested:
    def test_web_search_preview_tool_requests_action_sources(self):
        adapter = _make_adapter()
        response = SimpleNamespace(output=[], usage=None, status="completed", id="r1")
        params, _ = _capture_params(
            adapter, [{"type": "web_search_preview"}], response
        )
        assert params["include"] == ["web_search_call.action.sources"]

    def test_web_search_tool_variant_requests_action_sources(self):
        adapter = _make_adapter()
        response = SimpleNamespace(output=[], usage=None, status="completed", id="r1")
        params, _ = _capture_params(adapter, [{"type": "web_search"}], response)
        assert params["include"] == ["web_search_call.action.sources"]

    def test_no_web_search_tool_no_include(self):
        adapter = _make_adapter()
        response = SimpleNamespace(output=[], usage=None, status="completed", id="r1")
        params, _ = _capture_params(
            adapter, [{"type": "code_interpreter"}], response
        )
        assert "include" not in params


class TestActionSourcesFoldedIn:
    def test_uncited_search_fills_grounding_chunks_as_uncited(self):
        adapter = _make_adapter()
        response = SimpleNamespace(
            output=[
                _search_call("ws_1", "iceland population", ["https://a.example/1", "https://b.example/2"]),
                _message("The population is 395,000.", cited_urls=[]),
            ],
            usage=None,
            status="completed",
            id="r2",
        )
        _, result = _capture_params(adapter, [{"type": "web_search_preview"}], response)
        chunks = result.metadata["grounding_chunks"]
        assert {c["url"] for c in chunks} == {"https://a.example/1", "https://b.example/2"}
        assert all(c["cited"] is False for c in chunks)

    def test_cited_annotation_takes_precedence_and_stays_cited(self):
        adapter = _make_adapter()
        response = SimpleNamespace(
            output=[
                _search_call("ws_1", "sp500 close", ["https://apnews.example/x", "https://other.example/y"]),
                _message("Closed at 5,881.63.", cited_urls=["https://apnews.example/x"]),
            ],
            usage=None,
            status="completed",
            id="r3",
        )
        _, result = _capture_params(adapter, [{"type": "web_search_preview"}], response)
        chunks = result.metadata["grounding_chunks"]
        by_url = {c["url"]: c for c in chunks}
        # The cited URL keeps its citation entry (default cited=True, no
        # explicit key) in addition to the uncited fallback entry for the
        # other search result.
        assert "cited" not in by_url["https://apnews.example/x"]
        assert by_url["https://other.example/y"]["cited"] is False

    def test_web_search_call_records_query(self):
        adapter = _make_adapter()
        response = SimpleNamespace(
            output=[_search_call("ws_1", "iceland population", ["https://a.example/1"])],
            usage=None,
            status="completed",
            id="r4",
        )
        _, result = _capture_params(adapter, [{"type": "web_search_preview"}], response)
        call = result.metadata["native_tool_results"][0]
        assert call["type"] == "web_search_call"
        assert call["query"] == "iceland population"

    def test_non_search_action_has_no_sources(self):
        # action.type == "open_page" / "find_in_page" carries no `sources`
        # field; the adapter must not crash pulling it off a missing attr.
        adapter = _make_adapter()
        action = SimpleNamespace(type="open_page", url="https://a.example/1")
        call = SimpleNamespace(type="web_search_call", id="ws_2", action=action, status="completed")
        response = SimpleNamespace(output=[call], usage=None, status="completed", id="r5")
        _, result = _capture_params(adapter, [{"type": "web_search_preview"}], response)
        assert result.metadata["grounding_chunks"] == []
