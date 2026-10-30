"""Surfacing retrieved evidence on ``AgentResponse.sources`` / ``.citations``.

These are offline, deterministic tests of the *data-shaping* layer that turns
search/fetch tool results (and provider-native grounding metadata) into the
public ``sources``/``citations`` fields. No live API is called — the actual
end-to-end behavior with real models is exercised by the live integration
suite. Here we pin the conversions that previously left both fields empty:

* a bare list of search rows (``web_search`` → ``[{title, url, snippet}, ...]``)
* a single-document dict (``url_fetch`` → ``{url, title, text}``)
* provider-native grounded URLs in ``metadata["grounding_chunks"]``
  (OpenAI ``url_citation`` annotations and Gemini search grounding).
"""

from effgen.core.agent import AgentResponse
from effgen.presets import create_agent
from effgen.tools.base_tool import ToolResult
from effgen.tools.builtin.retrieval import Retrieval
from effgen.tools.builtin.url_fetch import URLFetchTool
from effgen.tools.builtin.web_search import WebSearch
from tests.fixtures.mock_models import MockModel


def _agent():
    # The research preset wires the search/fetch tools; the model is never
    # called here — we drive the citation helpers directly.
    return create_agent("research", MockModel(responses=["ok"]))


class TestCollectFromToolResults:
    def test_web_search_list_output_collects_sources(self):
        # A search can return URLs the model never references in its final
        # answer; those still widen `.sources` (a search that ran is never
        # silently unsourced) but only the one actually referenced in the
        # answer text becomes a `.citations` entry.
        agent = _agent()
        agent._collected_citations = []
        tool = WebSearch()
        result = ToolResult(
            success=True,
            output=[
                {"title": "A", "url": "https://example.com/a", "snippet": "snip a", "position": 1},
                {"title": "B", "url": "https://example.com/b", "snippet": "snip b", "position": 2},
            ],
        )
        agent._collect_citations(tool, "web_search", result)
        resp = AgentResponse(output="The answer draws on https://example.com/a.")
        agent._attach_citations(resp)
        assert resp.sources == ["https://example.com/a", "https://example.com/b"]
        assert len(resp.citations) == 1
        assert resp.citations[0].source == "https://example.com/a"

    def test_url_fetch_single_dict_collects_one_source(self):
        agent = _agent()
        agent._collected_citations = []
        tool = URLFetchTool()
        result = ToolResult(
            success=True,
            output={
                "url": "https://en.wikipedia.org/wiki/CrowdStrike",
                "title": "CrowdStrike",
                "text": "CrowdStrike is a cybersecurity company. " * 20,
                "content_length": 800,
            },
        )
        agent._collect_citations(tool, "url_fetch", result)
        resp = AgentResponse(output="Per https://en.wikipedia.org/wiki/CrowdStrike, it is a cybersecurity firm.")
        agent._attach_citations(resp)
        assert resp.sources == ["https://en.wikipedia.org/wiki/CrowdStrike"]
        assert len(resp.citations) == 1
        # Long body is truncated into a short quote, not dumped whole.
        assert len(resp.citations[0].quote) <= 200

    def test_web_search_url_not_referenced_in_answer_stays_source_only(self):
        # The direct regression case: a search result the model never draws
        # on must widen `.sources` but never manufacture a `.citations`
        # entry, exactly like the native-provider `cited: False` path.
        agent = _agent()
        agent._collected_citations = []
        tool = WebSearch()
        result = ToolResult(
            success=True,
            output=[
                {"title": "A", "url": "https://example.com/a", "snippet": "snip a", "position": 1},
            ],
        )
        agent._collect_citations(tool, "web_search", result)
        resp = AgentResponse(output="The answer never mentions that particular result.")
        agent._attach_citations(resp)
        assert resp.sources == ["https://example.com/a"]
        assert resp.citations == []

    def test_failed_tool_result_is_skipped(self):
        agent = _agent()
        agent._collected_citations = []
        tool = WebSearch()
        agent._collect_citations(
            tool, "web_search",
            ToolResult(success=False, output=None, error="boom"),
        )
        resp = AgentResponse(output="answer")
        agent._attach_citations(resp)
        assert resp.sources == []
        assert resp.citations == []

    def test_vector_store_metadata_doc_and_name_fall_back_before_opaque_id(self):
        # A VectorMemoryStore entry has no natural path/URL; its metadata may
        # use "doc" or "name" instead of "source"/"title" — both must resolve
        # to a human-readable citation rather than the internal mem_* id.
        agent = _agent()
        agent._collected_citations = []
        tool = Retrieval()
        result = ToolResult(
            success=True,
            output={
                "results": [
                    {
                        "id": "mem_1700000000000_0",
                        "content": "The Zephyr-9 turbine outputs 4.2 MW.",
                        "score": 0.9,
                        "metadata": {"doc": "zephyr_manual.pdf"},
                    },
                    {
                        "id": "mem_1700000000000_1",
                        "content": "Maintenance is every 6000 hours.",
                        "score": 0.5,
                        "metadata": {"name": "zephyr_manual_v2.pdf"},
                    },
                ]
            },
        )
        agent._collect_citations(tool, "retrieval", result)
        resp = AgentResponse(output="answer")
        agent._attach_citations(resp)
        assert resp.sources == ["zephyr_manual.pdf", "zephyr_manual_v2.pdf"]

    def test_dedup_across_calls(self):
        agent = _agent()
        agent._collected_citations = []
        tool = WebSearch()
        row = {"title": "A", "url": "https://example.com/a", "snippet": "s", "position": 1}
        agent._collect_citations(tool, "web_search", ToolResult(success=True, output=[row]))
        agent._collect_citations(tool, "web_search", ToolResult(success=True, output=[row]))
        resp = AgentResponse(output="answer")
        agent._attach_citations(resp)
        assert resp.sources == ["https://example.com/a"]


class TestNativeGroundingMetadata:
    def test_grounding_chunks_become_sources(self):
        # OpenAI url_citation annotations and Gemini grounding both land in
        # metadata["grounding_chunks"] as {url, title} dicts.
        agent = _agent()
        agent._collected_citations = []
        resp = AgentResponse(
            output="answer",
            metadata={
                "grounding_chunks": [
                    {"url": "https://news.example/x", "title": "X story"},
                    {"url": "https://news.example/y", "title": "Y story"},
                ]
            },
        )
        agent._attach_citations(resp)
        assert resp.sources == ["https://news.example/x", "https://news.example/y"]
        assert len(resp.citations) == 2
        assert resp.citations[0].quote == "X story"

    def test_explicit_caller_sources_win(self):
        # If the assembly path already populated sources, native metadata must
        # not clobber them.
        agent = _agent()
        agent._collected_citations = []
        resp = AgentResponse(
            output="answer",
            sources=["https://explicit.example/preset"],
            metadata={"grounding_chunks": [{"url": "https://news.example/x"}]},
        )
        agent._attach_citations(resp)
        assert resp.sources == ["https://explicit.example/preset"]

    def test_no_evidence_leaves_fields_empty(self):
        agent = _agent()
        agent._collected_citations = []
        resp = AgentResponse(output="answer")
        agent._attach_citations(resp)
        assert resp.sources == []
        assert resp.citations == []

    def test_uncited_grounding_chunk_fills_sources_not_citations(self):
        # A web search can return URLs the model never references inline
        # (OpenAI web_search_call action.sources). Those widen `.sources`
        # so a search that ran is never silently unsourced, but they must
        # not manufacture a Citation the model did not actually make.
        agent = _agent()
        agent._collected_citations = []
        resp = AgentResponse(
            output="answer",
            metadata={
                "grounding_chunks": [
                    {"url": "https://news.example/cited", "title": "Cited story"},
                    {"url": "https://news.example/searched-only", "cited": False},
                ]
            },
        )
        agent._attach_citations(resp)
        assert resp.sources == [
            "https://news.example/cited",
            "https://news.example/searched-only",
        ]
        assert len(resp.citations) == 1
        assert resp.citations[0].source == "https://news.example/cited"

    def test_all_uncited_chunks_fill_sources_only(self):
        agent = _agent()
        agent._collected_citations = []
        resp = AgentResponse(
            output="answer",
            metadata={
                "grounding_chunks": [
                    {"url": "https://news.example/a", "cited": False},
                    {"url": "https://news.example/b", "cited": False},
                ]
            },
        )
        agent._attach_citations(resp)
        assert resp.sources == ["https://news.example/a", "https://news.example/b"]
        assert resp.citations == []
