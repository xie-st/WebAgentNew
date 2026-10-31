"""Citation and source assembly for :class:`Agent`.

Mines retrieval and search tool results for source passages and populates
``AgentResponse.sources`` / ``AgentResponse.citations``. Mixed into
:class:`Agent` through :class:`~effgen.core.agent_react.AgentReActMixin`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .agent_response import AgentResponse


def _key_of(entry: dict[str, Any]) -> tuple[str, str, str]:
    """The identity of one mined passage: same source, chunk and opening text."""
    return (entry["source"], entry["chunk_id"], entry["quote"][:80])


class AgentCitationsMixin:
    """Source-mining and citation-assembly methods for :class:`Agent`."""

    # Tool tags / categories whose results carry retrievable evidence we can
    # turn into AgentResponse sources + inline citations.
    _RETRIEVAL_TOOL_TAGS = frozenset({
        "retrieval", "rag", "knowledge-base", "search", "web-search",
        "wikipedia", "documents", "semantic",
    })

    def _is_retrieval_tool(self, tool: Any, tool_name: str) -> bool:
        """True if a tool's results should be mined for sources/citations."""
        meta = getattr(tool, "metadata", None)
        if meta is not None:
            category = getattr(meta, "category", None)
            cat_val = getattr(category, "value", category)
            if isinstance(cat_val, str) and "retrieval" in cat_val.lower():
                return True
            tags = getattr(meta, "tags", None) or []
            if any(str(t).lower() in self._RETRIEVAL_TOOL_TAGS for t in tags):
                return True
        return tool_name.lower() in {"retrieval", "rag", "knowledge_base"}

    def _collect_citations(
        self, tool: Any, tool_name: str, result: Any
    ) -> list[dict[str, Any]]:
        """
        Mine a retrieval/search tool result for source passages and stash them
        on the per-run accumulator. The actual ``Citation`` objects and the
        deduplicated source list are assembled in :meth:`_attach_citations`.

        Returns the passages this call contributed, in the order they were
        mined, so the caller can show them to the model. When inline citations
        were asked for, each passage also carries the number it keeps for the
        rest of the run (``cite_index``): a passage retrieved twice keeps the
        number it was first given, so ``[n]`` means one passage and not one
        retrieval call.
        """
        if not self._is_retrieval_tool(tool, tool_name):
            return []

        # Unwrap ToolResult → output (skip failed calls).
        output = result
        if hasattr(result, "output"):
            if hasattr(result, "success") and not result.success:
                return []
            output = result.output

        # Find the list of source items. Tools surface them three ways:
        #   - a bare list of rows (web_search → [{title, url, snippet}, ...]);
        #   - a dict wrapping a list under a well-known key (RAG/retrieval);
        #   - a single-document dict (url_fetch → {url, title, text}).
        items = None
        if isinstance(output, list):
            items = output
        elif isinstance(output, dict):
            for key in ("results", "documents", "chunks", "matches", "passages", "sources"):
                val = output.get(key)
                if isinstance(val, list) and val:
                    items = val
                    break
            if items is None and (
                output.get("url") or output.get("source") or output.get("file_path")
            ):
                items = [output]
        if not items:
            return []

        cite_sources = self._cite_sources_requested()
        collected = self._collected_citations
        mined: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            content = (
                item.get("content")
                or item.get("text")
                or item.get("snippet")
                or item.get("summary")
                or ""
            )
            source = (
                meta.get("source")
                or meta.get("url")
                or meta.get("file_path")
                or meta.get("file")
                or meta.get("title")
                or meta.get("doc")
                or meta.get("name")
                or item.get("source")
                or item.get("url")
                or item.get("title")
                or item.get("id")
                or tool_name
            )
            quote = str(content).strip().replace("\n", " ")
            if len(quote) > 200:
                quote = quote[:197].rstrip() + "..."
            try:
                score = float(item.get("score", meta.get("score", 0.0)) or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            entry: dict[str, Any] = {
                "source": str(source),
                "chunk_id": str(item.get("id") or meta.get("chunk_id") or ""),
                "score": score,
                "quote": quote,
                "page": meta.get("page"),
                "section": meta.get("section"),
            }
            if cite_sources:
                identity = _key_of(entry)
                already = next(
                    (
                        e for e in collected
                        if e.get("cite_index") and _key_of(e) == identity
                    ),
                    None,
                )
                if already is not None:
                    mined.append(already)
                    continue
                entry["cite_index"] = 1 + max(
                    (e.get("cite_index") or 0 for e in collected), default=0
                )
                # The numbered list the model reads is the full passage, not the
                # short quote the citation entry carries.
                entry["text"] = str(content).strip()
            collected.append(entry)
            mined.append(entry)
        return mined

    @staticmethod
    def _numbered_passage_block(entries: list[dict[str, Any]]) -> str:
        """Render mined passages as the numbered list ``[n]`` markers index."""
        return "\n\n".join(
            f"[{e['cite_index']}] Source: {e['source']}\n{e.get('text') or e['quote']}"
            for e in entries
            if e.get("cite_index")
        )

    def _attach_citations(self, response: "AgentResponse") -> None:
        """
        Populate ``response.sources`` / ``response.citations`` from the evidence
        collected during the run (deduplicated, 1-based citation indices). Only
        fills fields the assembly path left empty, so explicit callers win.

        Two evidence sources are merged: passages mined from local retrieval/
        search tools (``_collected_citations``), and grounded URLs a provider
        surfaced natively (``metadata["grounding_chunks"]`` — OpenAI web_search
        ``url_citation`` annotations and Gemini search grounding). Only real,
        retrieved URLs land here — never URLs scraped from the model's prose.

        A grounding chunk may carry ``"cited": False`` to mark a URL a search
        returned but the model never referenced inline (e.g. OpenAI
        ``web_search_call`` action sources). Those widen ``response.sources``
        so a search that ran is never left unsourced, but they never
        manufacture a ``Citation`` entry the model did not actually make.

        Locally-mined URL sources (``_collect_citations``, e.g. a plain
        ``web_search`` tool) carry no such explicit marker; a search can
        return results the model never draws on. For those, "cited" is
        determined by whether the model's final answer text actually
        contains the URL — mirroring the native-provider distinction above.
        Non-URL sources (e.g. a RAG file path) use inline ``[N]`` bracket
        markers this generic mining path can't reliably map back to one
        item across multiple tool calls, so they keep the previous default
        (cited=True) rather than risk dropping a real citation.

        When the caller asked for inline citations (``cite_sources``) that
        guesswork is replaced by the numbering the model was actually shown:
        every passage presented to it as ``[n]`` becomes ``citations[n - 1]``,
        whichever retrieval call produced it and whether or not its source is a
        URL. Provider-native grounding chunks are unaffected in both modes —
        the model never saw a number for those.
        """
        cite_sources = self._cite_sources_requested()
        raw = list(getattr(self, "_collected_citations", None) or [])
        # Fold provider-native grounding chunks ({url, title}) into the same
        # accumulator shape so the dedup/assembly below handles every path.
        meta = response.metadata if isinstance(response.metadata, dict) else {}
        for chunk in meta.get("grounding_chunks") or []:
            if not isinstance(chunk, dict):
                continue
            url = chunk.get("url") or chunk.get("uri") or chunk.get("source")
            if not url:
                continue
            raw.append({
                "source": str(url),
                "chunk_id": "",
                "score": 0.0,
                "quote": str(chunk.get("title") or chunk.get("snippet") or "").strip(),
                "page": None,
                "section": None,
                "cited": chunk.get("cited", True),
            })
        if not raw:
            return

        from ..rag.attribution import Citation

        answer_text = response.output or ""
        seen: set[tuple[str, str, str]] = set()
        citations: list[Citation] = []
        sources: list[str] = []
        seen_sources: set[str] = set()
        for entry in raw:
            # A passage the model was shown as "[n] Source: ..." keeps that
            # number, so marker n in the answer is citations[n - 1].
            numbered = entry.get("cite_index") if cite_sources else None
            is_cited = entry.get("cited")
            if numbered is not None:
                is_cited = True
            elif is_cited is None:
                source = entry["source"]
                if source.startswith(("http://", "https://")):
                    is_cited = source in answer_text
                else:
                    is_cited = True
            if is_cited:
                key = _key_of(entry)
                if key not in seen:
                    seen.add(key)
                    citations.append(Citation(
                        index=numbered if numbered is not None else len(citations) + 1,
                        source=entry["source"],
                        chunk_id=entry["chunk_id"],
                        relevance_score=entry["score"],
                        quote=entry["quote"],
                        page=entry["page"],
                        section=entry["section"],
                    ))
            if entry["source"] not in seen_sources:
                seen_sources.add(entry["source"])
                sources.append(entry["source"])

        if not response.citations:
            response.citations = citations
        if not response.sources:
            response.sources = sources
