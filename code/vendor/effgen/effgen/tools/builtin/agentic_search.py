"""
Agentic Search tool using bash commands for exact query matching.

This module provides an advanced retrieval tool that uses bash commands
(grep, find, etc.) to search for exact matches in a knowledge base,
then provides context around the matches.

Features:
- ripgrep (rg) as preferred backend, with grep fallback
- File-type aware search (Python, JSON, Markdown)
- Multi-query search with cross-query correlation
- Search result summarization
- Search agent system prompt

This is an alternative to embedding-based RAG that works particularly
well for:
- Exact phrase matching
- Technical queries (code, formulas, specific terms)
- When semantic similarity might miss exact answers
- Large knowledge bases where indexing is impractical
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)

logger = logging.getLogger(__name__)


# Search agent system prompt
SEARCH_AGENT_SYSTEM_PROMPT = """You are a search agent. You have access to the following search tools:
- agentic_search: Search files using grep/ripgrep for exact matches

Strategy:
1. Start with broad keywords from the user's question
2. Narrow down based on initial results
3. Use exact phrases for precise matches
4. Search across multiple terms if needed
5. Combine results from different queries to form a complete answer

Tips:
- For factual questions, extract key nouns and proper nouns as search terms
- For technical content, include specific identifiers (function names, error codes)
- Use search_mode='keywords' for natural language queries
- Use search_mode='exact' when you know the precise phrase
- If no results found, try synonyms or related terms
"""


@dataclass
class SearchMatch:
    """A search match with context."""
    file_path: str
    line_number: int
    match_line: str
    context_before: list[str]
    context_after: list[str]
    relevance_score: float


class AgenticSearch(BaseTool):
    """
    Agentic Search tool using bash commands for precise retrieval.

    Unlike embedding-based RAG, this tool uses exact string matching
    with grep/ripgrep to locate information, then provides context.

    Features:
    - ripgrep (rg) preferred backend (auto-detected, with grep fallback)
    - File-type aware search (Python, JSON, Markdown)
    - Multi-query search with cross-query correlation
    - Exact string matching (grep-based)
    - Context extraction (configurable lines before/after)
    - Multiple file format support (txt, json, jsonl, md, csv)
    - Case-sensitive/insensitive search
    - Regular expression support
    - Keyword extraction and multi-term search
    - Relevance scoring based on match quality
    """

    def __init__(
        self,
        data_path: str | None = None,
        context_lines: int = 5,
        max_results: int = 10,
        supported_extensions: list[str] | None = None,
    ) -> None:
        """
        Initialize the agentic search tool.

        Args:
            data_path: Path to the knowledge base directory/file
            context_lines: Number of lines of context to include before/after match
            max_results: Maximum number of results to return
            supported_extensions: File extensions to search (default: txt, json, jsonl, md, csv)
        """
        super().__init__(
            metadata=ToolMetadata(
                name="agentic_search",
                description=(
                    "Search a knowledge base using exact string matching with grep/ripgrep. "
                    "Returns matching passages with surrounding context. "
                    "Use this tool when you need to find EXACT information, specific terms, "
                    "numbers, formulas, or technical content. Works better than semantic search "
                    "for precise queries."
                ),
                category=ToolCategory.INFORMATION_RETRIEVAL,
                parameters=[
                    ParameterSpec(
                        name="query",
                        type=ParameterType.STRING,
                        description=(
                            "The search query. Can be exact text, keywords, or a regex pattern. "
                            "For best results, use specific terms from the expected answer."
                        ),
                        required=True,
                        min_length=1,
                        max_length=500,
                    ),
                    ParameterSpec(
                        name="context_lines",
                        type=ParameterType.INTEGER,
                        description="Number of lines of context to show before and after each match",
                        required=False,
                        default=5,
                        min_value=0,
                        max_value=50,
                    ),
                    ParameterSpec(
                        name="case_sensitive",
                        type=ParameterType.BOOLEAN,
                        description="Whether to perform case-sensitive search",
                        required=False,
                        default=False,
                    ),
                    ParameterSpec(
                        name="use_regex",
                        type=ParameterType.BOOLEAN,
                        description="Whether to treat query as a regular expression",
                        required=False,
                        default=False,
                    ),
                    ParameterSpec(
                        name="max_results",
                        type=ParameterType.INTEGER,
                        description="Maximum number of results to return",
                        required=False,
                        default=10,
                        min_value=1,
                        max_value=50,
                    ),
                    ParameterSpec(
                        name="search_mode",
                        type=ParameterType.STRING,
                        description=(
                            "Search mode: 'exact' for exact phrase, "
                            "'keywords' to search for all keywords, "
                            "'any' to match any keyword"
                        ),
                        required=False,
                        default="keywords",
                        enum=["exact", "keywords", "any"],
                    ),
                    ParameterSpec(
                        name="file_type",
                        type=ParameterType.STRING,
                        description="Filter by file type: 'all', 'python', 'json', 'markdown', 'text'",
                        required=False,
                        default="all",
                        enum=["all", "python", "json", "markdown", "text"],
                    ),
                ],
                returns={
                    "type": "object",
                    "properties": {
                        "results": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "content": {"type": "string"},
                                    "file": {"type": "string"},
                                    "line_number": {"type": "integer"},
                                    "score": {"type": "number"},
                                },
                            },
                        },
                        "total_matches": {"type": "integer"},
                        "query_terms": {"type": "array"},
                    },
                },
                timeout_seconds=60,
                tags=["search", "retrieval", "grep", "ripgrep", "exact-match", "knowledge-base"],
                examples=[
                    {
                        "query": "photosynthesis converts",
                        "context_lines": 5,
                        "output": {
                            "results": [
                                {
                                    "content": "...context...\nPhotosynthesis converts light energy into chemical energy.\n...context...",
                                    "file": "biology.txt",
                                    "line_number": 42,
                                    "score": 0.95,
                                }
                            ],
                            "total_matches": 1,
                        },
                    }
                ],
            )
        )

        self.data_path = data_path
        self.default_context_lines = context_lines
        self.default_max_results = max_results
        self.supported_extensions = supported_extensions or [
            ".txt", ".json", ".jsonl", ".md", ".csv", ".tsv",
        ]

        # Detect backend
        self._use_ripgrep = False
        self._file_cache: dict[str, list[str]] = {}
        self._cache_enabled = False

    async def initialize(self) -> None:
        """Initialize the agentic search tool."""
        await super().initialize()

        if self.data_path:
            if not os.path.exists(self.data_path):
                logger.warning(f"Data path does not exist: {self.data_path}")
            else:
                logger.info(f"Agentic search initialized with data path: {self.data_path}")

        self._check_commands()

    def _check_commands(self):
        """Check available search commands."""
        if shutil.which("rg"):
            self._use_ripgrep = True
            logger.info("Using ripgrep for faster search")
        elif shutil.which("grep"):
            self._use_ripgrep = False
            logger.info("Using grep (install ripgrep for faster search: apt install ripgrep)")
        else:
            logger.warning("Neither ripgrep nor grep found")

    def set_data_path(self, path: str) -> None:
        """Set the data path for searching."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data path not found: {path}")
        self.data_path = path
        self._file_cache.clear()
        logger.info(f"Data path set to: {path}")

    @staticmethod
    def get_search_agent_prompt() -> str:
        """Get the system prompt for search agents."""
        return SEARCH_AGENT_SYSTEM_PROMPT

    def _get_file_type_extensions(self, file_type: str) -> list[str]:
        """Get file extensions for a given file type filter."""
        FILE_TYPE_MAP = {
            "all": self.supported_extensions,
            "python": [".py"],
            "json": [".json", ".jsonl"],
            "markdown": [".md", ".markdown"],
            "text": [".txt"],
        }
        return FILE_TYPE_MAP.get(file_type, self.supported_extensions)

    def _extract_keywords(self, query: str) -> list[str]:
        """Extract meaningful keywords from query."""
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into", "through",
            "during", "before", "after", "above", "below", "between", "under",
            "again", "further", "then", "once", "here", "there", "when",
            "where", "why", "how", "all", "each", "few", "more", "most",
            "other", "some", "such", "no", "nor", "not", "only", "own",
            "same", "so", "than", "too", "very", "just", "but", "and",
            "or", "if", "because", "until", "while", "what", "which", "who",
            "whom", "this", "that", "these", "those", "am", "it", "its"
        }

        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        original_words = re.findall(r'\b\w+\b', query)
        keyword_set = set(keywords)

        result = []
        for orig in original_words:
            if orig.lower() in keyword_set and orig.lower() not in [r.lower() for r in result]:
                result.append(orig)

        return result if result else words[:5]

    def _run_search(
        self,
        pattern: str,
        path: str,
        case_sensitive: bool = False,
        context_lines: int = 5,
        use_regex: bool = False,
        max_count: int = 100,
        file_extensions: list[str] | None = None,
    ) -> list[tuple[str, int, str, list[str], list[str]]]:
        """
        Run search using ripgrep or grep.

        Returns:
            List of (file, line_num, match_line, context_before, context_after)
        """
        if self._use_ripgrep:
            return self._run_ripgrep(
                pattern, path, case_sensitive, context_lines,
                use_regex, max_count, file_extensions
            )
        else:
            return self._run_grep(
                pattern, path, case_sensitive, context_lines,
                use_regex, max_count, file_extensions
            )

    def _run_ripgrep(
        self,
        pattern: str,
        path: str,
        case_sensitive: bool = False,
        context_lines: int = 5,
        use_regex: bool = False,
        max_count: int = 100,
        file_extensions: list[str] | None = None,
    ) -> list[tuple[str, int, str, list[str], list[str]]]:
        """Run ripgrep and parse results."""
        cmd = ["rg"]

        if not case_sensitive:
            cmd.append("-i")

        if not use_regex:
            cmd.append("-F")

        cmd.extend(["-C", str(context_lines)])
        cmd.append("-n")
        cmd.extend(["-m", str(max_count)])

        # File type filtering
        extensions = file_extensions or self.supported_extensions
        if os.path.isdir(path):
            for ext in extensions:
                cmd.extend(["-g", f"*{ext}"])

        cmd.append(pattern)
        cmd.append(path)

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30,
            )
            if result.returncode > 1:
                logger.warning(f"ripgrep error: {result.stderr}")
                return []
            return self._parse_grep_output(result.stdout, context_lines, path)
        except subprocess.TimeoutExpired:
            logger.warning("ripgrep timed out")
            return []
        except Exception as e:
            logger.error(f"Error running ripgrep: {e}")
            return []

    def _run_grep(
        self,
        pattern: str,
        path: str,
        case_sensitive: bool = False,
        context_lines: int = 5,
        use_regex: bool = False,
        max_count: int = 100,
        file_extensions: list[str] | None = None,
    ) -> list[tuple[str, int, str, list[str], list[str]]]:
        """Run grep command and parse results."""
        cmd = ["grep"]

        if not case_sensitive:
            cmd.append("-i")
        if not use_regex:
            cmd.append("-F")

        cmd.extend(["-B", str(context_lines), "-A", str(context_lines)])
        cmd.append("-n")

        if os.path.isdir(path):
            cmd.append("-r")
            extensions = file_extensions or self.supported_extensions
            for ext in extensions:
                cmd.extend(["--include", f"*{ext}"])

        cmd.extend(["-m", str(max_count)])
        cmd.append(pattern)
        cmd.append(path)

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30,
            )
            if result.returncode > 1:
                logger.warning(f"grep error: {result.stderr}")
                return []
            return self._parse_grep_output(result.stdout, context_lines, path)
        except subprocess.TimeoutExpired:
            logger.warning("grep command timed out")
            return []
        except Exception as e:
            logger.error(f"Error running grep: {e}")
            return []

    def _parse_grep_output(
        self,
        output: str,
        context_lines: int,
        source_file: str = "unknown",
    ) -> list[tuple[str, int, str, list[str], list[str]]]:
        """Parse grep/ripgrep output with context."""
        results = []
        current_match_line = None
        current_line_num = None
        context_before = []
        context_after = []
        in_after_context = False
        after_count = 0

        for line in output.split("\n"):
            if not line:
                continue

            if line == "--":
                if current_match_line is not None:
                    results.append((
                        source_file, current_line_num or 0,
                        current_match_line, context_before.copy(), context_after.copy(),
                    ))
                current_match_line = None
                current_line_num = None
                context_before = []
                context_after = []
                in_after_context = False
                after_count = 0
                continue

            # Try single-file format: LINE_NUM:CONTENT or LINE_NUM-CONTENT
            single_match = re.match(r'^(\d+)([:\-])(.*)$', line)
            if single_match:
                line_num = int(single_match.group(1))
                separator = single_match.group(2)
                content = single_match.group(3)
                file_path = source_file
            else:
                # Try format with filename: FILE:LINE_NUM:CONTENT
                multi_match = re.match(r'^(.+?):(\d+)([:\-])(.*)$', line)
                if multi_match:
                    file_path = multi_match.group(1)
                    line_num = int(multi_match.group(2))
                    separator = multi_match.group(3)
                    content = multi_match.group(4)
                else:
                    continue

            is_match = (separator == ":")

            if is_match and not in_after_context:
                if current_match_line is not None:
                    results.append((
                        file_path, current_line_num or 0,
                        current_match_line, context_before.copy(), context_after.copy(),
                    ))
                    context_before = []
                    context_after = []

                current_line_num = line_num
                current_match_line = content
                in_after_context = True
                after_count = 0

            elif in_after_context:
                context_after.append(content)
                after_count += 1
                if after_count >= context_lines:
                    in_after_context = False
            else:
                context_before.append(content)
                if len(context_before) > context_lines:
                    context_before.pop(0)

        if current_match_line is not None:
            results.append((
                source_file, current_line_num or 0,
                current_match_line, context_before, context_after,
            ))

        return results

    def _calculate_relevance(
        self,
        match_line: str,
        context: str,
        query_terms: list[str],
    ) -> float:
        """Calculate relevance score for a match."""
        if not query_terms:
            return 0.5

        full_text = (match_line + " " + context).lower()
        match_line_lower = match_line.lower()

        terms_in_match = sum(1 for t in query_terms if t.lower() in match_line_lower)
        terms_in_context = sum(1 for t in query_terms if t.lower() in full_text)

        match_score = terms_in_match / len(query_terms) if query_terms else 0
        context_score = terms_in_context / len(query_terms) if query_terms else 0

        score = 0.7 * match_score + 0.3 * context_score

        query_phrase = " ".join(query_terms).lower()
        if query_phrase in match_line_lower:
            score = min(1.0, score + 0.2)

        return round(score, 4)

    def _summarize_results(self, results: list[dict], query: str, max_length: int = 500) -> str:
        """Generate a simple extractive summary of search results."""
        if not results:
            return "No results found."

        snippets = []
        for r in results[:5]:
            match_line = r.get("match_line", "")
            if match_line:
                snippets.append(match_line.strip())

        summary = " ... ".join(snippets)
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."

        return summary

    def multi_query_search(
        self,
        queries: list[str],
        context_lines: int = 5,
        case_sensitive: bool = False,
        max_results: int = 10,
    ) -> dict[str, Any]:
        """
        Search for multiple queries and correlate results.

        Args:
            queries: List of search queries
            context_lines: Lines of context
            case_sensitive: Case sensitivity
            max_results: Max results per query

        Returns:
            Dict with per-query results and cross-query summary
        """
        if not self.data_path or not os.path.exists(self.data_path):
            return {"error": "No valid data path set", "results_by_query": {}}

        all_results = {}
        all_files = set()

        for query in queries:
            keywords = self._extract_keywords(query)
            matches = self._run_search(
                keywords[0] if keywords else query,
                self.data_path,
                case_sensitive=case_sensitive,
                context_lines=context_lines,
            )

            results = []
            for file_path, line_num, match_line, before, after in matches:
                context_str = "\n".join(before) + "\n>>> " + match_line + " <<<\n" + "\n".join(after)
                score = self._calculate_relevance(match_line, context_str, keywords)
                results.append({
                    "content": context_str.strip(),
                    "match_line": match_line,
                    "file": os.path.basename(file_path),
                    "line_number": line_num,
                    "score": score,
                })
                all_files.add(os.path.basename(file_path))

            results.sort(key=lambda x: x["score"], reverse=True)
            all_results[query] = results[:max_results]

        return {
            "results_by_query": all_results,
            "files_searched": sorted(all_files),
            "total_queries": len(queries),
        }

    async def _execute(
        self,
        query: str,
        context_lines: int | None = None,
        case_sensitive: bool = False,
        use_regex: bool = False,
        max_results: int | None = None,
        search_mode: str = "keywords",
        file_type: str = "all",
        **kwargs,
    ) -> dict[str, Any]:
        """Execute agentic search."""
        if not self.data_path:
            return {
                "results": [],
                "total_matches": 0,
                "error": "No data path set. Use set_data_path() to specify the knowledge base location.",
            }

        if not os.path.exists(self.data_path):
            return {
                "results": [],
                "total_matches": 0,
                "error": f"Data path not found: {self.data_path}",
            }

        context_lines = context_lines or self.default_context_lines
        max_results = max_results or self.default_max_results
        file_extensions = self._get_file_type_extensions(file_type)

        query_terms = self._extract_keywords(query)
        all_matches = []

        if search_mode == "exact":
            matches = self._run_search(
                query, self.data_path,
                case_sensitive=case_sensitive,
                context_lines=context_lines,
                use_regex=use_regex,
                file_extensions=file_extensions,
            )
            all_matches.extend(matches)

        elif search_mode == "keywords":
            if query_terms:
                first_term = query_terms[0]
                matches = self._run_search(
                    first_term, self.data_path,
                    case_sensitive=case_sensitive,
                    context_lines=context_lines,
                    use_regex=False,
                    file_extensions=file_extensions,
                )
                for match in matches:
                    file_path, line_num, match_line, before, after = match
                    full_context = " ".join(before) + " " + match_line + " " + " ".join(after)
                    full_context_lower = full_context.lower()
                    if all(term.lower() in full_context_lower for term in query_terms):
                        all_matches.append(match)

        elif search_mode == "any":
            seen = set()
            for term in query_terms:
                matches = self._run_search(
                    term, self.data_path,
                    case_sensitive=case_sensitive,
                    context_lines=context_lines,
                    use_regex=False,
                    file_extensions=file_extensions,
                )
                for match in matches:
                    key = (match[0], match[1])
                    if key not in seen:
                        seen.add(key)
                        all_matches.append(match)

        # Build results with relevance scoring
        results = []
        for file_path, line_num, match_line, before, after in all_matches:
            context_str = "\n".join(before) + "\n>>> " + match_line + " <<<\n" + "\n".join(after)
            score = self._calculate_relevance(match_line, context_str, query_terms)

            results.append({
                "content": context_str.strip(),
                "match_line": match_line,
                "file": os.path.basename(file_path),
                "file_path": file_path,
                "line_number": line_num,
                "score": score,
                "context_before": before,
                "context_after": after,
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:max_results]

        for i, r in enumerate(results):
            r["rank"] = i + 1

        # Generate summary
        summary = self._summarize_results(results, query)

        return {
            "results": results,
            "total_matches": len(all_matches),
            "query_terms": query_terms,
            "search_mode": search_mode,
            "data_path": self.data_path,
            "backend": "ripgrep" if self._use_ripgrep else "grep",
            "summary": summary,
        }

    def search_in_json(
        self,
        query: str,
        json_path: str,
        content_field: str = "text",
        context_lines: int = 5,
    ) -> list[dict[str, Any]]:
        """Search within a JSON/JSONL file's specific field.

        Args:
            query: The text to look for.
            json_path: Path to the JSON or JSONL file to search.
            content_field: The field within each record that is searched.
            context_lines: Lines of surrounding text kept with each match.

        Returns:
            One entry per match, with its location and surrounding text.
        """
        results = []
        path = Path(json_path)
        if not path.exists():
            return results

        query_lower = query.lower()
        keywords = self._extract_keywords(query)

        if path.suffix == ".jsonl":
            with open(path, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    try:
                        entry = json.loads(line.strip())
                        content = entry.get(content_field, "")
                        content_lower = str(content).lower()
                        if query_lower in content_lower or all(k.lower() in content_lower for k in keywords):
                            score = self._calculate_relevance(str(content), "", keywords)
                            results.append({
                                "content": content,
                                "entry": entry,
                                "line_index": i,
                                "score": score,
                            })
                    except json.JSONDecodeError:
                        continue

        elif path.suffix == ".json":
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                for i, entry in enumerate(data):
                    if isinstance(entry, dict):
                        content = entry.get(content_field, "")
                    else:
                        content = str(entry)
                    content_lower = content.lower()
                    if query_lower in content_lower or all(k.lower() in content_lower for k in keywords):
                        score = self._calculate_relevance(content, "", keywords)
                        results.append({
                            "content": content,
                            "entry": entry if isinstance(entry, dict) else {"text": entry},
                            "index": i,
                            "score": score,
                        })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    async def cleanup(self) -> None:
        """Clean up resources."""
        self._file_cache.clear()
        await super().cleanup()
