"""A message raised as a builtin exception is held to the same bar — ratcheted.

``test_error_message_actionability`` gates effGen's *typed* error classes: the
message says what happened and then what to do, it bounds any text it quotes,
and it redacts credentials. That contract was green while ``effgen/`` also raised
Python's own exception types directly — ``ValueError``, ``RuntimeError``,
``ImportError``, ``KeyError`` — whose messages reach exactly the same places: a
CLI panel, a ``ToolResult.error``, a server error envelope. Those were held to
nothing.

Rewording all of them is a sweep of its own; what this file does is make the
count **monotonically decreasing**. Every module carries a ceiling — the number
of non-actionable builtin raises it had when the gate landed — and the test
fails if a module goes above it. So new code must be actionable from the start,
and a module that improves has its ceiling lowered in the same commit, which is
the only way the number moves.

The ``protocols/`` subtree is included: its audience is a different one, but its
messages surface the same way, and excluding it would have meant a second rule
to remember.
"""
from __future__ import annotations

import ast
from pathlib import Path

from tests.unit.test_error_message_actionability import _template, is_actionable

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "effgen"

#: The builtin exception types effGen raises directly.
BUILTIN_NAMES = frozenset({
    "ValueError", "RuntimeError", "TypeError", "OSError", "FileNotFoundError",
    "NotImplementedError", "KeyError", "ImportError", "PermissionError",
    "IsADirectoryError", "NotADirectoryError", "TimeoutError", "LookupError",
    "AttributeError", "IndexError", "ConnectionError", "MemoryError",
})

#: Non-actionable builtin raises per module, recorded when this gate landed.
#: A number may only go **down**, and only in the commit that reworded the
#: messages. A module absent from this map must be at zero.
_UNREWORDED: dict[str, int] = {
    "effgen/__init__.py": 2,
    "effgen/api/pool.py": 4,
    "effgen/api/queue.py": 1,
    "effgen/api/tenancy.py": 1,
    "effgen/cache/prompt_cache.py": 1,
    "effgen/cache/result_cache.py": 1,
    "effgen/cli/code/git_actions.py": 1,
    "effgen/cli/commands/batch.py": 8,
    "effgen/cli/commands/eval.py": 2,
    "effgen/cli/loadtest.py": 1,
    "effgen/cli/playground.py": 1,
    "effgen/config/loader.py": 6,
    "effgen/core/_compat.py": 2,
    "effgen/core/agent.py": 4,
    "effgen/core/agent_generation.py": 1,
    "effgen/core/agent_orchestration.py": 2,
    "effgen/core/agent_runtime.py": 2,
    "effgen/core/agent_streaming.py": 1,
    "effgen/core/agent_tool_execution.py": 3,
    "effgen/core/aggregation.py": 2,
    "effgen/core/background.py": 1,
    "effgen/core/batch.py": 5,
    "effgen/core/checkpoint.py": 7,
    "effgen/core/complexity_analyzer.py": 1,
    "effgen/core/decomposition_engine.py": 1,
    "effgen/core/execution_tracker_render.py": 1,
    "effgen/core/feedback.py": 1,
    "effgen/core/lifecycle.py": 3,
    "effgen/core/orchestrator.py": 3,
    "effgen/core/router.py": 2,
    "effgen/core/session.py": 3,
    "effgen/core/state.py": 2,
    "effgen/core/structured_output.py": 2,
    "effgen/core/sub_agent_manager.py": 1,
    "effgen/core/task.py": 1,
    "effgen/core/workflow.py": 9,
    "effgen/domains/expander.py": 2,
    "effgen/eval/battle.py": 2,
    "effgen/eval/evaluator.py": 2,
    "effgen/execution/docker_sandbox.py": 4,
    "effgen/execution/sandbox.py": 5,
    "effgen/execution/validators.py": 1,
    "effgen/gpu/allocator.py": 3,
    "effgen/gpu/utils.py": 7,
    "effgen/jupyter/magics.py": 1,
    "effgen/memory/compaction.py": 1,
    "effgen/memory/token_budget.py": 3,
    "effgen/memory/vector_store.py": 2,
    "effgen/models/__init__.py": 2,
    "effgen/models/_catalog.py": 2,
    "effgen/models/_refresh.py": 2,
    "effgen/models/anthropic_adapter.py": 1,
    "effgen/models/base.py": 1,
    "effgen/models/batching.py": 3,
    "effgen/models/fireworks_models.py": 2,
    "effgen/models/gemini_adapter.py": 2,
    "effgen/models/gemini_files.py": 4,
    "effgen/models/gguf_engine.py": 2,
    "effgen/models/groq_adapter.py": 1,
    "effgen/models/mlx_vlm_engine.py": 1,
    "effgen/models/model_loader.py": 1,
    "effgen/models/model_loader_local.py": 2,
    "effgen/models/openai_adapter.py": 1,
    "effgen/models/registry.py": 4,
    "effgen/models/router.py": 2,
    "effgen/models/routing/retry.py": 1,
    "effgen/models/transformers_engine_generation.py": 2,
    "effgen/models/transformers_engine_placement.py": 2,
    "effgen/models/transformers_engine_streaming.py": 1,
    "effgen/models/vllm_engine.py": 1,
    "effgen/observability/alerting.py": 5,
    "effgen/observability/slo.py": 3,
    "effgen/observability/tracing.py": 1,
    "effgen/observability/tracing_samplers.py": 1,
    "effgen/presets/registry.py": 3,
    "effgen/prompts/chain_manager.py": 8,
    "effgen/prompts/library/registry.py": 11,
    "effgen/prompts/template_manager.py": 2,
    "effgen/rag/search.py": 1,
    "effgen/reliability/bulkhead.py": 1,
    "effgen/reliability/circuit.py": 1,
    "effgen/reliability/timeouts.py": 6,
    "effgen/security/sandbox.py": 1,
    "effgen/server/app.py": 1,
    "effgen/server/auth.py": 1,
    "effgen/server/rbac.py": 1,
    "effgen/tools/__init__.py": 1,
    "effgen/tools/builtin/__init__.py": 1,
    "effgen/tools/builtin/_repl_worker.py": 3,
    "effgen/tools/builtin/agentic_search.py": 1,
    "effgen/tools/builtin/arxiv.py": 8,
    "effgen/tools/builtin/audio_transcribe.py": 7,
    "effgen/tools/builtin/bash_tool.py": 4,
    "effgen/tools/builtin/code_executor.py": 1,
    "effgen/tools/builtin/data_analysis.py": 13,
    "effgen/tools/builtin/datetime_tool.py": 4,
    "effgen/tools/builtin/devops.py": 11,
    "effgen/tools/builtin/docx.py": 3,
    "effgen/tools/builtin/email_imap.py": 1,
    "effgen/tools/builtin/email_smtp.py": 2,
    "effgen/tools/builtin/excel.py": 3,
    "effgen/tools/builtin/file_ops.py": 15,
    "effgen/tools/builtin/finance.py": 6,
    "effgen/tools/builtin/geocode.py": 3,
    "effgen/tools/builtin/hackernews.py": 5,
    "effgen/tools/builtin/image_caption.py": 3,
    "effgen/tools/builtin/image_info.py": 6,
    "effgen/tools/builtin/json_tool.py": 1,
    "effgen/tools/builtin/knowledge.py": 3,
    "effgen/tools/builtin/news.py": 7,
    "effgen/tools/builtin/ocr.py": 8,
    "effgen/tools/builtin/openai_native.py": 2,
    "effgen/tools/builtin/pdf.py": 4,
    "effgen/tools/builtin/pubmed.py": 7,
    "effgen/tools/builtin/qr_generate.py": 2,
    "effgen/tools/builtin/qr_read.py": 1,
    "effgen/tools/builtin/reddit.py": 9,
    "effgen/tools/builtin/retrieval.py": 6,
    "effgen/tools/builtin/rss.py": 5,
    "effgen/tools/builtin/semantic_scholar.py": 11,
    "effgen/tools/builtin/text_processing.py": 6,
    "effgen/tools/builtin/translate.py": 5,
    "effgen/tools/builtin/weather.py": 5,
    "effgen/tools/builtin/web_search.py": 6,
    "effgen/tools/builtin/wikipedia_tool.py": 2,
    "effgen/tools/builtin/youtube_metadata.py": 10,
    "effgen/tools/builtin/youtube_transcript.py": 13,
    "effgen/tools/function_tool.py": 2,
    "effgen/tools/plugin.py": 1,
    "effgen/tools/protocols/__init__.py": 1,
    "effgen/tools/protocols/a2a/client.py": 10,
    "effgen/tools/protocols/acp/client.py": 15,
    "effgen/tools/protocols/acp/server.py": 6,
    "effgen/tools/protocols/mcp/client.py": 26,
    "effgen/tools/protocols/mcp/protocol.py": 6,
    "effgen/tools/protocols/mcp/server.py": 4,
    "effgen/tools/protocols/mcp_official/client.py": 12,
    "effgen/tools/registry.py": 3,
    "effgen/utils/embedding_backend.py": 4,
}

def _non_actionable(path: Path) -> list[tuple[int, str | None]]:
    """Every builtin raise in *path* whose literal message is not actionable."""
    found: list[tuple[int, str | None]] = []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call)):
            continue
        func = node.exc.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name not in BUILTIN_NAMES:
            continue
        message = _template(node.exc.args[0]) if node.exc.args else None
        if not (message and is_actionable(message)):
            found.append((node.lineno, message))
    return found


def _scan() -> dict[str, list[tuple[int, str | None]]]:
    out: dict[str, list[tuple[int, str | None]]] = {}
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        rel = path.relative_to(PACKAGE_ROOT.parent).as_posix()
        found = _non_actionable(path)
        if found:
            out[rel] = found
    return out


def test_no_module_raises_more_unactionable_builtins_than_it_used_to():
    """The ratchet. A module may improve; it may never get worse."""
    scanned = _scan()
    regressions = []
    for module, found in sorted(scanned.items()):
        ceiling = _UNREWORDED.get(module, 0)
        if len(found) > ceiling:
            lines = ", ".join(str(line) for line, _ in found[:6])
            regressions.append(
                f"  {module}: {len(found)} non-actionable builtin raises, "
                f"ceiling {ceiling} (lines {lines})"
            )
    assert not regressions, (
        "New builtin-exception messages must say what to do next.\n"
        + "\n".join(regressions)
        + "\n\nWrite the message as two sentences — what happened, then what "
        "to do — or reword the module and lower its entry in _UNREWORDED."
    )


def test_a_module_that_was_reworded_has_its_ceiling_lowered():
    """A stale ceiling is a ratchet that stopped ratcheting.

    If a module improves and its entry is left where it was, the gap becomes
    room for a later regression to hide in. Lowering it is part of the same
    commit.
    """
    scanned = _scan()
    stale = [
        f"  {module}: ceiling {ceiling}, actual {len(scanned.get(module, []))}"
        for module, ceiling in sorted(_UNREWORDED.items())
        if len(scanned.get(module, [])) < ceiling
    ]
    assert not stale, (
        "These modules are better than their recorded ceiling; lower it:\n"
        + "\n".join(stale)
    )


def test_the_ceiling_map_names_no_module_that_no_longer_exists():
    for module in _UNREWORDED:
        assert (PACKAGE_ROOT.parent / module).exists(), f"{module} is gone"


def test_the_tools_tree_is_where_the_sweep_starts():
    """The reworded modules stay reworded.

    ``effgen/tools/`` is the surface a user hits most, and the one where the
    message is often the only thing the model sees, so it is where the sweep
    began. ``calculator.py`` was the worst single module (15) and is at zero.
    """
    scanned = _scan()
    assert scanned.get("effgen/tools/builtin/calculator.py", []) == []
