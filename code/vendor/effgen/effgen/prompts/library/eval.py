"""
PromptEval — golden-string and live-model evaluation harness.
"""

from __future__ import annotations

import ast
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema import exceptions as jsonschema_exceptions

from .base import LibraryPrompt

logger = logging.getLogger(__name__)

GOLDENS_DIR = Path(__file__).parent.parent.parent.parent / "tests" / "prompts" / "goldens"


@dataclass
class RunOutput:
    """Text plus per-call metadata from a single model run.

    ``run_model`` returns this so a caller (the CLI ``prompts run`` path, the
    playground) can detect an empty/truncated billed result and read the
    token/cost/latency figures the model reported, rather than seeing only the
    text and treating an empty answer as a normal one.
    """

    text: str
    finish_reason: str = ""
    truncated: bool = False
    tokens_used: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: float | None = None
    max_tokens: int | None = None
    model_name: str = ""

    @property
    def is_empty(self) -> bool:
        """True when the model produced no visible (non-whitespace) text."""
        return not (self.text or "").strip()


@dataclass
class EvalResult:
    """Outcome of one prompt evaluation (golden render or live model run)."""

    name: str
    passed: bool
    kind: str  # 'golden' | 'live'
    message: str = ""
    rendered: str = ""
    model_output: str = ""


@dataclass
class EvalReport:
    """Collected evaluation results with pass/fail views and a text table."""

    results: list[EvalResult] = field(default_factory=list)

    def passed(self) -> list[EvalResult]:
        """The results that passed."""
        return [r for r in self.results if r.passed]

    def failed(self) -> list[EvalResult]:
        """The results that failed."""
        return [r for r in self.results if not r.passed]

    def as_table(self) -> str:
        """Render the results as an aligned plain-text table."""
        if not self.results:
            return "No prompts evaluated — registry is empty.\n"
        lines = [f"{'Name':<45} {'Kind':<8} {'Status'}"]
        lines.append("-" * 70)
        for r in self.results:
            status = "PASS" if r.passed else f"FAIL: {r.message}"
            lines.append(f"{r.name:<45} {r.kind:<8} {status}")
        lines.append("-" * 70)
        n_pass = len(self.passed())
        n_fail = len(self.failed())
        lines.append(f"Total: {len(self.results)}  Pass: {n_pass}  Fail: {n_fail}")
        return "\n".join(lines) + "\n"


class PromptEval:
    """Evaluation harness for LibraryPrompt instances."""

    def __init__(
        self,
        goldens_dir: Path | None = None,
        *,
        live_retries: int = 1,
        live_retry_delay: float = 65.0,
    ) -> None:
        self.goldens_dir = goldens_dir or GOLDENS_DIR
        self.goldens_dir.mkdir(parents=True, exist_ok=True)
        self.live_retries = max(0, live_retries)
        self.live_retry_delay = max(0.0, live_retry_delay)

    # ------------------------------------------------------------------
    # Golden evaluation
    # ------------------------------------------------------------------

    def eval_golden(self, prompt: LibraryPrompt) -> EvalResult:
        """Render with fixture; compare against stored golden string."""
        try:
            rendered = prompt.render_fixture()
        except Exception as exc:
            return EvalResult(
                name=prompt.name,
                passed=False,
                kind="golden",
                message=f"render error: {exc}",
            )

        golden_path = self.goldens_dir / f"{prompt.name}.txt"

        if not golden_path.exists():
            golden_path.parent.mkdir(parents=True, exist_ok=True)
            golden_path.write_text(rendered)
            logger.warning(
                "Golden file missing for '%s'; written to %s. "
                "Review it and commit.",
                prompt.name,
                golden_path,
            )
            return EvalResult(
                name=prompt.name,
                passed=True,
                kind="golden",
                message="(golden created on first run)",
                rendered=rendered,
            )

        expected = golden_path.read_text()
        if rendered.strip() == expected.strip():
            return EvalResult(
                name=prompt.name, passed=True, kind="golden", rendered=rendered
            )
        return EvalResult(
            name=prompt.name,
            passed=False,
            kind="golden",
            message="rendered output differs from golden",
            rendered=rendered,
        )

    # ------------------------------------------------------------------
    # Live evaluation
    # ------------------------------------------------------------------

    def eval_live(self, prompt: LibraryPrompt, model: str) -> EvalResult:
        """Run rendered prompt through a model; validate against expected_shape."""
        try:
            rendered = prompt.render_fixture()
        except Exception as exc:
            return EvalResult(
                name=prompt.name,
                passed=False,
                kind="live",
                message=f"render error: {exc}",
            )

        # Small models occasionally emit malformed or truncated structured
        # output. Allow up to ``live_retries`` extra attempts on shape failures
        # in addition to rate-limit retries.
        last_output = ""
        last_msg = ""
        for attempt in range(self.live_retries + 1):
            try:
                output = self._run_model(rendered, model)
            except Exception as exc:
                if (
                    attempt < self.live_retries
                    and self._is_rate_limit_error(exc)
                    and self.live_retry_delay > 0
                ):
                    time.sleep(self.live_retry_delay)
                    continue
                return EvalResult(
                    name=prompt.name,
                    passed=False,
                    kind="live",
                    message=f"model error: {exc}",
                )

            last_output = output
            if prompt.expected_shape is None:
                return EvalResult(
                    name=prompt.name,
                    passed=True,
                    kind="live",
                    rendered=rendered,
                    model_output=output,
                    message="(no shape check defined)",
                )

            ok, msg = self._check_shape(output, prompt.expected_shape)
            if ok or attempt >= self.live_retries:
                return EvalResult(
                    name=prompt.name,
                    passed=ok,
                    kind="live",
                    message=msg,
                    rendered=rendered,
                    model_output=output,
                )
            last_msg = msg

        return EvalResult(
            name=prompt.name,
            passed=False,
            kind="live",
            message=last_msg or "live eval failed",
            rendered=rendered,
            model_output=last_output,
        )

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def eval_all_golden(self, prompts: list[LibraryPrompt]) -> EvalReport:
        """Run the golden-render check for every prompt and collect a report."""
        report = EvalReport()
        for p in prompts:
            report.results.append(self.eval_golden(p))
        return report

    def eval_all_live(
        self, prompts: list[LibraryPrompt], model: str, delay: float = 35.0
    ) -> EvalReport:
        """Run each prompt against a live *model*, pausing *delay* seconds between runs.

        Args:
            prompts: The prompts to run.
            model: The model id each prompt is sent to.
            delay: Seconds to pause between runs, to stay inside rate limits.

        Returns:
            One report covering every prompt that was run.
        """
        report = EvalReport()
        for i, p in enumerate(prompts):
            if i > 0 and delay > 0:
                time.sleep(delay)
            report.results.append(self.eval_live(p, model))
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def run_model(
        self,
        prompt_text: str,
        model: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> RunOutput:
        """Call a model via effgen's model loader and return text + metadata.

        Routing rules (applied in order):
        1. ``provider:model_id`` prefix is handled natively by ``load_model``.
        2. A bare model id that lives in exactly one provider catalog is
           auto-routed to that provider. (Ambiguous ids raise as usual.)
        3. Otherwise the loader's default detection runs (OpenAI/Anthropic/
           Gemini/HF).

        ``max_tokens``/``temperature`` override the loaded model's own settings
        when supplied. With neither set and no model-level cap, the completion
        cap defaults to 4096 so structured-output prompts (JSON arrays,
        multi-section briefs) don't truncate mid-stream.

        Args:
            prompt_text: The rendered prompt to send.
            model: The model id to call.
            max_tokens: Output cap for the call.
            temperature: Sampling temperature for the call.
        """
        from effgen.models import load_model

        provider = None
        if isinstance(model, str) and ":" not in model:
            try:
                # Ensure adapters are imported so the registry is populated.
                import effgen.models.cerebras_adapter  # noqa: F401
                from effgen.models.registry import ProviderRegistry

                candidates = ProviderRegistry._model_index.get(model, [])
                if len(candidates) == 1:
                    provider = candidates[0]
            except Exception:
                provider = None

        loaded = load_model(model, provider=provider) if provider else load_model(model)

        resolved_max: int | None = None
        gen_kwargs: dict[str, Any] = {}
        try:
            from effgen.models.base import GenerationConfig

            existing_max = None
            existing_cfg = getattr(loaded, "config", None)
            if existing_cfg is not None:
                existing_max = getattr(existing_cfg, "max_tokens", None)
            resolved_max = max_tokens if max_tokens is not None else existing_max
            if resolved_max is None:
                resolved_max = 4096
            # Only build a fresh config when we change something; otherwise the
            # loaded model's own config (which may set a cap) is left untouched.
            if max_tokens is not None or temperature is not None or existing_max is None:
                cfg_kwargs: dict[str, Any] = {"max_tokens": resolved_max}
                if temperature is not None:
                    cfg_kwargs["temperature"] = temperature
                gen_kwargs["config"] = GenerationConfig(**cfg_kwargs)
        except Exception:
            pass

        result = loaded.generate(prompt_text, **gen_kwargs)
        return self._to_run_output(result, resolved_max, model)

    @staticmethod
    def _to_run_output(result: Any, max_tokens: int | None, model: str) -> RunOutput:
        """Fold a raw ``generate()`` return into a :class:`RunOutput`."""
        text = getattr(result, "text", None)
        if text is None and isinstance(result, dict):
            text = result.get("text") or result.get("content")
        if text is None:
            text = str(result)
        meta = getattr(result, "metadata", None) or {}
        finish_reason = getattr(result, "finish_reason", "") or ""
        truncated = bool(meta.get("truncated")) or finish_reason == "length"
        return RunOutput(
            text=text,
            finish_reason=finish_reason,
            truncated=truncated,
            tokens_used=getattr(result, "tokens_used", None),
            prompt_tokens=meta.get("prompt_tokens"),
            completion_tokens=meta.get("completion_tokens"),
            cost_usd=meta.get("cost_usd"),
            latency_ms=meta.get("latency_ms"),
            max_tokens=max_tokens,
            model_name=getattr(result, "model_name", None) or str(model),
        )

    def _run_model(
        self,
        prompt_text: str,
        model: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Return only the text output (used by the live-eval shape checks)."""
        return self.run_model(
            prompt_text, model, max_tokens=max_tokens, temperature=temperature
        ).text

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return (
            "429" in msg
            or "rate limit" in msg
            or "rate_limit" in msg
            or "request_quota_exceeded" in msg
            or "too_many_requests" in msg
        )

    @staticmethod
    def _escape_raw_controls_in_strings(text: str) -> str:
        """Escape raw newlines/tabs/CRs that appear inside JSON string literals.

        Some small models emit JSON where a string value spans multiple physical
        lines (e.g. a multi-line SQL query). Strict JSON disallows raw control
        characters inside strings, so ``json.loads`` rejects it. This helper
        rewrites unescaped newline/tab/CR characters that fall inside ``"..."``
        regions into their escaped equivalents so the result parses.
        """
        out: list[str] = []
        in_str = False
        escape = False
        for ch in text:
            if in_str:
                if escape:
                    out.append(ch)
                    escape = False
                    continue
                if ch == "\\":
                    out.append(ch)
                    escape = True
                    continue
                if ch == '"':
                    out.append(ch)
                    in_str = False
                    continue
                if ch == "\n":
                    out.append("\\n")
                    continue
                if ch == "\r":
                    out.append("\\r")
                    continue
                if ch == "\t":
                    out.append("\\t")
                    continue
                out.append(ch)
            else:
                out.append(ch)
                if ch == '"':
                    in_str = True
        return "".join(out)

    @staticmethod
    def _extract_json_object(output: str) -> dict[str, Any] | None:
        """Return the first valid JSON object embedded in model output."""
        text = output.strip()
        decoder = json.JSONDecoder()

        if text.startswith("```"):
            lines = text.splitlines()
            for start, line in enumerate(lines):
                if line.strip().startswith("```"):
                    for end in range(len(lines) - 1, start, -1):
                        if lines[end].strip() == "```":
                            fenced = "\n".join(lines[start + 1 : end]).strip()
                            parsed = PromptEval._extract_json_object(fenced)
                            if parsed is not None:
                                return parsed
                            try:
                                literal = ast.literal_eval(fenced)
                            except (SyntaxError, ValueError):
                                literal = None
                            if isinstance(literal, dict):
                                return literal
                            break

        for match in re.finditer(r"\{", text):
            try:
                parsed, _idx = decoder.raw_decode(text[match.start() :])
            except json.JSONDecodeError:
                # Try repairing raw control characters inside string literals.
                repaired = PromptEval._escape_raw_controls_in_strings(
                    text[match.start() :]
                )
                try:
                    parsed, _idx = decoder.raw_decode(repaired)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    pass
                try:
                    literal = ast.literal_eval(text[match.start() :])
                except (SyntaxError, ValueError):
                    continue
                if isinstance(literal, dict):
                    return literal
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    @staticmethod
    def _check_shape(output: str, spec: dict[str, Any]) -> tuple[bool, str]:
        """Validate model output against an expected_shape spec."""
        kind = spec.get("type", "")

        if kind == "json":
            try:
                parsed = json.loads(output)
            except json.JSONDecodeError:
                parsed = PromptEval._extract_json_object(output)
                if parsed is None:
                    return False, "output is not valid JSON"
            if not isinstance(parsed, dict):
                return False, "JSON output must be an object"

            schema = spec.get("schema")
            if schema:
                try:
                    Draft202012Validator.check_schema(schema)
                    Draft202012Validator(schema).validate(parsed)
                except jsonschema_exceptions.SchemaError as exc:
                    return False, f"invalid JSON schema: {exc.message}"
                except jsonschema_exceptions.ValidationError as exc:
                    return False, f"JSON schema validation failed: {exc.message}"
            return True, ""

        if kind == "regex":
            pattern = spec.get("pattern", "")
            if re.search(pattern, output, re.IGNORECASE | re.DOTALL):
                return True, ""
            return False, f"output did not match pattern: {pattern!r}"

        if kind == "callable":
            fn = spec.get("fn")
            if fn is None:
                return True, "(no callable provided)"
            try:
                result = fn(output)
                if result is True or result is None:
                    return True, ""
                return False, str(result)
            except Exception as exc:
                return False, f"shape-check raised: {exc}"

        return True, f"(unknown shape type '{kind}'; skipped)"
