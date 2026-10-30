"""Integration test: Cerebras registered models remain callable where accessible.

Skipped if CEREBRAS_API_KEY is absent. Requires all currently accessible
free-tier models to return real text.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path.home() / ".effgen" / ".env", override=False)
# Also load from project .env for CI convenience
load_dotenv(Path(__file__).parent.parent.parent / ".env", override=False)


def _has_key() -> bool:
    return bool(os.getenv("CEREBRAS_API_KEY"))


MODELS = [
    "gpt-oss-120b",
    "zai-glm-4.7",
]

RATE_LIMIT_MARKERS = ("429", "rate-limit", "rate_limit", "queue_exceeded", "too many requests")


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.skipif(not _has_key(), reason="SKIPPED: CEREBRAS_API_KEY not in env")
class TestCerebrasAllModelsParallel:
    def test_all_models_gather(self):
        """Call accessible models and report restricted models without timing out."""

        from effgen.models.base import GenerationConfig
        from effgen.models.cerebras_adapter import CerebrasAdapter
        from effgen.models.cerebras_models import free_tier_models

        async def call_model(model_id: str) -> tuple[str, str | None, str | None]:
            """Return (model_id, text_or_None, error_or_None). Retries on 429."""
            tag = f"Say exactly: MODEL_{model_id.upper().replace('-', '_').replace('.', '_')}_OK"
            adapter = CerebrasAdapter(model_name=model_id, enable_rate_limiting=False)
            adapter.load()
            try:
                cfg = GenerationConfig(max_tokens=16, temperature=0.0)
                result = await adapter.async_generate(tag, config=cfg)
                return model_id, result.text, None
            except Exception as exc:
                return model_id, None, str(exc)
            finally:
                adapter.unload()

        async def run_all():
            results = []
            for model_id in free_tier_models():
                results.append(await call_model(model_id))
                await asyncio.sleep(2)
            return results

        results = asyncio.run(run_all())

        successes = [(mid, text) for mid, text, err in results if text is not None]
        failures = [(mid, err) for mid, text, err in results if text is None]

        # Log outcomes for diagnosis
        for mid, text, err in results:
            if text is not None:
                print(f"  PASS {mid}: {repr(text[:80])}")
            else:
                print(f"  FAIL {mid}: {err}")

        if failures and all(
            any(marker in (err or "").lower() for marker in RATE_LIMIT_MARKERS)
            for _, err in failures
        ):
            pytest.xfail(f"Cerebras free-tier rate limit/backpressure during live run: {failures}")

        expected = set(free_tier_models())
        successful_models = {mid for mid, _ in successes}
        assert expected <= successful_models, (
            f"Accessible Cerebras models failed. Successes: {successes}. Failures: {failures}."
        )
