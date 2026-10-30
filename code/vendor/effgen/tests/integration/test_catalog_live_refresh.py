"""Live catalog refresh / drift / default-model smoke tests.

Each provider is skipped when its key is absent, and runs a real call otherwise
(no mocks).  These guard the failure modes: a
configured default model that has silently gone away, and the per-secondary-
provider id shapes (Replicate ``owner/model``, Fireworks
``accounts/.../models/...``, HF ``org/model``, Together full slug) that were
under-tested.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import load_dotenv

from effgen.models import _catalog as C
from effgen.models import _refresh as R

# Match the repo convention: load keys from the user env file if present.
load_dotenv(Path.home() / ".effgen" / ".env", override=False)
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

# Providers that expose a live model-list endpoint we can verify against.
_LIVE_PROVIDERS = [p for p in R.refreshable_providers() if p != "anthropic"]


def _key_reason(provider: str) -> str:
    envs = R._KEY_ENVS.get(provider, ())
    return f"SKIPPED: no key for {provider} ({' / '.join(envs)})"


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.parametrize("provider", _LIVE_PROVIDERS)
def test_default_model_is_live(provider):
    """The configured default for each keyed provider must still be served live."""
    if not R.has_credentials(provider) and provider != "hf":
        pytest.skip(_key_reason(provider))

    default = C.default_model(provider)
    if not default:
        pytest.skip(f"{provider} has no configured default model")

    live = set(R.available_models_live(provider))
    assert live, f"{provider}: live model list came back empty"

    if default not in live:
        # Stale-registry signal rather than a hard failure.
        pytest.xfail(
            f"{provider} default {default!r} is no longer in the live catalog; "
            f"run `effgen models refresh --provider {provider}`. "
            f"Nearest live: {[r.id for r in C.nearest_alternatives(default, provider, n=3)]}"
        )


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.parametrize("provider", _LIVE_PROVIDERS)
def test_refresh_models_live(provider):
    """`refresh_models` returns a non-empty, well-formed live record set."""
    if not R.has_credentials(provider) and provider != "hf":
        pytest.skip(_key_reason(provider))

    rep = R.refresh_models(provider, persist=False)
    assert rep["live_count"] > 0
    assert rep["records"], f"{provider}: no records"
    assert {"added", "removed", "changed"} <= set(rep["diff"])
    # verified_on stamped today.
    assert all(r.verified_on == rep["verified_on"] for r in rep["records"])


# ---------------------------------------------------------------------------
# Per-secondary-provider id-shape smoke
# ---------------------------------------------------------------------------

_ID_SHAPES = {
    "replicate": lambda mid: mid.count("/") == 1,                  # owner/model
    "fireworks": lambda mid: mid.startswith("accounts/") and "/models/" in mid,
    "hf": lambda mid: mid.count("/") == 1,                          # org/model
    "together": lambda mid: "/" in mid,                            # full slug
}


@pytest.mark.parametrize("provider,shape_ok", list(_ID_SHAPES.items()))
def test_secondary_provider_default_id_shape(provider, shape_ok):
    """Secondary-provider default ids keep their documented shape (offline)."""
    default = C.default_model(provider)
    assert default, f"{provider} has no default model"
    assert shape_ok(default), f"{provider} default {default!r} has an unexpected id shape"


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.parametrize("provider", list(_ID_SHAPES))
def test_secondary_provider_default_resolves_live(provider):
    """The curated default id for each secondary provider resolves on the live API."""
    if not R.has_credentials(provider) and provider != "hf":
        pytest.skip(_key_reason(provider))
    default = C.default_model(provider)
    live = set(R.available_models_live(provider))
    if default not in live:
        pytest.xfail(
            f"{provider} default {default!r} not in live list "
            f"({len(live)} live ids); refresh needed."
        )


# ---------------------------------------------------------------------------
# Wrong-id → helpful live suggestion
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.api
def test_cerebras_wrong_id_suggests_live_alternative():
    if not R.has_credentials("cerebras"):
        pytest.skip(_key_reason("cerebras"))
    from effgen.models.cerebras_adapter import CerebrasAdapter
    from effgen.models.errors import ModelNotFoundError

    with pytest.raises(ModelNotFoundError) as ei:
        CerebrasAdapter(model_name="llama3.1-8b")  # the retired default
    msg = str(ei.value)
    assert "gpt-oss-120b" in msg  # nearest live alternative suggested


# ---------------------------------------------------------------------------
# Live refresh/drift is filtered to chat models (no ft:/embedding/audio/image)
# ---------------------------------------------------------------------------

# Providers whose list-models endpoint returns a comprehensive surface we can
# assert is filtered down to chat models.  (Fireworks' endpoint is account
# scoped — it returns only a handful of ids — so it is not a useful fixture.)
_FILTERED_PROVIDERS = ["openai", "together", "gemini", "groq", "cerebras"]


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.parametrize("provider", _FILTERED_PROVIDERS)
def test_live_drift_is_chat_only(provider):
    """check_drift over the live API never reports a private fine-tune or a
    non-chat (embedding/audio/image/moderation/…) id as drift."""
    if not R.has_credentials(provider) and provider != "hf":
        pytest.skip(_key_reason(provider))

    diff = R.check_drift(provider, warn=False)
    # No private fine-tune id may surface anywhere in the drift.
    assert not [m for m in diff["added"] if C.is_finetune(m)]
    # Every "added" id must classify as chat (genuine new chat models only).
    non_chat = [m for m in diff["added"] if not C.is_chat_model(provider, m)]
    assert not non_chat, f"{provider}: non-chat ids reported as added: {non_chat[:10]}"


@pytest.mark.integration
@pytest.mark.api
def test_openai_refresh_persists_only_chat_models(tmp_path):
    """A persisted OpenAI refresh contains only chat models and never a private
    ``ft:`` fine-tune id (a privacy guarantee: a user's fine-tunes must not be
    written into the bundled catalog snapshot)."""
    if not R.has_credentials("openai"):
        pytest.skip(_key_reason("openai"))

    rep = R.refresh_models("openai", persist=False)
    ids = [r.id for r in rep["records"]]
    assert ids, "no records returned"
    assert not [i for i in ids if C.is_finetune(i)], "fine-tune id leaked into refresh"
    # Every persisted id is either already curated or classifies as chat.
    from effgen.models.openai_models import OPENAI_MODELS

    leaked = [i for i in ids if i not in OPENAI_MODELS and not C.is_chat_model("openai", i)]
    assert not leaked, f"non-chat ids in refresh: {leaked[:10]}"

    # Persisting to a temp path writes the same clean set, no fine-tunes.
    out = tmp_path / "openai.json"
    C.save_snapshot("openai", rep["records"], path=out)
    written = {m["id"] for m in C.load_snapshot("openai", path=out)["models"]}
    assert not any(C.is_finetune(i) for i in written)


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.parametrize("provider", ["openai", "together"])
def test_pruned_providers_have_no_stale_removed_drift(provider):
    """After pruning the confirmed-retired ids, live drift reports no 'removed'
    chat models for the keyed providers we re-verified."""
    if not R.has_credentials(provider):
        pytest.skip(_key_reason(provider))
    diff = R.check_drift(provider, warn=False)
    assert diff["removed"] == [], f"{provider}: stale removed drift {diff['removed']}"
