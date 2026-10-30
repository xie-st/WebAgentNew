"""Model-catalog payload for the OpenAI-compatible API.

Assembles the read-only payload served at ``GET /v1/models/catalog``: every
known provider model with pricing, capabilities and provenance, plus the models
present in the local cache. Every name is re-exported from
``effgen.api.openai_compat``; import from there.
"""
from __future__ import annotations

from typing import Any

_LOCAL_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".gguf", ".onnx")


def _local_cached_models() -> list[dict[str, Any]]:
    """List models present in the local HuggingFace cache (best-effort).

    Each entry marks whether the download is ``complete`` (has real weight
    files) so an incomplete snapshot isn't offered as ready. Returns an empty
    list when the cache can't be scanned (e.g. ``huggingface_hub`` absent).
    """
    out: list[dict[str, Any]] = []
    try:
        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir()
        for repo in sorted(info.repos, key=lambda r: r.repo_id):
            if repo.repo_type != "model":
                continue
            has_weights = any(
                f.file_name.endswith(_LOCAL_WEIGHT_SUFFIXES)
                and not f.file_name.endswith(".index.json")
                for rev in repo.revisions
                for f in rev.files
            )
            out.append(
                {
                    "id": repo.repo_id,
                    "provider": "local",
                    "engine": "transformers",
                    "size_gb": round(repo.size_on_disk / (1024**3), 2),
                    "complete": bool(has_weights),
                    "local": True,
                    "is_priced": False,
                }
            )
    except Exception:  # noqa: BLE001 - cache scan is best-effort
        pass
    return out


def build_model_catalog(provider: str | None = None) -> dict[str, Any]:
    """Assemble the catalog payload served at ``GET /v1/models/catalog``.

    Reads the in-package model catalog (the same source the ``effgen models``
    CLI uses) so a picker sees real ids, pricing, capabilities and provenance
    instead of the drop-in aliases ``GET /v1/models`` returns. Never raises: on
    any read failure it returns whatever parsed, with empty lists otherwise.
    """
    data: list[dict[str, Any]] = []
    providers_meta: list[dict[str, Any]] = []
    try:
        from effgen.models import _catalog

        names = (
            [provider]
            if provider and provider in _catalog.known_providers()
            else _catalog.known_providers()
        )
        for prov in names:
            try:
                records = _catalog.list_models(prov)
            except Exception:  # noqa: BLE001 - skip a provider whose catalog won't load
                continue
            meta = {}
            try:
                meta = _catalog.snapshot_meta(prov)
            except Exception:  # noqa: BLE001
                meta = {}
            providers_meta.append(
                {
                    "provider": prov,
                    "count": len(records),
                    "verified_on": meta.get("verified_on"),
                    "default_model": _catalog.default_model(prov),
                }
            )
            for rec in records:
                data.append(
                    {
                        "id": rec.id,
                        "provider": rec.provider,
                        "family": rec.family,
                        "context_window": rec.context_window,
                        "max_output": rec.max_output,
                        "price_in_per_1m": rec.price_in_per_1m,
                        "price_out_per_1m": rec.price_out_per_1m,
                        "supports_tools": rec.supports_tools,
                        "supports_vision": rec.supports_vision,
                        "free_tier": rec.free_tier,
                        "deprecated": rec.deprecated,
                        "is_priced": rec.is_priced,
                        "price_source": rec.price_source,
                        "verified_on": rec.verified_on or meta.get("verified_on"),
                        "local": False,
                    }
                )
    except Exception:  # noqa: BLE001 - degrade to whatever parsed
        pass

    local = _local_cached_models()
    return {
        "object": "list",
        "providers": providers_meta,
        "data": data,
        "local": local,
        "counts": {"catalog": len(data), "local": len(local)},
    }
