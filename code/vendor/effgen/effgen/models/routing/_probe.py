"""Warm-up probe: seed the LatencyTracker before first routing decision.

Fires 10-token probes to each capable candidate in parallel so the tracker
has real observations rather than seed values on the very first route call.

The result is cached per-process: once the probe has run, subsequent calls
return immediately.

Usage::

    from effgen.models.routing._probe import warm_up_providers
    from effgen.models.router import RoutingContext
    from effgen.models.capabilities import Capability

    warm_up_providers(
        context=RoutingContext(required_capabilities={Capability.chat}),
    )
"""

from __future__ import annotations

import concurrent.futures
import inspect
import logging
import os
import time
from threading import Lock

import effgen.models as _models  # noqa: F401  # ensure adapters self-register
from effgen.models.latency_tracker import LatencyTracker

logger = logging.getLogger(__name__)

_PROBE_PROMPT = "Hi"  # ~10 tokens — intentionally tiny to minimise cost
_PROBE_MAX_TOKENS = 10
_PROBE_TIMEOUT_S = 15.0  # per-probe wall-clock timeout

_probe_done: bool = False
_probe_lock: Lock = Lock()


def _probe_one(
    provider: str,
    model_id: str,
    tracker: LatencyTracker,
) -> None:
    """Fire a single probe call and record latency."""
    adapter = None
    try:
        from effgen.models.registry import ProviderRegistry
        rec = ProviderRegistry.get_provider_info(provider)
        env_keys = rec.get("env_keys", [])
        if env_keys and not any(os.environ.get(k) for k in env_keys):
            logger.debug("Probe skip %s/%s: no API key", provider, model_id)
            return

        adapter_cls = rec.get("adapter_cls") or rec.get("adapter_class")
        if adapter_cls is None:
            return

        try:
            # A probe measures latency, so it is one request and no more: a
            # provider that refuses must not turn a probe into a retry loop
            # that outlives the wall-clock budget below. The budget is asked
            # for at construction, because that is where each adapter applies
            # its own floor and where the ones that hand the budget to their
            # SDK client can still act on it.
            kwargs: dict[str, object] = {"model_name": model_id}
            params = inspect.signature(adapter_cls.__init__).parameters
            if "max_retries" in params:
                kwargs["max_retries"] = 0
            if "timeout" in params:
                kwargs["timeout"] = _PROBE_TIMEOUT_S
            adapter = adapter_cls(**kwargs)
            adapter.load()
            from effgen.models.base import GenerationConfig

            config = GenerationConfig(max_tokens=_PROBE_MAX_TOKENS, temperature=0.0)
            t0 = time.perf_counter()
            adapter.generate(_PROBE_PROMPT, config=config)
        except Exception as exc:
            logger.debug("Probe failed for %s/%s: %s", provider, model_id, exc)
            return
        finally:
            if adapter is not None:
                try:
                    adapter.unload()
                except Exception:
                    logger.debug("Probe unload failed for %s/%s", provider, model_id)

        if not tracker.has_data(provider, model_id):
            total_ms = (time.perf_counter() - t0) * 1000.0
            tracker.record(provider, model_id, total_ms)
        else:
            total_ms = tracker.p50(provider, model_id) or 0.0
        logger.info("Probe %s/%s: %.0f ms", provider, model_id, total_ms)

    except Exception as exc:
        logger.debug("Probe error %s/%s: %s", provider, model_id, exc)


def _select_probe_candidates(context_caps: set) -> list[tuple[str, str]]:
    """Return (provider, model_id) pairs eligible for probing."""
    from effgen.models.registry import ProviderRegistry

    candidates: list[tuple[str, str]] = []
    for provider in ProviderRegistry.list_providers():
        rec = ProviderRegistry.get_provider_info(provider)
        env_keys = rec.get("env_keys", [])
        if env_keys and not any(os.environ.get(k) for k in env_keys):
            continue

        provider_caps = ProviderRegistry.get_capabilities(provider)
        if context_caps - provider_caps:
            continue

        # Pick the smallest / fastest default model for this provider
        models = ProviderRegistry.list_models(provider)
        if not models:
            continue

        # Prefer explicitly free-tier models; fall back to first active model
        probe_model: str | None = None
        for m in models:
            if m.get("active") is False:
                continue
            if m.get("free_tier"):
                probe_model = m["model_id"]
                break
        if probe_model is None:
            for m in models:
                if m.get("active") is not False:
                    probe_model = m["model_id"]
                    break
        if probe_model:
            candidates.append((provider, probe_model))

    return candidates


def warm_up_providers(
    context_caps: set | None = None,
    tracker: LatencyTracker | None = None,
    force: bool = False,
    max_workers: int = 8,
) -> None:
    """Fire parallel 10-token probes to seed the latency tracker.

    Idempotent per process: the probe only runs once unless ``force=True``.

    Args:
        context_caps: Required capability set used to filter eligible
                      providers.  Defaults to empty (all providers eligible).
        tracker:      LatencyTracker to populate.  Defaults to global singleton.
        force:        Re-run the probe even if already done this session.
        max_workers:  Thread-pool size for parallel probes.
    """
    global _probe_done

    with _probe_lock:
        if _probe_done and not force:
            logger.debug("warm_up_providers: already done this session")
            return
        _probe_done = True

    if tracker is None:
        tracker = LatencyTracker.get()
    if context_caps is None:
        context_caps = set()

    candidates = _select_probe_candidates(context_caps)
    if not candidates:
        logger.debug("warm_up_providers: no eligible candidates")
        return

    logger.info(
        "warm_up_providers: probing %d providers in parallel: %s",
        len(candidates),
        [(p, m) for p, m in candidates],
    )

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    try:
        futures = {
            pool.submit(_probe_one, provider, model_id, tracker): (provider, model_id)
            for provider, model_id in candidates
        }
        done, pending = concurrent.futures.wait(futures, timeout=_PROBE_TIMEOUT_S)
        for future in done:
            provider, model_id = futures[future]
            try:
                future.result()
            except Exception as exc:
                logger.debug("warm_up_providers: probe %s/%s raised: %s", provider, model_id, exc)
        for future in pending:
            provider, model_id = futures[future]
            future.cancel()
            logger.debug(
                "warm_up_providers: probe %s/%s exceeded %.0fs timeout",
                provider,
                model_id,
                _PROBE_TIMEOUT_S,
            )
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    logger.info("warm_up_providers: complete")


def reset_probe_cache() -> None:
    """Reset the session cache so the next call will probe again (tests)."""
    global _probe_done
    with _probe_lock:
        _probe_done = False
