"""Dashboard data assembly for the effGen API server.

Builds the JSON payloads the dashboard SPA polls: ``data.json`` (metrics
summary, SLO burn rates, per-model breakdown, recent runs and spans) and
``history.json`` (stored runs and saved sessions). Metric samples come from
``prometheus_client``'s registry and effGen's native exporter; histogram
quantiles are interpolated the way Prometheus' ``histogram_quantile`` is.
"""
from __future__ import annotations

import logging
import math
import re
from typing import Any

logger = logging.getLogger(__name__)


def _build_dashboard_data() -> dict[str, Any]:
    """Assemble the JSON payload served at /dashboard/data.json."""
    import time

    raw_metrics, samples = _collect_dashboard_metrics()

    # --- Derive summary metrics from raw ---
    model_call_count = _sum_samples(samples, "effgen_model_call_latency_seconds_count")
    model_call_errors = _sum_samples(
        samples,
        "effgen_model_call_latency_seconds_count",
        lambda labels: labels.get("outcome") not in (None, "", "ok"),
    )
    legacy_requests = _sum_samples(samples, "effgen_requests_total")
    legacy_errors = _sum_samples(samples, "effgen_errors_total")

    total_requests = model_call_count or legacy_requests
    total_errors = model_call_errors or legacy_errors

    latency_sum = _sum_samples(samples, "effgen_model_call_latency_seconds_sum")
    latency_count = model_call_count
    if not latency_count:
        latency_sum = _sum_samples(samples, "effgen_response_latency_seconds_sum")
        latency_count = _sum_samples(samples, "effgen_response_latency_seconds_count")
    avg_latency_s = (latency_sum / latency_count) if latency_count else None

    # Token count
    model_tokens = _sum_samples(samples, "effgen_tokens_total")
    legacy_tokens = _sum_samples(samples, "effgen_tokens_used_total")
    total_tokens = model_tokens or legacy_tokens

    # --- Recent agent runs (from in-memory ring buffer if available) ---
    recent_runs = _get_recent_runs()

    # Cost is the sum of the real per-run ``cost_usd`` recorded for each run;
    # runs on unpriced models (or failed before pricing) contribute nothing and
    # are counted separately so the figure is never inflated by a flat estimate.
    priced_cost = 0.0
    priced_runs = 0
    unpriced_runs = 0
    for run in recent_runs:
        cost = run.get("cost_usd")
        if isinstance(cost, int | float):
            priced_cost += float(cost)
            priced_runs += 1
        else:
            unpriced_runs += 1
    session_cost_usd = round(priced_cost, 6) if priced_runs else None

    # --- Latency percentiles from the model-call histogram buckets ---
    percentiles = _latency_percentiles(samples)

    # --- SLO burn rates (burn = current / target; p99 burn uses the true p99) ---
    LATENCY_THRESHOLD = 2.0  # seconds — p99 target
    ERROR_RATE_TARGET = 0.01  # 1% errors allowed
    error_rate = (total_errors / total_requests) if total_requests > 0 else 0.0
    availability = 1.0 - error_rate
    p99_latency_s = percentiles.get("p99")

    slo: dict[str, float] = {
        # Burn rate now derives from the true p99 (falls back to the mean only
        # when no histogram buckets have been recorded yet).
        "p99_latency_burn": (
            (p99_latency_s / LATENCY_THRESHOLD)
            if p99_latency_s is not None
            else (avg_latency_s / LATENCY_THRESHOLD) if avg_latency_s else 0.0
        ),
        "error_rate_burn": error_rate / ERROR_RATE_TARGET if ERROR_RATE_TARGET > 0 else 0.0,
        "availability": availability,
        "latency_threshold_s": LATENCY_THRESHOLD,
        "p50_latency_s": percentiles.get("p50"),
        "p95_latency_s": percentiles.get("p95"),
        "p99_latency_s": p99_latency_s,
    }

    # --- HTTP responses by status code (from the request counter) ---
    by_status, http_client_errors, http_server_errors = _http_status_breakdown(samples)
    by_status_detail = _http_status_detail(samples)
    by_route = _http_route_breakdown(samples)

    # --- Per-model / per-provider breakdown ---
    by_model, unattributed_cost_usd = _model_breakdown(samples, recent_runs)

    # --- Recent spans ---
    recent_spans = _get_recent_spans()

    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "version": _effgen_version(),
        "metrics": {
            "total_requests": int(total_requests),
            "total_errors": int(total_errors),
            "avg_latency_s": round(avg_latency_s, 4) if avg_latency_s is not None else None,
            "total_tokens": int(total_tokens),
            # Real session cost (sum of per-run ``cost_usd``); ``None`` when no
            # run had a priced model. ``daily_cost_usd`` mirrors it for consumers
            # of the older key name — both are the true summed cost, not an
            # estimate.
            "cost_usd": session_cost_usd,
            "daily_cost_usd": session_cost_usd,
            "priced_runs": priced_runs,
            "unpriced_runs": unpriced_runs,
            "http_client_errors": http_client_errors,
            "http_server_errors": http_server_errors,
        },
        # ``slo``/``recent_spans`` are the canonical keys consumed by the SPA;
        # ``slos``/``spans`` are documented aliases so external consumers can use
        # either spelling.
        "slo": slo,
        "slos": slo,
        "by_model": by_model,
        # Spend recorded for a run that could not be placed on a per-model row.
        # Reported apart rather than spread across rows or duplicated, so the
        # cost column always sums to money actually attributed.
        "unattributed_cost_usd": unattributed_cost_usd,
        "by_status": by_status,
        # ``by_status`` keeps its shape; these two carry the route and method the
        # counter already records, plus a per-route denominator.
        "by_status_detail": by_status_detail,
        "by_route": by_route,
        "recent_runs": recent_runs,
        "recent_spans": recent_spans[:20],
        "spans": recent_spans[:20],
        "prompt_templates": _get_prompt_templates(),
        "raw_metrics": dict(sorted(raw_metrics.items())),
    }


def _effgen_version() -> str:
    """Return the running effGen version string (best-effort)."""
    try:
        from effgen import __version__

        return str(__version__)
    except Exception:  # noqa: BLE001
        return ""


def _histogram_quantile(bounds: list[tuple[float, float]], quantile: float) -> float | None:
    """Estimate a quantile from cumulative histogram buckets.

    ``bounds`` is a list of ``(upper_bound, cumulative_count)`` pairs sorted by
    upper bound ascending (the final bound is ``+Inf``). Linear interpolation
    within the matching bucket mirrors Prometheus' ``histogram_quantile``.
    Returns ``None`` when there is no observed count.
    """
    if not bounds:
        return None
    total = bounds[-1][1]
    if total <= 0:
        return None
    rank = quantile * total
    prev_bound = 0.0
    prev_count = 0.0
    for upper, cum in bounds:
        if cum >= rank:
            if upper == math.inf:
                return prev_bound if prev_bound > 0 else None
            if cum == prev_count:
                return upper
            frac = (rank - prev_count) / (cum - prev_count)
            return prev_bound + frac * (upper - prev_bound)
        prev_bound, prev_count = upper, cum
    return prev_bound


def _bucket_bounds(
    samples: list[tuple[str, dict[str, str], float]],
    predicate: Any = None,
) -> list[tuple[float, float]]:
    """Aggregate ``model_call_latency`` histogram buckets into sorted bounds."""
    acc: dict[float, float] = {}
    for name, labels, value in samples:
        if name != "effgen_model_call_latency_seconds_bucket":
            continue
        if predicate is not None and not predicate(labels):
            continue
        le_raw = labels.get("le", "")
        try:
            le = math.inf if le_raw in ("+Inf", "Inf", "inf") else float(le_raw)
        except (TypeError, ValueError):
            continue
        acc[le] = acc.get(le, 0.0) + value
    return sorted(acc.items(), key=lambda kv: kv[0])


def _latency_percentiles(
    samples: list[tuple[str, dict[str, str], float]],
) -> dict[str, float | None]:
    """Compute p50/p95/p99 latency (seconds) across all model-call buckets."""
    bounds = _bucket_bounds(samples)
    out: dict[str, float | None] = {}
    for label, q in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99)):
        val = _histogram_quantile(bounds, q)
        out[label] = round(val, 4) if val is not None else None
    return out


def _http_status_breakdown(
    samples: list[tuple[str, dict[str, str], float]],
) -> tuple[dict[str, int], int, int]:
    """Return ({status: count}, 4xx total, 5xx total) from the request counter."""
    by_status: dict[str, int] = {}
    client_errors = 0
    server_errors = 0
    for name, labels, value in samples:
        if name != "effgen_http_requests_total":
            continue
        status = str(labels.get("status", "")).strip()
        if not status:
            continue
        by_status[status] = by_status.get(status, 0) + int(value)
        if status.startswith("4"):
            client_errors += int(value)
        elif status.startswith("5"):
            server_errors += int(value)
    return dict(sorted(by_status.items())), client_errors, server_errors


def _http_status_class(status: str) -> str:
    """Return the ``2xx``/``4xx``-style class of a status code."""
    return f"{status[0]}xx" if status[:1].isdigit() else "other"


def _http_status_detail(
    samples: list[tuple[str, dict[str, str], float]],
) -> list[dict[str, Any]]:
    """Return one entry per (status, route, method) the request counter recorded.

    ``by_status`` collapses every route behind a status code, so three 404s from
    three different causes read as one number. This keeps the route and method
    the counter already carries, so a bad model id on ``/v1/chat/completions``
    and a probe of an unknown path stay separate.
    """
    acc: dict[tuple[str, str, str], int] = {}
    for name, labels, value in samples:
        if name != "effgen_http_requests_total":
            continue
        status = str(labels.get("status", "")).strip()
        if not status:
            continue
        key = (status, str(labels.get("route", "other")), str(labels.get("method", "")))
        acc[key] = acc.get(key, 0) + int(value)
    return [
        {
            "status": status,
            "class": _http_status_class(status),
            "route": route,
            "method": method,
            "count": count,
        }
        for (status, route, method), count in sorted(acc.items())
    ]


def _http_route_breakdown(
    samples: list[tuple[str, dict[str, str], float]],
) -> list[dict[str, Any]]:
    """Aggregate requests, errors and error rate per route and method.

    The denominator is what a status count on its own cannot give: a route that
    served eight requests and failed four is a different situation from a route
    that failed four out of four hundred. Rows are ordered worst-first.
    """
    acc: dict[tuple[str, str], dict[str, Any]] = {}
    for name, labels, value in samples:
        if name != "effgen_http_requests_total":
            continue
        status = str(labels.get("status", "")).strip()
        if not status:
            continue
        key = (str(labels.get("route", "other")), str(labels.get("method", "")))
        row = acc.setdefault(
            key,
            {
                "route": key[0],
                "method": key[1],
                "requests": 0,
                "errors": 0,
                "by_status": {},
            },
        )
        count = int(value)
        row["requests"] += count
        row["by_status"][status] = row["by_status"].get(status, 0) + count
        if status[:1] in ("4", "5"):
            row["errors"] += count

    rows: list[dict[str, Any]] = []
    for row in acc.values():
        requests = row["requests"]
        rows.append(
            {
                **row,
                "by_status": dict(sorted(row["by_status"].items())),
                "error_rate": round(row["errors"] / requests, 4) if requests else 0.0,
            }
        )
    rows.sort(key=lambda r: (-r["error_rate"], -r["requests"], r["route"], r["method"]))
    return rows


def _run_provider(run: dict[str, Any]) -> str | None:
    """Return the provider a recorded run belongs to, or ``None`` if unstated."""
    provider = run.get("provider")
    if isinstance(provider, str) and provider.strip():
        return provider.strip()
    prefix, sep, rest = str(run.get("model", "")).partition(":")
    if sep and prefix and rest:
        return prefix
    return None


def _cost_by_row(
    recent_runs: list[dict[str, Any]],
    row_keys: set[tuple[str, str]],
) -> tuple[dict[tuple[str, str], float], float]:
    """Attribute each run's recorded spend to one ``(model, provider)`` row.

    A run is placed by the provider it recorded, then by the ``provider:``
    prefix on its model id, then — only when the model name identifies exactly
    one row — by that name alone. Spend that still cannot be placed is returned
    separately rather than spread across rows or counted on every row that
    shares the model name.
    """
    rows_for_model: dict[str, list[tuple[str, str]]] = {}
    for row_key in row_keys:
        rows_for_model.setdefault(row_key[0], []).append(row_key)

    cost: dict[tuple[str, str], float] = {}
    unattributed = 0.0
    for run in recent_runs:
        amount = run.get("cost_usd")
        if not isinstance(amount, int | float):
            continue
        amount = float(amount)
        model = str(run.get("model", "")).split(":")[-1]
        provider = _run_provider(run)
        candidates = rows_for_model.get(model, [])
        key: tuple[str, str] | None = None
        if provider is not None and (model, provider) in row_keys:
            key = (model, provider)
        elif len(candidates) == 1:
            key = candidates[0]
        if key is None:
            unattributed += amount
        else:
            cost[key] = cost.get(key, 0.0) + amount
    return cost, unattributed


def _model_breakdown(
    samples: list[tuple[str, dict[str, str], float]],
    recent_runs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    """Aggregate calls, outcomes, error rate, p95 latency, tokens and cost per model.

    Returns the rows and the spend that could not be attributed to any of them.
    Every figure is scoped to the row's ``(model, provider)`` pair, so one model
    name served by two providers reports each provider's own latency tail and
    each provider's own spend.
    """
    agg: dict[tuple[str, str], dict[str, Any]] = {}

    def _row(model: str, provider: str) -> dict[str, Any]:
        key = (model, provider)
        if key not in agg:
            agg[key] = {
                "model": model,
                "provider": provider,
                "calls": 0,
                "errors": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "outcomes": {},
            }
        return agg[key]

    for name, labels, value in samples:
        if name != "effgen_model_call_latency_seconds_count":
            continue
        model = labels.get("model", "unknown")
        provider = labels.get("provider", "")
        row = _row(model, provider)
        row["calls"] += int(value)
        outcome = labels.get("outcome") or ""
        if outcome:
            row["outcomes"][outcome] = row["outcomes"].get(outcome, 0) + int(value)
        if outcome not in ("", "ok"):
            row["errors"] += int(value)

    for name, labels, value in samples:
        if name != "effgen_tokens_total":
            continue
        model = labels.get("model", "unknown")
        provider = labels.get("provider", "")
        row = _row(model, provider)
        if labels.get("kind") == "input":
            row["input_tokens"] += int(value)
        elif labels.get("kind") == "output":
            row["output_tokens"] += int(value)

    cost_by_row, unattributed = _cost_by_row(recent_runs, set(agg))

    rows: list[dict[str, Any]] = []
    for (model, provider), row in agg.items():
        p95 = _histogram_quantile(
            _bucket_bounds(
                samples,
                lambda lb, _m=model, _p=provider: lb.get("model") == _m
                and lb.get("provider", "") == _p,
            ),
            0.95,
        )
        calls = row["calls"]
        cost = cost_by_row.get((model, provider))
        top_error = _top_error(row["outcomes"])
        rows.append(
            {
                **row,
                "outcomes": dict(sorted(row["outcomes"].items())),
                "top_error": top_error,
                "top_error_hint": _remediation_for(top_error),
                "error_rate": round(row["errors"] / calls, 4) if calls else 0.0,
                "p95_latency_s": round(p95, 4) if p95 is not None else None,
                "cost_usd": round(cost, 6) if cost is not None else None,
            }
        )
    rows.sort(key=lambda r: r["calls"], reverse=True)
    return rows, round(unattributed, 6)


def _top_error(outcomes: dict[str, int]) -> str | None:
    """Return the most frequent non-``ok`` outcome label, or ``None``."""
    failures = {label: count for label, count in outcomes.items() if label != "ok"}
    if not failures:
        return None
    return sorted(failures.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _remediation_for(outcome: str | None) -> str | None:
    """Return the remediation sentence effGen already prints for *outcome*."""
    if not outcome:
        return None
    try:
        from effgen.models.errors import REMEDIATION_BY_CATEGORY
    except Exception:  # noqa: BLE001 - the hint is optional context
        return None
    return REMEDIATION_BY_CATEGORY.get(outcome)


_PROM_LINE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
)


def _collect_dashboard_metrics() -> tuple[dict[str, float], list[tuple[str, dict[str, str], float]]]:
    """Collect prometheus_client and effGen-native metric samples."""
    raw_metrics: dict[str, float] = {}
    samples: list[tuple[str, dict[str, str], float]] = []

    try:
        from prometheus_client import REGISTRY

        for metric in REGISTRY.collect():
            for sample in metric.samples:
                labels = {str(k): str(v) for k, v in dict(sample.labels).items()}
                value = float(sample.value)
                samples.append((sample.name, labels, value))
                raw_metrics[_metric_key(sample.name, labels)] = value
    except Exception:  # noqa: BLE001 - best-effort metrics scrape; return what parsed
        pass

    try:
        from effgen.observability.metrics import export_metrics

        for name, labels, value in _parse_prometheus_text(export_metrics()):
            samples.append((name, labels, value))
            raw_metrics[_metric_key(name, labels)] = value
    except Exception:  # noqa: BLE001 - best-effort metrics scrape; return what parsed
        pass

    return raw_metrics, samples


def _parse_prometheus_text(text: str) -> list[tuple[str, dict[str, str], float]]:
    """Parse simple Prometheus text-format sample lines."""
    parsed: list[tuple[str, dict[str, str], float]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PROM_LINE_RE.match(line)
        if not match:
            continue
        parsed.append(
            (
                match.group("name"),
                _parse_prometheus_labels(match.group("labels") or ""),
                float(match.group("value")),
            )
        )
    return parsed


def _parse_prometheus_labels(label_text: str) -> dict[str, str]:
    """Parse a Prometheus label set emitted by effGen's metric exporter."""
    labels: dict[str, str] = {}
    if not label_text:
        return labels
    for part in label_text.split(","):
        key, sep, value = part.partition("=")
        if sep:
            labels[key.strip()] = value.strip().strip('"')
    return labels


def _metric_key(name: str, labels: dict[str, str]) -> str:
    if not labels:
        return name
    label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return f"{name}{{{label_str}}}"


def _sum_samples(
    samples: list[tuple[str, dict[str, str], float]],
    name: str,
    predicate: Any = None,
) -> float:
    total = 0.0
    for sample_name, labels, value in samples:
        if sample_name != name:
            continue
        if predicate is not None and not predicate(labels):
            continue
        total += value
    return total


def _get_prompt_templates() -> list[dict[str, str]]:
    """Expose prompt-library entries for lightweight editor integrations."""
    try:
        from effgen.prompts.library import registry

        templates: list[dict[str, str]] = []
        for prompt in registry.all():
            templates.append(
                {
                    "name": prompt.name,
                    "description": prompt.description,
                    "template": f'registry.get("{prompt.name}").render(...)',
                    "category": prompt.domain,
                }
            )
        return templates
    except Exception:  # noqa: BLE001
        return []


def _build_dashboard_history(
    *,
    limit: int = 50,
    status: str | None = None,
    search: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Assemble the payload served at /dashboard/history.json."""
    limit = max(1, min(int(limit or 50), 500))
    payload: dict[str, Any] = {"runs": [], "sessions": [], "run": None}
    try:
        from effgen.observability import run_log

        if status == "failed":
            status = "error"
        payload["runs"] = run_log.read_runs(limit=limit, status=status, search=search)
        payload["runs_dir"] = str(run_log.history_dir())
        payload["persisted"] = run_log.history_enabled()
        if run_id:
            payload["run"] = run_log.get_run(run_id)
    except Exception:  # noqa: BLE001 - an empty history is a valid view
        logger.debug("Dashboard history: run store unavailable", exc_info=True)
    try:
        from effgen.core.session import SessionManager

        manager = SessionManager()
        sessions, unreadable = manager.scan()
        payload["sessions"] = sessions[:limit]
        payload["unreadable_sessions"] = unreadable
        payload["sessions_dir"] = str(manager.sessions_dir)
    except Exception:  # noqa: BLE001
        logger.debug("Dashboard history: session store unavailable", exc_info=True)
    return payload


def _get_recent_runs() -> list[dict[str, Any]]:
    """Return up to 50 recent agent runs from the in-memory run log."""
    try:
        from effgen.observability.run_log import get_recent_runs as _runs

        return _runs(limit=50)
    except Exception:  # noqa: BLE001
        return []


def _get_recent_spans() -> list[dict[str, Any]]:
    """Return up to 100 recent trace spans from the in-memory span buffer."""
    try:
        from effgen.observability.tracing import get_recent_spans as _spans

        return _spans(limit=100)
    except Exception:  # noqa: BLE001
        return []
