"""
effGen Observability -- structured logging, secret redaction, metrics, SLOs, and tracing.

Quick start
-----------
    from effgen.observability import get_logger, configure_logging

    # Configure once at application startup (call before first log line):
    configure_logging(level="INFO")

    # Then in any module:
    log = get_logger(__name__)
    log.event("model.call.started", model="gemini-3", cached_tokens=128)
    log.event("model.call.done",    model="gemini-3", latency_ms=340)
    log.info("something happened",  extra_key="value")

    # Metrics (Prometheus histograms + counters):
    from effgen.observability import record_model_call, record_tokens, export_metrics
    record_model_call(provider="cerebras", model="llama3.1-8b",
                      outcome="ok", latency=0.42)
    record_tokens(provider="cerebras", model="llama3.1-8b",
                  input_tokens=120, output_tokens=50)
    print(export_metrics())  # Prometheus text format

    # SLO tracking:
    from effgen.observability import SLO, get_slo_tracker
    tracker = get_slo_tracker()
    tracker.register(SLO("model_call_success", 99.0, 3600))
    tracker.record("model_call_success", ok=True)
    print(tracker.burn_rate("model_call_success"))

    # Tracing (OTel samplers + spans):
    from effgen.observability import (
        setup_tracing, ParentBasedSampler, TraceIdRatioSampler,
        start_agent_run, start_model_call, start_tool_call,
    )
    setup_tracing(sampler=ParentBasedSampler(TraceIdRatioSampler(0.1)))
    with start_agent_run(preset="my_agent", task="hello") as span:
        with start_model_call(provider="cerebras", model="llama3.1-8b") as mspan:
            ...

Sub-modules
-----------
- effgen.observability.logs    -- StructuredFormatter and EffGenLogger
- effgen.observability.redact  -- Redactor and get_redactor
- effgen.observability.metrics -- Prometheus histograms + counters
- effgen.observability.slo     -- SLO and SLOTracker
- effgen.observability.tracing -- OTel samplers + span context managers
- effgen.observability.spans   -- Span attribute constants
"""

from .alerting import (
    Alert,
    AlertSeverity,
    AlertWebhook,
    check_slo_and_alert,
    validate_alert_rules_yaml,
)
from .logs import (
    EffGenLogger,
    StructuredFormatter,
    clear_run_context,
    configure_logging,
    get_effgen_logger,
    set_run_context,
)
from .metrics import (
    LabeledCounter,
    LabeledHistogram,
    agent_iteration_latency,
    export_metrics,
    http_requests_total,
    model_call_latency,
    record_agent_iteration,
    record_http_request,
    record_model_call,
    record_tokens,
    record_tool_call,
    tokens_total,
    tool_call_latency,
)
from .metrics import (
    reset_all as reset_metrics,
)
from .redact import Redactor, get_redactor
from .slo import SLO, SLOTracker
from .slo import get_tracker as get_slo_tracker
from .spans import (
    AgentAttrs,
    ModelAttrs,
    RetryAttrs,
    RouterAttrs,
    SpanName,
    ToolAttrs,
)
from .tracing import (
    AlwaysOffSampler,
    AlwaysOnSampler,
    ParentBasedSampler,
    RateLimitedSampler,
    TraceIdRatioSampler,
    get_tracer,
    record_retry_attempt,
    reset_tracing,
    set_span_attribute,
    set_span_error,
    setup_tracing,
    shutdown_tracing,
    start_agent_iteration,
    start_agent_run,
    start_model_call,
    start_router_decision,
    start_tool_call,
)


def get_logger(name: str) -> "EffGenLogger":
    """
    Get a structured EffGenLogger for *name*.

    This is the canonical entry point for all effGen components.

    Args:
        name: Logger name -- pass ``__name__`` from the calling module.

    Returns:
        EffGenLogger instance (cached).

    Example::

        from effgen.observability import get_logger
        log = get_logger(__name__)
        log.event("model.call.started", model="gemini-3", cached_tokens=128)
    """
    return get_effgen_logger(name)


__all__ = [
    # Alerting
    "Alert",
    "AlertSeverity",
    "AlertWebhook",
    "check_slo_and_alert",
    "validate_alert_rules_yaml",
    # Primary entry point
    "get_logger",
    # Configuration
    "configure_logging",
    # Core logging classes
    "EffGenLogger",
    "StructuredFormatter",
    "Redactor",
    # Factories
    "get_redactor",
    "get_effgen_logger",
    # Run context helpers (used by agent loop)
    "set_run_context",
    "clear_run_context",
    # Metrics
    "model_call_latency",
    "tool_call_latency",
    "agent_iteration_latency",
    "tokens_total",
    "http_requests_total",
    "record_model_call",
    "record_tool_call",
    "record_agent_iteration",
    "record_tokens",
    "record_http_request",
    "export_metrics",
    "reset_metrics",
    "LabeledHistogram",
    "LabeledCounter",
    # SLO
    "SLO",
    "SLOTracker",
    "get_slo_tracker",
    # Tracing
    "setup_tracing",
    "shutdown_tracing",
    "reset_tracing",
    "get_tracer",
    "AlwaysOnSampler",
    "AlwaysOffSampler",
    "TraceIdRatioSampler",
    "RateLimitedSampler",
    "ParentBasedSampler",
    "start_agent_run",
    "start_agent_iteration",
    "start_model_call",
    "start_tool_call",
    "start_router_decision",
    "record_retry_attempt",
    "set_span_error",
    "set_span_attribute",
    # Span attribute constants
    "SpanName",
    "AgentAttrs",
    "ModelAttrs",
    "ToolAttrs",
    "RouterAttrs",
    "RetryAttrs",
]
