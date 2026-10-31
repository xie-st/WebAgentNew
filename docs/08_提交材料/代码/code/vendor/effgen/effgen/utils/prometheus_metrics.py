"""
Prometheus-compatible metrics collection for effGen.

.. note::
   This is the low-level Prometheus backend. The canonical metrics facade is
   ``effgen.observability.metrics`` (``record_model_call`` / ``record_tokens`` /
   ``export_metrics``), which wraps these counters so one scrape endpoint covers
   everything. Prefer the observability facade in new code.

Provides counters, histograms, and gauges for agent monitoring.
Uses a simple in-memory implementation that can export to Prometheus
text format. No external dependencies required.

Metrics:
    Histograms: response_latency, token_usage, tool_execution_time
    Counters: total_requests, tool_calls, errors, fallbacks, circuit_breaker_trips
    Gauges: active_agents, gpu_memory_used
    Labels: agent_name, model_name, tool_name

Usage:
    from effgen.utils.prometheus_metrics import metrics
    metrics.total_requests.inc(labels={"agent_name": "math_agent"})
    metrics.response_latency.observe(0.5, labels={"agent_name": "math_agent"})
    print(metrics.export())
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class Counter:
    """A monotonically increasing counter."""
    name: str
    help: str
    _value: float = 0.0
    _labels: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def inc(self, amount: float = 1.0, labels: dict[str, str] | None = None) -> None:
        """Add *amount* to the counter (to the labeled series when *labels* given)."""
        with self._lock:
            if labels:
                key = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
                self._labels[key] += amount
            else:
                self._value += amount

    def get(self, labels: dict[str, str] | None = None) -> float:
        """Return the current value (of the labeled series when *labels* given)."""
        if labels:
            key = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
            return self._labels.get(key, 0.0)
        return self._value

    def reset(self) -> None:
        """Zero the counter and drop every labeled series."""
        with self._lock:
            self._value = 0.0
            self._labels.clear()

    def export(self) -> str:
        """Return Prometheus text-format lines for this counter."""
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        if self._value > 0:
            lines.append(f"{self.name} {self._value}")
        for key, val in self._labels.items():
            lines.append(f"{self.name}{{{key}}} {val}")
        return "\n".join(lines)


@dataclass
class Histogram:
    """A histogram for measuring distributions."""
    name: str
    help: str
    _sum: float = 0.0
    _count: int = 0
    _values: list[float] = field(default_factory=list)
    _labels_sum: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    _labels_count: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _labels_values: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def observe(self, value: float, labels: dict[str, str] | None = None) -> None:
        """Record one observation (in the labeled series when *labels* given)."""
        with self._lock:
            if labels:
                key = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
                self._labels_sum[key] += value
                self._labels_count[key] += 1
                self._labels_values[key].append(value)
            else:
                self._sum += value
                self._count += 1
                self._values.append(value)

    def get_avg(self, labels: dict[str, str] | None = None) -> float:
        """Return the mean of observed values (0.0 with no observations)."""
        if labels:
            key = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
            count = self._labels_count.get(key, 0)
            return self._labels_sum.get(key, 0) / count if count else 0.0
        return self._sum / self._count if self._count else 0.0

    def get_percentile(
        self, p: float, labels: dict[str, str] | None = None
    ) -> float:
        """Get the p-th percentile (0-100) of observed values."""
        if labels:
            key = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
            values = sorted(self._labels_values.get(key, []))
        else:
            values = sorted(self._values)
        if not values:
            return 0.0
        idx = int(len(values) * p / 100.0)
        idx = min(idx, len(values) - 1)
        return values[idx]

    def reset(self) -> None:
        """Drop every observation, plain and labeled."""
        with self._lock:
            self._sum = 0.0
            self._count = 0
            self._values.clear()
            self._labels_sum.clear()
            self._labels_count.clear()
            self._labels_values.clear()

    def export(self) -> str:
        """Return Prometheus text-format lines for this histogram."""
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        if self._count > 0:
            lines.append(f"{self.name}_sum {self._sum}")
            lines.append(f"{self.name}_count {self._count}")
        for key in self._labels_sum:
            lines.append(f"{self.name}_sum{{{key}}} {self._labels_sum[key]}")
            lines.append(f"{self.name}_count{{{key}}} {self._labels_count[key]}")
        return "\n".join(lines)


@dataclass
class Gauge:
    """A gauge that can go up and down."""
    name: str
    help: str
    _value: float = 0.0
    _labels: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def set(self, value: float, labels: dict[str, str] | None = None) -> None:
        """Set the gauge (the labeled series when *labels* given) to *value*."""
        with self._lock:
            if labels:
                key = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
                self._labels[key] = value
            else:
                self._value = value

    def inc(self, amount: float = 1.0, labels: dict[str, str] | None = None) -> None:
        """Add *amount* to the gauge (to the labeled series when *labels* given)."""
        with self._lock:
            if labels:
                key = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
                self._labels[key] = self._labels.get(key, 0.0) + amount
            else:
                self._value += amount

    def dec(self, amount: float = 1.0, labels: dict[str, str] | None = None) -> None:
        """Subtract *amount* from the gauge (from the labeled series when given)."""
        with self._lock:
            if labels:
                key = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
                self._labels[key] = self._labels.get(key, 0.0) - amount
            else:
                self._value -= amount

    def get(self, labels: dict[str, str] | None = None) -> float:
        """Return the current value (of the labeled series when *labels* given)."""
        if labels:
            key = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
            return self._labels.get(key, 0.0)
        return self._value

    def reset(self) -> None:
        """Zero the gauge and drop every labeled series."""
        with self._lock:
            self._value = 0.0
            self._labels.clear()

    def export(self) -> str:
        """Return Prometheus text-format lines for this gauge."""
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} gauge"]
        if self._value != 0.0:
            lines.append(f"{self.name} {self._value}")
        for key, val in self._labels.items():
            lines.append(f"{self.name}{{{key}}} {val}")
        return "\n".join(lines)


class EffGenMetrics:
    """
    Collection of all effGen metrics.

    Metrics follow the naming convention: effgen_<subsystem>_<metric>_<unit>
    Labels: agent_name, model_name, tool_name
    """

    def __init__(self) -> None:
        # --- Counters ---
        self.total_requests = Counter(
            "effgen_requests_total",
            "Total number of agent run requests",
        )
        self.tool_calls = Counter(
            "effgen_tool_calls_total",
            "Total number of tool calls",
        )
        self.tool_failures = Counter(
            "effgen_tool_failures_total",
            "Total number of failed tool calls",
        )
        self.errors = Counter(
            "effgen_errors_total",
            "Total number of errors",
        )
        self.fallbacks = Counter(
            "effgen_fallbacks_total",
            "Total number of fallback invocations",
        )
        self.circuit_breaker_trips = Counter(
            "effgen_circuit_breaker_trips_total",
            "Total number of circuit breaker trips",
        )
        self.tokens_used = Counter(
            "effgen_tokens_used_total",
            "Total tokens consumed",
        )

        # --- Histograms ---
        self.response_latency = Histogram(
            "effgen_response_latency_seconds",
            "Agent response latency in seconds",
        )
        self.token_usage = Histogram(
            "effgen_token_usage",
            "Token usage per request",
        )
        self.tool_execution_time = Histogram(
            "effgen_tool_execution_seconds",
            "Tool execution duration in seconds",
        )

        # Backwards-compat aliases
        self.agent_runs = self.total_requests
        self.agent_duration = self.response_latency
        self.tool_duration = self.tool_execution_time

        # --- Gauges ---
        self.active_agents = Gauge(
            "effgen_active_agents",
            "Number of currently active agents",
        )
        self.gpu_memory_used = Gauge(
            "effgen_gpu_memory_used_bytes",
            "GPU memory used in bytes",
        )

    def export(self) -> str:
        """Export all metrics in Prometheus text format."""
        sections = [
            self.total_requests.export(),
            self.tool_calls.export(),
            self.tool_failures.export(),
            self.errors.export(),
            self.fallbacks.export(),
            self.circuit_breaker_trips.export(),
            self.tokens_used.export(),
            self.response_latency.export(),
            self.token_usage.export(),
            self.tool_execution_time.export(),
            self.active_agents.export(),
            self.gpu_memory_used.export(),
        ]
        return "\n\n".join(sections) + "\n"

    def reset(self) -> None:
        """Reset all metrics to zero."""
        self.total_requests.reset()
        self.tool_calls.reset()
        self.tool_failures.reset()
        self.errors.reset()
        self.fallbacks.reset()
        self.circuit_breaker_trips.reset()
        self.tokens_used.reset()
        self.response_latency.reset()
        self.token_usage.reset()
        self.tool_execution_time.reset()
        self.active_agents.reset()
        self.gpu_memory_used.reset()


# Global metrics instance
metrics = EffGenMetrics()
