"""
Health monitoring and infrastructure checking for effGen.

Provides health checks for:
- Website availability (effgen.org, docs.effgen.org)
- DNS resolution
- SSL certificate expiry
- PyPI package availability
- Netlify deployment status

Usage:
    from effgen.utils.health import HealthChecker
    checker = HealthChecker()
    results = checker.check_all()
"""

from __future__ import annotations

import logging
import socket
import ssl
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from effgen import __version__
except ImportError:
    __version__ = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a single health check."""
    name: str
    passed: bool
    message: str
    response_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class HealthChecker:
    """
    Infrastructure health checker for effGen.

    Checks website availability, DNS, SSL, PyPI, and deployment status.
    """

    DEFAULT_URLS = [
        "https://effgen.org",
        "https://docs.effgen.org",
    ]
    PYPI_URL = "https://pypi.org/pypi/effgen/json"
    TIMEOUT = 10

    def __init__(self, urls: list[str] | None = None, timeout: int = 8) -> None:
        self.urls = urls or self.DEFAULT_URLS
        self.timeout = timeout

    def check_all(self) -> list[HealthCheckResult]:
        """Run all health checks concurrently and return results.

        The checks are independent network probes, so they run in parallel with
        a bounded wall-clock budget. Run sequentially they would sum every
        per-check timeout (several unreachable targets could take a minute);
        in parallel the command stays responsive — wall time is roughly a
        single check's timeout regardless of how many targets are down.
        """
        from concurrent.futures import ThreadPoolExecutor, wait

        # (name, thunk) pairs; output order matches this list.
        checks: list[tuple[str, Callable[[], HealthCheckResult]]] = [
            (f"HTTP {url}", (lambda u=url: self.check_website(u))) for url in self.urls
        ]
        checks += [
            ("DNS effgen.org", lambda: self.check_dns("effgen.org")),
            ("SSL effgen.org", lambda: self.check_ssl("effgen.org")),
            ("PyPI effgen", lambda: self.check_pypi()),
        ]

        # A little headroom over a single check's timeout so a slow-but-bounded
        # probe still reports, while a wedged one can't stall the whole command.
        budget = self.timeout + 5
        pool = ThreadPoolExecutor(max_workers=max(1, len(checks)))
        futures = [pool.submit(thunk) for _, thunk in checks]
        wait(futures, timeout=budget)

        results: list[HealthCheckResult] = []
        for (name, _), fut in zip(checks, futures):
            if fut.done():
                try:
                    results.append(fut.result())
                except Exception as exc:  # noqa: BLE001 - report, don't crash
                    results.append(HealthCheckResult(name=name, passed=False, message=str(exc)))
            else:
                fut.cancel()
                results.append(HealthCheckResult(
                    name=name, passed=False, message=f"Timed out after {budget}s",
                ))
        # Don't block the CLI waiting on any still-running probe; per-check
        # timeouts mean stragglers finish on their own shortly after.
        pool.shutdown(wait=False, cancel_futures=True)
        return results

    def check_website(self, url: str) -> HealthCheckResult:
        """Check if a website returns HTTP 200."""
        if not REQUESTS_AVAILABLE:
            return HealthCheckResult(
                name=f"HTTP {url}",
                passed=False,
                message="requests library not installed",
            )

        try:
            start = time.time()
            response = requests.get(url, timeout=self.timeout, allow_redirects=True)
            elapsed_ms = (time.time() - start) * 1000
            passed = response.status_code == 200
            return HealthCheckResult(
                name=f"HTTP {url}",
                passed=passed,
                message=f"HTTP {response.status_code} ({elapsed_ms:.0f}ms)",
                response_time_ms=elapsed_ms,
            )
        except requests.Timeout:
            return HealthCheckResult(
                name=f"HTTP {url}",
                passed=False,
                message=f"Timeout after {self.timeout}s",
            )
        except Exception as e:
            return HealthCheckResult(
                name=f"HTTP {url}",
                passed=False,
                message=str(e),
            )

    def check_dns(self, domain: str) -> HealthCheckResult:
        """Check DNS resolution for a domain."""
        try:
            start = time.time()
            ip = socket.gethostbyname(domain)
            elapsed_ms = (time.time() - start) * 1000
            return HealthCheckResult(
                name=f"DNS {domain}",
                passed=True,
                message=f"{domain} -> {ip} ({elapsed_ms:.0f}ms)",
                response_time_ms=elapsed_ms,
            )
        except socket.gaierror as e:
            return HealthCheckResult(
                name=f"DNS {domain}",
                passed=False,
                message=f"DNS resolution failed: {e}",
            )
        except Exception as e:
            return HealthCheckResult(
                name=f"DNS {domain}",
                passed=False,
                message=str(e),
            )

    def check_ssl(self, domain: str, warn_days: int = 14) -> HealthCheckResult:
        """Check SSL certificate validity and expiry."""
        try:
            ctx = ssl.create_default_context()
            with ctx.wrap_socket(socket.socket(), server_hostname=domain) as s:
                s.settimeout(self.timeout)
                s.connect((domain, 443))
                cert = s.getpeercert()

            expires_str = cert.get("notAfter", "")
            if not expires_str:
                return HealthCheckResult(
                    name=f"SSL {domain}",
                    passed=False,
                    message="Could not determine certificate expiry",
                )

            expires = datetime.strptime(expires_str, "%b %d %H:%M:%S %Y %Z")
            # strptime with %Z returns naive datetime; treat as UTC
            if expires.tzinfo is None:
                expires = expires.replace(tzinfo=UTC)
            days_left = (expires - datetime.now(UTC)).days
            passed = days_left > warn_days

            return HealthCheckResult(
                name=f"SSL {domain}",
                passed=passed,
                message=f"Valid until {expires.strftime('%Y-%m-%d')} ({days_left} days left)",
            )
        except Exception as e:
            return HealthCheckResult(
                name=f"SSL {domain}",
                passed=False,
                message=str(e),
            )

    def check_pypi(self, package: str = "effgen") -> HealthCheckResult:
        """Check if the package is available on PyPI."""
        if not REQUESTS_AVAILABLE:
            return HealthCheckResult(
                name=f"PyPI {package}",
                passed=False,
                message="requests library not installed",
            )

        try:
            start = time.time()
            url = f"https://pypi.org/pypi/{package}/json"
            response = requests.get(url, timeout=self.timeout)
            elapsed_ms = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                version = data.get("info", {}).get("version", "unknown")
                return HealthCheckResult(
                    name=f"PyPI {package}",
                    passed=True,
                    message=f"{package} {version} available ({elapsed_ms:.0f}ms)",
                    response_time_ms=elapsed_ms,
                )
            else:
                return HealthCheckResult(
                    name=f"PyPI {package}",
                    passed=False,
                    message=f"HTTP {response.status_code}",
                    response_time_ms=elapsed_ms,
                )
        except Exception as e:
            return HealthCheckResult(
                name=f"PyPI {package}",
                passed=False,
                message=str(e),
            )

    def print_results(self, results: list[HealthCheckResult] | None = None) -> bool:
        """
        Print health check results in a formatted way.

        Returns True if all checks passed.
        """
        if results is None:
            results = self.check_all()

        all_passed = True
        for r in results:
            icon = "\u2705" if r.passed else "\u274c"
            print(f"{icon} {r.name} \u2014 {r.message}")
            if not r.passed:
                all_passed = False

        return all_passed
