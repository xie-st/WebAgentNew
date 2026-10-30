"""Unit tests for effgen.utils.health.HealthChecker.

Avoids real network calls by mocking `requests` and `socket` primitives.
"""
from __future__ import annotations

import socket
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

from effgen.utils.health import HealthChecker, HealthCheckResult


class TestHealthCheckResult:
    def test_default_timestamp_is_iso(self):
        r = HealthCheckResult(name="x", passed=True, message="ok")
        # Round-trip parse
        parsed = datetime.fromisoformat(r.timestamp)
        assert parsed.tzinfo is not None


class TestCheckWebsite:
    def test_returns_passed_on_200(self):
        checker = HealthChecker(urls=["https://example.test"], timeout=2)
        fake_resp = MagicMock(status_code=200)
        with patch("effgen.utils.health.requests") as mock_requests:
            mock_requests.get.return_value = fake_resp
            mock_requests.Timeout = Exception
            r = checker.check_website("https://example.test")
        assert r.passed is True
        assert "200" in r.message

    def test_returns_failed_on_500(self):
        checker = HealthChecker(timeout=2)
        fake_resp = MagicMock(status_code=500)
        with patch("effgen.utils.health.requests") as mock_requests:
            mock_requests.get.return_value = fake_resp
            mock_requests.Timeout = Exception
            r = checker.check_website("https://example.test")
        assert r.passed is False

    def test_handles_timeout(self):
        checker = HealthChecker(timeout=2)

        class FakeTimeout(Exception):
            pass

        with patch("effgen.utils.health.requests") as mock_requests:
            mock_requests.Timeout = FakeTimeout
            mock_requests.get.side_effect = FakeTimeout()
            r = checker.check_website("https://example.test")
        assert r.passed is False
        assert "Timeout" in r.message

    def test_handles_unexpected_exception(self):
        checker = HealthChecker(timeout=2)
        with patch("effgen.utils.health.requests") as mock_requests:
            mock_requests.Timeout = TimeoutError
            mock_requests.get.side_effect = ValueError("boom")
            r = checker.check_website("https://example.test")
        assert r.passed is False
        assert "boom" in r.message

    def test_passes_when_requests_unavailable(self):
        checker = HealthChecker()
        with patch("effgen.utils.health.REQUESTS_AVAILABLE", False):
            r = checker.check_website("https://example.test")
        assert r.passed is False
        assert "requests" in r.message


class TestCheckDNS:
    def test_returns_passed_on_resolve(self):
        checker = HealthChecker()
        with patch("effgen.utils.health.socket.gethostbyname", return_value="93.184.216.34"):
            r = checker.check_dns("example.test")
        assert r.passed is True
        assert "93.184.216.34" in r.message

    def test_returns_failed_on_resolution_failure(self):
        checker = HealthChecker()
        with patch(
            "effgen.utils.health.socket.gethostbyname",
            side_effect=socket.gaierror("nx domain"),
        ):
            r = checker.check_dns("not-a-real.test")
        assert r.passed is False
        assert "nx domain" in r.message

    def test_handles_unexpected_exception(self):
        checker = HealthChecker()
        with patch(
            "effgen.utils.health.socket.gethostbyname",
            side_effect=RuntimeError("weird"),
        ):
            r = checker.check_dns("x")
        assert r.passed is False
        assert "weird" in r.message


class TestCheckSSL:
    def _fake_cert(self, days_from_now: int) -> dict:
        expiry = datetime.now(UTC) + timedelta(days=days_from_now)
        return {"notAfter": expiry.strftime("%b %d %H:%M:%S %Y GMT")}

    def test_passes_when_far_from_expiry(self):
        checker = HealthChecker()
        cert = self._fake_cert(60)
        fake_socket_ctx = MagicMock()
        fake_socket_ctx.__enter__.return_value.getpeercert.return_value = cert
        with patch("effgen.utils.health.ssl.create_default_context") as mock_ctx, \
             patch("effgen.utils.health.socket.socket"):
            mock_ctx.return_value.wrap_socket.return_value = fake_socket_ctx
            r = checker.check_ssl("example.test", warn_days=14)
        assert r.passed is True
        assert "days left" in r.message

    def test_fails_when_close_to_expiry(self):
        checker = HealthChecker()
        cert = self._fake_cert(3)
        fake_socket_ctx = MagicMock()
        fake_socket_ctx.__enter__.return_value.getpeercert.return_value = cert
        with patch("effgen.utils.health.ssl.create_default_context") as mock_ctx, \
             patch("effgen.utils.health.socket.socket"):
            mock_ctx.return_value.wrap_socket.return_value = fake_socket_ctx
            r = checker.check_ssl("example.test", warn_days=14)
        assert r.passed is False

    def test_fails_on_missing_notafter(self):
        checker = HealthChecker()
        fake_socket_ctx = MagicMock()
        fake_socket_ctx.__enter__.return_value.getpeercert.return_value = {}
        with patch("effgen.utils.health.ssl.create_default_context") as mock_ctx, \
             patch("effgen.utils.health.socket.socket"):
            mock_ctx.return_value.wrap_socket.return_value = fake_socket_ctx
            r = checker.check_ssl("example.test")
        assert r.passed is False

    def test_handles_exception(self):
        checker = HealthChecker()
        with patch(
            "effgen.utils.health.ssl.create_default_context",
            side_effect=RuntimeError("ssl-err"),
        ):
            r = checker.check_ssl("example.test")
        assert r.passed is False
        assert "ssl-err" in r.message


class TestCheckPyPI:
    def test_returns_version_on_200(self):
        checker = HealthChecker(timeout=2)
        fake_resp = MagicMock(status_code=200)
        fake_resp.json.return_value = {"info": {"version": "0.2.1"}}
        with patch("effgen.utils.health.requests") as mock_requests:
            mock_requests.get.return_value = fake_resp
            r = checker.check_pypi()
        assert r.passed is True
        assert "0.2.1" in r.message

    def test_handles_non_200(self):
        checker = HealthChecker(timeout=2)
        fake_resp = MagicMock(status_code=404)
        with patch("effgen.utils.health.requests") as mock_requests:
            mock_requests.get.return_value = fake_resp
            r = checker.check_pypi()
        assert r.passed is False
        assert "404" in r.message

    def test_handles_exception(self):
        checker = HealthChecker(timeout=2)
        with patch("effgen.utils.health.requests") as mock_requests:
            mock_requests.get.side_effect = RuntimeError("dns-err")
            r = checker.check_pypi()
        assert r.passed is False

    def test_skips_when_requests_unavailable(self):
        checker = HealthChecker()
        with patch("effgen.utils.health.REQUESTS_AVAILABLE", False):
            r = checker.check_pypi()
        assert r.passed is False


class TestCheckAllAndPrint:
    def test_check_all_runs_one_per_default_url_plus_dns_ssl_pypi(self):
        checker = HealthChecker(urls=["https://a.test", "https://b.test"])
        with patch.object(checker, "check_website") as cw, \
             patch.object(checker, "check_dns") as cd, \
             patch.object(checker, "check_ssl") as cs, \
             patch.object(checker, "check_pypi") as cp:
            cw.return_value = HealthCheckResult(name="w", passed=True, message="ok")
            cd.return_value = HealthCheckResult(name="d", passed=True, message="ok")
            cs.return_value = HealthCheckResult(name="s", passed=True, message="ok")
            cp.return_value = HealthCheckResult(name="p", passed=True, message="ok")
            results = checker.check_all()
        assert len(results) == 2 + 1 + 1 + 1
        assert cw.call_count == 2

    def test_print_results_returns_true_when_all_pass(self, capsys):
        checker = HealthChecker()
        results = [
            HealthCheckResult(name="ok1", passed=True, message="ok"),
            HealthCheckResult(name="ok2", passed=True, message="ok"),
        ]
        assert checker.print_results(results) is True
        out = capsys.readouterr().out
        assert "ok1" in out and "ok2" in out

    def test_print_results_returns_false_when_any_fails(self, capsys):
        checker = HealthChecker()
        results = [
            HealthCheckResult(name="ok", passed=True, message="ok"),
            HealthCheckResult(name="bad", passed=False, message="boom"),
        ]
        assert checker.print_results(results) is False
