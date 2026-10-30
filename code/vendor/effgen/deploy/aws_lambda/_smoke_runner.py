"""Local smoke test for the effGen AWS Lambda handler.

Runs the handler against the bundled mock API Gateway HTTP API v2 event and
prints a JSON summary. Intended for quick local verification before deploying
via SAM — no AWS account or SAM CLI required.

Usage
-----
::

    EFFGEN_DEV_MODE=1 python deploy/aws_lambda/_smoke_runner.py

Expected output (status 200, body {"status": "ok", "version": "..."}).
Exits non-zero if the handler does not return 200.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Default to dev mode for the smoke test so JWT auth does not reject /health's
# siblings; /health itself is always public.
os.environ.setdefault("EFFGEN_DEV_MODE", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIXTURE = REPO_ROOT / "tests" / "deploy" / "fixtures" / "apigw-http-event.json"


class _FakeContext:
    """Minimal AWS Lambda context for local invocation."""

    function_name = "effgen-smoke"
    aws_request_id = "smoke-0001"

    def get_remaining_time_in_millis(self) -> int:
        return 29_000


def _mangum_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("mangum")
    except PackageNotFoundError:
        return "unknown"


def main() -> int:
    from deploy.aws_lambda.handler import handler

    event = json.loads(FIXTURE.read_text())

    start = time.monotonic()
    resp = handler(event, _FakeContext())
    elapsed_ms = round((time.monotonic() - start) * 1000, 1)

    status = resp.get("statusCode")
    try:
        body = json.loads(resp.get("body", "null"))
    except (json.JSONDecodeError, TypeError):
        body = resp.get("body")

    summary = {
        "event_path": event.get("rawPath"),
        "event_method": event.get("requestContext", {}).get("http", {}).get("method"),
        "event_format": "API Gateway HTTP API v2.0",
        "response_status_code": status,
        "response_body_parsed": body,
        "elapsed_ms": elapsed_ms,
        "mangum_version": _mangum_version(),
        "test": "PASS" if status == 200 else "FAIL",
    }
    print(json.dumps(summary, indent=2))
    return 0 if status == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
