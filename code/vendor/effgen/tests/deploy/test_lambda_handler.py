"""Tests for the effGen AWS Lambda handler (Mangum adapter).

Tests verify:

1. Handler module loads without errors (import smoke).
2. API Gateway HTTP API v2 event → valid ``/health`` response.
3. API Gateway REST API v1 event → valid ``/health`` response.
4. POST to ``/v1/chat/completions`` with invalid JSON returns 422 or 400.
5. Unknown path returns 404 (not 500).
6. Cold-start timing is recorded at module level.
7. SAM template YAML is syntactically valid and contains required resources.
8. Handler returns correct shape (statusCode, headers, body).
"""
from __future__ import annotations

import importlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

# ---------------------------------------------------------------------------
# CloudFormation / SAM YAML loader — handles intrinsic function tags
# (e.g. !Ref, !Sub, !If, !Equals, !GetAtt, !Not) that the stdlib SafeLoader
# does not understand.  We register all CF intrinsic tags as plain scalar or
# sequence constructors so the template can be loaded structurally.
# ---------------------------------------------------------------------------


def _make_cfn_loader() -> type:
    """Return a yaml.Loader subclass that handles CloudFormation intrinsic tags."""

    class _CfnLoader(yaml.SafeLoader):
        pass

    # Tags whose values are scalars or sequences — treat them all the same way:
    # return a dict {"tag": value} so tests can still inspect the structure.
    cfn_tags = [
        "!And", "!Base64", "!Cidr", "!Condition", "!Equals",
        "!FindInMap", "!GetAZs", "!GetAtt", "!If", "!ImportValue",
        "!Join", "!Not", "!Or", "!Ref", "!Select", "!Split",
        "!Sub", "!Transform", "!ValueOf", "!ValueOfAll",
    ]

    def _scalar_ctor(loader: yaml.SafeLoader, tag: str, node: yaml.Node) -> Any:
        if isinstance(node, yaml.ScalarNode):
            return {tag.lstrip("!"): loader.construct_scalar(node)}
        if isinstance(node, yaml.SequenceNode):
            return {tag.lstrip("!"): loader.construct_sequence(node, deep=True)}
        return {tag.lstrip("!"): loader.construct_mapping(node, deep=True)}

    for _tag in cfn_tags:
        _CfnLoader.add_multi_constructor(
            _tag,
            lambda loader, tag, node: _scalar_ctor(loader, tag, node),
        )
        # Also register as exact-match constructor for !Tag (not prefix)
        _CfnLoader.add_constructor(
            _tag,
            lambda loader, node, _t=_tag: _scalar_ctor(loader, _t, node),
        )

    return _CfnLoader


def _load_cfn_yaml(text: str) -> Any:
    """Load a CloudFormation/SAM YAML template tolerating intrinsic tags."""
    return yaml.load(text, Loader=_make_cfn_loader())  # noqa: S506 - controlled input


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
HANDLER_MODULE = "deploy.aws_lambda.handler"
SAM_TEMPLATE = REPO_ROOT / "deploy" / "aws_lambda" / "sam-template.yaml"
FIXTURES_DIR = Path(__file__).parent / "fixtures"
HTTP_EVENT_FILE = FIXTURES_DIR / "apigw-http-event.json"
V1_EVENT_FILE = FIXTURES_DIR / "apigw-v1-event.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeContext:
    """Minimal AWS Lambda context object."""

    function_name: str = "effgen-test"
    function_version: str = "$LATEST"
    invoked_function_arn: str = (
        "arn:aws:lambda:us-east-1:123456789012:function:effgen-test"
    )
    memory_limit_in_mb: str = "1024"
    aws_request_id: str = "test-request-id-000"
    log_group_name: str = "/aws/lambda/effgen-test"
    log_stream_name: str = "test-stream"

    def get_remaining_time_in_millis(self) -> int:  # noqa: D401
        """Fake: 29 s remaining."""
        return 29_000


def _make_http_v2_event(
    method: str = "GET",
    path: str = "/health",
    body: str | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a minimal API Gateway HTTP API (payload v2) event."""
    hdrs = {
        "accept": "application/json",
        "content-length": str(len(body or "")),
        "host": "abc123.execute-api.us-east-1.amazonaws.com",
        "user-agent": "effgen-test/1.0",
    }
    if headers:
        hdrs.update(headers)
    return {
        "version": "2.0",
        "routeKey": f"{method} {path}",
        "rawPath": path,
        "rawQueryString": "",
        "headers": hdrs,
        "requestContext": {
            "accountId": "123456789012",
            "apiId": "abc123",
            "domainName": "abc123.execute-api.us-east-1.amazonaws.com",
            "domainPrefix": "abc123",
            "http": {
                "method": method,
                "path": path,
                "protocol": "HTTP/1.1",
                "sourceIp": "1.2.3.4",
                "userAgent": "effgen-test/1.0",
            },
            "requestId": f"test-{method}-{path.replace('/', '-')}-001",
            "routeKey": f"{method} {path}",
            "stage": "$default",
            "time": "01/Jan/2026:00:00:00 +0000",
            "timeEpoch": 1767225600000,
        },
        "body": body,
        "isBase64Encoded": False,
    }


def _load_handler():
    """Import the handler module with dev mode enabled.

    Path-anchored at the repo root so this is robust to suite ordering: another
    test putting ``tests/`` on ``sys.path`` would otherwise shadow the repo-root
    ``deploy`` *namespace* package with the ``tests/deploy`` *regular* package
    (which has no ``aws_lambda`` submodule). We force the repo root to the front
    of ``sys.path`` and evict any cached ``deploy`` module that does not resolve
    under the repo root before importing.
    """
    os.environ.setdefault("EFFGEN_DEV_MODE", "1")
    repo_root_str = str(REPO_ROOT)
    # Force the repo root ahead of any stray tests/ entry on sys.path.
    while repo_root_str in sys.path:
        sys.path.remove(repo_root_str)
    sys.path.insert(0, repo_root_str)
    # Evict a stale top-level `deploy` (and submodules) cached from a different
    # location (e.g. tests/deploy) so import resolves the repo-root package.
    deploy_mod = sys.modules.get("deploy")
    deploy_path = getattr(deploy_mod, "__file__", None) or ""
    deploy_ns_paths = list(getattr(deploy_mod, "__path__", []) or [])
    anchored = str(REPO_ROOT / "deploy")
    if deploy_mod is not None and not (
        deploy_path.startswith(anchored)
        or any(p.startswith(anchored) for p in deploy_ns_paths)
    ):
        for name in list(sys.modules):
            if name == "deploy" or name.startswith("deploy."):
                del sys.modules[name]
    return importlib.import_module(HANDLER_MODULE)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def handler_module():
    """Return the imported handler module (dev mode, module-scoped).

    ``EFFGEN_DEV_MODE`` only matters at handler-import time (the module builds
    the app once at load). We set it for the import and then restore the prior
    value so it does not leak into other test modules (a stray ``=1`` flips
    RBAC out of strict mode for the rest of the session).
    """
    _prev = os.environ.get("EFFGEN_DEV_MODE")
    os.environ["EFFGEN_DEV_MODE"] = "1"
    try:
        module = _load_handler()
    finally:
        if _prev is None:
            os.environ.pop("EFFGEN_DEV_MODE", None)
        else:
            os.environ["EFFGEN_DEV_MODE"] = _prev
    return module


@pytest.fixture(scope="module")
def handler_fn(handler_module):
    """Return the ``handler`` callable from the module."""
    fn = handler_module.handler
    assert callable(fn), "handler must be callable"
    return fn


@pytest.fixture()
def ctx():
    """Return a fresh fake Lambda context for each test."""
    return _FakeContext()


# ---------------------------------------------------------------------------
# 1. Import smoke
# ---------------------------------------------------------------------------


class TestHandlerImport:
    """Verify the module loads without errors."""

    def test_module_importable(self, handler_module):
        assert handler_module is not None

    def test_handler_attribute_exists(self, handler_module):
        assert hasattr(handler_module, "handler"), (
            "handler module must expose a top-level 'handler' attribute"
        )

    def test_handler_is_callable(self, handler_fn):
        assert callable(handler_fn)

    def test_cold_start_ts_recorded(self, handler_module):
        """Module records the cold-start monotonic timestamp."""
        assert hasattr(handler_module, "_cold_start_ts")
        assert handler_module._cold_start_ts < time.monotonic()


# ---------------------------------------------------------------------------
# 2. API Gateway HTTP API v2 — GET /health
# ---------------------------------------------------------------------------


class TestHealthEndpointV2:
    """HTTP API payload v2 events."""

    def test_health_status_200(self, handler_fn, ctx):
        event = _make_http_v2_event("GET", "/health")
        resp = handler_fn(event, ctx)
        assert resp["statusCode"] == 200, f"expected 200, got {resp['statusCode']}: {resp}"

    def test_health_body_is_json(self, handler_fn, ctx):
        event = _make_http_v2_event("GET", "/health")
        resp = handler_fn(event, ctx)
        body = json.loads(resp["body"])
        assert isinstance(body, dict)

    def test_health_body_has_status_ok(self, handler_fn, ctx):
        event = _make_http_v2_event("GET", "/health")
        resp = handler_fn(event, ctx)
        body = json.loads(resp["body"])
        assert body.get("status") == "ok", f"unexpected body: {body}"

    def test_health_body_has_version(self, handler_fn, ctx):
        event = _make_http_v2_event("GET", "/health")
        resp = handler_fn(event, ctx)
        body = json.loads(resp["body"])
        assert "version" in body, f"version missing from body: {body}"

    def test_response_has_required_keys(self, handler_fn, ctx):
        event = _make_http_v2_event("GET", "/health")
        resp = handler_fn(event, ctx)
        for key in ("statusCode", "headers", "body"):
            assert key in resp, f"response missing key '{key}': {resp}"

    def test_response_status_code_is_int(self, handler_fn, ctx):
        event = _make_http_v2_event("GET", "/health")
        resp = handler_fn(event, ctx)
        assert isinstance(resp["statusCode"], int)

    def test_response_body_is_str(self, handler_fn, ctx):
        event = _make_http_v2_event("GET", "/health")
        resp = handler_fn(event, ctx)
        assert isinstance(resp["body"], str)


# ---------------------------------------------------------------------------
# 3. API Gateway REST API v1 — GET /health
# ---------------------------------------------------------------------------


class TestHealthEndpointV1:
    """REST API (v1) payload format events."""

    def _load_v1_event(self) -> dict[str, Any]:
        return json.loads(V1_EVENT_FILE.read_text())

    def test_v1_health_status_200(self, handler_fn, ctx):
        event = self._load_v1_event()
        resp = handler_fn(event, ctx)
        assert resp["statusCode"] == 200, f"expected 200, got {resp['statusCode']}: {resp}"

    def test_v1_body_is_json(self, handler_fn, ctx):
        event = self._load_v1_event()
        resp = handler_fn(event, ctx)
        body = json.loads(resp["body"])
        assert isinstance(body, dict)

    def test_v1_body_has_status_ok(self, handler_fn, ctx):
        event = self._load_v1_event()
        resp = handler_fn(event, ctx)
        body = json.loads(resp["body"])
        assert body.get("status") == "ok"


# ---------------------------------------------------------------------------
# 4. Invalid POST to /v1/chat/completions
# ---------------------------------------------------------------------------


class TestChatCompletionsEndpoint:
    """Verify the /v1/chat/completions endpoint returns an error for bad input."""

    def test_post_invalid_json_returns_error(self, handler_fn, ctx):
        event = _make_http_v2_event(
            "POST",
            "/v1/chat/completions",
            body="not-valid-json",
            headers={"content-type": "application/json"},
        )
        resp = handler_fn(event, ctx)
        # Expect 4xx (422 Unprocessable Entity or 400 Bad Request)
        assert 400 <= resp["statusCode"] < 500, (
            f"expected 4xx for invalid JSON body, got {resp['statusCode']}: {resp}"
        )

    def test_post_empty_body_returns_error(self, handler_fn, ctx):
        event = _make_http_v2_event(
            "POST",
            "/v1/chat/completions",
            body=None,
            headers={"content-type": "application/json"},
        )
        resp = handler_fn(event, ctx)
        # Unprocessable or bad request
        assert 400 <= resp["statusCode"] < 500, (
            f"expected 4xx for empty body, got {resp['statusCode']}: {resp}"
        )


# ---------------------------------------------------------------------------
# 5. Unknown path → 404
# ---------------------------------------------------------------------------


class TestNotFoundRoute:
    def test_unknown_path_404(self, handler_fn, ctx):
        event = _make_http_v2_event("GET", "/does-not-exist-xyz")
        resp = handler_fn(event, ctx)
        assert resp["statusCode"] == 404, (
            f"expected 404 for unknown path, got {resp['statusCode']}"
        )


# ---------------------------------------------------------------------------
# 6. /docs and /redoc serve HTML (OpenAPI UI)
# ---------------------------------------------------------------------------


class TestDocsEndpoints:
    def test_docs_200(self, handler_fn, ctx):
        event = _make_http_v2_event("GET", "/docs")
        resp = handler_fn(event, ctx)
        # FastAPI redirects /docs to /docs/ → 307, or serves HTML → 200
        assert resp["statusCode"] in (200, 307), (
            f"expected 200 or 307 for /docs, got {resp['statusCode']}"
        )


# ---------------------------------------------------------------------------
# 6b. Cold-start preload + warm-invocation timing (mock mode)
# ---------------------------------------------------------------------------


class TestColdStartTiming:
    """The first invocation pays cold-start cost; warm invocations are fast.

    Checklist 6.3: first call < 3 s in mock mode, subsequent warm call
    < 100 ms (the registry + app are already built at module load).
    """

    def test_first_call_under_3s(self, handler_fn, ctx):
        event = _make_http_v2_event("GET", "/health")
        start = time.monotonic()
        resp = handler_fn(event, ctx)
        elapsed = time.monotonic() - start
        assert resp["statusCode"] == 200
        assert elapsed < 3.0, f"first /health call took {elapsed:.3f}s (budget 3s)"

    def test_warm_call_under_100ms(self, handler_fn, ctx):
        # Warm the path once, then measure the best of a few warm calls to
        # avoid flaking on a single GC pause / scheduler hiccup.
        event = _make_http_v2_event("GET", "/health")
        handler_fn(event, ctx)
        best = min(
            (
                (lambda: (time.monotonic(), handler_fn(event, ctx), time.monotonic()))()
                for _ in range(5)
            ),
            key=lambda t: t[2] - t[0],
        )
        elapsed = best[2] - best[0]
        assert best[1]["statusCode"] == 200
        assert elapsed < 0.1, f"warm /health call took {elapsed * 1000:.1f}ms (budget 100ms)"

    def test_registry_preloaded_at_module_level(self, handler_module):
        """Cold-start preload runs at import time, before the first request."""
        assert hasattr(handler_module, "_cold_start_ts")
        # The adapter (or fallback handler) must already exist at module load.
        assert handler_module.handler is not None


# ---------------------------------------------------------------------------
# 6c. Per-invocation timeout budget enforcement → 504
# ---------------------------------------------------------------------------


class TestTimeoutBudget:
    """EFFGEN_TIMEOUT_SECONDS / remaining-Lambda-time is enforced (build 6.1)."""

    def test_slow_request_returns_504(self, handler_module, ctx):
        """A handler that overruns its budget returns a clean 504, not a crash.

        We rebuild a tiny handler around a deliberately slow adapter to exercise
        the timeout wrapper directly (no network / model call required).
        """

        def _slow_adapter(event, context):
            time.sleep(2.0)
            return {"statusCode": 200, "headers": {}, "body": "{}"}

        slow_handler = handler_module._make_handler(_slow_adapter)

        class _ShortBudgetCtx:
            def get_remaining_time_in_millis(self) -> int:
                # 1s remaining → budget = max(0.5, 1 - 0.5) = 0.5s < 2s sleep.
                return 1_000

        resp = slow_handler({"rawPath": "/slow"}, _ShortBudgetCtx())
        assert resp["statusCode"] == 504, f"expected 504 on overrun, got {resp}"
        body = json.loads(resp["body"])
        # Same error envelope the app's own routes return.
        assert set(body["error"]) == {"message", "type", "param", "code"}
        assert body["error"]["type"] == "timeout"
        assert body["error"]["code"] == "invocation_timeout"
        assert "timeout" in body["error"]["message"].lower()

    def test_fast_request_within_budget_ok(self, handler_module):
        def _fast_adapter(event, context):
            return {"statusCode": 200, "headers": {}, "body": '{"ok": true}'}

        fast_handler = handler_module._make_handler(_fast_adapter)

        class _AmpleBudgetCtx:
            def get_remaining_time_in_millis(self) -> int:
                return 29_000

        resp = fast_handler({"rawPath": "/fast"}, _AmpleBudgetCtx())
        assert resp["statusCode"] == 200


# ---------------------------------------------------------------------------
# 7. SAM template validation (structural / YAML only, no SAM CLI required)
# ---------------------------------------------------------------------------


class TestSamTemplate:
    """Verify the SAM template is structurally valid."""

    @pytest.fixture(scope="class")
    def template(self):
        assert SAM_TEMPLATE.exists(), f"SAM template not found: {SAM_TEMPLATE}"
        return _load_cfn_yaml(SAM_TEMPLATE.read_text())

    def test_template_loads(self, template):
        assert isinstance(template, dict)

    def test_has_transform(self, template):
        assert "Transform" in template
        assert "AWS::Serverless-2016-10-31" in template["Transform"]

    def test_has_resources(self, template):
        assert "Resources" in template
        assert isinstance(template["Resources"], dict)
        assert len(template["Resources"]) > 0

    def test_has_lambda_function(self, template):
        resources = template["Resources"]
        functions = [
            k for k, v in resources.items()
            if v.get("Type") == "AWS::Serverless::Function"
        ]
        assert len(functions) >= 1, "No AWS::Serverless::Function resource found"

    def test_has_http_api(self, template):
        resources = template["Resources"]
        apis = [
            k for k, v in resources.items()
            if v.get("Type") == "AWS::Serverless::HttpApi"
        ]
        assert len(apis) >= 1, "No AWS::Serverless::HttpApi resource found"

    def test_function_handler_points_to_module(self, template):
        resources = template["Resources"]
        fn = next(
            v for v in resources.values()
            if v.get("Type") == "AWS::Serverless::Function"
        )
        handler = fn["Properties"]["Handler"]
        assert "handler" in handler.lower(), (
            f"Function Handler should reference 'handler', got: {handler}"
        )

    def test_handler_module_avoids_python_keyword(self, template):
        """The handler module path must not use the reserved word ``lambda``.

        ``deploy.lambda.handler`` cannot be imported with an ``import`` statement
        (``lambda`` is a keyword), so the module dir must be ``aws_lambda``.
        """
        resources = template["Resources"]
        fn = next(
            v for v in resources.values()
            if v.get("Type") == "AWS::Serverless::Function"
        )
        handler = fn["Properties"]["Handler"]
        module_parts = handler.split(".")[:-1]  # drop the function name
        assert "lambda" not in module_parts, (
            f"Handler module path uses reserved keyword 'lambda': {handler}"
        )
        assert handler == "deploy.aws_lambda.handler.handler", (
            f"unexpected handler reference: {handler}"
        )

    def test_function_runtime_not_deprecated(self, template):
        # Runtime may be set at Globals level. python3.10/3.11 were deprecated by AWS
        # on 2026-06-30 (creation blocked 2026-07-31) — the template must track a
        # currently supported minor version.
        resources = template["Resources"]
        fn = next(
            v for v in resources.values()
            if v.get("Type") == "AWS::Serverless::Function"
        )
        runtime = fn["Properties"].get("Runtime") or template.get("Globals", {}).get("Function", {}).get("Runtime", "")
        match = re.fullmatch(r"python3\.(\d+)", runtime)
        assert match, f"Unexpected Lambda runtime format: {runtime}"
        assert int(match.group(1)) >= 12, f"Runtime {runtime} is deprecated; use python3.12 or newer"

    def test_has_outputs(self, template):
        assert "Outputs" in template
        assert "EffgenApiEndpoint" in template["Outputs"]

    def test_effgen_dev_mode_env_var_present(self, template):
        env_vars = (
            template.get("Globals", {})
            .get("Function", {})
            .get("Environment", {})
            .get("Variables", {})
        )
        assert "EFFGEN_DEV_MODE" in env_vars, (
            "EFFGEN_DEV_MODE must be set in Globals.Function.Environment.Variables"
        )

    def test_sandbox_backend_env_var(self, template):
        env_vars = (
            template.get("Globals", {})
            .get("Function", {})
            .get("Environment", {})
            .get("Variables", {})
        )
        # Lambda can't run Docker; we must explicitly set subprocess backend
        assert "EFFGEN_SANDBOX_BACKEND" in env_vars, (
            "EFFGEN_SANDBOX_BACKEND must be set in SAM template (Lambda can't run Docker)"
        )
        assert env_vars["EFFGEN_SANDBOX_BACKEND"] == "subprocess"

    def test_has_log_group_resource(self, template):
        resources = template["Resources"]
        log_groups = [
            k for k, v in resources.items()
            if v.get("Type") == "AWS::Logs::LogGroup"
        ]
        assert len(log_groups) >= 1, "No AWS::Logs::LogGroup resource found"

    def test_parameters_exist(self, template):
        params = template.get("Parameters", {})
        for expected in ("Environment", "EffgenTimeoutSeconds", "ApiKeySecretArn"):
            assert expected in params, f"Parameter '{expected}' missing from SAM template"


# ---------------------------------------------------------------------------
# 8. Fixture files exist
# ---------------------------------------------------------------------------


class TestFixtures:
    def test_http_v2_fixture_exists(self):
        assert HTTP_EVENT_FILE.exists()

    def test_v1_fixture_exists(self):
        assert V1_EVENT_FILE.exists()

    def test_http_v2_fixture_valid_json(self):
        data = json.loads(HTTP_EVENT_FILE.read_text())
        assert data["version"] == "2.0"
        assert "requestContext" in data

    def test_v1_fixture_valid_json(self):
        data = json.loads(V1_EVENT_FILE.read_text())
        assert "httpMethod" in data
        assert "requestContext" in data


# ---------------------------------------------------------------------------
# 9. sam validate (only when the SAM CLI is installed; build 6.5)
# ---------------------------------------------------------------------------


class TestSamCliValidate:
    """Validate the SAM template against the real CloudFormation/SAM schema.

    Prefers the AWS SAM CLI (`sam validate`); falls back to `cfn-lint` (which
    bundles aws-sam-translator and validates the ``AWS::Serverless`` transform).
    Skips only when neither tool can run — the structural YAML test in
    :class:`TestSamTemplate` still covers the resource/parameter shape.
    """

    @staticmethod
    def _usable(command: str) -> bool:
        """Whether *command* is on PATH and can actually start.

        A launcher script can be on PATH while the package it imports is not
        importable — a console entry point installed into a user site directory
        that the current ``HOME`` does not point at, for one. That is the same
        situation as the tool not being installed, so it reads as absent rather
        than as a validation failure.
        """
        import shutil
        import subprocess

        if shutil.which(command) is None:
            return False
        try:
            probe = subprocess.run(
                [command, "--version"], capture_output=True, text=True, timeout=60
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return probe.returncode == 0

    def test_sam_validate(self):
        import subprocess

        if self._usable("sam"):
            result = subprocess.run(
                ["sam", "validate", "--lint", "--template", str(SAM_TEMPLATE)],
                capture_output=True,
                text=True,
                cwd=str(SAM_TEMPLATE.parent),
            )
            assert result.returncode == 0, (
                f"sam validate failed:\n{result.stdout}\n{result.stderr}"
            )
            return

        if self._usable("cfn-lint"):
            result = subprocess.run(
                ["cfn-lint", str(SAM_TEMPLATE)],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, (
                f"cfn-lint failed:\n{result.stdout}\n{result.stderr}"
            )
            return

        pytest.skip(
            "Neither SAM CLI nor cfn-lint can run here — structural YAML test "
            "covers schema"
        )
