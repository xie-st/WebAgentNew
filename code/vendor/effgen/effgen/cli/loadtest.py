"""
effgen loadtest — CLI sub-command for the load-test harness.

Wired into the ``effgen`` CLI by ``effgen/cli/_main.py`` and invoked when the
user types ``effgen loadtest …``.

Examples
--------
    # 30-second mock run, concurrency=10, report printed to stdout
    effgen loadtest --duration 30 --concurrency 10

    # Live run against Cerebras (calls the provider adapter directly)
    effgen loadtest --provider cerebras --model gpt-oss-120b \\
        --concurrency 5 --duration 60

    # Drive a running `effgen serve` instance instead — auth, rate limiting
    # and the rest of the middleware stack are exercised, not just the
    # provider adapter
    effgen loadtest --url http://127.0.0.1:8000 --model openai:gpt-5-nano \\
        --concurrency 5 --duration 60

    # Synthetic scenario, also save the report to a file
    effgen loadtest --scenario synthetic --output /tmp/my-report.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..tools.loadgen import LoadConfig, LoadGenerator, LoadScenario
from ..ui.report_html import ReportError, write_html_report

if TYPE_CHECKING:
    import asyncio


def add_loadtest_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Attach ``loadtest`` to an existing argparse subparsers group."""
    p = subparsers.add_parser(
        "loadtest",
        help="Run a load test against a local mock or a live provider",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--concurrency", "-c",
        type=int,
        default=10,
        metavar="N",
        help="Number of concurrent virtual users (default: 10)",
    )
    p.add_argument(
        "--duration", "-d",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help="Test duration in seconds (default: 30)",
    )
    p.add_argument(
        "--scenario", "-s",
        choices=[s.value for s in LoadScenario],
        default=LoadScenario.FIXED.value,
        help="Workload scenario: fixed | synthetic | multi_tool (default: fixed)",
    )
    p.add_argument(
        "--ramp-up",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Ramp-up period: stagger VU starts over N seconds (default: 0)",
    )
    p.add_argument(
        "--think-time",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Simulated think-time between requests per VU (default: 0)",
    )
    p.add_argument(
        "--request-timeout",
        type=float,
        default=60.0,
        metavar="SECONDS",
        help="Per-request timeout in seconds (default: 60)",
    )
    p.add_argument(
        "--provider",
        default=None,
        metavar="NAME",
        help="Provider for live runs (e.g. cerebras, openai). Default: mock",
    )
    p.add_argument(
        "-m", "--model",
        default=None,
        metavar="MODEL_ID",
        help=(
            "Model id for live runs. With --provider, a bare id (e.g. "
            "gpt-oss-120b); with --url, the full id the server expects "
            "(e.g. openai:gpt-5-nano). Default: mock"
        ),
    )
    p.add_argument(
        "--url", "--target-url",
        dest="url",
        default=None,
        metavar="URL",
        help=(
            "Base URL of a running `effgen serve` instance (e.g. "
            "http://127.0.0.1:8000). When set, the load test drives "
            "POST {URL}/v1/chat/completions over HTTP -- exercising auth, "
            "rate limiting, and the rest of the middleware stack -- instead "
            "of calling a provider adapter directly. Requires --model. "
            "Mutually exclusive with --provider."
        ),
    )
    p.add_argument(
        "--api-key",
        default=None,
        metavar="KEY",
        help=(
            "Bearer token for --url mode. Defaults to $EFFGEN_API_KEY "
            "if unset."
        ),
    )
    p.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Write the report to PATH. The extension chooses the format: "
            ".html renders the self-contained HTML report, anything else "
            "writes JSON. If omitted, the report is only printed to stdout."
        ),
    )
    p.add_argument(
        "--report",
        type=Path,
        default=None,
        metavar="PATH.html",
        help=(
            "Write a self-contained HTML report to PATH — latency "
            "percentiles, throughput, error rate, and the error breakdown. "
            "The file opens offline with no external references."
        ),
    )
    p.set_defaults(func=run_loadtest_command)


def _set_future_result(fut: "asyncio.Future[str]", value: str) -> None:
    if not fut.done():
        fut.set_result(value)


def _set_future_exception(fut: "asyncio.Future[str]", exc: Exception) -> None:
    if not fut.done():
        fut.set_exception(exc)


def _build_live_target(provider: str, model: str):
    """Return an async callable that queries a real provider model.

    The adapter is loaded **once** here (not per request) so a load test does
    not pay model-load cost on every call.  Each virtual-user request runs the
    sync adapter call in its own daemon thread rather than
    ``asyncio.to_thread`` (which schedules onto the loop's default
    ``ThreadPoolExecutor``): ``asyncio.run()`` waits for every pending
    default-executor thread to finish before the process can exit, so a call
    still in flight past ``--request-timeout`` (including a provider SDK's own
    internal retries) would keep the process alive well past ``--duration``.
    A daemon thread is abandoned instead of joined when the load generator
    cancels the wait on timeout, so the process can exit as soon as the
    virtual-user loop finishes, regardless of how long a straggling call
    keeps running in the background.
    """
    import asyncio
    import threading

    from effgen import load_model  # noqa: PLC0415

    # Load the model adapter a single time; reused across all requests.
    mdl = load_model(model, provider=provider)
    if hasattr(mdl, "load"):
        try:
            mdl.load()
        except Exception:
            # Some adapters are ready on construction; loading is best-effort.
            pass

    def _generate(prompt: str) -> str:
        # The per-request timeout is enforced by the load generator via
        # asyncio.wait_for, so it is not duplicated in the adapter call here.
        result = mdl.generate(prompt)
        # Adapters return a GenerationResult; fall back to str() for plain text.
        return getattr(result, "text", None) or str(result)

    def _resolve(loop: asyncio.AbstractEventLoop, fut: "asyncio.Future[str]", prompt: str) -> None:
        try:
            value = _generate(prompt)
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller via the future
            try:
                loop.call_soon_threadsafe(_set_future_exception, fut, exc)
            except RuntimeError:
                pass  # loop already closed; the future's outcome no longer matters
            return
        try:
            loop.call_soon_threadsafe(_set_future_result, fut, value)
        except RuntimeError:
            pass

    async def _live_target(prompt: str) -> str:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[str] = loop.create_future()
        threading.Thread(
            target=_resolve, args=(loop, fut, prompt), daemon=True,
            name="effgen-loadtest-call",
        ).start()
        return await fut

    return _live_target


class _ServerTarget:
    """Async callable that drives a running ``effgen serve`` instance over HTTP.

    Unlike :func:`_build_live_target` (which calls a provider adapter
    in-process), this POSTs to ``{base_url}/v1/chat/completions`` so a load
    test exercises the deployed server surface -- auth, rate limiting, and
    the rest of the middleware stack -- not only the bare provider call.
    Uses ``httpx.AsyncClient`` (already a core dependency), which integrates
    with the event loop natively: a timed-out request is cancelled by
    ``asyncio.wait_for`` the same way any other coroutine is, with no
    background-thread lifetime to manage.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None,
        *,
        request_timeout: float = 60.0,
        transport: Any = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._request_timeout = request_timeout
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        # `transport` is a test-only injection point (an ASGI transport
        # driving an in-process app) -- production/CLI use always leaves it
        # None, which gets httpx's real network transport.
        self._transport = transport
        self._client: Any = None

    def _ensure_client(self) -> Any:
        import httpx  # noqa: PLC0415

        if self._client is None:
            # The load generator's own asyncio.wait_for(request_timeout) is the
            # authoritative per-request deadline. Give httpx an explicit timeout
            # set above that deadline so wait_for still fires first (the request
            # counts as a timeout, not a transport error), while the socket
            # itself can never hang without bound.
            from ..reliability.timeouts import make_httpx_timeout  # noqa: PLC0415
            client_timeout = make_httpx_timeout(self._request_timeout + 30.0)
            self._client = httpx.AsyncClient(
                timeout=client_timeout, transport=self._transport
            )
        return self._client

    async def __call__(self, prompt: str) -> str:
        import httpx  # noqa: PLC0415

        client = self._ensure_client()
        resp = await client.post(
            f"{self._base_url}/v1/chat/completions",
            headers=self._headers,
            json={"model": self._model, "messages": [{"role": "user", "content": prompt}]},
        )
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            # Surface the server's own error envelope in the message so a
            # failed run's error_breakdown/raw error text is actionable.
            raise RuntimeError(f"{exc} -- {resp.text[:300]}") from exc
        data = resp.json()
        choices = data.get("choices") or [{}]
        message = choices[0].get("message") or {}
        return message.get("content") or ""

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _build_server_target(
    url: str, model: str, api_key: str | None, *, request_timeout: float = 60.0
) -> _ServerTarget:
    """Return a :class:`_ServerTarget` driving *url* with *model*."""
    return _ServerTarget(url, model, api_key, request_timeout=request_timeout)


def run_loadtest_command(args: argparse.Namespace) -> int:
    """Execute the load test and print / save the report."""

    # Resolve output path: write a file only when --output is given,
    # otherwise the report is printed to stdout. An .html path is rendered
    # after the run rather than written as JSON by the generator.
    output_path: Path | None = Path(args.output) if args.output else None
    html_output = output_path if _is_html(output_path) else None
    if html_output is not None:
        output_path = None
    report_path: Path | None = Path(args.report) if getattr(args, "report", None) else None

    scenario = LoadScenario(args.scenario)

    if args.url and args.provider:
        print(
            "[loadtest] Error: --url and --provider are mutually exclusive "
            "(--url drives the server's HTTP surface; --provider calls the "
            "adapter directly).",
            file=sys.stderr,
        )
        return 2

    if args.url and not args.model:
        print(
            "[loadtest] Error: --url requires --model (the model id the "
            "server should route the request to, e.g. openai:gpt-5-nano).",
            file=sys.stderr,
        )
        return 2

    # --provider and --model must be given together for a live run.
    if not args.url and bool(args.provider) != bool(args.model):
        print(
            "[loadtest] Error: --provider and --model must be supplied "
            "together for a live run. Omit both to use the local mock.",
            file=sys.stderr,
        )
        return 2

    # Build target callable
    target = None
    server_target: _ServerTarget | None = None
    if args.url:
        api_key = args.api_key or os.environ.get("EFFGEN_API_KEY")
        server_target = _build_server_target(
            args.url, args.model, api_key, request_timeout=args.request_timeout
        )
        target = server_target
        print(
            f"Server mode: url={args.url}  model={args.model}",
            flush=True,
        )
    elif args.provider and args.model:
        try:
            target = _build_live_target(args.provider, args.model)
            print(
                f"Live mode: provider={args.provider}  model={args.model}",
                flush=True,
            )
        except Exception as exc:
            print(f"[loadtest] Warning: could not build live target: {exc}", file=sys.stderr)
            print("[loadtest] Falling back to mock target.", file=sys.stderr)
            target = None

    cfg = LoadConfig(
        concurrency=args.concurrency,
        duration=args.duration,
        scenario=scenario,
        ramp_up=args.ramp_up,
        request_timeout=args.request_timeout,
        think_time=args.think_time,
        provider=args.provider,
        model=args.model,
        output_path=output_path,
    )

    print(
        f"Starting load test  scenario={scenario.value}  "
        f"concurrency={cfg.concurrency}  duration={cfg.duration}s",
        flush=True,
    )

    gen = LoadGenerator(cfg, target=target)
    on_finish = server_target.aclose if server_target is not None else None
    report = gen.run(on_finish=on_finish)

    # Pretty-print report to stdout
    _print_report(report)

    for target_path in (html_output, report_path):
        if target_path is None:
            continue
        try:
            written = write_html_report(
                target_path,
                report.to_dict(),
                kind="loadtest",
                command=" ".join(["effgen", *sys.argv[1:]]).strip(),
            )
        except ReportError as exc:
            print(f"[loadtest] Error: {exc}", file=sys.stderr)
            return 2
        print(f"HTML report written to {written}")

    return 0 if report.error_rate < 1.0 else 1


def _is_html(path: Path | None) -> bool:
    return path is not None and path.suffix.lower() in (".html", ".htm")


def _print_report(report) -> None:  # type: ignore[no-untyped-def]
    """Render a load report to stdout."""
    d = report.to_dict()
    sep = "─" * 52
    print(sep)
    print(f"  Load Test Report — {d['scenario']}")
    print(sep)
    print(f"  Concurrency   : {d['concurrency']}")
    # Only surface the drain note when the overshoot is large enough to show at
    # the 0.1s precision used below; sub-0.05s scheduling jitter would otherwise
    # print a self-contradictory "incl. 0.0s draining" line on every fast run.
    if d["drain_s"] >= 0.05:
        print(
            f"  Duration      : requested {d['requested_duration_s']:.1f}s, "
            f"wall {d['duration_s']:.1f}s (incl. {d['drain_s']:.1f}s draining "
            f"in-flight requests after the window closed)"
        )
    else:
        print(f"  Duration      : {d['duration_s']:.1f}s")
    print(f"  Total requests: {d['total_requests']}")
    print(f"  Successful    : {d['successful_requests']}")
    print(f"  Failed        : {d['failed_requests']}")
    print(f"  Error rate    : {d['error_rate'] * 100:.2f}%")
    if d["error_breakdown"]:
        breakdown = ", ".join(f"{k}={v}" for k, v in d["error_breakdown"].items())
        print(f"  Error types   : {breakdown}")
    print(f"  Throughput    : {d['throughput_rps']:.2f} req/s")
    lat = d["latency"]
    print(f"  Latency p50   : {lat['p50'] * 1000:.1f}ms")
    print(f"  Latency p95   : {lat['p95'] * 1000:.1f}ms")
    print(f"  Latency p99   : {lat['p99'] * 1000:.1f}ms")
    print(f"  Latency mean  : {lat['mean'] * 1000:.1f}ms")
    print(f"  Latency stdev : {lat['stdev'] * 1000:.1f}ms")
    if d.get("provider"):
        print(f"  Provider      : {d['provider']}")
        print(f"  Model         : {d['model']}")
    print(sep)
