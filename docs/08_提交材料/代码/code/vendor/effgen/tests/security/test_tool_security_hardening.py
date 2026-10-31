"""Security regression tests for the built-in execution/IO tools.

Covers, with real subprocesses and real sockets (no mocks of the behaviour
under test):

* PythonREPL — a runaway (``while True``) is hard-killed at the wall-clock
  timeout; the worker runs out-of-process; persistence still works.
* URLFetch — private/loopback/link-local/metadata targets are blocked by
  default, redirects are re-validated, and the block is opt-out only.
* Retrieval — indexes persist as JSON; a malicious pickle is refused (no code
  execution) unless explicitly allowed.
* BashTool — a timeout kills the whole process group (no orphaned children);
  argv execution is preferred over the shell when no shell features are used.
* FileOperations — path traversal, symlink escape and out-of-root writes are
  rejected.
"""

from __future__ import annotations

import asyncio
import ipaddress
import os
import pickle
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from effgen.tools.builtin.bash_tool import BashTool
from effgen.tools.builtin.file_ops import FileOperations
from effgen.tools.builtin.python_repl import PythonREPL
from effgen.tools.builtin.retrieval import Document, Retrieval
from effgen.tools.builtin.url_fetch import URLFetchTool, _is_blocked_ip


# --------------------------------------------------------------------------- #
# PythonREPL                                                                   #
# --------------------------------------------------------------------------- #
class TestPythonREPLSandbox:
    async def test_infinite_loop_killed_at_timeout(self):
        repl = PythonREPL(timeout=2)
        await repl.initialize()
        try:
            t0 = time.time()
            res = await repl.execute(code="while True:\n    pass")
            elapsed = time.time() - t0
            assert res.success is False
            assert res.output.get("timeout") is True
            # Killed promptly at the deadline, not hanging indefinitely.
            assert elapsed < 6, f"timeout not enforced (took {elapsed:.1f}s)"
        finally:
            await repl.cleanup()

    async def test_basic_exec_and_persistence(self):
        repl = PythonREPL(timeout=10)
        await repl.initialize()
        try:
            a = await repl.execute(code="x = 21\ny = 21\nx + y")
            assert a.success and a.output["result"] == 42
            # State persists across calls in the same session worker.
            b = await repl.execute(code="x + y + 1")
            assert b.success and b.output["result"] == 43
            c = await repl.execute(code="print('hello')")
            assert c.output["stdout"].strip() == "hello"
        finally:
            await repl.cleanup()

    async def test_runs_out_of_process(self):
        # Unrestricted execution is a developer opt-in (allow_unrestricted=True);
        # a model can never reach it (see TestPythonREPLSandboxToggle).
        repl = PythonREPL(timeout=10, allow_unrestricted=True)
        await repl.initialize()
        try:
            res = await repl.execute(
                code="import os\nos.getpid()", restricted_mode=False
            )
            assert res.success
            assert res.output["result"] != os.getpid()
        finally:
            await repl.cleanup()

    async def test_import_gate_blocks_disallowed(self):
        repl = PythonREPL(timeout=10)
        await repl.initialize()
        try:
            res = await repl.execute(code="import os")
            assert res.success is False
            assert "not allowed" in str(res.output["error"]).lower()
        finally:
            await repl.cleanup()

    async def test_no_orphan_workers_after_cleanup(self):
        repl = PythonREPL(timeout=5)
        await repl.initialize()
        await repl.execute(code="1 + 1")
        await repl.execute(code="1 + 1", session_id="other")
        # The exact worker PIDs this instance owns.
        pids = [w.proc.pid for w in repl._workers.values()]
        assert len(pids) == 2
        await repl.cleanup()
        await asyncio.sleep(0.3)

        def _alive(pid: int) -> bool:
            try:
                os.kill(pid, 0)
                return True
            except ProcessLookupError:
                return False
            except PermissionError:  # pragma: no cover
                return True

        survivors = [pid for pid in pids if _alive(pid)]
        assert survivors == [], f"orphan REPL workers after cleanup: {survivors}"


# --------------------------------------------------------------------------- #
# PythonREPL — dynamically-constructed dunder attribute names                  #
# --------------------------------------------------------------------------- #
class TestPythonREPLDynamicAttributeEscape:
    """restricted_mode must block a dunder attribute name built at run time,
    not only the literal spelling caught by the parent-side text/AST scan."""

    async def test_dynamic_getattr_subclasses_walk_blocked(self):
        repl = PythonREPL(timeout=10)
        await repl.initialize()
        try:
            code = (
                "u = chr(95) * 2\n"
                "attr = u + 'subclasses' + u\n"
                "getattr(object, attr)()"
            )
            res = await repl.execute(code=code)
            assert res.success is False
            assert "blocked" in str(res.output["error"]).lower()
        finally:
            await repl.cleanup()

    async def test_dynamic_getattr_globals_chain_cannot_reach_os(self):
        """Walk every loaded subclass hunting for
        one whose __init__.__globals__ exposes a live `os` module, then call
        os.popen. Must never reach a real os.popen result."""
        repl = PythonREPL(timeout=10)
        await repl.initialize()
        try:
            code = """
u = chr(95) * 2
subs = getattr(object, u + 'subclasses' + u)()
osmod = None
for c in subs:
    try:
        g = getattr(c, u + 'init' + u, None)
        gl = getattr(g, u + 'globals' + u, None) if g else None
        if gl and 'os' in gl:
            osmod = gl['os']
            break
    except Exception:
        pass
osmod.popen('id').read() if osmod else 'NO OS MODULE FOUND'
"""
            res = await repl.execute(code=code)
            assert res.success is False
            assert "blocked" in str(res.output["error"]).lower()
        finally:
            await repl.cleanup()

    async def test_operator_attrgetter_chain_cannot_reach_os(self):
        """operator.attrgetter reaches attributes via the C attribute
        protocol directly, bypassing a namespace-local getattr override --
        must be blocked independently."""
        repl = PythonREPL(timeout=10)
        await repl.initialize()
        try:
            code = """
import operator
u = chr(95) * 2
subs = operator.attrgetter(u + 'subclasses' + u)(object)()
osmod = None
for c in subs:
    try:
        init = operator.attrgetter(u + 'init' + u)(c)
        gl = operator.attrgetter(u + 'globals' + u)(init)
        if gl and 'os' in gl:
            osmod = gl['os']
            break
    except Exception:
        pass
osmod.popen('id').read() if osmod else 'NO OS MODULE FOUND'
"""
            res = await repl.execute(code=code)
            # Either the call is rejected outright, or every attempt to reach
            # __globals__ raises internally and the loop's own except clause
            # swallows it, leaving no os module found.
            if res.success:
                assert res.output["result"] == "NO OS MODULE FOUND"
            else:
                assert "blocked" in str(res.output["error"]).lower()
        finally:
            await repl.cleanup()

    async def test_functools_update_wrapper_cannot_stash_globals(self):
        """A dunder value copied onto a plain object via functools.
        update_wrapper (which calls the real setattr internally) must not be
        readable back out as a live os module reference."""
        repl = PythonREPL(timeout=10)
        await repl.initialize()
        try:
            code = """
import functools
u = chr(95) * 2
subkey, initkey, globkey = u + 'subclasses' + u, u + 'init' + u, u + 'globals' + u
subs = type.__dict__[subkey](object)


class Box:
    pass


found = None
for c in subs:
    init = c.__dict__.get(initkey)
    if init is None:
        continue
    try:
        box = Box()
        functools.update_wrapper(box, init, assigned=(globkey,), updated=())
        gl = box.__dict__.get(globkey)
        if gl and 'os' in gl:
            found = gl['os']
            break
    except Exception:
        pass
found.popen('id').read() if found else 'NO OS MODULE FOUND'
"""
            res = await repl.execute(code=code)
            # Either the ``__dict__`` route to the subclass graph is refused
            # outright, or the walk completes but never recovers a live os
            # module (and the audit hook would refuse the popen regardless).
            if res.success:
                assert res.output["result"] == "NO OS MODULE FOUND"
            else:
                assert "blocked" in str(res.output["error"]).lower()
        finally:
            await repl.cleanup()

    async def test_descriptor_route_to_globals_cannot_execute_os(self):
        """A subclass-graph walk can recover a live ``os`` reference through a
        getset descriptor (``type(f).__dict__['__globals__'].__get__(f)``)
        without ever calling ``getattr`` or spelling a blocked dunder — the
        audit hook must still refuse the process/shell execution it leads to."""
        repl = PythonREPL(timeout=10)
        await repl.initialize()
        try:
            code = """
u = chr(95) * 2
subs = type.__dict__[u + 'subclasses' + u](object)
ftype = type(lambda: 0)
gdesc = ftype.__dict__[u + 'globals' + u]
initkey = u + 'init' + u
osmod = None
for c in subs:
    init = c.__dict__.get(initkey)
    if not isinstance(init, ftype):
        continue
    try:
        g = gdesc.__get__(init)
        if isinstance(g, dict) and 'os' in g:
            osmod = g['os']
            break
    except Exception:
        pass
osmod.popen('id').read() if osmod else 'NO OS MODULE FOUND'
"""
            res = await repl.execute(code=code)
            assert res.success is False
            assert "blocked in restricted mode" in str(res.output["error"]).lower()
        finally:
            await repl.cleanup()

    @pytest.mark.parametrize("call", [
        "osmod.system('id')",
        "osmod.execv('/bin/true', ['/bin/true'])",
    ])
    async def test_os_process_primitives_blocked_by_audit_hook(self, call):
        """Even with a live ``os`` reference in hand, the audit hook refuses the
        actual process/shell-execution primitives."""
        repl = PythonREPL(timeout=10)
        await repl.initialize()
        try:
            code = f"""
u = chr(95) * 2
subs = type.__dict__[u + 'subclasses' + u](object)
ftype = type(lambda: 0)
gdesc = ftype.__dict__[u + 'globals' + u]
initkey = u + 'init' + u
osmod = None
for c in subs:
    init = c.__dict__.get(initkey)
    if not isinstance(init, ftype):
        continue
    try:
        g = gdesc.__get__(init)
        if isinstance(g, dict) and 'os' in g:
            osmod = g['os']
            break
    except Exception:
        pass
{call} if osmod else 'NO OS MODULE FOUND'
"""
            res = await repl.execute(code=code)
            assert res.success is False
            assert "blocked in restricted mode" in str(res.output["error"]).lower()
        finally:
            await repl.cleanup()

    async def test_ordinary_getattr_and_operator_use_still_work(self):
        """The guard only rejects the specific dangerous dunder names, not
        every dunder or every getattr/operator call."""
        repl = PythonREPL(timeout=10)
        await repl.initialize()
        try:
            res = await repl.execute(
                code="class P:\n    x = 1\np = P()\nr = getattr(p, 'x')\nr"
            )
            assert res.success and res.output["result"] == 1

            res = await repl.execute(
                code="import operator\nr = operator.add(2, 3)\nr"
            )
            assert res.success and res.output["result"] == 5

            res = await repl.execute(code="import math\nr = math.sqrt(16)\nr")
            assert res.success and res.output["result"] == 4.0

            res = await repl.execute(
                code=(
                    "class Foo:\n"
                    "    def __init__(self, v):\n"
                    "        self.v = v\n"
                    "f = Foo(5)\n"
                    "f.v"
                )
            )
            assert res.success and res.output["result"] == 5
        finally:
            await repl.cleanup()


# --------------------------------------------------------------------------- #
# URLFetch / SSRF                                                              #
# --------------------------------------------------------------------------- #
class TestURLFetchSSRF:
    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1/",
            "http://localhost/",
            "http://169.254.169.254/latest/meta-data/",
            "http://10.0.0.1/",
            "http://192.168.1.1/",
            "http://0.0.0.0/",
            "http://[::1]/",
        ],
    )
    async def test_internal_targets_blocked_by_default(self, url):
        tool = URLFetchTool(timeout=5)
        res = await tool.execute(url=url)
        assert res.success is False
        assert "ssrf" in str(res.error).lower() or "resolve" in str(res.error).lower()

    async def test_allow_private_opt_in_attempts_connection(self):
        # With the override the SSRF guard must NOT short-circuit; the request is
        # attempted (and fails on connection refused at a closed port).
        tool = URLFetchTool(timeout=3, allow_private=True)
        res = await tool.execute(url="http://127.0.0.1:9/")
        assert "SSRF" not in str(res.error)

    async def test_redirect_to_internal_is_revalidated(self):
        """A public first hop must not be able to bounce us to an internal IP."""
        tool = URLFetchTool(timeout=5)

        class _Resp:
            def __init__(self, status, location=None, text=""):
                self.status_code = status
                self.is_redirect = status in (301, 302, 303, 307, 308)
                self.headers = {"Location": location} if location else {}
                self.text = text

            def raise_for_status(self):
                pass

        class _Session:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def get(self, url, **kw):
                return _Resp(302, location="http://169.254.169.254/latest/meta-data/")

        class _FakeRequests:
            @staticmethod
            def Session():
                return _Session()

        with pytest.raises(ValueError):
            # First hop (example.com) resolves public, redirect points internal.
            tool._fetch_with_requests(_FakeRequests, "https://example.com/")

    def test_is_blocked_ip_classification(self):
        assert _is_blocked_ip(ipaddress.ip_address("127.0.0.1"))
        assert _is_blocked_ip(ipaddress.ip_address("169.254.169.254"))
        assert _is_blocked_ip(ipaddress.ip_address("10.1.2.3"))
        assert _is_blocked_ip(ipaddress.ip_address("::1"))
        assert _is_blocked_ip(ipaddress.ip_address("::ffff:127.0.0.1"))
        assert not _is_blocked_ip(ipaddress.ip_address("8.8.8.8"))


# --------------------------------------------------------------------------- #
# Retrieval pickle gating                                                      #
# --------------------------------------------------------------------------- #
class TestRetrievalPickle:
    async def test_json_roundtrip(self, tmp_path):
        import numpy as np

        r = Retrieval()
        await r.initialize()
        r.documents["d1"] = Document(
            id="d1", content="hello world", metadata={"src": "x"},
            embedding=np.array([0.1, 0.2, 0.3]),
        )
        r.doc_ids = ["d1"]
        path = str(tmp_path / "index.json")
        r.save_index(path)
        # Stored as JSON text, not a pickle.
        assert open(path, encoding="utf-8").read(1) == "{"

        r2 = Retrieval(index_path=path)
        assert list(r2.documents) == ["d1"]
        assert r2.documents["d1"].embedding.tolist() == pytest.approx([0.1, 0.2, 0.3])

    def test_malicious_pickle_refused_no_rce(self, tmp_path):
        flag = tmp_path / "pwned"

        class Evil:
            def __reduce__(self):
                return (os.system, (f"touch {flag}",))

        pkl = tmp_path / "evil.pkl"
        with open(pkl, "wb") as f:
            pickle.dump({"documents": {}, "doc_ids": [], "x": Evil()}, f)

        with pytest.raises(ValueError) as exc:
            Retrieval(index_path=str(pkl))
        assert "pickle" in str(exc.value).lower()
        assert not flag.exists(), "pickle executed code — RCE not prevented"

    def test_allow_pickle_opt_in_loads(self, tmp_path):
        data = {"documents": {}, "doc_ids": [], "chunk_size": 500, "chunk_overlap": 100}
        pkl = tmp_path / "legacy.pkl"
        with open(pkl, "wb") as f:
            pickle.dump(data, f)
        r = Retrieval(index_path=str(pkl), allow_pickle=True)
        assert r.doc_ids == []


# --------------------------------------------------------------------------- #
# BashTool                                                                     #
# --------------------------------------------------------------------------- #
class TestBashSecurity:
    async def test_timeout_kills_process_group(self):
        # Use a process-unique sleep duration (a large float, valid for `sleep`)
        # as a sentinel so the system-wide pgrep matches ONLY this test's own
        # children — never a co-tenant's plain `sleep 120` on a shared host.
        # The bash kill itself is correct; this keeps the TEST hermetic.
        marker = f"7777.{os.getpid() % 1000:03d}{int(time.time() * 1000) % 1000:03d}"
        bash = BashTool(timeout=2)
        t0 = time.time()
        res = await bash.execute(command=f"sleep {marker} & exec sleep {marker}")
        assert res.success is False
        assert (time.time() - t0) < 6
        await asyncio.sleep(0.5)
        out = subprocess.run(
            ["pgrep", "-af", f"sleep {marker}"], capture_output=True, text=True
        ).stdout
        leftovers = [ln for ln in out.splitlines() if marker in ln and "pgrep" not in ln]
        assert leftovers == [], f"orphaned children after timeout: {leftovers}"

    async def test_argv_execution_when_no_shell_features(self):
        bash = BashTool(timeout=10)
        # Quotes are handled by argv parsing; no shell needed.
        res = await bash.execute(command="echo 'literal | not a pipe'")
        assert res.success
        assert res.output["stdout"].strip() == "literal | not a pipe"
        assert bash._needs_shell("echo hi") is False
        assert bash._needs_shell("ls *.py") is True
        assert bash._needs_shell("a | b") is True

    async def test_basic_command(self):
        bash = BashTool(timeout=10)
        res = await bash.execute(command="echo hello")
        assert res.success and res.output["stdout"].strip() == "hello"


# --------------------------------------------------------------------------- #
# FileOperations path safety                                                   #
# --------------------------------------------------------------------------- #
class TestFileOpsPathSafety:
    async def _tool(self):
        d = tempfile.mkdtemp(prefix="p17_fo_")
        with open(os.path.join(d, "ok.txt"), "w") as f:
            f.write("inside")
        return d, FileOperations(allowed_directories=[d])

    async def test_read_inside_allowed(self):
        d, tool = await self._tool()
        res = await tool.execute(operation="read", path=os.path.join(d, "ok.txt"))
        assert res.success and res.output["data"] == "inside"

    async def test_traversal_blocked(self):
        d, tool = await self._tool()
        res = await tool.execute(
            operation="read", path=os.path.join(d, "../../../../etc/passwd")
        )
        assert res.success is False

    async def test_absolute_outside_blocked(self):
        d, tool = await self._tool()
        res = await tool.execute(operation="read", path="/etc/passwd")
        assert res.success is False

    async def test_symlink_escape_blocked(self):
        d, tool = await self._tool()
        link = os.path.join(d, "evil.txt")
        os.symlink("/etc/passwd", link)
        res = await tool.execute(operation="read", path=link)
        assert res.success is False

    async def test_write_outside_blocked(self):
        d, tool = await self._tool()
        res = await tool.execute(
            operation="write", path="/tmp/p17_should_not_exist.txt", content="x"
        )
        assert res.success is False
        assert not os.path.exists("/tmp/p17_should_not_exist.txt")


# --------------------------------------------------------------------------- #
# PromptChain condition evaluation (no eval of model output)           #
# --------------------------------------------------------------------------- #
class TestChainConditionSafety:
    def _state(self):
        from effgen.prompts.chain_manager import ChainState

        return ChainState()

    def test_legit_conditions_still_work(self):
        st = self._state()
        st.set_variable("quality_score", 0.85)
        st.set_variable("result_type", "success")
        st.iteration_count = 1
        assert st.evaluate_condition("quality_score >= 0.8") is True
        assert st.evaluate_condition("result_type == 'success'") is True
        assert st.evaluate_condition("iteration_count < 3") is True
        assert st.evaluate_condition("{quality_score} >= 0.8") is True
        assert st.evaluate_condition(
            "quality_score >= 0.8 and result_type == 'success'"
        ) is True
        assert st.evaluate_condition("result_type in ['success', 'ok']") is True

    def test_model_output_is_data_not_code(self):
        # A prior step result of "1+1" must compare as the string, never compute.
        st = self._state()
        st.set_variable("step_result", "1+1")
        assert st.evaluate_condition("{step_result} == 2") is False
        assert st.evaluate_condition("step_result == '1+1'") is True

    @pytest.mark.parametrize(
        "payload",
        [
            "().__class__.__name__ == 'tuple'",
            "len(().__class__.__base__.__subclasses__()) > 0",
            "__import__('os').system('echo pwned') == 0",
            "''.__class__.__mro__[1].__subclasses__()",
            "(lambda: 1)() == 1",
        ],
    )
    def test_adversarial_payloads_inert(self, payload):
        st = self._state()
        # Either rejected (False) — never raises, never executes.
        assert st.evaluate_condition(payload) is False

    def test_injected_payload_does_not_execute(self, tmp_path):
        st = self._state()
        flag = tmp_path / "chain_pwned"
        st.set_variable(
            "evil", f"__import__('os').system('touch {flag}')"
        )
        # Interpolated into the condition the way a vulnerable eval() would see it.
        st.evaluate_condition("{evil} == 0")
        st.evaluate_condition("evil == 0")
        assert not flag.exists(), "chain condition executed code — injection not prevented"


# --------------------------------------------------------------------------- #
# shared SSRF guard used by every URL-taking tool                       #
# --------------------------------------------------------------------------- #
@pytest.fixture
def name_resolves_to_a_public_address(monkeypatch):
    """Answer every name lookup with one public address.

    The guard resolves a hostname before it decides, so a case about what it
    decides for a public host would otherwise also be a case about whether the
    machine running it has a name server. Pinning the answer keeps the assertion
    on the guard.
    """
    import socket as _socket

    def _resolve(host, port=None, *args, **kwargs):
        return [
            (
                _socket.AF_INET,
                _socket.SOCK_STREAM,
                _socket.IPPROTO_TCP,
                "",
                ("93.184.216.34", port or 443),
            )
        ]

    monkeypatch.setattr(_socket, "getaddrinfo", _resolve)


class TestSharedSSRFGuard:
    INTERNAL = [
        "http://127.0.0.1/",
        "http://localhost/",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.1/",
        "http://[::1]/",
    ]

    def test_net_helpers_classify(self):
        from effgen.tools.builtin import _net

        assert _net.is_blocked_ip(__import__("ipaddress").ip_address("127.0.0.1"))
        assert _net.is_blocked_ip(__import__("ipaddress").ip_address("169.254.169.254"))
        assert not _net.is_blocked_ip(__import__("ipaddress").ip_address("8.8.8.8"))
        for url in self.INTERNAL:
            with pytest.raises(_net.BlockedURLError):
                _net.check_url_safe(url)
        # opt-out lets it through the guard (no exception)
        _net.check_url_safe("http://127.0.0.1/", allow_private=True)

    def test_host_pin_rejects_other_hosts(self, name_resolves_to_a_public_address):
        from effgen.tools.builtin import _net

        with pytest.raises(_net.BlockedURLError):
            _net.check_url_safe(
                "https://evil.example.com/", allowed_hosts={"api.example.org"}
            )
        # wildcard suffix match
        _net.check_url_safe(
            "https://en.wikipedia.org/w/api.php", allowed_hosts={"*.wikipedia.org"}
        )

    async def test_rss_blocks_internal(self):
        from effgen.tools.builtin.rss import RSSFeedTool

        res = await RSSFeedTool().execute(operation="latest", url="http://127.0.0.1/x", n=1)
        assert res.success is False

    async def test_news_newsapi_blocks_internal(self):
        # NewsAPI host pin: an internal URL never reaches the network.
        from effgen.tools.builtin import news
        from effgen.tools.builtin._net import BlockedURLError

        with pytest.raises(BlockedURLError):
            news._newsapi_request("http://127.0.0.1:9/v2/top-headlines")

    async def test_devops_http_blocks_internal(self):
        from effgen.tools.builtin.devops import HTTPTool

        res = await HTTPTool().execute(url="http://169.254.169.254/latest/meta-data/")
        assert res.success is False

    async def test_webhook_override_blocks_internal(self):
        from effgen.tools.builtin.slack_webhook import SlackWebhookTool

        res = await SlackWebhookTool().execute(
            text="hi", webhook_url="http://127.0.0.1:9/services/x"
        )
        assert res.success is False

    def test_url_fetch_uses_shared_guard(self):
        # url_fetch must re-export the shared classifier (single source of truth).
        from effgen.tools.builtin import _net, url_fetch

        assert url_fetch._is_blocked_ip is _net.is_blocked_ip

    @pytest.mark.parametrize(
        "url,scheme",
        [
            ("file:///etc/passwd", "file"),
            ("ftp://example.com/x", "ftp"),
            ("gopher://evil.example.com/", "gopher"),
        ],
    )
    def test_url_fetch_rejects_non_http_scheme_by_name(self, url, scheme):
        # A non-http(s) scheme reaches the shared guard and is refused by name.
        # Normalizing it to "https://" + url first would make the scheme the
        # hostname, turning a scheme rejection into a DNS lookup failure.
        from effgen.tools.builtin._net import BlockedURLError
        from effgen.tools.builtin.url_fetch import URLFetchTool

        with pytest.raises(BlockedURLError) as exc:
            URLFetchTool()._validate_url(url)
        assert f"scheme '{scheme}'" in str(exc.value)

    def test_url_fetch_still_normalizes_a_bare_host(self, name_resolves_to_a_public_address):
        from effgen.tools.builtin.url_fetch import URLFetchTool

        tool = URLFetchTool()
        assert tool._validate_url("example.com/page") == "https://example.com/page"
        assert tool._validate_url("example.com:8080/page") == "https://example.com:8080/page"
        assert tool._validate_url("http://example.com/x") == "http://example.com/x"

    @pytest.mark.parametrize("url", ["https://host:not-a-port/x", "http://[::1/x"])
    def test_unparseable_url_is_refused_by_name(self, url):
        # urlparse raises on an unclosed IPv6 literal and defers a bad port to
        # attribute access, so an unparseable URL must come back as a refusal
        # naming the URL — not as a bare urllib ValueError.
        from effgen.tools.builtin._net import BlockedURLError, check_url_safe
        from effgen.tools.builtin.url_fetch import URLFetchTool

        with pytest.raises(BlockedURLError, match="unparseable URL"):
            check_url_safe(url)
        with pytest.raises(BlockedURLError, match="unparseable URL"):
            URLFetchTool()._validate_url(url)


# --------------------------------------------------------------------------- #
# file-path tools may not read sensitive locations; tighten on demand    #
# --------------------------------------------------------------------------- #
class TestFilePathConfinement:
    # Sensitive targets an attacker actually wants (refused by default).
    SENSITIVE = ["/etc/passwd", str(Path.home() / ".ssh" / "id_rsa")]

    async def test_image_info_blocks_sensitive(self):
        from effgen.tools.builtin.image_info import ImageInfoTool

        for target in self.SENSITIVE:
            res = await ImageInfoTool().execute(operation="info", image_path=target)
            assert res.success is False
            assert "protected" in str(res.error).lower() or "refus" in str(res.error).lower()

    async def test_pdf_blocks_etc_passwd(self):
        from effgen.tools.builtin.pdf import PDFTool

        res = await PDFTool().execute(operation="text", path="/etc/passwd")
        assert res.success is False
        assert "protected" in str(res.error).lower() or "refus" in str(res.error).lower()

    async def test_data_analysis_blocks_etc(self):
        from effgen.tools.builtin.data_analysis import DataFrameTool

        res = await DataFrameTool().execute(file_path="/etc/passwd", operation="head")
        assert res.success is False

    async def test_ocr_blocks_ssh_key(self):
        from effgen.tools.builtin.ocr import OCRTool

        res = await OCRTool().execute(
            operation="extract", image_path="/etc/shadow"
        )
        assert res.success is False
        assert "protected" in str(res.error).lower() or "refus" in str(res.error).lower()

    async def test_ordinary_file_allowed_by_default(self, tmp_path):
        # A normal (non-sensitive) file is readable by default — the tools stay
        # usable; confinement only fails on a *format* error here, never on the
        # path policy.
        from effgen.tools.builtin.data_analysis import DataFrameTool

        good = tmp_path / "data.csv"
        good.write_text("a,b\n1,2\n3,4\n")
        res = await DataFrameTool().execute(file_path=str(good), operation="head", n=1)
        assert res.success is True

    async def test_allowed_directories_strict_allowlist(self, tmp_path):
        # When configured, only the granted roots are readable; everything else
        # (even an ordinary temp file elsewhere) is refused.
        from effgen.tools.builtin.data_analysis import DataFrameTool

        good = tmp_path / "data.csv"
        good.write_text("a,b\n1,2\n")
        tool = DataFrameTool(allowed_directories=[str(tmp_path)])
        assert (await tool.execute(file_path=str(good), operation="head", n=1)).success

        other = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
        other.write("a,b\n1,2\n")
        other.close()
        try:
            res = await tool.execute(file_path=other.name, operation="head")
            assert res.success is False
            assert "outside" in str(res.error).lower()
        finally:
            os.unlink(other.name)

    def test_confine_path_helper(self, tmp_path):
        from effgen.tools.builtin._fs import (
            PathNotAllowedError,
            confine_path,
            normalize_allowed_dirs,
        )

        # default (deny-list) posture: ordinary file ok, sensitive refused
        inside = tmp_path / "ok.txt"
        inside.write_text("x")
        assert confine_path(str(inside), None) == inside.resolve()
        with pytest.raises(PathNotAllowedError):
            confine_path("/etc/passwd", None)
        # a symlink that resolves into a denied tree is refused
        link = tmp_path / "escape"
        link.symlink_to("/etc/passwd")
        with pytest.raises(PathNotAllowedError):
            confine_path(str(link), None)

        # strict allow-list posture
        roots = normalize_allowed_dirs([str(tmp_path)])
        assert confine_path(str(inside), roots) == inside.resolve()
        with pytest.raises(PathNotAllowedError):
            confine_path("/etc/passwd", roots)
        with pytest.raises(PathNotAllowedError):
            confine_path(str(link), roots)


# --------------------------------------------------------------------------- #
# Filesystem confinement — filename- and content-based secrets-file heuristic #
# --------------------------------------------------------------------------- #
class TestFilesystemSecretsFileHeuristic:
    """A tool built on ``_fs.py`` must refuse a credentials file even when its
    name doesn't match a known sensitive path, and even when it has been
    renamed to an innocuous extension."""

    def test_filename_pattern_blocks_dotenv_and_credentials_files(self, tmp_path):
        from effgen.tools.builtin._fs import PathNotAllowedError, confine_path

        for name in (".env", ".env.production", "credentials", "id_rsa", ".netrc"):
            f = tmp_path / name
            f.write_text("a,b\n1,2\n")
            with pytest.raises(PathNotAllowedError):
                confine_path(str(f), None)

    def test_content_sniff_blocks_renamed_dotenv(self, tmp_path):
        from effgen.tools.builtin._fs import check_content_not_credentials

        decoy = (
            "OPENAI_API_KEY=sk-decoyFAKEKEY1234567890abcdEFGH\n"
            "DATABASE_URL=postgres://user:pass@localhost:5432/db\n"
            "AWS_SECRET_ACCESS_KEY=decoyFAKEsecretvalueNOTREAL1234567890\n"
            "DEBUG=true\n"
        )
        with pytest.raises(ValueError, match="credentials file"):
            check_content_not_credentials(decoy, source="decoy.csv")

    def test_content_sniff_leaves_ordinary_data_alone(self):
        from effgen.tools.builtin._fs import check_content_not_credentials

        # Must not be tripped by ordinary tabular/text content, including a
        # stray line or two that happens to look "KEY=VALUE"-shaped.
        check_content_not_credentials("name,age,city\nAlice,30,NYC\nBob,25,LA\n")
        check_content_not_credentials("TOTAL=42\nsome other prose line here\n")

    async def test_data_analysis_refuses_dotenv_renamed_to_csv(self, tmp_path):
        """A decoy .env-shaped file renamed to .csv must not be read in full
        by DataFrameTool."""
        from effgen.tools.builtin.data_analysis import DataFrameTool

        decoy = tmp_path / "export.csv"
        decoy.write_text(
            "OPENAI_API_KEY=sk-decoyFAKEKEY1234567890abcdEFGH\n"
            "DATABASE_URL=postgres://user:pass@localhost:5432/db\n"
            "AWS_SECRET_ACCESS_KEY=decoyFAKEsecretvalueNOTREAL1234567890\n"
            "DEBUG=true\n"
        )
        res = await DataFrameTool().execute(file_path=str(decoy), operation="head")
        assert res.success is False
        assert "credentials file" in str(res.error).lower()

        # An ordinary CSV in the same directory still loads normally.
        good = tmp_path / "real_data.csv"
        good.write_text("name,age,city\nAlice,30,NYC\nBob,25,LA\n")
        res = await DataFrameTool().execute(file_path=str(good), operation="head")
        assert res.success is True


# --------------------------------------------------------------------------- #
# PythonREPL — the sandbox toggle is developer-controlled, never model-facing  #
# --------------------------------------------------------------------------- #
class TestPythonREPLSandboxToggle:
    def test_restricted_mode_not_in_model_schema(self):
        repl = PythonREPL()
        props = repl.metadata.to_json_schema()["parameters"]["properties"]
        # The safety toggle must never appear in the schema a model sees.
        assert "restricted_mode" not in props
        # ...but the normal parameters still do.
        assert "code" in props

    def test_restricted_mode_absent_from_react_prompt(self):
        from effgen.prompts.tool_prompt_generator import ToolPromptGenerator

        text = ToolPromptGenerator([PythonREPL()]).generate_tools_section(verbose=True)
        assert "restricted_mode" not in text

    async def test_model_supplied_unrestricted_is_ignored_by_default(self):
        # A prompt-injected model emitting restricted_mode=False must NOT escape:
        # without a developer opt-in the call stays sandboxed.
        repl = PythonREPL(timeout=10)
        await repl.initialize()
        try:
            res = await repl.execute(code="import os\nos.getpid()", restricted_mode=False)
            assert res.success is False
            assert "not allowed" in str(res.output.get("error")).lower()
        finally:
            await repl.cleanup()

    async def test_developer_optin_enables_unrestricted(self):
        repl = PythonREPL(timeout=10, allow_unrestricted=True)
        await repl.initialize()
        try:
            res = await repl.execute(code="import os\nos.getpid()", restricted_mode=False)
            assert res.success
            # restricted_mode=True is still honored even when opted in.
            blocked = await repl.execute(code="import os", restricted_mode=True)
            assert blocked.success is False
        finally:
            await repl.cleanup()

    async def test_env_optin_enables_unrestricted(self, monkeypatch):
        monkeypatch.setenv("EFFGEN_REPL_ALLOW_UNRESTRICTED", "1")
        repl = PythonREPL(timeout=10)
        await repl.initialize()
        try:
            res = await repl.execute(code="import os\nos.getpid()", restricted_mode=False)
            assert res.success
        finally:
            await repl.cleanup()


# --------------------------------------------------------------------------- #
# BashTool — secret env-strip is exhaustive; secret files are refused          #
# --------------------------------------------------------------------------- #
class TestBashSecretProtection:
    def test_every_provider_key_is_stripped(self, monkeypatch):
        from effgen.tools.builtin.bash_tool import BashTool

        keys = [
            "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY",
            "GROQ_API_KEY", "CEREBRAS_API_KEY", "TOGETHER_API_KEY",
            "FIREWORKS_API_KEY", "REPLICATE_API_TOKEN", "HF_TOKEN",
        ]
        for k in keys:
            monkeypatch.setenv(k, "DUMMY-secret-value-0123456789")
        # A provider added later (generic-pattern coverage, not in any list).
        monkeypatch.setenv("FUTUREPROVIDER_API_KEY", "DUMMY-secret-0123456789")
        monkeypatch.setenv("MY_DB_PASSWORD", "DUMMY-pw")
        safe = BashTool()._get_safe_env()
        for k in keys + ["FUTUREPROVIDER_API_KEY", "MY_DB_PASSWORD"]:
            assert k not in safe, f"{k} leaked into bash env"
        # Non-secret vars survive.
        monkeypatch.setenv("PATH_OK_VAR", "value")
        assert "PATH_OK_VAR" in BashTool()._get_safe_env()

    def test_secret_files_are_refused(self):
        from effgen.tools.builtin.bash_tool import BashTool

        bt = BashTool()
        for cmd in [
            "cat .env", "cat ./.env", "head .env.local",
            "cat ~/.ssh/id_rsa", "cat ~/.aws/credentials",
            "cat /home/u/id_ed25519", "cat ~/.netrc",
        ]:
            safe, _ = bt._is_command_safe(cmd)
            assert safe is False, f"{cmd!r} should be refused"
        # Ordinary commands still allowed.
        for cmd in ["ls -la", "echo hi", "cat README.md", "wc -l setup.py"]:
            safe, _ = bt._is_command_safe(cmd)
            assert safe is True, f"{cmd!r} should be allowed"

    async def test_cat_dotenv_blocked_at_execute(self):
        from effgen.tools.builtin.bash_tool import BashTool

        bt = BashTool()
        res = await bt.execute(command="cat .env")
        assert res.success is False

    def test_bash_description_states_it_is_not_a_sandbox(self):
        from effgen.presets.registry import PRESETS
        from effgen.tools.builtin.bash_tool import BashTool

        desc = BashTool().description.lower()
        assert "not a sandbox" in desc
        assert "safely" not in desc
        assert "bash" not in PRESETS["general"].tool_names
        assert "bash" in PRESETS["coding"].tool_names


# --------------------------------------------------------------------------- #
# BashTool — obfuscated secret-file reads (quote-concatenation, dotfile globs,#
# decode-then-execute) are refused, not just the literal `cat .env` case.     #
# --------------------------------------------------------------------------- #
class TestBashSecretBypassHardening:
    def _bt(self):
        return BashTool()

    @pytest.mark.parametrize(
        "cmd",
        [
            # Adjacent quoted-string concatenation builds ".env" without the
            # literal substring appearing in the raw command text.
            'F=".e""nv"; cat "$F"',
            "F='.e''nv'; cat \"$F\"",
        ],
    )
    def test_quote_concatenation_bypass_blocked(self, cmd):
        safe, reason = self._bt()._is_command_safe(cmd)
        assert safe is False, f"{cmd!r} should be refused"
        assert "credential" in reason.lower()

    @pytest.mark.parametrize(
        "cmd",
        ["cat .e*", "cat .en?", "cat .ssh*", "head .netrc*"],
    )
    def test_dotfile_glob_bypass_blocked(self, cmd):
        safe, reason = self._bt()._is_command_safe(cmd)
        assert safe is False, f"{cmd!r} should be refused"

    def test_decode_then_execute_bypass_blocked(self):
        cmd = (
            "echo Y2F0IC5lbnY= | base64 -d > /tmp/_x.sh && bash /tmp/_x.sh; "
            "rm -f /tmp/_x.sh"
        )
        safe, reason = self._bt()._is_command_safe(cmd)
        assert safe is False, f"{cmd!r} should be refused"
        assert "decode" in reason.lower()

    def test_ordinary_globs_and_decodes_still_allowed(self):
        bt = self._bt()
        for cmd in [
            "ls -la",
            "cat *.py",
            "wc -l *.py",
            "echo aGVsbG8= | base64 -d",  # decode alone, no execution
            "ls .git*",  # dotfile glob but no wildcard glued to the leading dot run — still a dotfile
        ]:
            safe, reason = bt._is_command_safe(cmd)
            # "ls .git*" is a dotfile glob and intentionally denied (same
            # conservative posture as the literal .ssh/.env denials); the
            # rest must stay allowed.
            if cmd == "ls .git*":
                assert safe is False
            else:
                assert safe is True, f"{cmd!r} should be allowed: {reason}"

    async def test_quote_concatenation_blocked_at_execute(self):
        bt = self._bt()
        res = await bt.execute(command='F=".e""nv"; cat "$F"')
        assert res.success is False


# --------------------------------------------------------------------------- #
# FileOperations — credential-shaped filenames/content are refused even      #
# inside an allowed directory (parity with the _fs.py deny-list).             #
# --------------------------------------------------------------------------- #
class TestFileOpsCredentialAwareness:
    async def _tool_dir(self):
        d = tempfile.mkdtemp(prefix="p12_fo_")
        return d, FileOperations(allowed_directories=[d])

    async def test_credential_shaped_filename_refused(self):
        d, tool = await self._tool_dir()
        path = os.path.join(d, "credentials")
        with open(path, "w") as f:
            f.write("plain text, not even secret-shaped content\n")
        res = await tool.execute(operation="read", path=path)
        assert res.success is False
        assert "credential" in res.output["message"].lower()

    async def test_renamed_dotenv_content_refused_despite_allowed_extension(self):
        d, tool = await self._tool_dir()
        path = os.path.join(d, "export.csv")
        with open(path, "w") as f:
            f.write(
                "OPENAI_API_KEY=sk-decoyFAKEKEY1234567890abcdEFGH\n"
                "DATABASE_URL=postgres://user:pass@localhost:5432/db\n"
                "AWS_SECRET_ACCESS_KEY=decoyFAKEsecretvalueNOTREAL1234567890\n"
            )
        res = await tool.execute(operation="read", path=path)
        assert res.success is False
        assert "credential" in res.output["message"].lower()

    async def test_credential_file_excluded_from_directory_listing(self):
        d, tool = await self._tool_dir()
        with open(os.path.join(d, "credentials"), "w") as f:
            f.write("secret-shaped\n")
        with open(os.path.join(d, "notes.txt"), "w") as f:
            f.write("ordinary\n")
        res = await tool.execute(operation="list", path=d)
        names = [e["name"] for e in res.output["data"]]
        assert "credentials" not in names
        assert "notes.txt" in names

    async def test_ordinary_file_still_readable(self):
        d, tool = await self._tool_dir()
        path = os.path.join(d, "report.txt")
        with open(path, "w") as f:
            f.write("just a normal report, nothing sensitive here\n")
        res = await tool.execute(operation="read", path=path)
        assert res.success is True
        assert res.output["data"] == "just a normal report, nothing sensitive here\n"
