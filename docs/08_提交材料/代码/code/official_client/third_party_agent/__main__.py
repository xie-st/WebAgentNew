"""Run with: python -m third_party_agent serve / check."""
import argparse
import hmac
import importlib
import json
import os
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VERSION = "1.0"
MAX_BYTES = 8 * 1024 * 1024


def request_json(endpoint, path, payload=None, timeout=0):
    token = os.environ.get("WEBARENA_AGENT_TOKEN", "")
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if token:
        headers["Authorization"] = "Bearer " + token
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(endpoint.rstrip("/") + path, data=body, headers=headers)
    try:
        # Local agent calls must not be routed through a machine's HTTP proxy.
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=None if timeout == 0 else timeout) as response:
            content = response.read(MAX_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Agent HTTP {exc.code} at {path}") from None
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(f"Agent request failed at {path}: {type(exc).__name__}") from None
    if len(content) > MAX_BYTES:
        raise RuntimeError("Agent response exceeds 8 MiB")
    try:
        result = json.loads(content)
    except (ValueError, UnicodeError):
        raise RuntimeError("Agent returned invalid JSON") from None
    if not isinstance(result, dict):
        raise RuntimeError("Agent response must be a JSON object")
    return result


def make_server(host, port, factory):
    sessions = {}
    lock = threading.RLock()
    token = os.environ.get("WEBARENA_AGENT_TOKEN", "")
    if host not in ("127.0.0.1", "localhost", "::1") and not token:
        raise ValueError("Set WEBARENA_AGENT_TOKEN before listening outside localhost")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Do not log observations or credentials.

        def send_json(self, status, body):
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def authorized(self):
            return not token or hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token)

        def do_GET(self):
            if not self.authorized():
                return self.send_json(401, {"error": "unauthorized"})
            if self.path != "/health":
                return self.send_json(404, {"error": "unknown route"})
            self.send_json(200, {"protocol_version": VERSION, "observation_types": ["accessibility_tree"], "action_format": "webarena_id"})

        def do_POST(self):
            if not self.authorized():
                return self.send_json(401, {"error": "unauthorized"})
            self.connection.settimeout(120)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BYTES:
                    return self.send_json(413, {"error": "invalid body length"})
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict) or body.get("protocol_version") != VERSION:
                    raise ValueError("unsupported protocol")
                run_id = body.get("run_id")
                if not isinstance(run_id, str) or not run_id or len(run_id) > 128:
                    raise ValueError("invalid run_id")
                # Serialize calls in this small SDK; use multiple processes for parallel agents.
                with lock:
                    if self.path == "/reset":
                        previous = sessions.pop(run_id, None)
                        if previous is not None and hasattr(previous, "close"):
                            previous.close()
                        agent = factory()
                        agent.reset(body)
                        sessions[run_id] = agent
                        result = {"ok": True}
                    elif self.path == "/step":
                        if run_id not in sessions:
                            return self.send_json(409, {"error": "reset required"})
                        result = sessions[run_id].step(body)
                        if not isinstance(result, dict) or not isinstance(result.get("action"), str) or not result["action"].strip():
                            raise ValueError("step must return a nonempty action string")
                        result = {"action": result["action"]}
                    elif self.path == "/close":
                        agent = sessions.pop(run_id, None)
                        if agent is not None and hasattr(agent, "close"):
                            agent.close()
                        result = {"ok": True}
                    else:
                        return self.send_json(404, {"error": "unknown route"})
                self.send_json(200, result)
            except (ValueError, KeyError, TypeError, UnicodeError):
                self.send_json(400, {"error": "invalid request or agent output"})
            except Exception:
                self.send_json(500, {"error": "agent execution failed; inspect your agent locally"})

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(description="Third-party WebArena Agent SDK (no external dependencies)")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--agent", default="third_party_agent.example:ExampleAgent", help="Python module:class")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    check = sub.add_parser("check")
    check.add_argument("--endpoint", default="http://127.0.0.1:8765")
    outbound = sub.add_parser("connect", help="Connect outbound to a self-hosted evaluation platform")
    outbound.add_argument("--connection-file", required=True)
    outbound.add_argument("--agent", default="my_agent:MyAgent")
    outbound.add_argument("--allow-http", action="store_true")
    args = parser.parse_args()
    if args.command == "connect":
        from .outbound import run_from_args
        run_from_args(args)
        return
    if args.command == "serve":
        module, name = args.agent.split(":", 1)
        factory = getattr(importlib.import_module(module), name)
        server = make_server(args.host, args.port, factory)
        print(f"Agent SDK listening at http://{args.host}:{args.port}; Ctrl+C to stop", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    else:
        import uuid
        run_id = "check-" + uuid.uuid4().hex
        health = request_json(args.endpoint, "/health")
        if health.get("protocol_version") != VERSION or health.get("action_format") != "webarena_id":
            raise SystemExit("Incompatible agent protocol")
        base = {"protocol_version": VERSION, "run_id": run_id}
        try:
            request_json(args.endpoint, "/reset", {**base, "task_id": None})
            output = request_json(args.endpoint, "/step", {**base, "step": 0, "task": "Connection test only. Return stop [ok].", "observation": {"type": "accessibility_tree", "url": "about:blank", "text": "SDK test page"}, "history": [], "action_history": []})
            if not isinstance(output.get("action"), str) or not output["action"].strip():
                raise SystemExit("Agent action must be a nonempty string")
            print("PASS: health/reset/step. JSON transport verified; no browser, website or evaluator used.")
        finally:
            request_json(args.endpoint, "/close", base)


if __name__ == "__main__":
    main()
