"""Outbound long-poll transport: the user never exposes an inbound port."""

import importlib
import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from .session import AgentProcess


class ClientError(RuntimeError):
    def __init__(self, code):
        self.code = code
        super().__init__(f"Platform HTTP {code}")


def connect(connection_file, factory, stop=None, allow_http=False):
    config = json.loads(Path(connection_file).read_text(encoding="utf-8"))
    server, token = config["server"].rstrip("/"), config["token"]
    parsed = urllib.parse.urlsplit(server)
    if (
        parsed.scheme not in ("http", "https")
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Invalid platform URL")
    if (
        parsed.scheme == "http"
        and parsed.hostname not in ("localhost", "127.0.0.1", "::1")
        and not allow_http
    ):
        raise ValueError(
            "Remote connections require HTTPS; --allow-http is only for controlled private-network testing"
        )
    stop = stop or threading.Event()
    sessions = {}
    active_from_heartbeat = None
    heartbeat_lock = threading.Lock()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def call(path, body):
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        if len(data) > 8 * 1024 * 1024:
            raise ValueError("Agent response exceeds 8 MiB")
        req = urllib.request.Request(
            server + path,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + token,
            },
        )
        try:
            with opener.open(req, timeout=25) as response:
                data = response.read(8 * 1024 * 1024 + 1)
        except urllib.error.HTTPError as error:
            raise ClientError(error.code) from None
        if len(data) > 8 * 1024 * 1024:
            raise ValueError("Platform request exceeds 8 MiB")
        return json.loads(data)

    def heartbeat():
        nonlocal active_from_heartbeat
        while not stop.wait(8):
            try:
                reply = call("/api/client/heartbeat", {})
                with heartbeat_lock:
                    active_from_heartbeat = set(reply.get('active_jobs', []))
            except Exception:
                pass

    call("/api/client/heartbeat", {})
    threading.Thread(target=heartbeat, daemon=True).start()
    print(
        "Agent connected to "
        + server
        + ". Select this Agent in the website to start an evaluation.",
        flush=True,
    )
    try:
        while not stop.is_set():
            try:
                batch = call("/api/client/poll", {})
                active = set(batch["active_jobs"])
                for jid in list(sessions):
                    if jid not in active:
                        previous = sessions.pop(jid)
                        if hasattr(previous, "close"):
                            previous.close()
                        del previous
                command = batch.get("command")
                if not command:
                    continue
                jid, method, payload = (
                    command["job_id"],
                    command["method"],
                    command["payload"],
                )
                try:
                    if method == "reset":
                        previous = sessions.pop(jid, None)
                        if previous is not None and hasattr(previous, "close"):
                            previous.close()
                        del previous
                        def cancelled(jid=jid):
                            with heartbeat_lock:
                                return stop.is_set() or (active_from_heartbeat is not None and jid not in active_from_heartbeat)
                        # A stale heartbeat may precede this freshly assigned job.
                        with heartbeat_lock:
                            if active_from_heartbeat is not None:
                                active_from_heartbeat.add(jid)
                        agent = AgentProcess(factory, server, token, jid, cancelled)
                        agent.reset(payload)
                        sessions[jid] = agent
                        result = {"ok": True}
                    elif method == "step":
                        result = sessions[jid].step(payload)
                        if (
                            not isinstance(result, dict)
                            or not isinstance(result.get("action"), str)
                            or not result["action"].strip()
                        ):
                            raise ValueError("Agent must return an action string")
                    elif method == "close":
                        previous = sessions.pop(jid, None)
                        if previous is not None and hasattr(previous, "close"):
                            previous.close()
                        del previous
                        result = {"ok": True}
                    else:
                        raise ValueError("Unknown method")
                    response = {"ok": True, "result": result}
                except Exception as exc:
                    # Do not transmit arbitrary exception text, which can contain model credentials.
                    print(
                        f"Agent command failed: {type(exc).__name__}; inspect your Agent's local logs",
                        flush=True,
                    )
                    response = {"ok": False, "error_type": type(exc).__name__}
                while not stop.is_set():
                    try:
                        call("/api/client/result/" + command["id"], response)
                        break
                    except ClientError as exc:
                        if exc.code == 409:
                            print(
                                "Result expired or already accepted; action will not be repeated.",
                                flush=True,
                            )
                            break
                        if exc.code in (401, 403):
                            raise
                        print("Result delivery temporarily unavailable; retrying the same result.", flush=True)
                        stop.wait(1)
                    except OSError:
                        # Retain the completed decision until acknowledged; never rerun it.
                        stop.wait(1)
            except ClientError as exc:
                if exc.code in (401, 403):
                    print(
                        "Connection revoked or invalid. Download a new connection file.",
                        flush=True,
                    )
                    break
                print(str(exc) + "; reconnecting", flush=True)
                stop.wait(3)
            except OSError:
                print(
                    "Platform unreachable; reconnecting. No action will be re-executed automatically.",
                    flush=True,
                )
                stop.wait(3)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        for agent in sessions.values():
            if hasattr(agent, "close"):
                try:
                    agent.close()
                except Exception:
                    pass


def run_from_args(args):
    module, name = args.agent.split(":", 1)
    factory = getattr(importlib.import_module(module), name)
    connect(args.connection_file, factory, allow_http=args.allow_http)
