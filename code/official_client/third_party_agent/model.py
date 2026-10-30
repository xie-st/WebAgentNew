"""Contestant inference API: messages in, text out; policy lives on server."""
import contextlib
import contextvars
import json
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass

_context = contextvars.ContextVar('webarena_model_session', default=None)


@dataclass(frozen=True)
class ModelResult:
    text: str
    request_id: str


@contextlib.contextmanager
def session(server, token, job_id, run_id, deadline):
    binding = _context.set((server, token, job_id, run_id, deadline))
    try:
        yield
    finally:
        _context.reset(binding)


def generate(messages):
    config = _context.get()
    if config is None:
        raise RuntimeError('model.generate is available only during an active Agent invocation')
    server, token, jid, rid, deadline = config
    remaining = deadline-time.time()
    if remaining <= 0:
        raise TimeoutError('Task execution deadline reached')
    body = {'job_id': jid, 'run_id': rid, 'request_id': uuid.uuid4().hex, 'messages': messages}
    request = urllib.request.Request(server.rstrip('/')+'/api/client/model', data=json.dumps(body).encode(),
                                    headers={'Content-Type':'application/json','Authorization':'Bearer '+token})
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=remaining) as response:
            raw = response.read(4*1024*1024+1)
        if len(raw)>4*1024*1024:
            raise RuntimeError('Model response exceeds size limit')
        value = json.loads(raw)
        return ModelResult(text=value['text'], request_id=value['request_id'])
    except urllib.error.HTTPError as exc:
        # Supplier metadata and credentials are never included in this error.
        raise RuntimeError(f'Platform model request failed: HTTP {exc.code}; request {body["request_id"]}') from None
    except (OSError, ValueError, KeyError, TypeError):
        raise RuntimeError(f'Platform model transport failed; request {body["request_id"]}') from None
