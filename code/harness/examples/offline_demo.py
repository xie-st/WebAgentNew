"""Run a deterministic offline session and write an example evidence directory.

This demo uses only the fake site and fake backend. It never calls a model, a
browser or the official platform. Run from the harness directory::

    python examples/offline_demo.py examples/sample_run
"""

from __future__ import annotations

import sys
from pathlib import Path

from webagent.clock import FakeClock
from webagent.protocol import Finish, ToolRequest
from webagent.session import Session
from webagent.testing import FakeBackend, FakeSite
from webagent.tools import guard_requests

HERE = Path(__file__).resolve().parent
SITE = HERE.parent / "tests" / "fixtures" / "fake_site.json"


def main() -> Path:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "sample_run"
    site = FakeSite.from_json(SITE)
    clock = FakeClock()
    session = Session("offline-demo", runtime_dir=output, clock=clock)
    backend = FakeBackend("offline-demo", site, start_url=site.default_url)
    try:
        session.start_task("Type the sample query and open the first result.")
        session.ingest_observation(backend.current_observation)
        for request in (
            ToolRequest("web_type", {"target": "1", "text": "sample query"}),
            ToolRequest("web_click", {"target": "2"}),
            ToolRequest("web_click", {"target": "1"}),
        ):
            guarded = guard_requests([request], session.observation)
            if guarded.action is None:
                raise RuntimeError(f"demo request did not produce an action: {request.tool}")
            action = guarded.action
            session.propose(action)
            session.consume()
            result = backend.execute(action)
            session.ingest_observation(result.observation)
        session.record_model_call(
            request_id="offline-demo-request-1",
            input_tokens=420,
            output_tokens=24,
            token_count_source="estimate:chars_div_4",
            latency_ms=0.0,
            model_name="fake-scripted",
        )
        session.finish(Finish(claim="offline demo complete", message="reached the detail page", reason="completed"))
    finally:
        session.close()
        backend.close()
    run_dir = session.evidence.run_dir
    print(f"example run: {run_dir}")
    print(f"events: {session.evidence.events_path}")
    return run_dir


if __name__ == "__main__":
    main()
