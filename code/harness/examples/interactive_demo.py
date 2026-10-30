"""Run the A2 interactive loop offline and write an example evidence directory.

One scripted EffGen model drives the vendored interactive generator; the A1 fake
backend executes each page action and returns its observation. No real model,
browser or platform is contacted. Run from the harness directory::

    python examples/interactive_demo.py examples/sample_run_a2
"""

from __future__ import annotations

import sys
from pathlib import Path

from webagent.clock import FakeClock
from webagent.effgen_bridge import BackendLoopRunner, LoopSettings, create_driver
from webagent.models.fake import ScriptedModel
from webagent.session import Session
from webagent.testing import FakeBackend, FakeSite

HERE = Path(__file__).resolve().parent
SITE = HERE.parent / "tests" / "fixtures" / "fake_site.json"


def main() -> Path:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "sample_run_a2"
    site = FakeSite.from_json(SITE)
    clock = FakeClock()
    session = Session("a2-demo", runtime_dir=output, clock=clock)
    backend = FakeBackend("a2-demo", site, start_url=site.default_url)
    model = ScriptedModel([
        "Thought: Record the task condition first.\n"
        "Action: task_note\nAction Input: {\"note\": \"open the sample result\"}",
        "Thought: Enter the query.\n"
        "Action: web_type\nAction Input: {\"target\": \"1\", \"text\": \"sample\"}",
        "Thought: Submit the search.\n"
        "Action: web_click\nAction Input: {\"target\": \"2\"}",
        "Thought: Open the first result.\n"
        "Action: web_click\nAction Input: {\"target\": \"1\"}",
        "Thought: The detail page is open.\nFinal Answer: opened the sample result",
    ])
    try:
        session.start_task("Type the sample query and open the first result.")
        session.ingest_observation(backend.current_observation)
        driver = create_driver(session, model, settings=LoopSettings())
        step = BackendLoopRunner(driver, backend).run()
        print(f"loop status: {step.status.value}")
        print(f"answer: {step.message}")
    finally:
        session.close()
        backend.close()
    run_dir = session.evidence.run_dir
    print(f"example run: {run_dir}")
    print(f"events: {session.evidence.events_path}")
    return run_dir


if __name__ == "__main__":
    main()
