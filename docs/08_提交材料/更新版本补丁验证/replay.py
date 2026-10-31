"""Replay the recorded XHS observations through EffGen, without network calls."""
import dataclasses
import json
from pathlib import Path
import sys
import uuid

from effgen.core.thread_budget import count_text_tokens
from effgen.models.base import ModelType
from webagent.config import load_config
from webagent.effgen_bridge import DriverStatus, create_driver
from webagent.events import read_events
from webagent.models.fake import ScriptedModel
from webagent.protocol import ArtifactSet, Observation
from webagent.session import Session


source = Path(sys.argv[1]).resolve()
output = Path(__file__).resolve().parent
config = load_config("code/harness/configs/real.yaml")
observations = [Observation.from_dict(json.loads(p.read_text(encoding="utf-8")))
                for p in sorted((source / "observations").glob("obs-*.json"))]
task = json.loads((source / "result.json").read_text(encoding="utf-8"))["task"]


def replay(project):
    mode = "projected" if project else "baseline"
    evidence_root = output / f"{mode}-{uuid.uuid4().hex[:8]}"
    session = Session(observations[0].run_id, runtime_dir=evidence_root)
    session.start_task(task)
    session.ingest_observation(dataclasses.replace(observations[0], artifacts=ArtifactSet()))
    model = ScriptedModel([
        'Thought: close popup.\nAction: web_press\nAction Input: {"keys":"Escape"}',
        'Thought: search.\nAction: web_type\nAction Input: {"target":"2","text":"mate90"}',
        'Final Answer: observation replay reached the third model call',
    ], model_type=ModelType.OPENAI)
    driver = create_driver(session, model, app_config=config)
    if not project:
        driver.runtime.observation_input_tokens = None
    try:
        step = driver.start()
        for original in observations[1:]:
            if step.status is not DriverStatus.ACTION:
                break
            action = driver.take_action()
            observation = dataclasses.replace(
                original, previous_action_id=action.action_id,
                feedback=dataclasses.replace(original.feedback, action_id=action.action_id),
                artifacts=ArtifactSet(),
            )
            step = driver.accept_observation(observation)
        events = read_events(session.evidence.events_path)
        return {
            "status": step.status.value, "message": step.message,
            "model_calls": len(model.calls),
            "native_prompt_tokens": [count_text_tokens(p) for p in model.prompts],
            "project_prompt_tokens": [session.budget.counter.count_text(p) for p in model.prompts],
            "projections": [e["result"] for e in events if e["event_type"] == "observation_projection"],
            "input_budget": driver.agent.config.context_budget,
            "evidence_root": str(evidence_root),
        }
    finally:
        driver.close()
        session.close()


result = {"source_run": source.name, "mode": "offline_actual_observation_replay",
          "live_site_or_model_called": False, "baseline": replay(False), "fixed": replay(True)}
assert result["baseline"]["status"] == "error", result
assert "ContextBudgetExceededError" in result["baseline"]["message"], result
assert result["fixed"]["status"] == "completed", result
assert result["fixed"]["model_calls"] == 3, result
assert max(result["fixed"]["native_prompt_tokens"]) <= 24000, result
assert max(result["fixed"]["project_prompt_tokens"]) <= 24000, result
(output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
