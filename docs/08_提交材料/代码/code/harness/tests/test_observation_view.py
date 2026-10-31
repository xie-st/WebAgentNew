"""Dense real pages must reach the next model turn under the unchanged budget."""

from __future__ import annotations

import dataclasses
import json
import re

import pytest
from effgen.core.thread_budget import count_text_tokens
from effgen.models.base import ModelType

from webagent.budget import render_observation
from webagent.config import AppConfig
from webagent.effgen_bridge import DriverStatus, create_driver
from webagent.errors import ContextOverflow
from webagent.events import read_events
from webagent.models.fake import ScriptedModel
from webagent.observation_view import render_browser_view
from webagent.protocol import ElementRef, Feedback, FeedbackStatus, Observation
from webagent.session import Session


def dense_observation(index=0, action=None):
    oid = f"obs-{index}"
    elements = {}
    rows = []
    for i in range(1, 115):
        number = str(i)
        role = "textbox" if i == 2 else "link"
        name = "搜索笔记" if i == 2 else f"笔记分享真实体验 {i}"
        localization = (
            {"value": "mate90", "css_path": "#search"} if i == 2 else
            {"href": f"https://fixture.local/explore/{i}?token=" + "a7BcD9eF01" * 16}
        )
        elements[number] = ElementRef(
            page_id="page-1", observation_id=oid, number=number,
            role=role, name=name, localization=localization,
        )
        row = f"[{number}] role={role} name={json.dumps(name, ensure_ascii=False)}"
        for key, value in localization.items():
            if key != "css_path":
                row += f" {key}={json.dumps(value)}"
        rows.append(row)
    feedback = None if action is None else Feedback(
        status=FeedbackStatus.OBSERVED, action_id=action.action_id,
        observation_id=oid, message="typed without clearing" if index == 2 else "pressed Escape",
    )
    return Observation(
        run_id="dense-page", page_id="page-1", observation_id=oid,
        url="https://fixture.local/explore",
        ax_text="TITLE: Dense feed\nTEXT:\n- 浏览笔记\nINTERACTIVE ELEMENTS (114):\n" + "\n".join(rows),
        elements=elements, previous_action_id=None if action is None else action.action_id,
        feedback=feedback,
    )


def action(tool, args):
    return f"Thought: continue.\nAction: {tool}\nAction Input: {json.dumps(args)}"


def test_projection_preserves_controls_bindings_feedback_and_raw_evidence():
    observation = dense_observation()
    before = observation.to_local_dict()
    view = render_browser_view(observation, max_tokens=4000, count_tokens=count_text_tokens)
    assert view.tokens <= 4000 < view.original_tokens
    assert 0 < view.shown_elements < view.total_elements
    assert 'role=textbox name="搜索笔记" value="mate90"' in view.text
    assert "PARTIAL OBSERVATION" in view.text
    for number in re.findall(r"^\[(\d+)\]", view.text, re.MULTILINE):
        assert number in observation.elements
    assert observation.to_local_dict() == before


def test_small_observations_keep_existing_prompt(make_observation):
    observation = make_observation()
    view = render_browser_view(observation, max_tokens=4000, count_tokens=count_text_tokens)
    assert view.text == render_observation(observation)


def test_scrolling_changes_projected_controls_even_below_element_capacity():
    observation = dense_observation()
    # 114 elements fit the raw 150-element table but not the prompt view.
    # Scrolling must promote later visible cards, not repeat the first rows.
    elements = {
        n: dataclasses.replace(e, localization={**e.localization, "in_viewport": int(n) >= 110})
        for n, e in observation.elements.items()
    }
    observation = dataclasses.replace(observation, elements=elements)
    view = render_browser_view(observation, max_tokens=1000, count_tokens=count_text_tokens)
    assert "[110] role=link" in view.text
    assert "[114] role=link" in view.text
    assert view.tokens <= 1000


def test_identity_and_feedback_are_never_silently_cut():
    with pytest.raises(ContextOverflow) as caught:
        render_browser_view(dense_observation(), max_tokens=5, count_tokens=count_text_tokens)
    assert caught.value.required_tokens > caught.value.available_tokens


@pytest.mark.parametrize("project", [False, True])
def test_native_loop_dense_page_third_call_regression(tmp_path, project):
    session = Session("dense-page", runtime_dir=tmp_path)
    task = "搜索 mate90；只浏览，不发送评论。"
    session.start_task(task)
    first = dense_observation()
    session.ingest_observation(first)
    model = ScriptedModel([
        action("web_press", {"keys": "Escape"}),
        action("web_type", {"target": "2", "text": "mate90"}),
        action("web_press", {"keys": "Enter"}),
        "Final Answer: replay complete",
    ], model_type=ModelType.OPENAI)
    driver = create_driver(session, model, app_config=AppConfig(profile="real", backend="playwright"))
    if not project:
        driver.runtime.observation_input_tokens = None
    try:
        step = driver.start()
        while step.status is DriverStatus.ACTION and len(model.calls) < 4:
            pending = driver.take_action()
            step = driver.accept_observation(dense_observation(len(model.calls), pending))
        if project:
            assert step.status is DriverStatus.COMPLETED, step.message
            assert len(model.calls) == 4
            assert all(count_text_tokens(prompt) <= 24000 for prompt in model.prompts)
            assert all(task in prompt for prompt in model.prompts)
            assert 'value="mate90"' in model.prompts[2]
            assert session.observation is not None
            assert len(session.observation.elements) == 114
            # web_observe uses the same projection as initial/action results.
            assert count_text_tokens(driver.runtime.observe()) <= 4000
            events = read_events(session.evidence.events_path)
            assert any(e["event_type"] == "observation_projection" for e in events)
        else:
            assert step.status is DriverStatus.ERROR, step.message
            assert "ContextBudgetExceededError" in step.message
            assert len(model.calls) < 3
    finally:
        driver.close()
        session.close()


def test_official_entry_retains_full_observation(tmp_path):
    session = Session("dense-page", runtime_dir=tmp_path)
    session.start_task("read the page")
    observation = dense_observation()
    session.ingest_observation(observation)
    driver = create_driver(session, ScriptedModel([]), app_config=AppConfig(profile="official", backend="official"))
    try:
        assert driver.runtime.observe() == render_observation(observation)
    finally:
        driver.close()
        session.close()


def test_long_run_with_native_calibration_still_compacts_history(tmp_path):
    class CalibratedModel(ScriptedModel):
        def generate(self, prompt, config=None, **kwargs):
            result = super().generate(prompt, config=config, **kwargs)
            return dataclasses.replace(result, metadata={
                **result.metadata, "prompt_tokens": count_text_tokens(prompt) * 2,
            })

    session = Session("dense-page", runtime_dir=tmp_path)
    session.start_task("read the feed without submitting")
    session.ingest_observation(dense_observation())
    model = CalibratedModel([
        *[action("web_press", {"keys": "Escape"}) for _ in range(20)],
        "Final Answer: completed",
    ], model_type=ModelType.OPENAI)
    driver = create_driver(session, model, app_config=AppConfig(profile="real", backend="playwright"))
    try:
        step = driver.start()
        while step.status is DriverStatus.ACTION:
            pending = driver.take_action()
            step = driver.accept_observation(dense_observation(len(model.calls), pending))
        assert step.status is DriverStatus.COMPLETED, step.message
        assert len(model.calls) == 21
        assert max(count_text_tokens(p) * 2 for p in model.prompts) <= 24000
        assert step.response.metadata["context_budget"]["calibration"] == 2.0
    finally:
        driver.close()
        session.close()
