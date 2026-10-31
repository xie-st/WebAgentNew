"""G1: necessary-context views, the final-send check and the ``context`` section.

These are pure program-rule tests: no model, no page, no network. They prove the
four necessary regions are rendered deterministically, measured byte-for-byte on
the final prompt, and never dropped or silently shortened when the budget is
exceeded.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from webagent.budget import (
    BudgetLimits,
    HeuristicTokenCounter,
    render_observation,
    serialized_request_bytes,
)
from webagent.clock import FakeClock
from webagent.config import (
    ConfigError,
    ContextSettings,
    OfficialBehaviors,
    load_config,
    parse_config,
)
from webagent.context import (
    ContextBudgetExceeded,
    ContextProtectionError,
    ContextProtector,
    SectionKind,
    build_necessary_sections,
    effects_block,
    ledger_block,
)
from webagent.context.guard import (
    VERDICT_MISSING,
    VERDICT_OK,
    VERDICT_OVER_LIMIT,
    describe,
    evaluate,
)
from webagent.context.sections import (
    EFFECTS_HEADER,
    LEDGER_HEADER,
    NO_EFFECTS,
    NO_LEDGER,
    byte_length,
)
from webagent.protocol import Observation
from webagent.session import Session
from webagent.state.records import EffectStatus
from webagent.state.store import StateStore
from webagent.state.tools import StateToolRuntime

CONFIGS = Path(__file__).parents[2] / "configs"
RUN_ID = "context-run-1"


# ------------------------------------------------------------------- fixtures
def _raw_config() -> dict:
    return yaml.safe_load((CONFIGS / "official.yaml").read_text(encoding="utf-8"))


def _store(tmp_path: Path, task: str = "book the room for two nights"):
    store = StateStore(tmp_path / "runtime", RUN_ID, clock=FakeClock())
    store.load_or_initialize(task)
    return store


def _add_goal(store: StateStore) -> None:
    runtime = StateToolRuntime(store)
    runtime.update(
        {
            "base_revision": store.state.revision,
            "plan_version": store.state.plan_version,
            "branch": "root",
            "goals": [
                {
                    "id": "g1",
                    "requirement": "the reservation shows two nights and the agreed room",
                    "acceptance_criteria": [
                        "the visible date range is the agreed one",
                        "the room name is the agreed one",
                    ],
                    "status": "running",
                }
            ],
        }
    )


def _session(
    tmp_path: Path, observation: Observation, task: str = "book the room for two nights"
) -> Session:
    session = Session(RUN_ID, runtime_dir=tmp_path / "runtime", clock=FakeClock())
    session.start_task(task)
    session.ingest_observation(observation)
    return session


def _protector(
    tmp_path: Path,
    observation: Observation,
    store: StateStore | None,
    *,
    max_input_tokens: int = 24_000,
    ledger_max_chars: int = 2_000,
    effects_max_chars: int = 1_200,
) -> ContextProtector:
    session = _session(tmp_path, observation)
    return ContextProtector(
        session,
        store,
        settings=ContextSettings(
            enabled=True,
            max_input_tokens=max_input_tokens,
            ledger_max_chars=ledger_max_chars,
            effects_max_chars=effects_max_chars,
        ),
        counter=HeuristicTokenCounter(),
        limits=BudgetLimits(),
        branch="root",
    )


def _prompt(task: str, observation: Observation, ledger, effects, tail: str = "") -> str:
    """A prompt shaped like the official send: task, AX view, then the blocks."""

    text = (
        "Question: "
        + task
        + "\n\n"
        + render_observation(observation)
        + "\n\n"
        + ledger.text
        + "\n\n"
        + effects.text
    )
    return text + tail


# --------------------------------------------------------------------- config
def test_context_section_is_off_by_default_and_leaves_existing_keys_alone() -> None:
    official = load_config(CONFIGS / "official.yaml")
    assert official.context == ContextSettings(
        enabled=False,
        max_input_tokens=24_000,
        max_request_bytes=1024 * 1024,
        ledger_max_chars=2_000,
        effects_max_chars=1_200,
    )
    assert official.summary()["context"] == {
        "enabled": False,
        "max_input_tokens": 24_000,
        "max_request_bytes": 1024 * 1024,
        "ledger_max_chars": 2_000,
        "effects_max_chars": 1_200,
    }
    # The new top-level section changed no existing key.
    assert official.budget.input_tokens == 24_000
    assert official.budget_limits().input_tokens == 24_000
    assert official.limits.max_request_bytes == 1024 * 1024
    assert official.official.behaviors == OfficialBehaviors(
        verification=True, state_tools=True, recovery=False
    )
    assert official.verification.max_semantic_calls == 20
    assert official.recovery.stagnation_window == 6


def test_context_section_parses_explicit_values_and_rejects_bad_ones() -> None:
    raw = _raw_config()
    raw["context"] = {
        "enabled": True,
        "max_input_tokens": 4096,
        "max_request_bytes": 65536,
        "ledger_max_chars": 256,
        "effects_max_chars": 128,
    }
    parsed = parse_config(raw)
    assert parsed.context.enabled is True
    assert parsed.context.max_input_tokens == 4096
    assert parsed.summary()["context"]["enabled"] is True

    for broken in (
        {"enabled": "yes"},
        {"max_input_tokens": 0},
        {"max_request_bytes": -1},
        {"ledger_max_chars": 0},
        {"unknown": True},
    ):
        raw = _raw_config()
        raw["context"] = dict(broken)
        with pytest.raises(ConfigError):
            parse_config(raw)

    # The project final check may not claim a looser ceiling than the budget the
    # native compaction path receives.
    raw = _raw_config()
    raw["budget"]["input_tokens"] = 8_000
    raw["context"] = {"enabled": True, "max_input_tokens": 24_000}
    with pytest.raises(ConfigError):
        parse_config(raw)


# ---------------------------------------------------------------- ledger view
def test_ledger_view_is_deterministic_and_carries_the_mandatory_core(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    _add_goal(store)
    first = ledger_block(store, max_chars=2_000, branch="root")
    second = ledger_block(store, max_chars=2_000, branch="root")
    assert first.text == second.text
    assert first.to_dict() == second.to_dict()
    assert first.kind is SectionKind.LEDGER
    assert first.text.startswith(LEDGER_HEADER)
    assert "revision=1 plan_version=0" in first.text
    assert "- g1 [running]" in first.text
    assert "criteria=2" in first.text
    assert "two nights and the agreed room" in first.text
    # The block kept through the real send is exactly these bytes.
    assert first.byte_length == len(first.text.encode("utf-8"))


def test_ledger_view_drops_only_optional_detail_at_its_size_bound(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    _add_goal(store)
    for index in range(40):
        StateToolRuntime(store).update(
            {
                "base_revision": store.state.revision,
                "plan_version": store.state.plan_version,
                "branch": "root",
                "facts": [
                    {
                        "id": f"f{index:02d}",
                        "object": "reservation",
                        "field": f"field-{index:02d}",
                        "value": "x" * 60,
                        "kind": "inferred",
                        "source": "page",
                    }
                ],
            }
        )
    wide = ledger_block(store, max_chars=20_000, branch="root")
    narrow = ledger_block(store, max_chars=400, branch="root")
    # The mandatory part is identical; only the optional fact list shrank.
    assert narrow.required == wide.required
    assert narrow.trimmed_optional_chars > 0
    assert narrow.optional_chars < wide.optional_chars
    assert len(narrow.text) <= 400
    assert "- g1 [running]" in narrow.required


def test_ledger_view_without_a_ledger_says_so(tmp_path: Path) -> None:
    block = ledger_block(None, max_chars=500)
    assert block.text == LEDGER_HEADER + "\n" + NO_LEDGER
    assert block.optional == ""
    assert block.trimmed_optional_chars == 0


# --------------------------------------------------------------- effects view
def test_effects_view_lists_only_unconfirmed_records(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.prepare_effect(
        action_id="a-confirmed",
        target="click [1]",
        object_id="reservation",
        branch_id="root",
        observation_id="obs-1",
        note="save pressed",
    )
    store.confirm_effect("a-confirmed", status=EffectStatus.CONFIRMED)
    store.prepare_effect(
        action_id="a-pending",
        target="click [2]",
        object_id="reservation",
        branch_id="root",
        observation_id="obs-2",
        note="confirm pressed",
    )
    store.prepare_effect(
        action_id="a-unknown",
        target="type [3]",
        object_id="note",
        branch_id="root",
        observation_id="obs-3",
        note="no feedback returned",
    )
    store.confirm_effect("a-unknown", status=EffectStatus.UNKNOWN)

    block = effects_block(store, max_chars=1_200)
    assert block.text.startswith(EFFECTS_HEADER)
    assert "a-pending" in block.text and "status=pending" in block.text
    assert "a-unknown" in block.text and "status=unknown" in block.text
    assert "a-confirmed" not in block.text
    assert block.optional == ""

    empty = effects_block(_store(tmp_path / "empty"), max_chars=500)
    assert NO_EFFECTS in empty.text
    assert effects_block(None, max_chars=500).text.endswith(NO_EFFECTS)


# ---------------------------------------------------------------- final check
def test_complete_final_send_passes_and_every_region_is_byte_complete(
    tmp_path: Path, make_observation
) -> None:
    observation = make_observation(run_id=RUN_ID, ax_text="[1] textbox 'Nights'\n[2] button 'Book'")
    store = _store(tmp_path)
    _add_goal(store)
    ledger = ledger_block(store, max_chars=2_000, branch="root")
    effects = effects_block(store, max_chars=1_200)
    sections = build_necessary_sections(
        task="book the room for two nights",
        observation=observation,
        ledger=ledger,
        effects=effects,
    )
    prompt = _prompt("book the room for two nights", observation, ledger, effects)

    report = evaluate(
        prompt=prompt,
        sections=sections,
        limits=BudgetLimits(input_tokens=24_000),
        counter=HeuristicTokenCounter(),
    )
    assert report.verdict == VERDICT_OK
    assert report.mandatory_intact and not report.over_limit
    assert [check.kind for check in report.sections] == list(SectionKind)
    for check in report.sections:
        assert check.byte_complete
        assert check.matched_bytes == check.expected_bytes
    assert report.section(SectionKind.LEDGER).matched_bytes == ledger.byte_length
    # The same inputs always produce the same report.
    again = evaluate(
        prompt=prompt,
        sections=sections,
        limits=BudgetLimits(input_tokens=24_000),
        counter=HeuristicTokenCounter(),
    )
    assert again.to_dict() == report.to_dict()
    assert "ledger=" in describe(report)


def test_final_send_with_a_lost_region_is_refused_not_accepted(
    tmp_path: Path, make_observation
) -> None:
    observation = make_observation(run_id=RUN_ID)
    store = _store(tmp_path)
    _add_goal(store)
    protector = _protector(tmp_path, observation, store)
    ledger, effects = protector.current_blocks()
    task = "book the room for two nights"
    healthy = _prompt(task, observation, ledger, effects, tail="\n\nhistoric tail")
    assert protector.check_final_send(healthy).mandatory_intact

    dropped = healthy.replace(ledger.text, "LEDGER VIEW (trimmed away)")
    with pytest.raises(ContextProtectionError) as raised:
        protector.check_final_send(dropped)
    report = raised.value.report
    assert report is not None and not report.mandatory_intact
    assert report.verdict == VERDICT_MISSING
    assert "ledger" in report.missing
    assert report.section(SectionKind.TASK).byte_complete
    assert report.section(SectionKind.OBSERVATION).byte_complete


def test_overlong_input_still_reports_all_four_regions_byte_complete(
    tmp_path: Path, make_observation
) -> None:
    observation = make_observation(run_id=RUN_ID, ax_text="[1] button 'Book'\n" + ("long ax line\n" * 4_000))
    store = _store(tmp_path)
    _add_goal(store)
    protector = _protector(tmp_path, observation, store, max_input_tokens=512)
    ledger, effects = protector.current_blocks()
    task = "book the room for two nights"
    prompt = _prompt(task, observation, ledger, effects)

    with pytest.raises(ContextBudgetExceeded) as raised:
        protector.check_final_send(prompt)
    report = raised.value.report
    assert report is not None
    assert report.verdict == VERDICT_OVER_LIMIT
    assert report.over_limit and report.mandatory_intact
    assert [check.kind for check in report.sections] == list(SectionKind)
    for check in report.sections:
        # Byte-complete: every protected byte is found in the prompt that was
        # measured, even though the request is refused for exceeding the budget.
        assert check.present and check.byte_complete
        assert check.matched_bytes == check.expected_bytes
    assert report.section(SectionKind.OBSERVATION).expected_bytes == byte_length(
        render_observation(observation)
    )
    assert report.tokens > report.max_input_tokens
    assert report.request_bytes > report.protected_bytes
    # Nothing was silently shortened to fit: the blocks and the AX text are whole.
    assert ledger.text in prompt and effects.text in prompt
    assert observation.ax_text in prompt


def test_shortened_optional_detail_is_reported_not_hidden(
    tmp_path: Path, make_observation
) -> None:
    observation = make_observation(run_id=RUN_ID)
    store = _store(tmp_path)
    _add_goal(store)
    for index in range(30):
        StateToolRuntime(store).update(
            {
                "base_revision": store.state.revision,
                "plan_version": store.state.plan_version,
                "branch": "root",
                "facts": [
                    {
                        "id": f"f{index:02d}",
                        "object": "reservation",
                        "field": f"field-{index:02d}",
                        "value": "y" * 50,
                        "kind": "inferred",
                        "source": "page",
                    }
                ],
            }
        )
    protector = _protector(
        tmp_path, observation, store, ledger_max_chars=400
    )
    ledger, effects = protector.current_blocks()
    assert ledger.trimmed_optional_chars > 0
    prompt = _prompt("book the room for two nights", observation, ledger, effects)
    report = protector.check_final_send(prompt)
    assert report.mandatory_intact
    assert report.section(SectionKind.LEDGER).byte_complete
    assert report.trimmed_optional_chars == ledger.trimmed_optional_chars
    assert report.trimmed_optional_chars > 0
    assert report.verdict == VERDICT_OK


def test_protector_turn_note_is_stable_for_the_same_state(
    tmp_path: Path, make_observation
) -> None:
    observation = make_observation(run_id=RUN_ID)
    store = _store(tmp_path)
    _add_goal(store)
    protector = _protector(tmp_path, observation, store)
    first = protector.turn_note()
    second = protector.turn_note()
    assert first == second
    assert first.startswith(LEDGER_HEADER)
    assert EFFECTS_HEADER in first
    assert serialized_request_bytes([{"role": "user", "content": first}]) > 0
