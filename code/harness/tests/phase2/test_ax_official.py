"""P02: official AX field-name and child-StaticText value adaptation."""

from __future__ import annotations

import json
from pathlib import Path

from webagent.protocol import ElementRef, Observation
from webagent.state.store import StateStore
from webagent.verification.ax_official import (
    child_static_text_value,
    extract_quoted_name,
    find_element_line,
    name_matches,
)
from webagent.verification.checks import (
    DeterministicChecker,
    visible_element,
    visible_value,
)
from webagent.verification.types import (
    VerificationCondition,
    VerificationRequest,
    VerificationStatus,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "official_ax"
APPENDED = "t2_5adwlxvnI am a robot"
CORRECT = "I am a robot"


def _load(name: str) -> Observation:
    data = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return Observation.from_dict(data)


def _condition() -> VerificationCondition:
    return VerificationCondition(
        condition_id="biography",
        description="the visible Biography field equals the requested value",
        check="visible_value_equals",
        expected=CORRECT,
        role="textbox",
        name="Biography",
    )


def test_official_name_extraction_ignores_attribute_suffix() -> None:
    raw = "'Biography' required: False describedby: user_biography_biography_help"
    assert extract_quoted_name(raw) == "Biography"
    assert name_matches(raw, "Biography")
    assert name_matches("Biography", "Biography")
    assert not name_matches(raw, "Formatting help +")
    # A local extractor name without the official quotes still matches.
    assert name_matches("Biography", "Biography")


def test_saved_official_observations_read_the_child_static_text_value() -> None:
    appended = _load("official_399_obs5_appended.json")
    correct = _load("official_399_obs10_correct.json")

    assert visible_element(appended, role="textbox", name="Biography") is not None
    assert visible_value(appended, role="textbox", name="Biography") == APPENDED
    assert visible_value(correct, role="textbox", name="Biography") == CORRECT

    # The 399 samples must be distinguished by the value, not by the "updated"
    # banner: both pages contain the same success alert.
    assert "The biography was updated." in appended.ax_text
    assert "The biography was updated." in correct.ax_text
    assert visible_value(appended, role="textbox", name="Biography") != CORRECT


def test_official_equals_check_fails_the_appended_batch_and_passes_the_correct_one(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "runtime", "run-a")
    store.initialize("edit the biography to say I am a robot")
    checker = DeterministicChecker(store)
    condition = _condition()

    outcomes = {}
    for name, observation in (
        ("obs-5", _load("official_399_obs5_appended.json")),
        ("obs-10", _load("official_399_obs10_correct.json")),
    ):
        request = VerificationRequest(
            goal_id="g1",
            subject_id="Marvin profile",
            conditions=(condition,),
            observation=observation,
        )
        outcomes[name] = checker.evaluate(condition, request)

    assert outcomes["obs-5"].status is VerificationStatus.FAIL
    assert outcomes["obs-10"].status is VerificationStatus.PASS
    assert APPENDED in outcomes["obs-5"].detail
    assert CORRECT in outcomes["obs-10"].detail


def test_a_same_named_label_sibling_is_never_the_value_source() -> None:
    ax_text = "\n".join(
        (
            "[1] RootWebArea 'Editing biography'",
            "\t[2] StaticText 'Biography'",
            "\t[3] textbox 'Biography' required: False describedby: help",
            "\t\t[4] StaticText 'I am a robot'",
            "\t[5] checkbox 'Formatting help +' checked: false",
        )
    )
    elements = {
        "2": ElementRef(
            page_id="official-page",
            observation_id="obs-x",
            number="2",
            role="StaticText",
            name="'Biography'",
        ),
        "3": ElementRef(
            page_id="official-page",
            observation_id="obs-x",
            number="3",
            role="textbox",
            name="'Biography' required: False describedby: help",
        ),
        "5": ElementRef(
            page_id="official-page",
            observation_id="obs-x",
            number="5",
            role="checkbox",
            name="'Formatting help +' checked: false",
        ),
    }
    observation = Observation(
        run_id="run-a",
        page_id="official-page",
        observation_id="obs-x",
        url="http://127.0.0.1/user/edit_biography",
        ax_text=ax_text,
        elements=elements,
    )

    assert visible_value(observation, role="textbox", name="Biography") == CORRECT
    assert find_element_line(observation.ax_text, "3").lstrip("\t").startswith(
        "[3] textbox"
    )
    # An element with no value-bearing child is a real empty value, not the
    # sibling label and not a missing field.
    assert visible_value(observation, role="checkbox", name="Formatting help +") == ""
    no_value_ax = "\n".join(
        (
            "[2] StaticText 'Biography'",
            "[3] textbox 'Biography' required: False describedby: help",
            "[5] StaticText 'Biography'",
        )
    )
    empty = Observation(
        run_id="run-a",
        page_id="official-page",
        observation_id="obs-y",
        url="http://127.0.0.1/user/edit_biography",
        ax_text=no_value_ax,
        elements={
            "3": ElementRef(
                page_id="official-page",
                observation_id="obs-y",
                number="3",
                role="textbox",
                name="'Biography' required: False describedby: help",
            )
        },
    )
    assert visible_value(empty, role="textbox", name="Biography") == ""
    assert child_static_text_value(no_value_ax, "3") is None
