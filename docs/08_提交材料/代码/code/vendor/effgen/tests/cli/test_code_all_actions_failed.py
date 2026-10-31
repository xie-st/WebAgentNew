"""A coding turn whose every action failed did not do what it was asked.

The model can end a turn having written nothing, with an answer that truthfully
*describes* the tool failure it hit — and the run was still reported
``success=true`` with `files_written=[]`. Nothing in the text distinguishes that
paragraph from an answer, so the check is on the recorded actions: a turn that
ran at least one action, had every one of them come back an error, and wrote
nothing, is at best a partial outcome.

It is reported the way the iteration cap is: ``success=False``, ``partial``, the
model's own paragraph under ``partial_output``, and a typed error naming the
failing action.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from effgen.cli.code.engine import CodeEngine, _every_action_failed
from effgen.cli.code.permissions import ActionRecord, PermissionMode


def _engine(tmp_path):
    return CodeEngine(
        model="fake:model", workspace=str(tmp_path),
        mode=PermissionMode.YES, interactive=False,
    )


def _response(answer: str):
    return SimpleNamespace(
        output=answer, success=True, metadata={}, iterations=1,
        tool_calls=1, tokens_used=10, execution_time=0.1,
    )


NARRATION = (
    "The final answer to the user's question is: The code_executor tool failed "
    "to execute the code because the module 'greet' was not found."
)


class TestEveryActionFailed:
    def test_a_turn_that_only_failed_is_not_a_success(self, tmp_path):
        engine = _engine(tmp_path)
        engine.gate.actions.append(ActionRecord(
            kind="run", summary="python greet.py", decision="allowed",
            outcome="error", detail="ModuleNotFoundError: greet",
        ))

        result = engine.result_from_response("write and run greet.py", _response(NARRATION))

        assert result.success is False
        assert result.partial is True
        assert result.files_written == []

    def test_the_narration_travels_as_progress_not_as_the_answer(self, tmp_path):
        engine = _engine(tmp_path)
        engine.gate.actions.append(ActionRecord(
            kind="run", summary="python greet.py", decision="allowed",
            outcome="error", detail="ModuleNotFoundError: greet",
        ))

        result = engine.result_from_response("write and run greet.py", _response(NARRATION))

        assert result.partial_output == NARRATION
        assert result.answer != NARRATION
        assert "changed nothing" in result.answer

    def test_the_outcome_is_typed_and_names_the_failure(self, tmp_path):
        engine = _engine(tmp_path)
        engine.gate.actions.append(ActionRecord(
            kind="run", summary="python greet.py", decision="allowed",
            outcome="error", detail="ModuleNotFoundError: greet",
        ))

        result = engine.result_from_response("write and run greet.py", _response(NARRATION))

        assert result.error["type"] == "NoActionSucceeded"
        assert result.error["last_action"] == "python greet.py"
        assert "greet" in result.error["last_error"]
        assert result.reason == "all_actions_failed"

    def test_one_success_among_failures_is_still_a_success(self, tmp_path):
        """Only a turn where *nothing* worked is reported this way."""
        engine = _engine(tmp_path)
        engine.gate.actions.extend([
            ActionRecord(kind="run", summary="first try", decision="allowed",
                         outcome="error", detail="boom"),
            ActionRecord(kind="run", summary="second try", decision="allowed",
                         outcome="ok"),
        ])

        result = engine.result_from_response("run it", _response("Done: it printed 4."))

        assert result.success is True
        assert result.partial is False

    def test_a_turn_that_wrote_a_file_is_still_a_success(self, tmp_path):
        engine = _engine(tmp_path)
        engine.gate.actions.append(ActionRecord(
            kind="write", summary="write greet.py", decision="allowed",
            outcome="ok", target="greet.py",
        ))

        result = engine.result_from_response("write greet.py", _response("Wrote greet.py."))

        assert result.success is True
        assert result.files_written == ["greet.py"]


class TestOnlyActionsThatRanCount:
    @pytest.mark.parametrize("decision", ["withheld", "declined", "refused"])
    def test_an_action_that_never_ran_is_not_a_failure(self, decision):
        """Plan mode proposes edits without running them; nothing failed there."""
        assert not _every_action_failed([
            ActionRecord(kind="write", summary="write x.py", decision=decision),
        ])

    def test_a_turn_with_no_actions_at_all_is_not_this_case(self):
        """Deliberately out of scope — see the follow-up's 'what was left'."""
        assert not _every_action_failed([])

    def test_every_allowed_action_failing_is_this_case(self):
        assert _every_action_failed([
            ActionRecord(kind="run", summary="a", decision="allowed", outcome="error"),
            ActionRecord(kind="write", summary="b", decision="refused"),
            ActionRecord(kind="run", summary="c", decision="allowed", outcome="error"),
        ])
