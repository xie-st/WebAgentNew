"""Coercion of string-encoded parameter values from model tool calls.

Small models frequently send a value as a string: an integer as ``"3"``, or a
whole object as ``"{}"``. The object form is what Llama 3.2 fills optional
object parameters with, and rejecting it stops the tool from running at all.
Coercion applies only when the string parses as JSON *and* yields the declared
type — anything else is left for validation to reject.
"""

from __future__ import annotations

from typing import Any

import pytest

from effgen.tools.base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)


def _tool(*params: ParameterSpec) -> BaseTool:
    class _T(BaseTool):
        async def _execute(self, **kwargs: Any) -> Any:
            return kwargs

    return _T(
        metadata=ToolMetadata(
            name="coerce_probe",
            description="Parameter coercion probe",
            category=ToolCategory.COMPUTATION,
            parameters=list(params),
        )
    )


OBJ = ParameterSpec("filters", ParameterType.OBJECT, "object param", required=False)
ARR = ParameterSpec("items", ParameterType.ARRAY, "array param", required=False)


class TestObjectAndArrayCoercion:
    @pytest.mark.parametrize(
        "raw, expected",
        [("{}", {}), ('{"type": "txt"}', {"type": "txt"})],
    )
    def test_stringified_object_becomes_an_object(self, raw, expected):
        assert _tool(OBJ)._coerce_parameters({"filters": raw}) == {"filters": expected}

    @pytest.mark.parametrize(
        "raw, expected", [("[]", []), ('["a", "b"]', ["a", "b"])]
    )
    def test_stringified_array_becomes_an_array(self, raw, expected):
        assert _tool(ARR)._coerce_parameters({"items": raw}) == {"items": expected}

    def test_a_real_object_is_left_alone(self):
        value = {"type": "txt"}
        assert _tool(OBJ)._coerce_parameters({"filters": value}) == {"filters": value}

    @pytest.mark.parametrize("raw", ["not json", "", "{unclosed", '"a string"', "42"])
    def test_a_value_that_is_not_the_declared_type_is_left_for_validation(self, raw):
        """Coercion never invents a value; validation still reports the error."""
        tool = _tool(OBJ)
        assert tool._coerce_parameters({"filters": raw}) == {"filters": raw}
        ok, error = tool.validate_parameters(filters=raw)
        assert not ok
        assert "filters" in error

    def test_an_array_string_does_not_satisfy_an_object_parameter(self):
        tool = _tool(OBJ)
        assert tool._coerce_parameters({"filters": "[]"}) == {"filters": "[]"}
        ok, _ = tool.validate_parameters(filters="[]")
        assert not ok

    def test_an_object_string_does_not_satisfy_an_array_parameter(self):
        tool = _tool(ARR)
        assert tool._coerce_parameters({"items": "{}"}) == {"items": "{}"}
        ok, _ = tool.validate_parameters(items="{}")
        assert not ok


class TestCoercedCallExecutes:
    @pytest.mark.asyncio
    async def test_tool_runs_with_a_stringified_object_argument(self):
        """The call a small model actually emits reaches the tool body."""
        result = await _tool(OBJ).execute(filters="{}")
        assert result.success, result.error
        assert result.output == {"filters": {}}

    @pytest.mark.asyncio
    async def test_unparseable_object_argument_reports_a_typed_failure(self):
        result = await _tool(OBJ).execute(filters="not json")
        assert not result.success
        assert "filters" in result.error
