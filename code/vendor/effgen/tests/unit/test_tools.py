"""Unit tests for built-in tools."""

import pytest

from effgen.tools.base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
)


class TestCalculator:
    """Tests for Calculator tool."""

    def test_instantiation(self, calculator):
        assert calculator.name == "calculator"
        assert isinstance(calculator, BaseTool)

    def test_has_metadata(self, calculator):
        meta = calculator.metadata
        assert meta is not None
        assert meta.name == "calculator"
        assert len(meta.description) > 0

    @pytest.mark.asyncio
    async def test_basic_addition(self, calculator):
        result = await calculator.execute(expression="2 + 2")
        assert "4" in str(result)

    @pytest.mark.asyncio
    async def test_multiplication(self, calculator):
        result = await calculator.execute(expression="7 * 8")
        assert "56" in str(result)

    @pytest.mark.asyncio
    async def test_complex_expression(self, calculator):
        result = await calculator.execute(expression="(10 + 5) * 3")
        assert "45" in str(result)

    @pytest.mark.asyncio
    async def test_division(self, calculator):
        result = await calculator.execute(expression="100 / 4")
        assert "25" in str(result)


class TestDateTimeTool:
    """Tests for DateTimeTool."""

    def test_instantiation(self, datetime_tool):
        assert datetime_tool.name == "datetime"
        assert isinstance(datetime_tool, BaseTool)

    @pytest.mark.asyncio
    async def test_current_time(self, datetime_tool):
        result = await datetime_tool.execute(operation="now")
        assert "202" in str(result)

    @pytest.mark.asyncio
    async def test_current_time_utc(self, datetime_tool):
        result = await datetime_tool.execute(operation="now", timezone="UTC")
        assert "202" in str(result)


class TestJSONTool:
    """Tests for JSONTool."""

    def test_instantiation(self, json_tool):
        assert json_tool.name == "json_tool"
        assert isinstance(json_tool, BaseTool)

    @pytest.mark.asyncio
    async def test_keys_operation(self, json_tool):
        result = await json_tool.execute(
            operation="keys",
            data='{"a": 1, "b": 2, "c": 3}'
        )
        result_str = str(result)
        assert "a" in result_str

    @pytest.mark.asyncio
    async def test_length_operation(self, json_tool):
        result = await json_tool.execute(
            operation="length",
            data='{"a": 1, "b": 2, "c": 3}'
        )
        assert "3" in str(result)

    @pytest.mark.asyncio
    async def test_validate_valid_json(self, json_tool):
        result = await json_tool.execute(
            operation="validate",
            data='{"name": "Alice"}'
        )
        result_str = str(result)
        assert result.success or "valid" in result_str.lower()


class TestTextProcessingTool:
    """Tests for TextProcessingTool."""

    def test_instantiation(self, text_tool):
        assert text_tool.name == "text_processing"
        assert isinstance(text_tool, BaseTool)

    @pytest.mark.asyncio
    async def test_word_count(self, text_tool):
        result = await text_tool.execute(
            operation="word_count",
            text="Hello world this is a test"
        )
        assert "6" in str(result)

    @pytest.mark.asyncio
    async def test_word_count_empty(self, text_tool):
        result = await text_tool.execute(
            operation="word_count",
            text=""
        )
        assert "0" in str(result)


class TestToolMetadata:
    """Tests for ToolMetadata and ParameterSpec."""

    def test_parameter_spec_validation(self):
        spec = ParameterSpec(
            name="test",
            type=ParameterType.STRING,
            description="A test parameter",
            required=True,
        )
        valid, error = spec.validate("hello")
        assert valid is True
        assert error is None

    def test_parameter_spec_required_missing(self):
        spec = ParameterSpec(
            name="test",
            type=ParameterType.STRING,
            description="A test parameter",
            required=True,
        )
        valid, error = spec.validate(None)
        assert valid is False
        assert "required" in error.lower()

    def test_parameter_spec_optional_none(self):
        spec = ParameterSpec(
            name="test",
            type=ParameterType.STRING,
            description="A test parameter",
            required=False,
        )
        valid, error = spec.validate(None)
        assert valid is True

    def test_model_facing_false_hidden_from_schema(self):
        from effgen.tools.base_tool import ToolCategory, ToolMetadata

        meta = ToolMetadata(
            name="t",
            description="d",
            category=ToolCategory.COMPUTATION,
            parameters=[
                ParameterSpec(name="code", type=ParameterType.STRING, description="c", required=True),
                ParameterSpec(
                    name="unsafe_toggle",
                    type=ParameterType.BOOLEAN,
                    description="developer-only",
                    model_facing=False,
                ),
            ],
        )
        props = meta.to_json_schema()["parameters"]["properties"]
        assert "code" in props
        # The non-model-facing toggle must not appear in the model schema.
        assert "unsafe_toggle" not in props
        # ...but it is still part of the tool's real parameter list.
        assert {p.name for p in meta.parameters} == {"code", "unsafe_toggle"}
        assert [p.name for p in meta.model_facing_parameters] == ["code"]
