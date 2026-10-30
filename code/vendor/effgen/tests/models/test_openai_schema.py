"""Unit tests for effgen.models.openai_schema.to_openai_schema."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from effgen.models.openai_schema import to_openai_schema


class SimpleAnswer(BaseModel):
    sentiment: Literal["pos", "neg", "neu"]
    confidence: float


class Address(BaseModel):
    street: str
    city: str


class Person(BaseModel):
    name: str
    age: int
    address: Address


class WithList(BaseModel):
    tags: list[str]
    score: float


def test_simple_schema_has_required_type_and_props():
    schema = to_openai_schema(SimpleAnswer)
    assert schema["type"] == "object"
    assert "sentiment" in schema["properties"]
    assert "confidence" in schema["properties"]


def test_simple_schema_has_no_defs():
    schema = to_openai_schema(SimpleAnswer)
    assert "$defs" not in schema
    assert "definitions" not in schema
    assert "$ref" not in str(schema)


def test_simple_schema_additional_properties_false():
    schema = to_openai_schema(SimpleAnswer)
    assert schema["additionalProperties"] is False


def test_simple_schema_required_lists_all_fields():
    schema = to_openai_schema(SimpleAnswer)
    assert set(schema["required"]) == {"sentiment", "confidence"}


def test_nested_schema_inlines_ref():
    schema = to_openai_schema(Person)
    addr_schema = schema["properties"]["address"]
    # $ref must be resolved
    assert "$ref" not in str(addr_schema)
    assert addr_schema["type"] == "object"
    assert "street" in addr_schema["properties"]
    assert "city" in addr_schema["properties"]


def test_nested_schema_additional_properties_on_inner_object():
    schema = to_openai_schema(Person)
    addr_schema = schema["properties"]["address"]
    assert addr_schema["additionalProperties"] is False


def test_nested_schema_required_on_inner_object():
    schema = to_openai_schema(Person)
    addr_schema = schema["properties"]["address"]
    assert set(addr_schema["required"]) == {"street", "city"}


def test_list_field_preserved():
    schema = to_openai_schema(WithList)
    tags_schema = schema["properties"]["tags"]
    assert tags_schema["type"] == "array"
    assert tags_schema["items"]["type"] == "string"


def test_literal_field_becomes_enum():
    schema = to_openai_schema(SimpleAnswer)
    sent = schema["properties"]["sentiment"]
    assert "enum" in sent or "const" in sent or sent.get("type") == "string"


def test_no_dollar_ref_anywhere():
    schema = to_openai_schema(Person)
    import json
    assert "$ref" not in json.dumps(schema)


def test_top_level_additional_properties():
    for model in (SimpleAnswer, Person, WithList):
        schema = to_openai_schema(model)
        assert schema.get("additionalProperties") is False, f"Missing for {model.__name__}"
