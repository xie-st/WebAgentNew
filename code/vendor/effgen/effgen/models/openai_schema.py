"""
Pydantic → OpenAI structured-outputs JSON Schema helper.

OpenAI's structured-outputs API requires:
  - All ``$ref`` inlined (no top-level ``$defs`` / ``definitions``).
  - ``additionalProperties: false`` on every object node.
  - Explicit ``required`` arrays listing all properties (no optional fields
    unless they carry a nullable union type, which we preserve as-is).

Usage::

    from pydantic import BaseModel
    from typing import Literal
    from effgen.models.openai_schema import to_openai_schema

    class Answer(BaseModel):
        sentiment: Literal["pos", "neg", "neu"]
        confidence: float

    schema = to_openai_schema(Answer)
    # {"type": "object", "properties": {...}, "required": [...],
    #  "additionalProperties": false}
"""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel


def _resolve_refs(schema: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    """Recursively replace ``$ref`` with the inline definition."""
    if "$ref" in schema:
        ref_path = schema["$ref"]
        # "#/$defs/Foo" or "#/definitions/Foo"
        parts = ref_path.lstrip("#/").split("/")
        node: Any = defs
        for part in parts[1:]:  # skip the leading key ("$defs" or "definitions")
            node = node[part]
        resolved = copy.deepcopy(node)
        # Merge any sibling keys (rare in Pydantic output but keep them).
        extra = {k: v for k, v in schema.items() if k != "$ref"}
        resolved.update(extra)
        return _resolve_refs(resolved, defs)

    result: dict[str, Any] = {}
    for key, value in schema.items():
        if key in ("$defs", "definitions"):
            continue  # strip definition tables from the output
        if isinstance(value, dict):
            result[key] = _resolve_refs(value, defs)
        elif isinstance(value, list):
            result[key] = [
                _resolve_refs(item, defs) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[key] = value
    return result


def _enforce_additional_properties(schema: dict[str, Any]) -> dict[str, Any]:
    """Add ``additionalProperties: false`` and complete ``required`` on all objects."""
    if schema.get("type") == "object" or "properties" in schema:
        props = schema.get("properties", {})
        schema["properties"] = {
            k: _enforce_additional_properties(v) for k, v in props.items()
        }
        schema.setdefault("required", list(props.keys()))
        schema["additionalProperties"] = False

    if "items" in schema and isinstance(schema["items"], dict):
        schema["items"] = _enforce_additional_properties(schema["items"])

    # Handle anyOf / oneOf / allOf (e.g. Optional[X] → anyOf[X, {type: null}])
    for combiner in ("anyOf", "oneOf", "allOf"):
        if combiner in schema:
            schema[combiner] = [
                _enforce_additional_properties(sub) if isinstance(sub, dict) else sub
                for sub in schema[combiner]
            ]

    return schema


def to_openai_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic *model* to an OpenAI-compatible JSON Schema dict.

    The returned dict is suitable for use as the value of ``json_schema`` in
    ``response_format={"type": "json_schema", "json_schema": {...}, "strict": True}``.

    Transformations applied:
    1. All ``$ref`` / ``$defs`` are inlined — no external references remain.
    2. Every object node gets ``"additionalProperties": false``.
    3. Every object node gets an explicit ``"required"`` listing all properties.

    Args:
        model: A Pydantic ``BaseModel`` subclass.

    Returns:
        A plain ``dict`` with no ``$ref`` or ``$defs``.

    Example::

        class Answer(BaseModel):
            sentiment: Literal["pos", "neg", "neu"]
            confidence: float

        schema = to_openai_schema(Answer)
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "Answer",
                "schema": schema,
                "strict": True,
            },
        }
    """
    raw = model.model_json_schema()
    defs: dict[str, Any] = {}
    for key in ("$defs", "definitions"):
        if key in raw:
            defs.update({k: copy.deepcopy(v) for k, v in raw[key].items()})

    resolved = _resolve_refs(copy.deepcopy(raw), defs)
    return _enforce_additional_properties(resolved)
