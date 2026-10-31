"""
Data prompt: SQL from Natural Language — structured-output variant.

Returns JSON: {sql: str, warnings: [str]}
Live eval: sqlglot.parse(sql, dialect=dialect) must succeed.

Inputs: schema_ddl (str), question (str), dialect (str)
"""

from __future__ import annotations

from effgen.prompts.library.base import LibraryPrompt
from effgen.prompts.library.eval import PromptEval
from effgen.prompts.library.registry import registry

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "schema_ddl": {
            "type": "string",
            "description": "CREATE TABLE statements describing the schema.",
            "minLength": 1,
        },
        "question": {
            "type": "string",
            "description": "Natural-language question to translate to SQL.",
            "minLength": 1,
        },
        "dialect": {
            "type": "string",
            "description": "SQL dialect: '' (default/standard) | 'bigquery' | 'duckdb' | 'mysql' | 'postgres' | 'sqlite' | 'tsql'",
            "default": "",
        },
    },
    "required": ["schema_ddl", "question"],
}

_FIXTURE = {
    "schema_ddl": (
        "CREATE TABLE orders (\n"
        "    order_id    INTEGER PRIMARY KEY,\n"
        "    customer_id INTEGER NOT NULL,\n"
        "    order_date  DATE    NOT NULL,\n"
        "    total_amt   NUMERIC(12,2) NOT NULL\n"
        ");\n\n"
        "CREATE TABLE customers (\n"
        "    customer_id INTEGER PRIMARY KEY,\n"
        "    name        VARCHAR(120) NOT NULL,\n"
        "    country     VARCHAR(60)  NOT NULL\n"
        ");"
    ),
    "question": "Which customers placed more than 3 orders in 2023, sorted by total spend descending?",
    "dialect": "",
}

_EXPECTED_SCHEMA = {
    "type": "object",
    "properties": {
        "sql": {"type": "string", "minLength": 10},
        "warnings": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["sql", "warnings"],
    "additionalProperties": False,
}


def _json_sqlglot_check(output: str) -> bool | str:
    """Validate structured output and parse its SQL with sqlglot."""
    try:
        import sqlglot
        from jsonschema import Draft202012Validator
        from jsonschema import exceptions as jsonschema_exceptions
    except ImportError:
        return (
            "sqlglot and jsonschema are required for SQL parse validation; "
            "install effgen[prompts-data]"
        )

    parsed = PromptEval._extract_json_object(output)
    if parsed is None:
        return "output is not valid JSON"
    try:
        Draft202012Validator(_EXPECTED_SCHEMA).validate(parsed)
    except jsonschema_exceptions.ValidationError as exc:
        return f"JSON schema validation failed: {exc.message}"
    sql = parsed["sql"]
    try:
        statements = sqlglot.parse(sql)
    except Exception as exc:
        return f"sqlglot could not parse generated SQL: {exc}"
    if not statements:
        return "sqlglot parsed no SQL statements"
    return True


def _sql_from_nl_structured(
    schema_ddl: str,
    question: str,
    dialect: str = "",
) -> str:
    dialect_label = dialect.upper() if dialect else "STANDARD SQL"
    return (
        "You are an expert SQL engineer. Your task is to translate a natural-language question "
        "into a syntactically correct SQL query for the given schema.\n\n"
        f"Target dialect: {dialect_label}\n\n"
        "Schema:\n"
        f"```sql\n{schema_ddl}\n```\n\n"
        f"Question: {question}\n\n"
        "Requirements:\n"
        "  - Produce a single, complete SQL SELECT statement.\n"
        "  - Use only tables and columns present in the schema above.\n"
        "  - Copy table and column names exactly as written in the schema; do not invent or misspell identifiers.\n"
        "  - Use balanced single quotes around date literals, e.g. '2023-01-01'.\n"
        "  - Mentally check that the SQL would parse before returning it.\n"
        "  - If the question cannot be answered exactly (e.g. missing columns), add a warning.\n"
        "  - Do NOT use dialect-specific syntax unless the dialect is not 'ansi'.\n\n"
        "Return a JSON object with exactly two keys:\n"
        '  "sql"      — the SQL query string as one JSON string value (no markdown fences and no adjacent string fragments)\n'
        '  "warnings" — an array of strings (empty array if no warnings)\n\n'
        "Output only the JSON object, no markdown fences, no extra text, and no explanation after the object."
    )


sql_from_nl_v1 = LibraryPrompt(
    name="data.sql_from_nl.v1",
    domain="data",
    variant="structured",
    description=(
        "Translate a natural-language question to SQL given a DDL schema. "
        "Returns JSON {sql, warnings[]}. Live eval validates via sqlglot.parse()."
    ),
    template=_sql_from_nl_structured,
    input_schema=_INPUT_SCHEMA,
    fixture=_FIXTURE,
    expected_shape={
        "type": "callable",
        "fn": _json_sqlglot_check,
        "schema": _EXPECTED_SCHEMA,
    },
    tags=["data", "sql", "structured", "json", "nlp"],
)

registry.register(sql_from_nl_v1)
