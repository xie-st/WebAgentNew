"""
Data prompt: SQL Explain — zero-shot variant.

Explains what a SQL query does in plain English.

Inputs: sql (str), schema_ddl (str), audience (str)
"""

from __future__ import annotations

from effgen.prompts.library.base import LibraryPrompt
from effgen.prompts.library.registry import registry

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "sql": {
            "type": "string",
            "description": "SQL query to explain.",
            "minLength": 1,
        },
        "audience": {
            "type": "string",
            "description": "Intended reader: 'technical' | 'business'",
            "default": "technical",
        },
        "schema_ddl": {
            "type": "string",
            "description": (
                "Optional CREATE TABLE statements for referenced tables. "
                "When provided, the explanation should use these table names."
            ),
            "default": "",
        },
    },
    "required": ["sql"],
}

_FIXTURE = {
    "schema_ddl": (
        "CREATE TABLE customers (\n"
        "    customer_id INTEGER PRIMARY KEY,\n"
        "    name VARCHAR(120) NOT NULL\n"
        ");\n\n"
        "CREATE TABLE orders (\n"
        "    order_id INTEGER PRIMARY KEY,\n"
        "    customer_id INTEGER NOT NULL,\n"
        "    order_date DATE NOT NULL,\n"
        "    total_amt NUMERIC(12,2) NOT NULL\n"
        ");"
    ),
    "sql": (
        "SELECT c.name, COUNT(o.order_id) AS order_count,\n"
        "       SUM(o.total_amt) AS total_spend\n"
        "FROM customers c\n"
        "JOIN orders o ON c.customer_id = o.customer_id\n"
        "WHERE o.order_date >= '2023-01-01'\n"
        "  AND o.order_date <  '2024-01-01'\n"
        "GROUP BY c.customer_id, c.name\n"
        "HAVING COUNT(o.order_id) > 3\n"
        "ORDER BY total_spend DESC;"
    ),
    "audience": "business",
}


def _sql_explain_zero_shot(
    sql: str,
    audience: str = "technical",
    schema_ddl: str = "",
) -> str:
    if audience == "business":
        style = (
            "Explain this SQL query to a business stakeholder with no SQL background. "
            "Avoid jargon; describe what data the query retrieves and why it might be useful."
        )
    else:
        style = (
            "Explain this SQL query to a technical developer. "
            "Describe each clause, the join logic, filtering, grouping, and ordering."
        )
    schema_block = (
        f"Input schema:\n```sql\n{schema_ddl}\n```\n\n" if schema_ddl.strip() else ""
    )
    return (
        f"You are a database expert and technical writer.\n\n"
        f"{style}\n\n"
        f"{schema_block}"
        f"SQL query:\n```sql\n{sql}\n```\n\n"
        "Structure your explanation as:\n"
        "  1. One-sentence summary of what the query does.\n"
        "  2. Step-by-step breakdown of each clause.\n"
        "  3. What the result set looks like (columns, row meaning).\n\n"
        "Write in clear prose. Mention the referenced table names from the input schema "
        "or SQL query when describing the result."
    )


sql_explain_v1 = LibraryPrompt(
    name="data.sql_explain.v1",
    domain="data",
    variant="zero_shot",
    description=(
        "Explain a SQL query in plain English, targeting either a technical developer "
        "or a business stakeholder."
    ),
    template=_sql_explain_zero_shot,
    input_schema=_INPUT_SCHEMA,
    fixture=_FIXTURE,
    expected_shape={
        "type": "regex",
        "pattern": r"(customer|order|select|join|group|filter|sort|result)",
    },
    tags=["data", "sql", "explain", "zero_shot"],
)

registry.register(sql_explain_v1)
