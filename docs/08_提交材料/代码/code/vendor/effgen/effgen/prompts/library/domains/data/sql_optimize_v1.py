"""
Data prompt: SQL Optimize — chain-of-thought variant.

Analyzes a slow SQL query and produces optimization recommendations
with step-by-step reasoning.

Inputs: sql (str), schema_ddl (str), dialect (str), observed_issue (str)
"""

from __future__ import annotations

from effgen.prompts.library.base import LibraryPrompt
from effgen.prompts.library.registry import registry

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "sql": {
            "type": "string",
            "description": "The slow or inefficient SQL query to optimize.",
            "minLength": 1,
        },
        "schema_ddl": {
            "type": "string",
            "description": "CREATE TABLE statements for all referenced tables.",
        },
        "dialect": {
            "type": "string",
            "description": "SQL dialect: '' (standard) | 'postgres' | 'mysql' | 'bigquery' | 'duckdb' | 'tsql' | etc.",
            "default": "",
        },
        "observed_issue": {
            "type": "string",
            "description": "Optional: describe the observed performance symptom.",
        },
    },
    "required": ["sql"],
}

_FIXTURE = {
    "sql": (
        "SELECT *\n"
        "FROM orders o, customers c\n"
        "WHERE o.customer_id = c.customer_id\n"
        "  AND YEAR(o.order_date) = 2023\n"
        "  AND c.country = 'US'\n"
        "ORDER BY o.total_amt DESC;"
    ),
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
    "dialect": "",
    "observed_issue": "Full table scan on orders; query takes >10 s on 5 M rows.",
}


def _sql_optimize_cot(
    sql: str,
    schema_ddl: str = "",
    dialect: str = "",
    observed_issue: str = "",
) -> str:
    schema_block = (
        f"\nSchema:\n```sql\n{schema_ddl}\n```\n" if schema_ddl.strip() else ""
    )
    issue_block = (
        f"\nObserved performance issue: {observed_issue}\n" if observed_issue.strip() else ""
    )
    dialect_label = dialect.upper() if dialect else "standard SQL"
    return (
        "You are a senior database performance engineer. "
        f"The following SQL query runs on a {dialect_label} database and needs optimization.\n"
        f"{schema_block}"
        f"{issue_block}\n"
        "Query to optimize:\n"
        f"```sql\n{sql}\n```\n\n"
        "Think step by step:\n"
        "  Step 1 — Identify anti-patterns (e.g. SELECT *, implicit joins, non-sargable predicates, "
        "missing indexes, N+1 sub-queries, unnecessary DISTINCT/ORDER BY).\n"
        "  Step 2 — Explain the execution plan impact of each anti-pattern.\n"
        "  Step 3 — Propose a rewritten, optimized query.\n"
        "  Step 4 — List any indexes that should be created.\n"
        "  Step 5 — Estimate the expected improvement (order-of-magnitude, not exact numbers).\n\n"
        "Use the exact headings 'Step 1', 'Step 2', ..., 'Step 5' in your response."
    )


sql_optimize_v1 = LibraryPrompt(
    name="data.sql_optimize.v1",
    domain="data",
    variant="cot",
    description=(
        "Chain-of-thought SQL optimization: identify anti-patterns, explain execution impact, "
        "produce a rewritten query, and suggest indexes."
    ),
    template=_sql_optimize_cot,
    input_schema=_INPUT_SCHEMA,
    fixture=_FIXTURE,
    expected_shape={
        "type": "regex",
        "pattern": r"Step\s+[12345]",
    },
    tags=["data", "sql", "optimize", "cot", "performance"],
)

registry.register(sql_optimize_v1)
