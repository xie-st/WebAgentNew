"""
Data prompt: Data Profile — tool-augmented variant.

Consumes ExcelTool/CSV column statistics output and produces a structured
data-quality profile report.

Inputs: column_stats (str), dataset_name (str), row_count (int)
"""

from __future__ import annotations

from effgen.prompts.library.base import LibraryPrompt
from effgen.prompts.library.registry import registry

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "column_stats": {
            "type": "string",
            "description": (
                "JSON or tabular summary of column statistics produced by ExcelTool "
                "or a similar profiling tool. Include dtype, null_count, unique_count, "
                "min, max, mean where applicable."
            ),
            "minLength": 1,
        },
        "dataset_name": {
            "type": "string",
            "description": "Human-readable name of the dataset being profiled.",
        },
        "row_count": {
            "type": "integer",
            "description": "Total number of rows in the dataset.",
            "minimum": 0,
        },
    },
    "required": ["column_stats"],
}

_FIXTURE = {
    "column_stats": (
        '[\n'
        '  {"column": "order_id",    "dtype": "int64",   "null_count": 0,    "unique_count": 10000, "min": 1,        "max": 10000},\n'
        '  {"column": "customer_id", "dtype": "int64",   "null_count": 0,    "unique_count": 842,   "min": 1,        "max": 999},\n'
        '  {"column": "order_date",  "dtype": "object",  "null_count": 12,   "unique_count": 730,   "min": "2022-01-03", "max": "2024-12-30"},\n'
        '  {"column": "total_amt",   "dtype": "float64", "null_count": 35,   "unique_count": 9187,  "min": 0.01,     "max": 49999.99, "mean": 312.45},\n'
        '  {"column": "status",      "dtype": "object",  "null_count": 0,    "unique_count": 5,     "top": "shipped"}\n'
        ']'
    ),
    "dataset_name": "orders_2022_2024",
    "row_count": 10000,
}


def _data_profile_tool(
    column_stats: str,
    dataset_name: str = "dataset",
    row_count: int = 0,
) -> str:
    row_clause = f" ({row_count:,} rows)" if row_count > 0 else ""
    return (
        "You are a senior data analyst and data-quality engineer.\n\n"
        f"Profile the dataset '{dataset_name}'{row_clause} using the column statistics below. "
        "These statistics were produced by a data-profiling tool (e.g. ExcelTool or pandas describe).\n\n"
        "Column statistics:\n"
        f"```\n{column_stats}\n```\n\n"
        "Produce a structured data-quality report covering:\n"
        "  1. **Schema Overview** — list each column with its inferred type and role.\n"
        "  2. **Completeness** — flag any columns with missing values; compute null rate (%).\n"
        "  3. **Uniqueness** — identify potential key columns; flag unexpectedly high or low cardinality.\n"
        "  4. **Range & Distribution** — note suspicious min/max values, extreme outliers, or skewed distributions.\n"
        "  5. **Data Quality Issues** — summarize the top issues found, ranked by severity (critical / warning / info).\n"
        "  6. **Recommended Actions** — concrete next steps to fix or investigate each issue.\n\n"
        "Use the headings above exactly. Be specific — reference column names and actual values."
    )


data_profile_v1 = LibraryPrompt(
    name="data.data_profile.v1",
    domain="data",
    variant="tool",
    description=(
        "Tool-augmented data profiling: takes ExcelTool/CSV column statistics and produces "
        "a structured data-quality report (completeness, uniqueness, range, issues, actions)."
    ),
    template=_data_profile_tool,
    input_schema=_INPUT_SCHEMA,
    fixture=_FIXTURE,
    expected_shape={
        "type": "regex",
        "pattern": r"(Completeness|Uniqueness|Quality|Issue|Action|null|missing)",
    },
    tags=["data", "profiling", "quality", "tool", "excel"],
)

registry.register(data_profile_v1)
