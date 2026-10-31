"""
Data prompt: ETL Plan — few-shot variant.

Two exemplar ETL designs are included to guide output style.

Inputs: source_description (str), target_description (str), transformations (str), schedule (str)
"""

from __future__ import annotations

from effgen.prompts.library.base import LibraryPrompt
from effgen.prompts.library.registry import registry

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "source_description": {
            "type": "string",
            "description": "Description of the source system/dataset.",
            "minLength": 1,
        },
        "target_description": {
            "type": "string",
            "description": "Description of the target system/table.",
            "minLength": 1,
        },
        "transformations": {
            "type": "string",
            "description": "Business rules and transformations required.",
        },
        "schedule": {
            "type": "string",
            "description": "Desired pipeline schedule (e.g. 'daily at midnight', 'real-time').",
            "default": "daily",
        },
    },
    "required": ["source_description", "target_description"],
}

_FIXTURE = {
    "source_description": (
        "PostgreSQL orders table (10 M rows, updated continuously). "
        "Columns: order_id, customer_id, order_date, total_amt, status."
    ),
    "target_description": (
        "BigQuery data warehouse table `analytics.daily_order_summary` used by dashboards."
    ),
    "transformations": (
        "Aggregate orders by day and customer: compute order_count, total_revenue, avg_order_value. "
        "Exclude cancelled orders. Map status codes to human-readable labels."
    ),
    "schedule": "daily at 02:00 UTC",
}

_EXEMPLAR_1 = """\
--- ETL Example 1: CSV file → SQLite ---
Source: Daily CSV export from a CRM system (~50 k rows).
Target: SQLite analytics.contacts table.
Schedule: Daily at 03:00 local time.

Design:
  Extract  — Download CSV via SFTP to local staging directory.
  Transform — Parse dates (MM/DD/YYYY → ISO-8601). Deduplicate on email.
              Normalize phone numbers to E.164. Flag rows with missing name.
  Load     — UPSERT into contacts using email as the natural key.
  Validate — Row count delta should be ±5% of yesterday; alert otherwise.
  Cleanup  — Delete staging file after successful load.
"""

_EXEMPLAR_2 = """\
--- ETL Example 2: REST API → Parquet → Snowflake ---
Source: GitHub Events API (paginated, up to 300 events/min).
Target: Snowflake raw_events table via Snowpipe.
Schedule: Near-real-time (15-minute micro-batches).

Design:
  Extract  — Poll API every 15 min; cursor stored in Redis for dedup.
  Transform — Flatten nested JSON. Drop PII (author email). Add batch_id, ingested_at.
  Load     — Write Parquet to S3 staging bucket; trigger Snowpipe COPY INTO.
  Validate — Assert no duplicate event_id in the last 24 h window.
  Cleanup  — Expire S3 objects after 7 days via lifecycle rule.
"""


def _etl_plan_few_shot(
    source_description: str,
    target_description: str,
    transformations: str = "",
    schedule: str = "daily",
) -> str:
    transform_block = (
        f"\nRequired transformations:\n{transformations}\n"
        if transformations.strip()
        else ""
    )
    return (
        "You are a senior data engineer. Design a production-ready ETL pipeline.\n\n"
        "Here are two examples of well-structured ETL designs to follow:\n\n"
        f"{_EXEMPLAR_1}\n"
        f"{_EXEMPLAR_2}\n"
        "--- End of examples ---\n\n"
        "Now design an ETL pipeline for the following requirements:\n\n"
        f"Source: {source_description}\n"
        f"Target: {target_description}\n"
        f"Schedule: {schedule}\n"
        f"{transform_block}\n"
        "Follow the same structure as the examples:\n"
        "  Extract → Transform → Load → Validate → Cleanup\n\n"
        "Be specific about:\n"
        "  - Technology choices (e.g. Airflow, dbt, Spark, pandas).\n"
        "  - Error handling and retry strategy.\n"
        "  - Idempotency: how to safely re-run the pipeline.\n"
        "  - Monitoring: what metrics or alerts to add."
    )


etl_plan_v1 = LibraryPrompt(
    name="data.etl_plan.v1",
    domain="data",
    variant="few_shot",
    description=(
        "Few-shot ETL pipeline design: two exemplar designs guide output style. "
        "Covers Extract → Transform → Load → Validate → Cleanup with technology choices."
    ),
    template=_etl_plan_few_shot,
    input_schema=_INPUT_SCHEMA,
    fixture=_FIXTURE,
    expected_shape={
        "type": "regex",
        "pattern": r"(Extract|Transform|Load|Validate|Cleanup)",
    },
    tags=["data", "etl", "pipeline", "few_shot", "engineering"],
)

registry.register(etl_plan_v1)
