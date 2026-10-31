# Data / SQL Prompts

Five prompt templates for SQL generation, explanation, optimization, data profiling, and ETL design.

Install the optional dependency:

```bash
pip install "effgen[prompts-data]"
```

---

## Templates

### `data.sql_from_nl.v1` — SQL from Natural Language (structured)

Translates a natural-language question into SQL given a DDL schema. Returns `{"sql": "...", "warnings": [...]}`.

**Inputs:**

| Field | Type | Description |
|-------|------|-------------|
| `schema_ddl` | str | `CREATE TABLE` statements for all referenced tables |
| `question` | str | Natural-language question to translate |
| `dialect` | str | SQL dialect: standard SQL, `postgres`, `mysql`, `bigquery`, `duckdb`, `sqlite`, `tsql` |

**Live eval:** `sqlglot.parse(sql, dialect=dialect)` must succeed.

```python
from effgen.prompts.library.domains.data import sql_from_nl_v1

rendered = sql_from_nl_v1.render(
    schema_ddl="CREATE TABLE orders (id INT, total NUMERIC, date DATE);",
    question="What is the total revenue by month in 2023?",
    dialect="postgres",
)
```

---

### `data.sql_explain.v1` — SQL Explain (zero-shot)

Explains what a SQL query does in plain English, targeting either a technical developer or a business stakeholder.

**Inputs:**

| Field | Type | Description |
|-------|------|-------------|
| `sql` | str | SQL query to explain |
| `schema_ddl` | str | Optional `CREATE TABLE` statements for referenced tables |
| `audience` | str | `technical` or `business` (default: `technical`) |

```python
from effgen.prompts.library.domains.data import sql_explain_v1

rendered = sql_explain_v1.render(
    schema_ddl="CREATE TABLE orders (id INT, customer_id INT);",
    sql="SELECT name, COUNT(*) FROM orders GROUP BY name HAVING COUNT(*) > 5",
    audience="business",
)
```

---

### `data.sql_optimize.v1` — SQL Optimizer (CoT)

Chain-of-thought analysis of a slow SQL query. Identifies anti-patterns, explains execution impact, produces a rewritten query, and suggests indexes.

**Inputs:**

| Field | Type | Description |
|-------|------|-------------|
| `sql` | str | The slow SQL query |
| `schema_ddl` | str | DDL for referenced tables (optional but recommended) |
| `dialect` | str | SQL dialect (default: `ansi`) |
| `observed_issue` | str | Observed performance symptom (optional) |

**Output structure:** Five numbered steps: anti-patterns → execution plan impact → rewritten query → indexes → expected improvement.

```python
from effgen.prompts.library.domains.data import sql_optimize_v1

rendered = sql_optimize_v1.render(
    sql="SELECT * FROM orders, customers WHERE orders.cid = customers.id AND YEAR(date) = 2024",
    dialect="mysql",
    observed_issue="Full table scan, 8 seconds on 2M rows",
)
```

---

### `data.data_profile.v1` — Data Profile (tool-augmented)

Takes column statistics produced by ExcelTool or a similar profiling tool and produces a structured data-quality report.

**Inputs:**

| Field | Type | Description |
|-------|------|-------------|
| `column_stats` | str | JSON or tabular column statistics (dtype, null_count, unique_count, min, max, mean) |
| `dataset_name` | str | Human-readable dataset name |
| `row_count` | int | Total row count |

**Output sections:** Schema Overview → Completeness → Uniqueness → Range & Distribution → Data Quality Issues → Recommended Actions.

```python
from effgen.prompts.library.domains.data import data_profile_v1
import json

stats = json.dumps([
    {"column": "id",    "dtype": "int64",   "null_count": 0,  "unique_count": 5000},
    {"column": "email", "dtype": "object",  "null_count": 42, "unique_count": 4901},
])
rendered = data_profile_v1.render(
    column_stats=stats,
    dataset_name="users_export",
    row_count=5000,
)
```

---

### `data.etl_plan.v1` — ETL Pipeline Design (few-shot)

Designs an end-to-end ETL pipeline. Two exemplar ETL designs guide output style. Covers Extract → Transform → Load → Validate → Cleanup with technology choices and error-handling guidance.

**Inputs:**

| Field | Type | Description |
|-------|------|-------------|
| `source_description` | str | Description of the source system/dataset |
| `target_description` | str | Description of the target system/table |
| `transformations` | str | Business rules and required transformations |
| `schedule` | str | Pipeline schedule (default: `daily`) |

```python
from effgen.prompts.library.domains.data import etl_plan_v1

rendered = etl_plan_v1.render(
    source_description="MySQL legacy CRM (5M contacts, updated in real-time)",
    target_description="Snowflake analytics.contacts table",
    transformations="Deduplicate on email. Normalize phone to E.164. Exclude test accounts.",
    schedule="hourly",
)
```

---

## Evaluation

Run golden tests:

```bash
pytest tests/prompts/test_data.py -v
```

Run live eval (requires `CEREBRAS_API_KEY`):

```bash
effgen prompts eval --domain data --live --model gpt-oss-120b
```

The live eval for `data.sql_from_nl.v1` validates that the generated JSON contains `sql` and `warnings`, and that the generated SQL parses cleanly with `sqlglot.parse`.
The live eval for `data.sql_explain.v1` checks that output mentions at least one table name from the input schema or query.
