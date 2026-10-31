"""Tests for data/SQL domain prompt templates."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests._harness.live_eval import assert_live_eval_passed

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "data"
GOLDENS_DIR = Path(__file__).parent / "goldens"


# ---------------------------------------------------------------------------
# Fixture files
# ---------------------------------------------------------------------------

class TestFixtures:
    def test_fixtures_exist(self):
        assert FIXTURES_DIR.exists(), f"Fixtures dir missing: {FIXTURES_DIR}"
        sql_files = list(FIXTURES_DIR.glob("*.sql"))
        assert len(sql_files) >= 2, "Expected at least 2 SQL fixture files"

    def test_tpch_fixture_has_tables(self):
        p = FIXTURES_DIR / "tpch_subset.sql"
        assert p.exists()
        content = p.read_text()
        for table in ("customer", "orders", "lineitem"):
            assert table in content.lower(), f"Table '{table}' missing from TPC-H fixture"

    def test_orders_simple_fixture(self):
        p = FIXTURES_DIR / "orders_simple.sql"
        assert p.exists()
        content = p.read_text()
        assert "customers" in content.lower()
        assert "orders" in content.lower()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestDataRegistration:
    def test_global_registry_has_data_prompts(self):
        from effgen.prompts.library.registry import registry
        prompts = registry.search(domain="data")
        assert len(prompts) == 5, f"Expected exactly 5 data prompts, got {len(prompts)}"

    def test_all_required_names_present(self):
        from effgen.prompts.library.registry import registry
        names = {p.name for p in registry.search(domain="data")}
        required = {
            "data.sql_from_nl.v1",
            "data.sql_explain.v1",
            "data.sql_optimize.v1",
            "data.data_profile.v1",
            "data.etl_plan.v1",
        }
        missing = required - names
        assert not missing, f"Missing data prompts: {missing}"

    def test_variants(self):
        from effgen.prompts.library.registry import registry
        prompts = {p.name: p for p in registry.search(domain="data")}
        assert prompts["data.sql_from_nl.v1"].variant == "structured"
        assert prompts["data.sql_explain.v1"].variant == "zero_shot"
        assert prompts["data.sql_optimize.v1"].variant == "cot"
        assert prompts["data.data_profile.v1"].variant == "tool"
        assert prompts["data.etl_plan.v1"].variant == "few_shot"


# ---------------------------------------------------------------------------
# sql_from_nl_v1
# ---------------------------------------------------------------------------

class TestSqlFromNl:
    def test_renders_with_fixture(self):
        from effgen.prompts.library.domains.data.sql_from_nl_v1 import sql_from_nl_v1
        rendered = sql_from_nl_v1.render_fixture()
        assert "orders" in rendered
        assert "customers" in rendered
        assert "2023" in rendered

    def test_json_output_instruction(self):
        from effgen.prompts.library.domains.data.sql_from_nl_v1 import sql_from_nl_v1
        rendered = sql_from_nl_v1.render_fixture()
        assert '"sql"' in rendered
        assert '"warnings"' in rendered

    def test_no_markdown_fences_instruction(self):
        from effgen.prompts.library.domains.data.sql_from_nl_v1 import sql_from_nl_v1
        rendered = sql_from_nl_v1.render_fixture()
        assert "no markdown fences" in rendered.lower()

    def test_dialect_in_rendered_output(self):
        from effgen.prompts.library.domains.data.sql_from_nl_v1 import sql_from_nl_v1
        rendered = sql_from_nl_v1.render(
            schema_ddl="CREATE TABLE t (id INT);",
            question="Count all rows",
            dialect="postgres",
        )
        assert "POSTGRES" in rendered

    def test_default_dialect_renders_standard_sql_label(self):
        from effgen.prompts.library.domains.data.sql_from_nl_v1 import sql_from_nl_v1
        rendered = sql_from_nl_v1.render(
            schema_ddl="CREATE TABLE t (id INT);",
            question="Count all rows",
            dialect="",
        )
        assert "STANDARD SQL" in rendered

    def test_expected_shape_validates_json_schema_and_sql_parse(self):
        from effgen.prompts.library.domains.data.sql_from_nl_v1 import sql_from_nl_v1
        assert sql_from_nl_v1.expected_shape is not None
        assert sql_from_nl_v1.expected_shape["type"] == "callable"
        assert callable(sql_from_nl_v1.expected_shape["fn"])
        schema = sql_from_nl_v1.expected_shape["schema"]
        assert "sql" in schema["required"]
        assert "warnings" in schema["required"]

    def test_schema_validation_passes_for_valid_output(self):
        from effgen.prompts.library.domains.data.sql_from_nl_v1 import sql_from_nl_v1
        from effgen.prompts.library.eval import PromptEval
        evaluator = PromptEval()
        valid_output = json.dumps(
            {
                "sql": (
                    "SELECT c.name, COUNT(*) FROM customers c JOIN orders o "
                    "ON c.customer_id = o.customer_id WHERE o.order_date >= "
                    "'2023-01-01' GROUP BY c.customer_id, c.name HAVING COUNT(*) > 3 "
                    "ORDER BY SUM(o.total_amt) DESC"
                ),
                "warnings": [],
            }
        )
        ok, msg = evaluator._check_shape(valid_output, sql_from_nl_v1.expected_shape)
        assert ok, msg

    def test_schema_validation_fails_for_missing_sql(self):
        from effgen.prompts.library.domains.data.sql_from_nl_v1 import sql_from_nl_v1
        from effgen.prompts.library.eval import PromptEval
        evaluator = PromptEval()
        invalid_output = json.dumps({"warnings": []})
        ok, _ = evaluator._check_shape(invalid_output, sql_from_nl_v1.expected_shape)
        assert not ok

    def test_schema_validation_fails_for_extra_field(self):
        from effgen.prompts.library.domains.data.sql_from_nl_v1 import sql_from_nl_v1
        from effgen.prompts.library.eval import PromptEval
        evaluator = PromptEval()
        invalid_output = json.dumps({"sql": "SELECT 1", "warnings": [], "extra": "bad"})
        ok, _ = evaluator._check_shape(invalid_output, sql_from_nl_v1.expected_shape)
        assert not ok

    def test_schema_validation_fails_for_unparseable_sql(self):
        from effgen.prompts.library.domains.data.sql_from_nl_v1 import sql_from_nl_v1
        from effgen.prompts.library.eval import PromptEval

        evaluator = PromptEval()
        invalid_output = json.dumps({"sql": "SELECT FROM", "warnings": []})
        ok, msg = evaluator._check_shape(invalid_output, sql_from_nl_v1.expected_shape)
        assert not ok
        assert "sqlglot" in msg

    def test_sqlglot_parse_on_known_good_sql(self):
        """sqlglot must parse typical SELECT output that the model would emit."""
        import sqlglot
        sample_sql = (
            "SELECT c.name, COUNT(o.order_id) AS order_count, SUM(o.total_amt) AS total_spend "
            "FROM customers c JOIN orders o ON c.customer_id = o.customer_id "
            "WHERE o.order_date >= '2023-01-01' AND o.order_date < '2024-01-01' "
            "GROUP BY c.customer_id, c.name HAVING COUNT(o.order_id) > 3 "
            "ORDER BY total_spend DESC"
        )
        result = sqlglot.parse(sample_sql)
        assert result and len(result) > 0, "sqlglot failed to parse valid SQL"


# ---------------------------------------------------------------------------
# sql_explain_v1
# ---------------------------------------------------------------------------

class TestSqlExplain:
    def test_renders_with_fixture(self):
        from effgen.prompts.library.domains.data.sql_explain_v1 import sql_explain_v1
        rendered = sql_explain_v1.render_fixture()
        assert "customer" in rendered.lower()
        assert "order" in rendered.lower()
        assert "Input schema" in rendered

    def test_business_audience_style(self):
        from effgen.prompts.library.domains.data.sql_explain_v1 import sql_explain_v1
        rendered = sql_explain_v1.render(sql="SELECT 1", audience="business")
        assert "business stakeholder" in rendered.lower()

    def test_technical_audience_style(self):
        from effgen.prompts.library.domains.data.sql_explain_v1 import sql_explain_v1
        rendered = sql_explain_v1.render(sql="SELECT 1", audience="technical")
        assert "technical developer" in rendered.lower()

    def test_three_part_structure_in_prompt(self):
        from effgen.prompts.library.domains.data.sql_explain_v1 import sql_explain_v1
        rendered = sql_explain_v1.render_fixture()
        assert "1." in rendered
        assert "2." in rendered
        assert "3." in rendered

    def test_schema_context_can_be_provided(self):
        from effgen.prompts.library.domains.data.sql_explain_v1 import sql_explain_v1

        rendered = sql_explain_v1.render(
            sql="SELECT COUNT(*) FROM invoices",
            schema_ddl="CREATE TABLE invoices (invoice_id INT);",
            audience="business",
        )
        assert "CREATE TABLE invoices" in rendered
        assert "referenced table names" in rendered

    def test_variant_is_zero_shot(self):
        from effgen.prompts.library.domains.data.sql_explain_v1 import sql_explain_v1
        assert sql_explain_v1.variant == "zero_shot"

    def test_expected_shape_is_regex(self):
        from effgen.prompts.library.domains.data.sql_explain_v1 import sql_explain_v1
        assert sql_explain_v1.expected_shape is not None
        assert sql_explain_v1.expected_shape["type"] == "regex"


# ---------------------------------------------------------------------------
# sql_optimize_v1
# ---------------------------------------------------------------------------

class TestSqlOptimize:
    def test_renders_with_fixture(self):
        from effgen.prompts.library.domains.data.sql_optimize_v1 import sql_optimize_v1
        rendered = sql_optimize_v1.render_fixture()
        assert "Step 1" in rendered
        assert "Step 5" in rendered

    def test_anti_pattern_mentioned(self):
        from effgen.prompts.library.domains.data.sql_optimize_v1 import sql_optimize_v1
        rendered = sql_optimize_v1.render_fixture()
        assert "anti-pattern" in rendered.lower() or "anti_pattern" in rendered.lower()

    def test_schema_block_included_when_provided(self):
        from effgen.prompts.library.domains.data.sql_optimize_v1 import sql_optimize_v1
        rendered = sql_optimize_v1.render_fixture()
        assert "CREATE TABLE" in rendered

    def test_schema_block_omitted_when_empty(self):
        from effgen.prompts.library.domains.data.sql_optimize_v1 import sql_optimize_v1
        rendered = sql_optimize_v1.render(
            sql="SELECT * FROM t",
            schema_ddl="",
            dialect="ansi",
            observed_issue="",
        )
        assert "CREATE TABLE" not in rendered

    def test_variant_is_cot(self):
        from effgen.prompts.library.domains.data.sql_optimize_v1 import sql_optimize_v1
        assert sql_optimize_v1.variant == "cot"


# ---------------------------------------------------------------------------
# data_profile_v1
# ---------------------------------------------------------------------------

class TestDataProfile:
    def test_renders_with_fixture(self):
        from effgen.prompts.library.domains.data.data_profile_v1 import data_profile_v1
        rendered = data_profile_v1.render_fixture()
        assert "orders_2022_2024" in rendered
        assert "Completeness" in rendered
        assert "Uniqueness" in rendered

    def test_six_sections_present(self):
        from effgen.prompts.library.domains.data.data_profile_v1 import data_profile_v1
        rendered = data_profile_v1.render_fixture()
        for heading in ("Schema Overview", "Completeness", "Uniqueness",
                        "Range", "Quality", "Recommended Actions"):
            assert heading in rendered, f"Missing section heading: {heading}"

    def test_row_count_formatted(self):
        from effgen.prompts.library.domains.data.data_profile_v1 import data_profile_v1
        rendered = data_profile_v1.render(
            column_stats='[{"column": "id", "dtype": "int64", "null_count": 0, "unique_count": 5000}]',
            dataset_name="test_ds",
            row_count=10000,
        )
        assert "10,000" in rendered

    def test_row_count_omitted_when_zero(self):
        from effgen.prompts.library.domains.data.data_profile_v1 import data_profile_v1
        rendered = data_profile_v1.render(
            column_stats='[{"column": "id"}]',
            dataset_name="test_ds",
            row_count=0,
        )
        assert "(0 rows)" not in rendered

    def test_variant_is_tool(self):
        from effgen.prompts.library.domains.data.data_profile_v1 import data_profile_v1
        assert data_profile_v1.variant == "tool"


# ---------------------------------------------------------------------------
# etl_plan_v1
# ---------------------------------------------------------------------------

class TestEtlPlan:
    def test_renders_with_fixture(self):
        from effgen.prompts.library.domains.data.etl_plan_v1 import etl_plan_v1
        rendered = etl_plan_v1.render_fixture()
        assert "Extract" in rendered
        assert "Transform" in rendered
        assert "Load" in rendered

    def test_has_two_exemplars(self):
        from effgen.prompts.library.domains.data.etl_plan_v1 import etl_plan_v1
        rendered = etl_plan_v1.render_fixture()
        assert "ETL Example 1" in rendered
        assert "ETL Example 2" in rendered

    def test_five_stage_structure_in_prompt(self):
        from effgen.prompts.library.domains.data.etl_plan_v1 import etl_plan_v1
        rendered = etl_plan_v1.render_fixture()
        for stage in ("Extract", "Transform", "Load", "Validate", "Cleanup"):
            assert stage in rendered, f"Missing stage: {stage}"

    def test_transformations_block_included_when_provided(self):
        from effgen.prompts.library.domains.data.etl_plan_v1 import etl_plan_v1
        rendered = etl_plan_v1.render(
            source_description="CSV file",
            target_description="Postgres table",
            transformations="Normalize phone numbers.",
            schedule="daily",
        )
        assert "Normalize phone numbers" in rendered

    def test_transformations_block_omitted_when_empty(self):
        from effgen.prompts.library.domains.data.etl_plan_v1 import etl_plan_v1
        rendered = etl_plan_v1.render(
            source_description="CSV file",
            target_description="Postgres table",
            transformations="",
            schedule="daily",
        )
        assert "Required transformations:" not in rendered

    def test_variant_is_few_shot(self):
        from effgen.prompts.library.domains.data.etl_plan_v1 import etl_plan_v1
        assert etl_plan_v1.variant == "few_shot"


# ---------------------------------------------------------------------------
# Golden eval
# ---------------------------------------------------------------------------

class TestGoldenEval:
    def test_all_goldens_pass(self):
        from effgen.prompts.library.eval import PromptEval
        from effgen.prompts.library.registry import registry

        evaluator = PromptEval()
        prompts = registry.search(domain="data")
        assert len(prompts) >= 5

        report = evaluator.eval_all_golden(prompts)
        failed = report.failed()
        assert not failed, f"Golden failures: {[r.name + ': ' + r.message for r in failed]}"


# ---------------------------------------------------------------------------
# sqlglot parse assertion
# ---------------------------------------------------------------------------

class TestSqlglotParse:
    """Unit tests for sqlglot parse validation logic."""

    def test_sqlglot_parses_simple_select(self):
        import sqlglot
        sql = "SELECT id, name FROM users WHERE active = 1"
        result = sqlglot.parse(sql)
        assert result and len(result) > 0

    def test_sqlglot_parses_join_query(self):
        import sqlglot
        sql = (
            "SELECT c.name, COUNT(*) AS cnt "
            "FROM customers c JOIN orders o ON c.customer_id = o.customer_id "
            "GROUP BY c.customer_id, c.name"
        )
        result = sqlglot.parse(sql)
        assert result and len(result) > 0

    def test_sqlglot_parses_having_clause(self):
        import sqlglot
        sql = (
            "SELECT customer_id, SUM(total_amt) AS total "
            "FROM orders "
            "GROUP BY customer_id "
            "HAVING SUM(total_amt) > 1000"
        )
        result = sqlglot.parse(sql)
        assert result and len(result) > 0

    def test_sqlglot_parses_subquery(self):
        import sqlglot
        sql = (
            "SELECT name FROM customers "
            "WHERE customer_id IN (SELECT customer_id FROM orders WHERE total_amt > 500)"
        )
        result = sqlglot.parse(sql)
        assert result and len(result) > 0

    def test_sqlglot_parses_mysql_dialect(self):
        import sqlglot
        sql = "SELECT id, name FROM users WHERE active = 1 LIMIT 10"
        result = sqlglot.parse(sql, dialect="mysql")
        assert result and len(result) > 0

    def test_sqlglot_parse_extracts_sql_from_json_output(self):
        """Simulate extracting and parsing SQL from model JSON output."""
        import sqlglot
        model_output = json.dumps({
            "sql": (
                "SELECT c.name, COUNT(o.order_id) AS order_count "
                "FROM customers c "
                "JOIN orders o ON c.customer_id = o.customer_id "
                "WHERE o.order_date >= '2023-01-01' "
                "GROUP BY c.customer_id, c.name "
                "HAVING COUNT(o.order_id) > 3 "
                "ORDER BY order_count DESC"
            ),
            "warnings": [],
        })
        parsed_json = json.loads(model_output)
        sql = parsed_json["sql"]
        result = sqlglot.parse(sql)
        assert result and len(result) > 0, "sqlglot could not parse extracted SQL"


# ---------------------------------------------------------------------------
# Live eval
# ---------------------------------------------------------------------------

class TestLiveEval:
    @pytest.mark.skipif(
        not os.environ.get("CEREBRAS_API_KEY"),
        reason="CEREBRAS_API_KEY not set",
    )
    def test_sql_from_nl_live_json_and_sqlglot(self):
        """sql_from_nl_v1 live output must produce valid JSON with parseable SQL."""
        import sqlglot

        from effgen.prompts.library.domains.data.sql_from_nl_v1 import sql_from_nl_v1
        from effgen.prompts.library.eval import PromptEval

        evaluator = PromptEval()
        result = evaluator.eval_live(sql_from_nl_v1, model="gpt-oss-120b")
        assert_live_eval_passed(result)
        parsed = PromptEval._extract_json_object(result.model_output)
        assert parsed is not None, f"No JSON object found in output: {result.model_output[:300]}"
        assert "sql" in parsed, "Missing 'sql' key"
        sql = parsed["sql"]
        parsed_sql = sqlglot.parse(sql)
        assert parsed_sql and len(parsed_sql) > 0, f"sqlglot could not parse: {sql}"

    @pytest.mark.skipif(
        not os.environ.get("CEREBRAS_API_KEY"),
        reason="CEREBRAS_API_KEY not set",
    )
    def test_sql_explain_live_mentions_tables(self):
        """sql_explain_v1 live output must reference table names from the input."""
        from effgen.prompts.library.domains.data.sql_explain_v1 import sql_explain_v1
        from effgen.prompts.library.eval import PromptEval

        evaluator = PromptEval()
        result = evaluator.eval_live(sql_explain_v1, model="gpt-oss-120b")
        assert_live_eval_passed(result)
        output = result.model_output.lower()
        assert "customer" in output or "order" in output, (
            f"Output doesn't mention any table names: {result.model_output[:300]}"
        )

    @pytest.mark.skipif(
        not os.environ.get("CEREBRAS_API_KEY"),
        reason="CEREBRAS_API_KEY not set",
    )
    def test_sql_optimize_live_has_steps(self):
        """sql_optimize_v1 live output must contain step headings."""
        from effgen.prompts.library.domains.data.sql_optimize_v1 import sql_optimize_v1
        from effgen.prompts.library.eval import PromptEval

        evaluator = PromptEval()
        result = evaluator.eval_live(sql_optimize_v1, model="gpt-oss-120b")
        assert_live_eval_passed(result)

    @pytest.mark.skipif(
        not os.environ.get("CEREBRAS_API_KEY"),
        reason="CEREBRAS_API_KEY not set",
    )
    def test_etl_plan_live_has_stages(self):
        """etl_plan_v1 live output must contain ETL stage headings."""
        from effgen.prompts.library.domains.data.etl_plan_v1 import etl_plan_v1
        from effgen.prompts.library.eval import PromptEval

        evaluator = PromptEval()
        result = evaluator.eval_live(etl_plan_v1, model="gpt-oss-120b")
        assert_live_eval_passed(result)
