"""Tests for PromptEval: golden and live harness."""

from __future__ import annotations

from effgen.prompts.library.base import LibraryPrompt
from effgen.prompts.library.eval import EvalReport, PromptEval


def _make_prompt(name="test.eval.v1", template=None, fixture=None, expected_shape=None):
    if template is None:

        def template(topic="AI"):
            return f"Explain {topic} in one sentence."

    if fixture is None:
        fixture = {"topic": "machine learning"}
    return LibraryPrompt(
        name=name,
        domain="test",
        variant="zero_shot",
        description="Test prompt for eval harness",
        template=template,
        input_schema={"type": "object", "properties": {"topic": {"type": "string"}}},
        fixture=fixture,
        expected_shape=expected_shape,
        tags=["test"],
    )


class TestGoldenEval:
    def test_missing_golden_creates_file(self, tmp_path):
        ev = PromptEval(goldens_dir=tmp_path)
        p = _make_prompt()
        result = ev.eval_golden(p)
        golden_file = tmp_path / "test.eval.v1.txt"
        assert golden_file.exists()
        assert result.passed
        assert "created on first run" in result.message

    def test_matching_golden_passes(self, tmp_path):
        ev = PromptEval(goldens_dir=tmp_path)
        p = _make_prompt()
        rendered = p.render_fixture()
        (tmp_path / "test.eval.v1.txt").write_text(rendered)
        result = ev.eval_golden(p)
        assert result.passed

    def test_mismatched_golden_fails(self, tmp_path):
        ev = PromptEval(goldens_dir=tmp_path)
        p = _make_prompt()
        (tmp_path / "test.eval.v1.txt").write_text("something completely different")
        result = ev.eval_golden(p)
        assert not result.passed
        assert "differs" in result.message

    def test_render_error_fails(self, tmp_path):
        def bad_template(**kw):
            raise ValueError("boom")
        p = _make_prompt(template=bad_template)
        ev = PromptEval(goldens_dir=tmp_path)
        result = ev.eval_golden(p)
        assert not result.passed
        assert "render error" in result.message

    def test_whitespace_insensitive_match(self, tmp_path):
        ev = PromptEval(goldens_dir=tmp_path)
        p = _make_prompt()
        rendered = p.render_fixture()
        (tmp_path / "test.eval.v1.txt").write_text(rendered + "\n\n")
        result = ev.eval_golden(p)
        assert result.passed


class TestEvalReport:
    def test_empty_report_table(self):
        report = EvalReport()
        table = report.as_table()
        assert "empty" in table.lower() or "No prompts" in table

    def test_pass_fail_counts(self, tmp_path):
        ev = PromptEval(goldens_dir=tmp_path)
        p1 = _make_prompt(name="a.p1.v1")
        p2 = _make_prompt(name="b.p2.v1")
        rendered = p1.render_fixture()
        (tmp_path / "a.p1.v1.txt").write_text(rendered)
        (tmp_path / "b.p2.v1.txt").write_text("wrong content")
        report = ev.eval_all_golden([p1, p2])
        assert len(report.passed()) == 1
        assert len(report.failed()) == 1

    def test_table_format(self, tmp_path):
        ev = PromptEval(goldens_dir=tmp_path)
        p = _make_prompt()
        report = ev.eval_all_golden([p])
        table = report.as_table()
        assert "golden" in table
        assert "test.eval.v1" in table


class TestShapeChecks:
    def test_json_shape_valid(self, tmp_path):
        shape = {"type": "json", "schema": {"required": ["answer"]}}
        ev = PromptEval(goldens_dir=tmp_path)

        # Directly test _check_shape
        ok, msg = ev._check_shape('{"answer": "yes"}', shape)
        assert ok

    def test_json_shape_missing_key(self, tmp_path):
        shape = {"type": "json", "schema": {"required": ["answer"]}}
        ev = PromptEval(goldens_dir=tmp_path)
        ok, msg = ev._check_shape('{"other": "x"}', shape)
        assert not ok
        assert "answer" in msg

    def test_json_shape_validates_schema_types(self, tmp_path):
        shape = {
            "type": "json",
            "schema": {
                "type": "object",
                "properties": {"answers": {"type": "array"}},
                "required": ["answers"],
            },
        }
        ev = PromptEval(goldens_dir=tmp_path)
        ok, msg = ev._check_shape('{"answers": "not a list"}', shape)
        assert not ok
        assert "schema validation failed" in msg

    def test_json_shape_rejects_non_object(self, tmp_path):
        shape = {"type": "json"}
        ev = PromptEval(goldens_dir=tmp_path)
        ok, msg = ev._check_shape('["answer"]', shape)
        assert not ok
        assert "object" in msg

    def test_json_shape_invalid_json(self, tmp_path):
        shape = {"type": "json"}
        ev = PromptEval(goldens_dir=tmp_path)
        ok, msg = ev._check_shape("not json", shape)
        assert not ok

    def test_json_shape_accepts_fenced_python_style_string_fragments(self, tmp_path):
        shape = {"type": "json", "schema": {"required": ["sql", "warnings"]}}
        ev = PromptEval(goldens_dir=tmp_path)
        output = (
            "```json\n"
            "{\n"
            '  "sql": "SELECT * " "FROM customers",\n'
            '  "warnings": []\n'
            "}\n"
            "```\n"
            "extra text"
        )
        ok, msg = ev._check_shape(output, shape)
        assert ok, msg

    def test_regex_shape_match(self, tmp_path):
        shape = {"type": "regex", "pattern": r"\d+ papers"}
        ev = PromptEval(goldens_dir=tmp_path)
        ok, msg = ev._check_shape("Found 42 papers in total", shape)
        assert ok

    def test_regex_shape_no_match(self, tmp_path):
        shape = {"type": "regex", "pattern": r"\d+ papers"}
        ev = PromptEval(goldens_dir=tmp_path)
        ok, msg = ev._check_shape("No mention", shape)
        assert not ok

    def test_callable_shape(self, tmp_path):
        shape = {"type": "callable", "fn": lambda s: len(s) > 5 or "too short"}
        ev = PromptEval(goldens_dir=tmp_path)
        ok, msg = ev._check_shape("hello world", shape)
        assert ok
        ok2, msg2 = ev._check_shape("hi", shape)
        assert not ok2

    def test_unknown_shape_type_passes(self, tmp_path):
        ev = PromptEval(goldens_dir=tmp_path)
        ok, msg = ev._check_shape("anything", {"type": "unknown_xyz"})
        assert ok
