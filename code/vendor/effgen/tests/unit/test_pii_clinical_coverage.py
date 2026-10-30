"""Coverage for label-anchored PII redaction, custom patterns, strict mode,
the de-identification preset, and the agent-side redaction/budget wiring.

All checks run offline: the regex redactor is deterministic, and the agent-level
tests use a canned model so no live inference is required.
"""

from __future__ import annotations

import re

import pytest

from effgen.guardrails import (
    GuardrailChain,
    PIIGuardrail,
    get_guardrail_preset,
)
from tests.fixtures.mock_models import MockModel

INLINE_NOTE = (
    "Patient Maria Gonzalez, DOB 04/12/1978, MRN 00847213, SSN 412-88-1043, "
    "phone (617) 555-0142, email maria.g@gmail.com, "
    "address 42 Oakwood Ave Boston MA 02118, insurance member ID BCBS-9932-1187. "
    "Summarize the clinical facts in one sentence."
)

MULTILINE_NOTE = (
    "PATIENT NOTE\n"
    "Name: Maria Gonzalez  DOB: 04/12/1978  MRN: 00847213\n"
    "Address: 42 Oakwood Ave, Boston, MA 02118\n"
    "Insurance member ID: BCBS-9932-1187\n"
    "Follow-up with Dr. Stephen Alvarez in 2 weeks."
)

PHI_TOKENS = [
    "Maria", "Gonzalez", "04/12/1978", "00847213", "412-88-1043",
    "maria.g@gmail.com", "Oakwood", "BCBS-9932-1187", "Stephen", "Alvarez",
]


class TestLabeledFieldRedaction:
    def test_inline_note_removes_all_identifiers(self):
        r = PIIGuardrail(action="redact").check(INLINE_NOTE)
        out = r.modified_content
        for tok in ["Maria", "Gonzalez", "04/12/1978", "00847213",
                    "412-88-1043", "maria.g@gmail.com", "Oakwood", "BCBS-9932-1187"]:
            assert tok not in out, f"leaked {tok!r}: {out}"
        types = r.metadata["pii_types"]
        for expected in ("name", "DOB", "MRN", "member_id", "address"):
            assert expected in types, f"{expected} not detected: {types}"

    def test_trailing_instruction_survives_redaction(self):
        # The redactor must not swallow non-PHI text that follows an address on
        # the same line — the instruction after the ZIP is kept.
        r = PIIGuardrail(action="redact").check(INLINE_NOTE)
        assert "Summarize the clinical facts in one sentence." in r.modified_content

    def test_multiline_labeled_fields(self):
        r = PIIGuardrail(action="redact").check(MULTILINE_NOTE)
        out = r.modified_content
        for tok in PHI_TOKENS:
            assert tok not in out, f"leaked {tok!r}: {out}"
        # Labels are preserved; only values are replaced.
        assert "Name: [NAME REDACTED]" in out
        assert "MRN: [MRN REDACTED]" in out
        assert "Dr. [NAME REDACTED]" in out

    @pytest.mark.parametrize("text,leaks", [
        # Middle initial + apostrophe/hyphen surnames must be captured whole,
        # not truncated to the first token.
        ("Name: John Q. O'Brien is stable.", ["John", "O'Brien"]),
        ("Patient name: Robert A. Smith-Jones Jr", ["Robert", "Smith-Jones"]),
        # Abbreviated / punctuated labels: D.O.B.: and Medical Record #: and the
        # dash separator (DOB - ...) and #-then-colon combos.
        ("D.O.B.: 1965-03-02", ["1965-03-02"]),
        ("DOB - 04/12/1978", ["04/12/1978"]),
        ("Medical Record #: A19-4432", ["A19-4432"]),
        ("MRN# 00847213 filed", ["00847213"]),
        # A name value must stop at the next field label, not run into it.
        ("Name: Maria Gonzalez  MRN: 00847213", ["Maria", "Gonzalez", "00847213"]),
    ])
    def test_label_variant_formats_redacted(self, text, leaks):
        out = PIIGuardrail(action="redact").check(text).modified_content or text
        for tok in leaks:
            assert tok not in out, f"leaked {tok!r}: {out}"

    def test_name_value_stops_at_next_label(self):
        # The name must not swallow the following MRN label ("...Gonzalez  M").
        out = PIIGuardrail(action="redact").check(
            "Name: Maria Gonzalez  MRN: 00847213"
        ).modified_content
        assert "MRN: [MRN REDACTED]" in out

    def test_labeled_detection_can_be_disabled(self):
        r = PIIGuardrail(action="redact", detect_labeled=False).check(
            "Name: Maria Gonzalez  MRN: 00847213"
        )
        assert r.metadata.get("pii_types") in (None, [])
        assert r.modified_content in (None, "Name: Maria Gonzalez  MRN: 00847213")

    @pytest.mark.parametrize("text", [
        "What is the name of the CEO of Apple?",
        "Name three benefits of regular exercise.",
        "The patient presents with a cough and fever.",
        "Please summarize the patient education leaflet.",
        "The address of the venue is not yet decided.",
        "Record number of visitors reached a new high.",
    ])
    def test_no_false_positive_on_benign_prose(self, text):
        r = PIIGuardrail(action="redact").check(text)
        assert not r.metadata.get("pii_types"), f"false positive on {text!r}"
        assert r.modified_content in (None, text)

    @pytest.mark.parametrize("text", [
        # A shared trailing \b after the whole alternation cannot match right
        # after "#" (a non-word char can't satisfy \b next to another
        # non-word char), so these silently failed to redact.
        "Record #: 12345",
        "Record # 12345",
        "Record #:12345",
    ])
    def test_bare_record_hash_variants_redacted(self, text):
        out = PIIGuardrail(action="redact").check(text).modified_content or text
        assert "12345" not in out, f"leaked MRN value: {out}"
        assert "[MRN REDACTED]" in out

    @pytest.mark.parametrize("text", [
        "Record No 12345",
        "Record Number: 12345",
    ])
    def test_bare_record_no_number_variants_still_redacted(self, text):
        # The word-ending alternatives ("no"/"number") must keep their own
        # trailing \b after the split from the "#" alternative.
        out = PIIGuardrail(action="redact").check(text).modified_content or text
        assert "12345" not in out, f"leaked MRN value: {out}"
        assert "[MRN REDACTED]" in out


class TestInternationalPhoneNumbers:
    @pytest.mark.parametrize("phone", [
        "+44 20 7946 0958",   # UK, space-grouped
        "+91 98765 43210",    # India, space-grouped
        "+33 1 42 68 53 00",  # France, multi-group
        "+442079460958",      # ungrouped — must keep working
        "+1-800-555-0199",    # dash-grouped
    ])
    def test_grouped_and_ungrouped_international_numbers_redacted(self, phone):
        r = PIIGuardrail(action="redact").check(f"Call me at {phone} tomorrow.")
        assert phone not in (r.modified_content or ""), f"leaked {phone!r}"
        assert "[PHONE REDACTED]" in r.modified_content

    def test_short_digit_run_not_flagged_as_phone(self):
        # A country-code-shaped prefix with too few trailing digits (e.g. a
        # version-like token) should not be swept up.
        r = PIIGuardrail(action="redact").check("see spec +1 2")
        assert r.modified_content in (None, "see spec +1 2")


class TestSSNFormats:
    @pytest.mark.parametrize("text", [
        "SSN 412-88-1043 on file.",
        "SSN 412 88 1043 on file.",
        "SSN: 412881043 on file.",
        "Social Security Number 412-88-1043.",
    ])
    def test_ssn_variants_redacted(self, text):
        r = PIIGuardrail(action="redact").check(text)
        assert "SSN" in r.metadata.get("pii_types", []), text
        assert "412" not in r.modified_content, text

    def test_bare_nine_digits_without_cue_not_redacted_as_ssn(self):
        # A plain 9-digit number with no SSN cue is ambiguous with other ids and
        # must not be redacted as an SSN.
        r = PIIGuardrail(action="redact").check("account 412881043 balance")
        assert "SSN" not in r.metadata.get("pii_types", [])


class TestCustomPatternsAndTerms:
    def test_custom_regex_pattern_with_label(self):
        g = PIIGuardrail(action="redact",
                         custom_patterns=[(r"BCBS-\d{4}-\d{4}", "member_id")])
        r = g.check("member BCBS-9932-1187 active")
        assert "member_id" in r.metadata["pii_types"]
        assert "[MEMBER_ID REDACTED]" in r.modified_content

    def test_custom_pattern_plain_string_defaults_to_custom(self):
        g = PIIGuardrail(action="redact", custom_patterns=[r"CASE-\d+"])
        r = g.check("see CASE-42 for details")
        assert "custom" in r.metadata["pii_types"]
        assert "CASE-42" not in r.modified_content

    def test_compiled_pattern_accepted(self):
        g = PIIGuardrail(action="redact",
                         custom_patterns=[re.compile(r"Z\d{3}")])
        assert "Z123" not in g.check("code Z123 here").modified_content

    def test_custom_terms_case_insensitive_whole_word(self):
        g = PIIGuardrail(action="redact",
                         custom_terms=[("Springfield Clinic", "facility")])
        r = g.check("Seen at springfield clinic today.")
        assert "[FACILITY REDACTED]" in r.modified_content
        assert "facility" in r.metadata["pii_types"]


class TestStrictMode:
    def test_strict_redact_fails_closed_on_detection(self):
        g = PIIGuardrail(action="redact", strict=True)
        r = g.check("Name: Maria Gonzalez  MRN: 00847213")
        assert r.passed is False
        # The redacted text is still available for inspection.
        assert "Maria" not in (r.modified_content or "")
        assert "MRN" in r.metadata["pii_types"]

    def test_strict_passes_when_nothing_detected(self):
        g = PIIGuardrail(action="redact", strict=True)
        r = g.check("The weather is fine today.")
        assert r.passed is True

    def test_non_strict_redact_still_passes(self):
        g = PIIGuardrail(action="redact")
        r = g.check("Name: Maria Gonzalez")
        assert r.passed is True


class TestRedactionCountsAndChain:
    def test_counts_reported(self):
        r = PIIGuardrail(action="redact").check(
            "email a@b.com and email c@d.com"
        )
        assert r.metadata["pii_counts"]["email"] == 2

    def test_chain_aggregates_pii_types(self):
        chain = get_guardrail_preset("phi")
        r = chain.check("Name: Maria Gonzalez  MRN: 00847213")
        assert "name" in r.metadata.get("pii_types", [])
        assert "MRN" in r.metadata.get("pii_types", [])
        assert "Maria" not in (r.modified_content or "")

    def test_plain_chain_without_pii_has_no_pii_metadata(self):
        chain = GuardrailChain([PIIGuardrail(action="redact")])
        r = chain.check("nothing sensitive here")
        assert "pii_types" not in r.metadata


class TestPhiPreset:
    def test_phi_and_hipaa_resolve(self):
        assert len(get_guardrail_preset("phi").guardrails) == 6
        assert len(get_guardrail_preset("hipaa").guardrails) == 6
        assert len(get_guardrail_preset("deidentify").guardrails) == 6

    def test_phi_preset_redacts_not_blocks_by_default(self):
        r = get_guardrail_preset("phi").check("MRN: 00847213")
        assert r.passed is True
        assert "00847213" not in (r.modified_content or "")


class TestAgentRedactionWiring:
    def test_input_redaction_summary_in_metadata(self):
        model = MockModel(responses=["The patient has pneumonia."])
        from effgen.core.agent import Agent, AgentConfig
        cfg = AgentConfig(model=model, guardrails="phi",
                          enable_sub_agents=False, enable_memory=False)
        r = Agent(config=cfg).run(
            "Patient Maria Gonzalez, MRN 00847213 has pneumonia. Summarize."
        )
        red = r.metadata.get("input_redaction")
        assert red is not None
        assert "name" in red["types"] and "MRN" in red["types"]
        # The model never received the identifiers.
        assert "Maria" not in model._generate_calls[-1]["prompt"]

    def test_config_max_tokens_is_used(self):
        model = MockModel(responses=["ok"])
        from effgen.core.agent import Agent, AgentConfig
        cfg = AgentConfig(model=model, max_tokens=256,
                          enable_sub_agents=False, enable_memory=False)
        Agent(config=cfg).run("hello")
        assert model._generate_calls[-1]["config"].max_tokens == 256

    def test_run_max_tokens_overrides_config(self):
        model = MockModel(responses=["ok"])
        from effgen.core.agent import Agent, AgentConfig
        cfg = AgentConfig(model=model, max_tokens=256,
                          enable_sub_agents=False, enable_memory=False)
        Agent(config=cfg).run("hello", max_tokens=999)
        assert model._generate_calls[-1]["config"].max_tokens == 999


class TestStructuredOutputEmptyHeuristic:
    def test_all_empty_object_flagged(self):
        from effgen.core.agent import Agent
        assert Agent._structured_all_empty({"dx": "", "meds": []}) is True
        assert Agent._structured_all_empty({"a": {"b": ""}}) is True
        assert Agent._structured_all_empty([]) is True

    def test_populated_object_not_flagged(self):
        from effgen.core.agent import Agent
        assert Agent._structured_all_empty({"dx": "pneumonia"}) is False
        assert Agent._structured_all_empty({"n": 0}) is False
        assert Agent._structured_all_empty([{"x": 1}]) is False
