"""G-wave context protection: necessary-region views and the final-send check.

The package keeps the four protected regions (task text, current AX observation,
ledger short view, unconfirmed side effects) and the one deterministic check that
runs before each model send. It is inert while ``context.enabled`` is false: the
official entry builds no protector and the driver is unchanged.
"""

from .errors import (
    ContextBudgetExceeded,
    ContextError,
    ContextProtectionError,
)
from .guard import (
    VERDICT_MISSING,
    VERDICT_OK,
    VERDICT_OVER_LIMIT,
    ContextCheckReport,
    SectionCheck,
    check_sections,
    describe,
    evaluate,
    prompt_messages_and_text,
)
from .protector import RECORDS_FILE_NAME, ContextProtector
from .sections import (
    EFFECTS_HEADER,
    LEDGER_HEADER,
    NO_EFFECTS,
    NO_LEDGER,
    SECTION_LABELS,
    SECTION_ORDER,
    UNCONFIRMED_EFFECT_STATUSES,
    ContextBlock,
    NecessarySection,
    SectionKind,
    build_necessary_sections,
    byte_length,
    effects_block,
    ledger_block,
    total_bytes,
)

__all__ = [
    "EFFECTS_HEADER",
    "LEDGER_HEADER",
    "NO_EFFECTS",
    "NO_LEDGER",
    "RECORDS_FILE_NAME",
    "SECTION_LABELS",
    "SECTION_ORDER",
    "UNCONFIRMED_EFFECT_STATUSES",
    "VERDICT_MISSING",
    "VERDICT_OK",
    "VERDICT_OVER_LIMIT",
    "ContextBlock",
    "ContextBudgetExceeded",
    "ContextCheckReport",
    "ContextError",
    "ContextProtectionError",
    "ContextProtector",
    "NecessarySection",
    "SectionCheck",
    "SectionKind",
    "build_necessary_sections",
    "byte_length",
    "check_sections",
    "describe",
    "effects_block",
    "evaluate",
    "ledger_block",
    "prompt_messages_and_text",
    "total_bytes",
]
