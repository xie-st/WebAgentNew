"""
Business prompt: OKR Generate — CoT variant.

Inputs: team_name (str), mission (str), strategic_priorities (list[str]),
        timeframe (str), num_objectives (int)
"""

from __future__ import annotations

from effgen.prompts.library.base import LibraryPrompt
from effgen.prompts.library.registry import registry

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "team_name": {
            "type": "string",
            "description": "Name of the team or organization.",
            "minLength": 1,
        },
        "mission": {
            "type": "string",
            "description": "Team mission statement or purpose.",
            "minLength": 10,
        },
        "strategic_priorities": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Top strategic priorities for this period.",
            "minItems": 1,
        },
        "timeframe": {
            "type": "string",
            "description": "OKR cycle, e.g. 'Q3 2025', 'H2 2025'.",
            "minLength": 1,
        },
        "num_objectives": {
            "type": "integer",
            "description": "Number of objectives to generate (typically 3-5).",
            "minimum": 1,
            "maximum": 7,
        },
    },
    "required": ["team_name", "mission", "strategic_priorities", "timeframe", "num_objectives"],
}

_FIXTURE = {
    "team_name": "Platform Engineering",
    "mission": "Build reliable, scalable infrastructure that enables product teams to ship fast.",
    "strategic_priorities": [
        "reduce system downtime to under 0.1% monthly",
        "cut developer onboarding time in half",
        "migrate 80% of services to Kubernetes",
    ],
    "timeframe": "Q3 2025",
    "num_objectives": 3,
}


def _okr_generate_cot(
    team_name: str,
    mission: str,
    strategic_priorities: list[str],
    timeframe: str,
    num_objectives: int,
) -> str:
    priorities_str = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(strategic_priorities))
    return (
        f"You are an expert OKR (Objectives and Key Results) coach. "
        f"Generate high-quality, ambitious yet achievable OKRs for the following team.\n\n"
        f"Team: {team_name}\n"
        f"Mission: {mission}\n"
        f"Timeframe: {timeframe}\n"
        f"Strategic Priorities:\n{priorities_str}\n\n"
        f"Think through the OKR creation step by step:\n\n"
        f"Step 1 — Alignment check: How do the strategic priorities connect to the team mission? "
        f"Identify the top {num_objectives} themes to focus on.\n\n"
        f"Step 2 — Objective drafting: For each theme, write an inspiring, qualitative Objective "
        f"(a direction, not a metric). Objectives should be memorable and motivating.\n\n"
        f"Step 3 — Key Results: For each Objective, write 3 specific, measurable, time-bound "
        f"Key Results. Each KR must have a clear baseline, target, and unit of measurement.\n\n"
        f"Step 4 — Ambition calibration: Review each KR. Is it a stretch goal (70% confidence "
        f"of achieving)? Adjust if too safe or too unrealistic.\n\n"
        f"Step 5 — Output: Present the final {num_objectives} OKRs in this format:\n"
        f"  O1: [Objective statement]\n"
        f"    KR1: [Metric] from [baseline] to [target] by {timeframe}\n"
        f"    KR2: ...\n"
        f"    KR3: ...\n\n"
        f"Work through each step before presenting the final OKRs."
    )


okr_generate_v1 = LibraryPrompt(
    name="business.okr_generate.v1",
    domain="business",
    variant="cot",
    description=(
        "Chain-of-thought OKR generator. Produces aligned objectives and measurable key results "
        "from team mission and strategic priorities."
    ),
    template=_okr_generate_cot,
    input_schema=_INPUT_SCHEMA,
    fixture=_FIXTURE,
    expected_shape={
        "type": "regex",
        "pattern": r"(?i)O\d+:|objective|KR\d+:|key result",
    },
    tags=["business", "okr", "strategy", "planning", "cot"],
)

registry.register(okr_generate_v1)
