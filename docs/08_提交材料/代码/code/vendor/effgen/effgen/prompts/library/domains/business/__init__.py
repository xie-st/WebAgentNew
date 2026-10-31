"""Business domain prompts: meeting summaries, emails, OKRs, SWOT, elevator pitch."""

from .elevator_pitch_v1 import elevator_pitch_v1
from .email_draft_v1 import email_draft_v1
from .meeting_summary_v1 import meeting_summary_v1
from .okr_generate_v1 import okr_generate_v1
from .swot_analysis_v1 import swot_analysis_v1

__all__ = [
    "meeting_summary_v1",
    "email_draft_v1",
    "okr_generate_v1",
    "swot_analysis_v1",
    "elevator_pitch_v1",
]
