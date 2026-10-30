"""Phase-1 WebAgent harness.

A1 provides the immutable protocol, session lifecycle, the single tool schema,
budget helpers, events/evidence writing, configuration and offline fixtures.
A2 adds the interactive EffGen loop; B1 adds the real text-model adapter,
Playwright/browser-use backends and the doctor/run CLI; C1 adds the official
model/backend adapter and the SDK reset/step/close entry.
"""

from .protocol import (
    SCHEMA_VERSION,
    ActionOp,
    Artifact,
    ArtifactSet,
    Capabilities,
    ElementRef,
    ExecutionResult,
    Feedback,
    FeedbackStatus,
    Finish,
    Observation,
    ScrollDirection,
    TerminationReason,
    ToolRequest,
    WebAction,
    new_action_id,
)

__all__ = [
    "SCHEMA_VERSION",
    "ActionOp",
    "Artifact",
    "ArtifactSet",
    "Capabilities",
    "ElementRef",
    "ExecutionResult",
    "Feedback",
    "FeedbackStatus",
    "Finish",
    "Observation",
    "ScrollDirection",
    "TerminationReason",
    "ToolRequest",
    "WebAction",
    "new_action_id",
]

__version__ = "0.1.0"
