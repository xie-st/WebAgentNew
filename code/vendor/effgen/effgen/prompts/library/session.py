"""
PlaygroundSession — save/restore format for the prompt playground.

A session stores:
- the selected prompt name
- current variable bindings (inputs)
- render history (rendered prompt strings)
- run history (model outputs)

Serialized as JSON under ~/.effgen/playground/<timestamp>.json
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SESSIONS_DIR = Path.home() / ".effgen" / "playground"


@dataclass
class RunEntry:
    """One playground run: the model, the rendered prompt, and its output."""

    model: str
    rendered: str
    output: str
    timestamp: str = field(default_factory=lambda: _utcnow())


@dataclass
class PlaygroundSession:
    """Prompt-playground state: selected prompt, variables and history."""

    prompt_name: str = ""
    variables: dict[str, Any] = field(default_factory=dict)
    render_history: list[str] = field(default_factory=list)
    run_history: list[RunEntry] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: _utcnow())
    updated_at: str = field(default_factory=lambda: _utcnow())

    # ------------------------------------------------------------------

    def touch(self) -> None:
        """Refresh ``updated_at``."""
        self.updated_at = _utcnow()

    def set_var(self, key: str, value: Any) -> None:
        """Set a template variable."""
        self.variables[key] = value
        self.touch()

    def unset_var(self, key: str) -> bool:
        """Remove a template variable; returns whether it existed."""
        if key in self.variables:
            del self.variables[key]
            self.touch()
            return True
        return False

    def add_render(self, rendered: str) -> None:
        """Record a rendered prompt."""
        self.render_history.append(rendered)
        self.touch()

    def add_run(self, model: str, rendered: str, output: str) -> None:
        """Record a model run.

        Args:
            model: The model that was called.
            rendered: The prompt text that was sent.
            output: What the model returned.
        """
        self.run_history.append(RunEntry(model=model, rendered=rendered, output=output))
        self.touch()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | None = None) -> Path:
        """Serialize session to JSON and write to *path*.

        If *path* is omitted a timestamped file under
        ``~/.effgen/playground/`` is created automatically.
        """
        if path is None:
            _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            safe = re.sub(r"[^a-zA-Z0-9._-]", "_", self.prompt_name) or "session"
            path = _SESSIONS_DIR / f"{ts}_{safe}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str) -> "PlaygroundSession":
        """Deserialize a session from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        run_history = [
            RunEntry(**r) for r in data.pop("run_history", [])
        ]
        obj = cls(**{k: v for k, v in data.items() if k != "run_history"})
        obj.run_history = run_history
        return obj

    def _to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["run_history"] = [asdict(r) for r in self.run_history]
        return d

    def summary(self) -> str:
        """Return a one-line summary of the session's contents."""
        return (
            f"prompt={self.prompt_name!r}  "
            f"vars={len(self.variables)}  "
            f"renders={len(self.render_history)}  "
            f"runs={len(self.run_history)}"
        )


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------

def _utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
