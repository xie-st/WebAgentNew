from __future__ import annotations

from pathlib import Path

import webagent.recovery
import webagent.verification


def _sources(package: object) -> list[Path]:
    root = Path(package.__file__).resolve().parent  # type: ignore[arg-type]
    return sorted(path for path in root.glob("*.py"))


def test_verification_and_recovery_do_not_touch_site_internals_or_browsers() -> None:
    forbidden = (
        "sqlite3",
        "HiddenStateReader",
        "webagent.testing",
        "sync_playwright",
        "PlaywrightBackend",
    )
    for package in (webagent.verification, webagent.recovery):
        for path in _sources(package):
            text = path.read_text(encoding="utf-8")
            for word in forbidden:
                assert word not in text, f"{path.name} imports or names {word!r}"
