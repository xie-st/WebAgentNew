from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest

sync_api = pytest.importorskip("playwright.sync_api")
sync_playwright = sync_api.sync_playwright

from webagent.testing import start


@pytest.fixture
def local_runtime(tmp_path: Path) -> Path:
    return tmp_path / "runtime"


@pytest.fixture
def make_case(local_runtime: Path):
    prepared = []

    def factory(
        case_id: str,
        seed: int = 101,
        *,
        faults: list[Mapping[str, Any]] | None = None,
        run_id: str | None = None,
    ):
        run = start(
            case_id,
            seed,
            runtime_dir=local_runtime,
            run_id=run_id,
            faults=faults,
        )
        prepared.append(run)
        return run

    yield factory

    for run in prepared:
        run.stop()


@pytest.fixture
def browser():
    with sync_playwright() as playwright:
        instance = playwright.chromium.launch(headless=True)
        try:
            yield instance
        finally:
            instance.close()
