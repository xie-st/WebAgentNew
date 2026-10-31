from __future__ import annotations

import functools
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping

import pytest

from webagent.clock import FakeClock
from webagent.protocol import ElementRef, Observation
from webagent.testing import FakeBackend, FakeSite

FIXTURES = Path(__file__).parent / "fixtures"
LOCAL_SITE = FIXTURES / "site"


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return


@pytest.fixture
def site_server():
    handler = functools.partial(_QuietHandler, directory=str(LOCAL_SITE))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield base
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def runtime(tmp_path: Path) -> Path:
    return tmp_path / "runtime"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def make_observation():
    def factory(
        *,
        run_id: str = "run-a",
        page_id: str = "page-1",
        observation_id: str = "obs-1",
        url: str = "http://fixture.local/",
        ax_text: str = "Search page.",
        elements: Mapping[str, Any] | None = None,
        previous_action_id: str | None = None,
        feedback=None,
    ) -> Observation:
        if elements is None:
            elements = {"1": ("textbox", "Search query"), "2": ("button", "Search")}
        refs: dict[str, ElementRef] = {}
        for number, value in elements.items():
            if isinstance(value, ElementRef):
                refs[number] = value
                continue
            role, name = value
            refs[number] = ElementRef(
                page_id=page_id,
                observation_id=observation_id,
                number=number,
                role=role,
                name=name,
                localization={"fake_index": int(number)},
            )
        return Observation(
            run_id=run_id,
            page_id=page_id,
            observation_id=observation_id,
            url=url,
            ax_text=ax_text,
            elements=refs,
            previous_action_id=previous_action_id,
            feedback=feedback,
        )

    return factory


@pytest.fixture
def site(fixtures_dir: Path) -> FakeSite:
    return FakeSite.from_json(fixtures_dir / "fake_site.json")


@pytest.fixture
def backend(site: FakeSite) -> FakeBackend:
    return FakeBackend("run-a", site, start_url=site.default_url)
