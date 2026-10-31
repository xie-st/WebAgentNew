"""Live notebook smoke for the effGen Jupyter magics.

Drives the magics inside a REAL IPython kernel via ``jupyter_client`` (no
mocking of model behaviour) against a cheap cloud model, and asserts that the
rich result card (``AgentResponse._repr_html_``) is rendered in the
notebook output. Skipped when no OpenAI key or when the kernel stack is absent.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env", override=False)
load_dotenv(Path.home() / ".effgen" / ".env", override=False)

jupyter_client = pytest.importorskip("jupyter_client")
pytest.importorskip("ipykernel")


def _has_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def _run_cell(kc, code, timeout=180):
    msg_id = kc.execute(code)
    outputs = []
    status = "ok"
    while True:
        try:
            msg = kc.get_iopub_msg(timeout=timeout)
        except Exception:
            status = "timeout"
            break
        if msg["parent_header"].get("msg_id") != msg_id:
            continue
        mt = msg["msg_type"]
        content = msg["content"]
        if mt == "status" and content.get("execution_state") == "idle":
            break
        if mt in ("stream", "display_data", "execute_result", "error"):
            outputs.append((mt, content))
        if mt == "error":
            status = "error"
    return status, outputs


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.skipif(not _has_key(), reason="SKIPPED: OPENAI_API_KEY not set")
def test_magics_render_rich_card_in_real_kernel():
    from jupyter_client.manager import start_new_kernel

    km, kc = start_new_kernel(kernel_name="python3")
    try:
        status, _ = _run_cell(kc, "%load_ext effgen.jupyter")
        assert status == "ok"

        status, outputs = _run_cell(
            kc,
            "%effgen_chat -m gpt-5-nano What is the capital of France? One word.",
        )
        assert status == "ok"

        # The HTML result card must be among the rendered outputs.
        html_blobs = [
            content["data"]["text/html"]
            for mt, content in outputs
            if mt in ("display_data", "execute_result")
            and "text/html" in content.get("data", {})
        ]
        assert html_blobs, "no HTML output rendered by the magic"
        assert any("effGen result" in h for h in html_blobs), (
            "AgentResponse._repr_html_ card not rendered"
        )
    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.skipif(not _has_key(), reason="SKIPPED: OPENAI_API_KEY not set")
def test_magic_no_model_shows_friendly_message_in_real_kernel():
    from jupyter_client.manager import start_new_kernel

    km, kc = start_new_kernel(kernel_name="python3")
    try:
        _run_cell(kc, "%load_ext effgen.jupyter")
        status, outputs = _run_cell(
            kc,
            "import os\n"
            "os.environ.pop('EFFGEN_JUPYTER_MODEL', None)\n"
            "os.environ.pop('EFFGEN_DEFAULT_MODEL', None)\n"
            "get_ipython().run_line_magic('effgen_chat', 'hi with no model')",
        )
        assert status == "ok"
        md_blobs = [
            content["data"].get("text/markdown", "")
            for mt, content in outputs
            if mt in ("display_data", "execute_result")
        ]
        assert any("No model configured" in m for m in md_blobs)
    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)
