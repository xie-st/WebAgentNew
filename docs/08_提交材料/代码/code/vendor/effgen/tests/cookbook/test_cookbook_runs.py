"""
Cookbook snippet tests — extracted from docs/cookbook/multimodal_0*.md.

Unit tests run on every platform (no API keys, no special hardware).
Live tests are guarded by ``pytest.mark.live`` and the relevant env-var check.
Run all live tests:
    pytest tests/cookbook/test_cookbook_runs.py -m live -v
Skip live tests (CI default):
    pytest tests/cookbook/test_cookbook_runs.py -m "not live" -v
"""

from __future__ import annotations

import io
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from effgen._env import load_env
from tests._harness.provider_unavailable import skip_if_provider_refused

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent.parent
_FIXTURES_DIR = _REPO_ROOT / "tests/fixtures/multimodal"

# The documented search path, so a snippet runs against the same keys the CLI
# finds: reading only the repository's own file misses a key configured a level
# up and silently sends the run to a different provider.
load_env()


def _extract_python_block(doc_path: Path) -> str:
    """Return the single runnable Python block from a cookbook document."""
    text = doc_path.read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)\n```", text, flags=re.DOTALL)
    assert len(blocks) == 1, f"{doc_path.name} must contain exactly one Python block"
    return blocks[0]


def _run_doc_python_block(doc_name: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Execute a cookbook's runnable Python block in a subprocess."""
    code = _extract_python_block(_REPO_ROOT / "docs/cookbook" / doc_name)
    script_path = tmp_path / f"{Path(doc_name).stem}.py"
    script_path.write_text(code, encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(script_path)],
        cwd=_REPO_ROOT,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        timeout=240,
        check=False,
    )


def _make_image_bytes(color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    """Return a small PNG from a solid-color PIL image."""
    from PIL import Image

    buf = io.BytesIO()
    img = Image.new("RGB", (64, 64), color=color)
    img.save(buf, format="PNG")
    return buf.getvalue()


def _have_openai() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def _have_google() -> bool:
    return bool(os.getenv("GOOGLE_API_KEY"))


def _have_vision_key() -> bool:
    return _have_openai() or _have_google()


def _have_audio_key() -> bool:
    return _have_openai() or _have_google()


def _load_vision_model():
    """Load cheapest available vision model for live tests."""
    if _have_openai():
        from effgen.models.openai_adapter import OpenAIAdapter
        m = OpenAIAdapter(model_name="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    else:
        from effgen.models.gemini_adapter import GeminiAdapter
        m = GeminiAdapter(model_name="gemini-3.1-flash-lite", api_key=os.getenv("GOOGLE_API_KEY"))
    m.load()
    return m


def _load_audio_model():
    """Load cheapest available audio/transcription model for live tests.
    Prefer OpenAI (Whisper) over Gemini to avoid free-tier rate limits."""
    if _have_openai():
        from effgen.models.openai_adapter import OpenAIAdapter
        m = OpenAIAdapter(model_name="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    else:
        from effgen.models.gemini_adapter import GeminiAdapter
        m = GeminiAdapter(model_name="gemini-3.1-flash-lite", api_key=os.getenv("GOOGLE_API_KEY"))
    m.load()
    return m


class TestCookbookDocs:
    """Documentation-level checks for runnable cookbook snippets."""

    @pytest.mark.parametrize(
        "doc_name",
        [
            "multimodal_01_image_qa.md",
            "multimodal_02_audio_transcribe_reason.md",
            "multimodal_03_video_summarize.md",
            "multimodal_04_ocr_plus_llm.md",
            "multimodal_05_bullet_chart_read.md",
        ],
    )
    def test_each_cookbook_has_one_runnable_python_block(self, doc_name):
        block = _extract_python_block(_REPO_ROOT / "docs/cookbook" / doc_name)
        assert "load_dotenv()" in block
        assert "model.load()" in block

    @pytest.mark.live
    @pytest.mark.cookbook
    @pytest.mark.skipif(not _have_vision_key(), reason="No vision API key")
    def test_cookbook_01_snippet_executes_end_to_end(self, tmp_path):
        result = _run_doc_python_block("multimodal_01_image_qa.md", tmp_path)
        assert result.returncode == 0, result.stderr[-2000:] + result.stdout[-2000:]
        assert "Direct model call" in result.stdout
        assert "Agent with multimodal preset" in result.stdout

    @pytest.mark.live
    @pytest.mark.cookbook
    @pytest.mark.skipif(not _have_vision_key(), reason="No vision API key")
    def test_cookbook_04_snippet_executes_end_to_end(self, tmp_path):
        result = _run_doc_python_block("multimodal_04_ocr_plus_llm.md", tmp_path)
        assert result.returncode == 0, result.stderr[-2000:] + result.stdout[-2000:]
        assert "Raw OCR output" in result.stdout
        assert "Parsed JSON" in result.stdout


# ===========================================================================
# Cookbook 01 — Image Q&A
# ===========================================================================

class TestCookbook01ImageQA:
    """Tests for docs/cookbook/multimodal_01_image_qa.md."""

    # ── Unit tests ──────────────────────────────────────────────────────────

    def test_image_from_bytes_produces_image_part(self):
        """image_from() accepts raw bytes and returns an ImagePart."""
        from effgen.core.messages import ImagePart
        from effgen.core.multimodal import image_from

        png_bytes = _make_image_bytes()
        part = image_from(png_bytes)
        assert isinstance(part, ImagePart)
        assert part.mime == "image/png"
        assert len(part.image) > 0

    def test_image_from_path_produces_image_part(self, tmp_path):
        """image_from() accepts a local path string or Path."""
        from PIL import Image

        from effgen.core.messages import ImagePart
        from effgen.core.multimodal import image_from

        img = Image.new("RGB", (32, 32), (0, 255, 0))
        p = tmp_path / "green.png"
        img.save(str(p))

        part = image_from(str(p))
        assert isinstance(part, ImagePart)
        assert part.mime == "image/png"

    def test_multimodal_message_has_image(self):
        """A Message built with an ImagePart reports has_image=True."""
        from effgen.core.messages import ImagePart, Message, Role, TextPart

        img = ImagePart(image=_make_image_bytes(), mime="image/png")
        msg = Message(
            role=Role.USER,
            content=[img, TextPart(text="What color is the square?")],
        )
        assert msg.has_image
        assert not msg.has_audio
        assert not msg.has_video

    def test_multimodal_preset_exists(self):
        """multimodal preset is registered and create_agent returns an Agent."""
        from effgen.core.agent import Agent
        from effgen.presets import create_agent, list_presets

        assert "multimodal" in list_presets()

        # Inject fake model so no API call is made
        fake_model = _make_fake_model()
        agent = create_agent("multimodal", fake_model)
        assert isinstance(agent, Agent)

    # ── Live tests ───────────────────────────────────────────────────────────

    @pytest.mark.live
    @pytest.mark.cookbook
    @pytest.mark.skipif(not _have_vision_key(), reason="No vision API key")
    def test_image_qa_live_with_public_url(self, tmp_path):
        """Ask a question about a red square image (created locally, no external fetch)."""
        from PIL import Image

        from effgen.core.messages import ImagePart, Message, Role, TextPart

        # Create a simple red square locally — no external URL needed
        img = Image.new("RGB", (128, 128), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_part = ImagePart(image=buf.getvalue(), mime="image/png")

        model = _load_vision_model()

        msg = Message(
            role=Role.USER,
            content=[img_part, TextPart(text="What color is the dominant color of this image? One word.")],
        )
        result = model.generate([msg])
        assert result.text
        assert "red" in result.text.lower(), (
            f"Expected 'red' in response; got: {result.text[:300]}"
        )

    @pytest.mark.live
    @pytest.mark.cookbook
    @pytest.mark.skipif(not _have_vision_key(), reason="No vision API key")
    def test_image_qa_live_with_agent(self, tmp_path):
        """Agent run with a red-square image returns a colour description."""
        from PIL import Image

        from effgen.presets import create_agent

        img = Image.new("RGB", (128, 128), color=(255, 0, 0))
        img_path = tmp_path / "red.png"
        img.save(str(img_path))

        from effgen.core.multimodal import image_from
        img_part = image_from(str(img_path))
        model = _load_vision_model()
        agent = create_agent("multimodal", model)

        result = agent.run(
            "What color is this image? One word answer.",
            inputs=[img_part],
        )
        skip_if_provider_refused(result)
        assert result.success, f"Agent failed: {result.output}"
        assert "red" in result.output.lower(), (
            f"Expected 'red'; got: {result.output[:200]}"
        )


# ===========================================================================
# Cookbook 02 — Audio Transcription + Reasoning
# ===========================================================================

class TestCookbook02AudioTranscribeReason:
    """Tests for docs/cookbook/multimodal_02_audio_transcribe_reason.md."""

    # ── Unit tests ──────────────────────────────────────────────────────────

    def test_audio_from_path_produces_audio_part(self):
        """audio_from() accepts the sample MP3 fixture and returns AudioPart."""
        from effgen.core.messages import AudioPart
        from effgen.core.multimodal import audio_from

        audio_fixture = _FIXTURES_DIR / "sample_audio.mp3"
        if not audio_fixture.exists():
            pytest.skip("sample_audio.mp3 fixture not found")

        part = audio_from(str(audio_fixture))
        assert isinstance(part, AudioPart)
        assert part.mime in ("audio/mpeg", "audio/mp3", "audio/mp4")
        assert len(part.audio) > 0

    def test_audio_message_has_audio(self):
        """A Message with an AudioPart reports has_audio=True."""
        from effgen.core.messages import AudioPart, Message, Role, TextPart

        audio_bytes = b"\xff\xfb" + b"\x00" * 64  # minimal MP3 magic
        part = AudioPart(audio=audio_bytes, mime="audio/mpeg")
        msg = Message(
            role=Role.USER,
            content=[part, TextPart(text="Transcribe this.")],
        )
        assert msg.has_audio
        assert not msg.has_image
        assert not msg.has_video

    def test_multimodal_preset_has_audio_transcribe_tool(self):
        """multimodal preset always includes the audio_transcribe tool."""
        from effgen.presets.registry import get_preset

        cfg = get_preset("multimodal")
        assert "audio_transcribe" in cfg.tool_names

    # ── Live tests ───────────────────────────────────────────────────────────

    @pytest.mark.live
    @pytest.mark.cookbook
    @pytest.mark.skipif(not _have_audio_key(), reason="No audio API key")
    def test_audio_transcribe_live(self):
        """Transcribe sample_audio.mp3 and verify non-empty output."""
        import shutil
        if not shutil.which("ffmpeg"):
            pytest.skip("ffmpeg not on PATH — required for audio preprocessing")

        from effgen.core.multimodal import audio_from
        from effgen.presets import create_agent

        audio_fixture = _FIXTURES_DIR / "sample_audio.mp3"
        if not audio_fixture.exists():
            pytest.skip("sample_audio.mp3 fixture not found")

        model = _load_audio_model()
        agent = create_agent("multimodal", model)
        part = audio_from(str(audio_fixture))

        result = agent.run(
            "Use the audio_transcribe tool to transcribe this recording, "
            "then give a one-line sentiment summary.",
            inputs=[part],
        )
        skip_if_provider_refused(result)
        assert result.success, f"Audio agent failed: {result.output}"
        assert len(result.output) > 10, "Expected non-trivial output"


# ===========================================================================
# Cookbook 03 — Video Summarisation
# ===========================================================================

class TestCookbook03VideoSummarize:
    """Tests for docs/cookbook/multimodal_03_video_summarize.md."""

    # ── Unit tests ──────────────────────────────────────────────────────────

    def test_video_from_path_produces_video_part(self):
        """video_from() uses the sample_video.mp4 fixture."""
        pytest.importorskip("PIL")

        video_fixture = _FIXTURES_DIR / "sample_video.mp4"
        if not video_fixture.exists():
            pytest.skip("sample_video.mp4 fixture not found")

        from effgen.core.messages import VideoPart
        from effgen.core.multimodal import video_from

        try:
            part = video_from(str(video_fixture), fps=1, max_frames=3)
        except Exception as e:
            if "ffmpeg" in str(e).lower() or "MissingSystemDependency" in type(e).__name__:
                pytest.skip(f"ffmpeg not available: {e}")
            raise

        assert isinstance(part, VideoPart)
        assert len(part.frames) >= 1

    def test_video_message_has_video(self):
        """A Message with a VideoPart reports has_video=True."""
        from effgen.core.messages import Message, Role, TextPart, VideoPart

        fake_frame = _make_image_bytes()
        part = VideoPart(frames=[fake_frame], fps=1.0, mime="image/jpeg")
        msg = Message(
            role=Role.USER,
            content=[part, TextPart(text="Summarise this video.")],
        )
        assert msg.has_video
        assert not msg.has_image
        assert not msg.has_audio

    def test_multimodal_describe_tool_registered(self):
        """multimodal preset includes the multimodal_describe tool."""
        from effgen.presets.registry import get_preset
        cfg = get_preset("multimodal")
        assert "multimodal_describe" in cfg.tool_names

    @pytest.mark.live
    @pytest.mark.cookbook
    @pytest.mark.skipif(not _have_vision_key(), reason="No vision API key")
    def test_video_summary_reaches_model_call_live(self):
        """Send synthetic video frames to the model without requiring ffmpeg."""
        from effgen.core.messages import Message, Role, TextPart, VideoPart

        video = VideoPart(
            frames=[
                _make_image_bytes(color=(255, 0, 0)),
                _make_image_bytes(color=(0, 0, 255)),
            ],
            fps=1.0,
            mime="image/png",
        )
        model = _load_vision_model()
        result = model.generate(
            [
                Message(
                    role=Role.USER,
                    content=[
                        video,
                        TextPart(
                            text=(
                                "These are sampled frames from a short video. "
                                "Summarize the visual change in one sentence."
                            ),
                        ),
                    ],
                ),
            ],
        )
        assert result.text
        assert len(result.text.strip()) > 10


# ===========================================================================
# Cookbook 04 — OCR + LLM Structured Extraction
# ===========================================================================

class TestCookbook04OCRPlusLLM:
    """Tests for docs/cookbook/multimodal_04_ocr_plus_llm.md."""

    # ── Unit tests ──────────────────────────────────────────────────────────

    def test_contract_template_exists(self):
        """legal.contract_summarize.v1 is in the prompt library."""
        from effgen.prompts.library import registry
        all_names = [p.name for p in registry.all()]
        assert "legal.contract_summarize.v1" in all_names, (
            f"Expected 'legal.contract_summarize.v1'; found legal prompts: "
            f"{[n for n in all_names if 'legal' in n.lower()]}"
        )

    def test_contract_template_renders(self):
        """legal.contract_summarize.v1 renders with contract_text input."""
        from effgen.prompts.library import registry
        tmpl = registry.get("legal.contract_summarize.v1")
        rendered = tmpl.render(contract_text="Agreement between A and B. Payment: $100/mo.")
        # Should be a non-empty prompt string
        assert len(rendered) > 20

    def test_image_part_with_text_content(self):
        """ImagePart can represent a document image."""
        from effgen.core.messages import ImagePart, Message, Role, TextPart

        doc_bytes = _make_image_bytes(color=(255, 255, 255))  # white page
        img = ImagePart(image=doc_bytes, mime="image/png")
        msg = Message(
            role=Role.USER,
            content=[img, TextPart(text="Extract all text from this document.")],
        )
        assert msg.has_image
        assert "Extract" in msg.text

    def test_multimodal_preset_has_ocr_tool(self):
        """multimodal preset includes the ocr tool."""
        from effgen.presets.registry import get_preset
        cfg = get_preset("multimodal")
        assert "ocr" in cfg.tool_names

    # ── Live tests ───────────────────────────────────────────────────────────

    @pytest.mark.live
    @pytest.mark.cookbook
    @pytest.mark.skipif(not _have_vision_key(), reason="No vision API key")
    def test_ocr_plus_llm_live(self, tmp_path):
        """Generate a synthetic contract image, OCR it, extract structured data."""
        from PIL import Image, ImageDraw

        # Build a simple text image
        img = Image.new("RGB", (512, 256), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        contract_lines = [
            "SERVICE AGREEMENT",
            "Parties: Acme Corp. and Beta LLC.",
            "Effective Date: 2026-06-01",
            "Payment: $5,000/month",
            "Governing Law: California",
        ]
        y = 20
        for line in contract_lines:
            draw.text((20, y), line, fill=(0, 0, 0))
            y += 40

        img_path = tmp_path / "contract.png"
        img.save(str(img_path))

        from effgen.core.multimodal import image_from
        from effgen.presets import create_agent

        img_part = image_from(str(img_path))
        model = _load_vision_model()
        agent = create_agent("multimodal", model)

        # Step 1: OCR
        ocr_result = agent.run(
            "Use the ocr tool to extract all visible text from this document image.",
            inputs=[img_part],
        )
        skip_if_provider_refused(ocr_result)
        assert ocr_result.success, f"OCR step failed: {ocr_result.output}"
        raw_text = ocr_result.output
        assert len(raw_text) > 10, "Expected non-empty OCR output"

        # Step 2: Structured extraction (direct model call with prompt template)
        from effgen.core.messages import Message, Role, TextPart
        from effgen.prompts.library import registry

        # Use the library template if available, otherwise build manually
        try:
            tmpl = registry.get("legal.contract_summarize.v1")
            extraction_prompt = tmpl.render(contract_text=raw_text)
        except KeyError:
            extraction_prompt = (
                f"Given this contract text, extract a JSON object with keys: "
                f"parties (list), effective_date, payment_terms.\n\n"
                f"Contract:\n{raw_text}"
            )
        extraction_result = model.generate(
            [Message(role=Role.USER, content=[TextPart(text=extraction_prompt)])],
        )
        assert extraction_result.text, "Expected extraction output"
        import json
        import re

        match = re.search(r"\{[\s\S]+\}", extraction_result.text)
        assert match, f"Expected JSON object; got: {extraction_result.text[:500]}"
        parsed = json.loads(match.group())
        expected_keys = {"parties", "term", "obligations", "termination", "risks"}
        assert expected_keys <= set(parsed), (
            f"Missing keys from contract summary: {sorted(expected_keys - set(parsed))}"
        )
        assert parsed["parties"], "Expected extracted parties"
        assert parsed["term"], "Expected extracted term/date information"


# ===========================================================================
# Cookbook 05 — Bar Chart Reading
# ===========================================================================

class TestCookbook05BulletChartRead:
    """Tests for docs/cookbook/multimodal_05_bullet_chart_read.md."""

    # ── Unit tests ──────────────────────────────────────────────────────────

    def test_image_part_from_bytes(self):
        """ImagePart can be constructed directly from bytes and MIME."""
        from effgen.core.messages import ImagePart

        png_bytes = _make_image_bytes()
        img = ImagePart(image=png_bytes, mime="image/png")
        assert img.mime == "image/png"
        assert len(img.image) > 0

    def test_generate_chart_image(self):
        """matplotlib bar chart can be rendered to in-memory PNG bytes."""
        pytest.importorskip("matplotlib")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        categories = ["A", "B", "C"]
        values = [10, 20, 15]
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(categories, values)
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)

        chart_bytes = buf.getvalue()
        assert len(chart_bytes) > 0
        assert chart_bytes[:4] == b"\x89PNG"

    def test_chart_image_part_is_valid(self):
        """ImagePart accepts matplotlib-generated PNG."""
        pytest.importorskip("matplotlib")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from effgen.core.messages import ImagePart

        fig, ax = plt.subplots()
        ax.bar(["X", "Y"], [5, 10])
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)

        img = ImagePart(image=buf.getvalue(), mime="image/png")
        assert img.mime == "image/png"

    # ── Live tests ───────────────────────────────────────────────────────────

    @pytest.mark.live
    @pytest.mark.cookbook
    @pytest.mark.skipif(not _have_vision_key(), reason="No vision API key")
    def test_chart_read_live(self):
        """Generate a bar chart and ask which bar is tallest."""
        pytest.importorskip("matplotlib")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from effgen.core.messages import ImagePart, Message, Role, TextPart

        categories = ["Alpha", "Beta", "Gamma"]
        values = [30, 85, 50]  # Beta is clearly highest

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(categories, values)
        ax.set_title("Sales by Region")
        for i, (_cat, val) in enumerate(zip(categories, values)):
            ax.text(i, val + 1, str(val), ha="center")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)

        img = ImagePart(image=buf.getvalue(), mime="image/png")
        model = _load_vision_model()

        msg = Message(
            role=Role.USER,
            content=[
                img,
                TextPart(text=(
                    "Look at this bar chart. "
                    "Which region (bar) has the highest value? "
                    "Just name the region."
                )),
            ],
        )
        result = model.generate([msg])
        assert result.text, "Expected non-empty response"
        assert "beta" in result.text.lower(), (
            f"Expected 'Beta' to be identified as highest; got: {result.text[:200]}"
        )


# ===========================================================================
# Helpers
# ===========================================================================

def _make_fake_model():
    """Minimal stub model that satisfies Agent.__init__ checks."""
    from effgen.models.base import BaseModel, GenerationResult, ModelType, TokenCount

    class FakeModel(BaseModel):
        def __init__(self):
            super().__init__(model_name="fake-cookbook", model_type=ModelType.OPENAI)
            self._is_loaded = True
            self.config = type("C", (), {"temperature": 0.0})()

        def load(self): self._is_loaded = True
        def unload(self): self._is_loaded = False
        @property
        def loaded(self): return True

        def generate(self, messages, config=None, **kw):
            return GenerationResult(
                text="mocked response",
                tokens_used=3,
                finish_reason="stop",
                model_name=self.model_name,
                metadata={},
            )

        def generate_stream(self, messages, config=None, **kw):
            yield "mocked response"

        def count_tokens(self, text):
            return TokenCount(count=len(text.split()), model_name=self.model_name)

        def get_context_length(self):
            return 4096

    return FakeModel()
