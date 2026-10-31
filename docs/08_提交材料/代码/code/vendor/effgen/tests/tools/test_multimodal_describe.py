"""
Tests for MultimodalDescribeTool.

Unit: media type inference, parameter validation, error paths.
Integration: routes to image/audio/video backends (mocked).
Live: skipped unless API keys are present.
"""

from __future__ import annotations

import asyncio
import importlib.util
import io
import os
from pathlib import Path
from unittest.mock import patch

import pytest

# Pillow is an optional dependency (the ``tools`` extra); these tests build real
# PIL images, so skip cleanly when it is absent (e.g. the lean ``[dev]`` install
# used by the offline CI lane).
pytest.importorskip("PIL", reason="Pillow not installed (pip install effgen[tools])")


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_png(tmp_path: Path) -> Path:
    from PIL import Image
    img = Image.new("RGB", (64, 64), color=(100, 200, 50))
    p = tmp_path / "sample.png"
    img.save(str(p))
    return p


@pytest.fixture()
def sample_jpg(tmp_path: Path) -> Path:
    from PIL import Image
    img = Image.new("RGB", (64, 64), color=(50, 100, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    p = tmp_path / "sample.jpg"
    p.write_bytes(buf.getvalue())
    return p


@pytest.fixture()
def audio_fixture() -> Path:
    p = Path(__file__).parent.parent.parent / "tests/fixtures/multimodal/sample_audio.mp3"
    if not p.exists():
        pytest.skip("sample_audio.mp3 fixture not found")
    return p


@pytest.fixture()
def video_fixture() -> Path:
    p = Path(__file__).parent.parent.parent / "tests/fixtures/multimodal/sample_video.mp4"
    if not p.exists():
        pytest.skip("sample_video.mp4 fixture not found")
    return p


# ---------------------------------------------------------------------------
# Unit — instantiation and metadata
# ---------------------------------------------------------------------------

def test_multimodal_describe_tool_instantiation():
    from effgen.tools.builtin.multimodal_describe import MultimodalDescribeTool
    tool = MultimodalDescribeTool()
    assert tool.metadata.name == "multimodal_describe"
    assert "multimodal" in tool.metadata.tags


def test_multimodal_describe_parameters():
    from effgen.tools.builtin.multimodal_describe import MultimodalDescribeTool
    tool = MultimodalDescribeTool()
    param_names = {p.name for p in tool.metadata.parameters}
    assert {"file_path", "media_type", "prompt", "operation", "max_frames"} <= param_names


# ---------------------------------------------------------------------------
# Unit — media type inference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("suffix,expected", [
    (".png", "image"),
    (".jpg", "image"),
    (".jpeg", "image"),
    (".webp", "image"),
    (".gif", "image"),
    (".mp3", "audio"),
    (".wav", "audio"),
    (".flac", "audio"),
    (".ogg", "audio"),
    (".m4a", "audio"),
    (".mp4", "video"),
    (".mov", "video"),
    (".avi", "video"),
    (".mkv", "video"),
    (".webm", "video"),
])
def test_infer_media_type(tmp_path, suffix, expected):
    from effgen.tools.builtin.multimodal_describe import _infer_media_type
    p = tmp_path / f"test{suffix}"
    p.touch()
    assert _infer_media_type(p) == expected


# ---------------------------------------------------------------------------
# Unit — file not found returns error result
# ---------------------------------------------------------------------------

def test_execute_file_not_found():
    from effgen.tools.builtin.multimodal_describe import MultimodalDescribeTool
    tool = MultimodalDescribeTool()
    result = run(tool._execute(file_path="/nonexistent/path/image.png"))
    assert result["success"] is False
    assert "not found" in result["error"].lower()


# ---------------------------------------------------------------------------
# Unit — image routing (mocked backend)
# ---------------------------------------------------------------------------

def test_execute_image_routing(sample_png):
    from effgen.tools.builtin.multimodal_describe import MultimodalDescribeTool

    mock_result = {
        "success": True,
        "data": {"caption": "A green square on white background."},
        "caption": "A green square on white background.",
        "error": None,
    }

    with patch(
        "effgen.tools.builtin.multimodal_describe._describe_image_sync",
        return_value=mock_result,
    ):
        tool = MultimodalDescribeTool()
        result = run(tool._execute(file_path=str(sample_png), media_type="image"))

    assert result["success"] is True
    assert result["media_type"] == "image"
    assert "green" in result["description"].lower()


# ---------------------------------------------------------------------------
# Unit — audio routing (mocked backend)
# ---------------------------------------------------------------------------

def test_execute_audio_routing(audio_fixture):
    from effgen.tools.builtin.multimodal_describe import MultimodalDescribeTool

    mock_result = {
        "success": True,
        "data": {"text": "Hello world, this is a test."},
        "text": "Hello world, this is a test.",
        "error": None,
    }

    with patch(
        "effgen.tools.builtin.multimodal_describe._transcribe_audio_sync",
        return_value=mock_result,
    ):
        tool = MultimodalDescribeTool()
        result = run(tool._execute(file_path=str(audio_fixture), media_type="audio"))

    assert result["success"] is True
    assert result["media_type"] == "audio"
    assert "hello" in result["transcript"].lower()


# ---------------------------------------------------------------------------
# Unit — video routing (mocked backend)
# ---------------------------------------------------------------------------

def test_execute_video_routing(video_fixture):
    from effgen.tools.builtin.multimodal_describe import MultimodalDescribeTool

    mock_result = {
        "success": True,
        "data": {"description": "Frame 1: A person walking.", "frames_sampled": 3},
        "description": "Frame 1: A person walking.",
        "error": None,
    }

    with patch(
        "effgen.tools.builtin.multimodal_describe._describe_video_sync",
        return_value=mock_result,
    ):
        tool = MultimodalDescribeTool()
        result = run(tool._execute(file_path=str(video_fixture), media_type="video"))

    assert result["success"] is True
    assert result["media_type"] == "video"
    assert "walking" in result["description"].lower()


# ---------------------------------------------------------------------------
# Unit — auto type detection via extension
# ---------------------------------------------------------------------------

def test_execute_auto_type_detection_image(sample_png):
    from effgen.tools.builtin.multimodal_describe import MultimodalDescribeTool

    mock_result = {
        "success": True,
        "data": {"caption": "A colorful test image."},
        "caption": "A colorful test image.",
        "error": None,
    }

    with patch(
        "effgen.tools.builtin.multimodal_describe._describe_image_sync",
        return_value=mock_result,
    ):
        tool = MultimodalDescribeTool()
        # media_type="auto" should detect image from .png extension
        result = run(tool._execute(file_path=str(sample_png), media_type="auto"))

    assert result["success"] is True
    assert result["media_type"] == "image"


# ---------------------------------------------------------------------------
# Unit — custom prompt is forwarded
# ---------------------------------------------------------------------------

def test_execute_custom_prompt(sample_png):
    from effgen.tools.builtin.multimodal_describe import MultimodalDescribeTool

    captured_args: list = []

    def mock_describe(file_path, prompt, operation):
        captured_args.extend([file_path, prompt, operation])
        return {
            "success": True,
            "data": {"caption": "Custom prompt used."},
            "caption": "Custom prompt used.",
            "error": None,
        }

    with patch(
        "effgen.tools.builtin.multimodal_describe._describe_image_sync",
        side_effect=mock_describe,
    ):
        tool = MultimodalDescribeTool()
        run(tool._execute(
            file_path=str(sample_png),
            media_type="image",
            prompt="Count the pixels.",
        ))

    assert "Count the pixels." in captured_args


# ---------------------------------------------------------------------------
# Unit — exception in backend returns error dict (no raise)
# ---------------------------------------------------------------------------

def test_execute_backend_exception_returns_error(sample_png):
    from effgen.tools.builtin.multimodal_describe import MultimodalDescribeTool

    with patch(
        "effgen.tools.builtin.multimodal_describe._describe_image_sync",
        side_effect=RuntimeError("Vision API down"),
    ):
        tool = MultimodalDescribeTool()
        result = run(tool._execute(file_path=str(sample_png), media_type="image"))

    assert result["success"] is False
    assert "Vision API down" in result["error"]


# ---------------------------------------------------------------------------
# Live — image description (skipped unless API key present)
# ---------------------------------------------------------------------------

@pytest.mark.live
@pytest.mark.skipif(
    not (os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")),
    reason="No vision API key available",
)
def test_multimodal_describe_image_live(sample_png):
    from effgen.tools.builtin.multimodal_describe import MultimodalDescribeTool
    tool = MultimodalDescribeTool()
    result = run(tool._execute(
        file_path=str(sample_png),
        media_type="image",
        prompt="What color is the dominant color in this image?",
    ))
    assert result["success"] is True, f"Error: {result.get('error')}"
    assert result["description"], "Empty description returned"


@pytest.mark.live
@pytest.mark.skipif(
    not os.path.exists(
        str(Path(__file__).parent.parent.parent / "tests/fixtures/multimodal/sample_audio.mp3")
    ),
    reason="sample_audio.mp3 fixture not found",
)
@pytest.mark.skipif(
    importlib.util.find_spec("faster_whisper") is None,
    reason="faster-whisper (local STT backend) not installed",
)
def test_multimodal_describe_audio_live():
    fixture = (
        Path(__file__).parent.parent.parent
        / "tests/fixtures/multimodal/sample_audio.mp3"
    )
    from effgen.tools.builtin.multimodal_describe import MultimodalDescribeTool
    tool = MultimodalDescribeTool()
    result = run(tool._execute(
        file_path=str(fixture),
        media_type="audio",
        operation="transcribe",
    ))
    # The fixture is a synthetic 440 Hz sine tone (no speech), so the
    # end-to-end pipeline (decode MP3 -> faster-whisper -> structured result)
    # is what we validate: the call succeeds and returns a well-formed string
    # transcript. An empty transcript is the correct result for speechless audio
    # and is not a failure.
    assert result["success"] is True, f"Error: {result.get('error')}"
    assert isinstance(result["transcript"], str)
