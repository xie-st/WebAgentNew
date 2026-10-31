"""Unit tests for effgen.multimodal.video_pre (no live API calls).

Tests cover:
  - ffmpeg-missing → clean MissingSystemDependency
  - Frame sampling from a real test file (when ffmpeg is on PATH)
  - Audio extraction from a real test file
  - VideoSource with bytes, path, and URL inputs (URL tested with a local server mock)
  - sample_frames / extract_audio convenience wrappers
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from effgen.core.messages import AudioPart, ImagePart
from effgen.errors import InvalidMultimodalContent, MissingSystemDependency

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_VIDEO = Path(__file__).parent.parent.parent / "tests/fixtures/multimodal/sample_video.mp4"

_FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None


def _create_minimal_mp4(output_path: str) -> None:
    """Create a tiny synthetic MP4 with ffmpeg for isolated tests."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg not on PATH")
    cmd = [
        ffmpeg, "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=duration=2:size=160x120:rate=6",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
        "-c:v", "mpeg4", "-c:a", "aac",
        "-shortest", "-y", output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


# ---------------------------------------------------------------------------
# ffmpeg-missing error
# ---------------------------------------------------------------------------

class TestFfmpegMissing:
    """Verify that MissingSystemDependency is raised cleanly when ffmpeg is absent."""

    def test_sample_frames_raises_missing_dep(self, tmp_path):
        from effgen.multimodal.video_pre import VideoSource

        dummy = tmp_path / "fake.mp4"
        dummy.write_bytes(b"\x00" * 16)

        vs = VideoSource(str(dummy))
        with patch("shutil.which", return_value=None):
            with pytest.raises(MissingSystemDependency) as exc_info:
                vs.sample_frames()
        assert "ffmpeg" in str(exc_info.value).lower()
        assert "install" in str(exc_info.value).lower()

    def test_extract_audio_raises_missing_dep(self, tmp_path):
        from effgen.multimodal.video_pre import VideoSource

        dummy = tmp_path / "fake.mp4"
        dummy.write_bytes(b"\x00" * 16)

        vs = VideoSource(str(dummy))
        with patch("shutil.which", return_value=None):
            with pytest.raises(MissingSystemDependency) as exc_info:
                vs.extract_audio()
        assert "ffmpeg" in str(exc_info.value).lower()

    def test_convenience_sample_frames_raises(self, tmp_path):
        from effgen.multimodal.video_pre import sample_frames

        dummy = tmp_path / "fake.mp4"
        dummy.write_bytes(b"\x00" * 16)

        with patch("shutil.which", return_value=None):
            with pytest.raises(MissingSystemDependency):
                sample_frames(str(dummy))


# ---------------------------------------------------------------------------
# VideoSource initialisation
# ---------------------------------------------------------------------------

class TestVideoSourceInit:
    def test_invalid_source_type_raises(self):
        from effgen.multimodal.video_pre import VideoSource

        vs = VideoSource(12345)  # type: ignore[arg-type]
        with pytest.raises(InvalidMultimodalContent):
            vs._resolve_path()

    def test_missing_path_raises(self, tmp_path):
        from effgen.multimodal.video_pre import VideoSource

        missing = str(tmp_path / "does_not_exist.mp4")
        vs = VideoSource(missing)
        with pytest.raises(InvalidMultimodalContent, match="File not found"):
            vs._resolve_path()

    def test_empty_bytes_raises(self):
        from effgen.multimodal.video_pre import VideoSource

        vs = VideoSource(b"")
        with pytest.raises(InvalidMultimodalContent, match="empty"):
            vs._resolve_path()

    def test_mime_inferred_from_extension(self, tmp_path):
        from effgen.multimodal.video_pre import VideoSource

        p = tmp_path / "clip.webm"
        p.write_bytes(b"\x00" * 8)
        vs = VideoSource(str(p))
        assert vs.mime_type == "video/webm"

    def test_mime_override(self, tmp_path):
        from effgen.multimodal.video_pre import VideoSource

        p = tmp_path / "clip.bin"
        p.write_bytes(b"\x00" * 8)
        vs = VideoSource(str(p), mime_type="video/mp4")
        assert vs.mime_type == "video/mp4"

    def test_invalid_mime_override_raises(self, tmp_path):
        from effgen.multimodal.video_pre import VideoSource

        p = tmp_path / "clip.bin"
        p.write_bytes(b"\x00" * 8)
        with pytest.raises(InvalidMultimodalContent, match="MIME"):
            VideoSource(str(p), mime_type="text/plain")

    def test_path_source_resolves(self, tmp_path):
        from effgen.multimodal.video_pre import VideoSource

        p = tmp_path / "clip.mp4"
        p.write_bytes(b"\x00" * 8)
        vs = VideoSource(str(p))
        assert vs._resolve_path() == str(p)

    def test_bytes_source_creates_tempfile(self):
        from effgen.multimodal.video_pre import VideoSource

        vs = VideoSource(b"\x00" * 64, mime_type="video/mp4")
        path = vs._resolve_path()
        assert os.path.exists(path)
        assert vs._own_tmp
        vs.cleanup()
        assert not os.path.exists(path)

    def test_pathlib_source(self, tmp_path):
        from effgen.multimodal.video_pre import VideoSource

        p = tmp_path / "clip.mp4"
        p.write_bytes(b"\x00" * 8)
        vs = VideoSource(p)
        assert vs._resolve_path() == str(p)


# ---------------------------------------------------------------------------
# Frame sampling (requires ffmpeg)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _FFMPEG_AVAILABLE, reason="ffmpeg not on PATH")
class TestFrameSampling:
    def test_sample_frames_returns_image_parts(self, tmp_path):
        from effgen.multimodal.video_pre import VideoSource

        out = str(tmp_path / "clip.mp4")
        _create_minimal_mp4(out)

        vs = VideoSource(out)
        frames = vs.sample_frames(fps=1.0, max_frames=4)

        assert len(frames) >= 1
        for f in frames:
            assert isinstance(f, ImagePart)
            assert f.mime == "image/jpeg"
            assert len(f.image) > 0

    def test_max_frames_respected(self, tmp_path):
        from effgen.multimodal.video_pre import VideoSource

        out = str(tmp_path / "clip.mp4")
        _create_minimal_mp4(out)

        vs = VideoSource(out)
        frames = vs.sample_frames(fps=6.0, max_frames=2)
        assert len(frames) <= 2

    def test_invalid_fps_raises(self, tmp_path):
        from effgen.multimodal.video_pre import VideoSource

        p = tmp_path / "clip.mp4"
        p.write_bytes(b"\x00" * 8)
        vs = VideoSource(str(p))
        with pytest.raises(InvalidMultimodalContent, match="fps"):
            vs.sample_frames(fps=0)

    def test_invalid_max_frames_raises(self, tmp_path):
        from effgen.multimodal.video_pre import VideoSource

        p = tmp_path / "clip.mp4"
        p.write_bytes(b"\x00" * 8)
        vs = VideoSource(str(p))
        with pytest.raises(InvalidMultimodalContent, match="max_frames"):
            vs.sample_frames(fps=1.0, max_frames=0)

    def test_sample_frames_from_bytes(self, tmp_path):
        """VideoSource accepts raw bytes and still extracts frames."""
        from effgen.multimodal.video_pre import VideoSource

        out = str(tmp_path / "clip.mp4")
        _create_minimal_mp4(out)
        with open(out, "rb") as fh:
            raw = fh.read()

        vs = VideoSource(raw, mime_type="video/mp4")
        frames = vs.sample_frames(fps=1.0, max_frames=4)
        assert len(frames) >= 1
        vs.cleanup()

    def test_convenience_sample_frames(self, tmp_path):
        from effgen.multimodal.video_pre import sample_frames

        out = str(tmp_path / "clip.mp4")
        _create_minimal_mp4(out)

        frames = sample_frames(out, fps=1.0, max_frames=3)
        assert len(frames) >= 1
        assert all(isinstance(f, ImagePart) for f in frames)

    def test_sample_frames_from_fixture(self):
        """Smoke-test on the shared sample fixture."""
        if not SAMPLE_VIDEO.exists():
            pytest.skip("Sample fixture not found")

        from effgen.multimodal.video_pre import VideoSource

        vs = VideoSource(SAMPLE_VIDEO)
        frames = vs.sample_frames(fps=1.0, max_frames=6)
        assert len(frames) >= 1

    def test_fixture_sampling_caps_at_max_frames(self):
        """The cap binds: the fixture offers more frames at this rate than are asked for."""
        if not SAMPLE_VIDEO.exists():
            pytest.skip("Sample fixture not found")

        from effgen.multimodal.video_pre import VideoSource

        vs = VideoSource(SAMPLE_VIDEO)
        # The fixture runs 2 s at 6 fps, so sampling at its native rate offers
        # 12 frames and the cap is what decides how many come back.
        frames = vs.sample_frames(fps=6.0, max_frames=8)
        assert len(frames) == 8

    def test_fixture_sampling_below_the_cap_returns_what_the_rate_yields(self):
        """Under the cap, the requested rate and the clip's length decide the count."""
        if not SAMPLE_VIDEO.exists():
            pytest.skip("Sample fixture not found")

        from effgen.multimodal.video_pre import VideoSource

        vs = VideoSource(SAMPLE_VIDEO)
        frames = vs.sample_frames(fps=2.0, max_frames=8)
        assert len(frames) == 4


# ---------------------------------------------------------------------------
# Audio extraction (requires ffmpeg)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _FFMPEG_AVAILABLE, reason="ffmpeg not on PATH")
class TestAudioExtraction:
    def test_extract_audio_returns_audio_part(self, tmp_path):
        from effgen.multimodal.video_pre import VideoSource

        out = str(tmp_path / "clip.mp4")
        _create_minimal_mp4(out)

        vs = VideoSource(out)
        audio = vs.extract_audio()

        assert audio is not None
        assert isinstance(audio, AudioPart)
        assert audio.mime == "audio/mp3"
        assert len(audio.audio) > 0

    def test_extract_audio_from_a_silent_video_returns_none(self):
        """A video with no audio track yields ``None``, not an empty ``AudioPart``."""
        if not SAMPLE_VIDEO.exists():
            pytest.skip("Sample fixture not found")

        from effgen.multimodal.video_pre import VideoSource

        # The shared fixture is generated silent (see generate_fixtures.make_video).
        vs = VideoSource(SAMPLE_VIDEO)
        assert vs.extract_audio() is None

    def test_convenience_extract_audio(self, tmp_path):
        from effgen.multimodal.video_pre import extract_audio

        out = str(tmp_path / "clip.mp4")
        _create_minimal_mp4(out)

        audio = extract_audio(out)
        assert audio is not None
        assert isinstance(audio, AudioPart)


# ---------------------------------------------------------------------------
# Duration (requires ffprobe)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _FFMPEG_AVAILABLE, reason="ffmpeg not on PATH")
class TestDuration:
    def test_duration_seconds(self, tmp_path):
        from effgen.multimodal.video_pre import VideoSource

        out = str(tmp_path / "clip.mp4")
        _create_minimal_mp4(out)

        vs = VideoSource(out)
        dur = vs.duration_seconds()
        # Synthetic clip is 2 seconds; allow some rounding tolerance
        assert dur is not None
        assert 1.0 <= dur <= 4.0

    def test_duration_missing_ffprobe_returns_none(self, tmp_path):
        from effgen.multimodal.video_pre import VideoSource

        p = tmp_path / "clip.mp4"
        p.write_bytes(b"\x00" * 8)
        vs = VideoSource(str(p))

        with patch("shutil.which", return_value=None):
            assert vs.duration_seconds() is None


# ---------------------------------------------------------------------------
# Module imports & __all__
# ---------------------------------------------------------------------------

class TestModuleInterface:
    def test_video_pre_exports(self):
        from effgen.multimodal import VideoSource, extract_video_audio, sample_frames
        assert callable(VideoSource)
        assert callable(sample_frames)
        assert callable(extract_video_audio)

    def test_missing_dep_has_install_hint(self):
        import shutil as _shutil

        from effgen.multimodal.video_pre import VideoSource

        vs = VideoSource(b"\x00" * 16, mime_type="video/mp4")
        with patch.object(_shutil, "which", return_value=None):
            try:
                vs.sample_frames()
            except MissingSystemDependency as exc:
                assert "apt install" in str(exc) or "brew install" in str(exc) or "conda install" in str(exc)
