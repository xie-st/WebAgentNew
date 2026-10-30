"""Bare path/URL coercion in ``Agent.run(inputs=[...])`` and ``part_from``.

``inputs=["photo.png"]`` should "just work": a bare path or URL string (or a
``Path``) is wrapped into the matching image/audio/video part by extension,
mirroring the CLI's smart path detection. Offline tests — no model is called.
"""

import pytest

from effgen.core.messages import AudioPart, ImagePart
from effgen.core.multimodal import _media_kind_from_name, part_from
from effgen.errors import InvalidMultimodalContent


class TestMediaKindFromName:
    @pytest.mark.parametrize(
        "name, kind",
        [
            ("photo.png", "image"),
            ("/abs/path/IMG_001.JPG", "image"),
            ("https://site.test/a/b.webp?width=200", "image"),
            ("clip.mp4", "video"),
            ("movie.MOV", "video"),
            ("speech.mp3", "audio"),
            ("note.wav", "audio"),
            ("https://x.test/song.flac#t=10", "audio"),
            ("notes.txt", None),
            ("data.json", None),
            ("no_extension", None),
        ],
    )
    def test_classification(self, name, kind):
        assert _media_kind_from_name(name) == kind


class TestPartFrom:
    def test_image_path(self, tmp_path):
        # 1x1 PNG.
        p = tmp_path / "x.png"
        p.write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00"
            b"\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        part = part_from(str(p))
        assert isinstance(part, ImagePart)
        # A Path object works identically.
        assert isinstance(part_from(p), ImagePart)

    def test_audio_path(self, tmp_path):
        p = tmp_path / "a.wav"
        p.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")
        assert isinstance(part_from(p), AudioPart)

    def test_unknown_extension_raises_helpful(self, tmp_path):
        p = tmp_path / "mystery.dat"
        p.write_bytes(b"\x00\x01")
        with pytest.raises(InvalidMultimodalContent, match="infer the media type"):
            part_from(str(p))

    @pytest.mark.parametrize("name", ["report.pdf", "notes.docx", "budget.xlsx", "data.csv"])
    def test_document_extension_points_to_rag_not_media_wrappers(self, name):
        # A document handed to the image/audio/video path gets a hint toward
        # RAG ingestion / the pdf/excel tools, not the unrelated media wrappers.
        with pytest.raises(InvalidMultimodalContent) as exc:
            part_from(f"/tmp/{name}")
        msg = str(exc.value)
        assert "rag" in msg.lower()
        assert "image_from" not in msg


class TestRunInputsCoercion:
    """`_build_multimodal_prompt` accepts bare paths and gives clear errors."""

    def _agent(self):
        from effgen.presets import create_agent
        from tests.fixtures.mock_models import MockModel

        return create_agent("general", MockModel(responses=["ok"]))

    def test_bare_path_is_wrapped(self, tmp_path):
        p = tmp_path / "pic.png"
        p.write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00"
            b"\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        messages = self._agent()._build_multimodal_prompt("describe", [str(p)])
        user_parts = messages[-1].content
        assert any(isinstance(part, ImagePart) for part in user_parts)

    def test_unknown_type_error_names_helpers(self):
        agent = self._agent()
        with pytest.raises(TypeError) as exc:
            agent._build_multimodal_prompt("x", [123])
        assert "image_from" in str(exc.value)

    def test_missing_file_error_mentions_import(self):
        agent = self._agent()
        with pytest.raises(TypeError) as exc:
            agent._build_multimodal_prompt("x", ["/no/such/file.png"])
        msg = str(exc.value)
        assert "from effgen import image_from" in msg

    def test_document_input_error_points_to_rag_not_media_helpers(self):
        # inputs=["report.pdf"] is the natural (wrong) call for a document. The
        # error should route to RAG / the pdf-tool, without the image_from import
        # line that only fits an image/audio/video mistake.
        agent = self._agent()
        with pytest.raises(TypeError) as exc:
            agent._build_multimodal_prompt("x", ["/tmp/report.pdf"])
        msg = str(exc.value)
        assert "rag" in msg.lower()
        assert "from effgen import image_from" not in msg
