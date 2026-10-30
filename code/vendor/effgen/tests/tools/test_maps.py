"""Tests for MapsTool — unit + integration (live OSM tile fetch)."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import pytest

from effgen.tools.builtin.maps import MapsTool

# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMapsToolUnit:
    def test_instantiation(self):
        t = MapsTool()
        assert t.metadata.name == "maps"

    @pytest.mark.asyncio
    async def test_render_missing_coords(self):
        t = MapsTool()
        result = await t._execute(operation="render")
        assert result["success"] is False
        assert "lat" in result["error"] or "lon" in result["error"]

    @pytest.mark.asyncio
    async def test_bounding_box_missing_coords(self):
        t = MapsTool()
        result = await t._execute(operation="bounding_box", south=40.0, north=41.0)
        assert result["success"] is False
        assert "south" in result["error"] or "west" in result["error"]

    @pytest.mark.asyncio
    async def test_unknown_operation(self):
        t = MapsTool()
        result = await t._execute(operation="teleport", lat=0.0, lon=0.0)
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_staticmap_not_installed(self):
        t = MapsTool()
        with patch("effgen.tools.builtin.maps._require_staticmap") as mock_req:
            mock_req.side_effect = ImportError("staticmap not installed")
            result = await t._execute(operation="render", lat=0.0, lon=0.0)
        assert result["success"] is False
        assert "staticmap" in result["error"]

    @pytest.mark.asyncio
    async def test_render_uses_default_marker(self):
        """When no markers given, a single marker at centre is auto-added."""
        t = MapsTool()

        captured = {}

        def fake_render(lat, lon, zoom, width, height, markers, dest):
            captured["markers"] = markers
            # Write a tiny valid PNG
            import struct
            import zlib
            def _png():
                sig = b'\x89PNG\r\n\x1a\n'
                def chunk(tp, data):
                    c = struct.pack('>I', len(data)) + tp + data
                    return c + struct.pack('>I', zlib.crc32(tp + data) & 0xFFFFFFFF)
                ihdr = chunk(b'IHDR', struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0))
                raw = b'\x00\xff\xff\xff'
                idat = chunk(b'IDAT', zlib.compress(raw))
                iend = chunk(b'IEND', b'')
                return sig + ihdr + idat + iend
            with open(dest, 'wb') as f:
                f.write(_png())
            return {"file": dest, "width": width, "height": height, "zoom": zoom,
                    "center": {"lat": lat, "lon": lon}, "markers": len(markers),
                    "file_size_bytes": os.path.getsize(dest)}

        with patch.object(t, "_render_map", side_effect=fake_render):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                dest = f.name
            result = await t._execute(operation="render", lat=10.0, lon=20.0, dest=dest)

        assert result["success"] is True
        assert len(captured["markers"]) == 1
        assert captured["markers"][0]["lat"] == 10.0

        os.unlink(dest)


# ---------------------------------------------------------------------------
# Integration tests (fetches live OSM tiles — requires network)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.api
class TestMapsToolIntegration:
    @pytest.mark.asyncio
    async def test_render_google_hq(self):
        t = MapsTool()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            dest = f.name
        try:
            result = await t._execute(
                operation="render",
                lat=37.422,
                lon=-122.084,
                zoom=14,
                width=400,
                height=300,
                dest=dest,
            )
            assert result["success"] is True, result.get("error")
            data = result["data"]
            assert data["file"] == dest
            assert os.path.isfile(dest)
            assert data["file_size_bytes"] > 1000, "PNG should be non-trivial"
        finally:
            if os.path.exists(dest):
                os.unlink(dest)

    @pytest.mark.asyncio
    async def test_render_with_custom_markers(self):
        t = MapsTool()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            dest = f.name
        try:
            markers = [
                {"lat": 48.8584, "lon": 2.2945, "color": "blue", "radius": 10},
                {"lat": 48.8606, "lon": 2.3376, "color": "green", "radius": 8},
            ]
            result = await t._execute(
                operation="render",
                lat=48.8566,
                lon=2.3522,
                zoom=13,
                markers=markers,
                dest=dest,
            )
            assert result["success"] is True, result.get("error")
            assert result["data"]["markers"] == 2
        finally:
            if os.path.exists(dest):
                os.unlink(dest)

    @pytest.mark.asyncio
    async def test_bounding_box(self):
        t = MapsTool()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            dest = f.name
        try:
            result = await t._execute(
                operation="bounding_box",
                south=37.38,
                west=-122.10,
                north=37.47,
                east=-122.06,
                dest=dest,
            )
            assert result["success"] is True, result.get("error")
            assert os.path.isfile(dest)
            assert result["data"]["file_size_bytes"] > 1000
        finally:
            if os.path.exists(dest):
                os.unlink(dest)

    @pytest.mark.asyncio
    async def test_auto_tmp_file(self):
        """When dest is omitted a temp file should be created and returned."""
        t = MapsTool()
        result = await t._execute(
            operation="render",
            lat=51.5074,
            lon=-0.1278,
            zoom=10,
            width=200,
            height=200,
        )
        assert result["success"] is True, result.get("error")
        fpath = result["data"]["file"]
        assert os.path.isfile(fpath)
        os.unlink(fpath)


class TestTileFailureIsBounded:
    """`staticmap` names every tile it could not fetch, one URL each.

    At zoom 13 that is a dozen tiles and around 900 characters, and it grows
    with the render. This string is what a caller prints and what a log line
    carries, so it is restated rather than forwarded.
    """

    FAILURE = (
        "could not download 12 tiles: ["
        + ", ".join(
            f"(1316, {3175 + i}, "
            f"'https://a.tile.openstreetmap.org/13/1316/{3175 + i}.png')"
            for i in range(12)
        )
        + "]"
    )

    def test_the_message_is_shorter_than_what_it_restates(self):
        from effgen.tools.builtin.maps import _restate

        restated = _restate(RuntimeError(self.FAILURE))
        assert len(restated) < len(self.FAILURE) / 3

    def test_it_keeps_the_count_the_host_and_one_example(self):
        from effgen.tools.builtin.maps import _restate

        restated = _restate(RuntimeError(self.FAILURE))
        assert "12" in restated
        assert "a.tile.openstreetmap.org" in restated
        assert "13/1316/3175.png" in restated

    def test_it_says_what_to_do_about_it(self):
        from effgen.tools.builtin.maps import _restate

        restated = _restate(RuntimeError(self.FAILURE))
        assert "zoom" in restated

    def test_only_one_tile_url_survives(self):
        from effgen.tools.builtin.maps import _restate

        restated = _restate(RuntimeError(self.FAILURE))
        assert restated.count("openstreetmap.org") == 1

    def test_an_unrelated_failure_passes_through_unchanged(self):
        from effgen.tools.builtin.maps import _restate

        assert _restate(ValueError("dest is not writable")) == "dest is not writable"
