"""Tests for GeocodeTool — unit + integration (live Nominatim API)."""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from effgen.tools.builtin.geocode import GeocodeTool, _fetch_json, _TokenBucket, _user_agent

# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGeocodeToolUnit:
    def test_instantiation(self):
        t = GeocodeTool()
        assert t.metadata.name == "geocode"

    @pytest.mark.asyncio
    async def test_geocode_missing_address(self):
        t = GeocodeTool()
        result = await t._execute(operation="geocode")
        assert result["success"] is False
        assert "address" in result["error"]

    @pytest.mark.asyncio
    async def test_reverse_missing_coords(self):
        t = GeocodeTool()
        result = await t._execute(operation="reverse")
        assert result["success"] is False
        assert "lat" in result["error"]

    @pytest.mark.asyncio
    async def test_unknown_operation(self):
        t = GeocodeTool()
        result = await t._execute(operation="fly_me_to_the_moon")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_geocode_mocked(self):
        fake_nominatim = [
            {
                "display_name": "Paris, France",
                "lat": "48.8566",
                "lon": "2.3522",
                "type": "city",
                "place_rank": 16,
                "osm_id": 7444,
                "osm_type": "relation",
                "boundingbox": ["48.8155", "48.9021", "2.2242", "2.4699"],
                "address": {"city": "Paris", "country": "France"},
            }
        ]
        with patch("effgen.tools.builtin.geocode._fetch_json", return_value=fake_nominatim):
            t = GeocodeTool()
            result = await t._execute(operation="geocode", address="Paris")
        assert result["success"] is True
        assert result["data"]["lat"] == pytest.approx(48.8566, abs=0.01)
        assert "Paris" in result["data"]["display_name"]

    @pytest.mark.asyncio
    async def test_reverse_mocked(self):
        fake_nominatim = {
            "display_name": "Eiffel Tower, Paris, France",
            "lat": "48.8584",
            "lon": "2.2945",
            "type": "attraction",
            "osm_id": 5013364,
            "osm_type": "way",
            "address": {"attraction": "Eiffel Tower", "city": "Paris"},
        }
        with patch("effgen.tools.builtin.geocode._fetch_json", return_value=fake_nominatim):
            t = GeocodeTool()
            result = await t._execute(operation="reverse", lat=48.8584, lon=2.2945)
        assert result["success"] is True
        assert "Eiffel" in result["data"]["display_name"]

    def test_token_bucket_throttle(self):
        """Bucket should delay three immediate acquisitions by at least two seconds."""
        bucket = _TokenBucket(rate=1.0)
        t0 = time.monotonic()
        bucket.acquire()
        bucket.acquire()
        bucket.acquire()
        elapsed = time.monotonic() - t0
        assert elapsed >= 1.9

    def test_fetch_json_sets_user_agent_header(self):
        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self):
                return json.dumps([]).encode()

        def fake_safe_urlopen(url, *, headers=None, timeout=None, **kwargs):
            captured["user_agent"] = (headers or {}).get("User-Agent")
            captured["timeout"] = timeout
            return FakeResponse()

        with patch("effgen.tools.builtin.geocode._BUCKET") as bucket:
            bucket.acquire = MagicMock()
            with patch(
                "effgen.tools.builtin.geocode.safe_urlopen", side_effect=fake_safe_urlopen
            ):
                _fetch_json("https://nominatim.openstreetmap.org/search?q=test")

        assert captured["user_agent"] == _user_agent()
        assert captured["user_agent"].startswith("effGen/")
        assert captured["timeout"] == 10

    @pytest.mark.asyncio
    async def test_no_results_error(self):
        with patch("effgen.tools.builtin.geocode._fetch_json", return_value=[]):
            t = GeocodeTool()
            result = await t._execute(operation="geocode", address="zzzzzzzz_nonexistent_place")
        assert result["success"] is False
        assert "No results" in result["error"]


# ---------------------------------------------------------------------------
# Integration tests (live Nominatim — 1 req/sec rate limit enforced)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.api
class TestGeocodeToolIntegration:
    @pytest.mark.asyncio
    async def test_geocode_google_hq(self):
        t = GeocodeTool()
        result = await t._execute(
            operation="geocode",
            address="1600 Amphitheatre Pkwy, Mountain View, CA",
        )
        assert result["success"] is True, result.get("error")
        data = result["data"]
        assert data["lat"] == pytest.approx(37.42, abs=0.1)
        assert data["lon"] == pytest.approx(-122.08, abs=0.1)

    @pytest.mark.asyncio
    async def test_geocode_eiffel_tower(self):
        t = GeocodeTool()
        result = await t._execute(operation="geocode", address="Eiffel Tower, Paris")
        assert result["success"] is True, result.get("error")
        data = result["data"]
        assert data["lat"] == pytest.approx(48.858, abs=0.1)
        assert data["lon"] == pytest.approx(2.294, abs=0.1)

    @pytest.mark.asyncio
    async def test_reverse_paris(self):
        t = GeocodeTool()
        result = await t._execute(operation="reverse", lat=48.8566, lon=2.3522)
        assert result["success"] is True, result.get("error")
        assert "France" in result["data"]["display_name"]

    @pytest.mark.asyncio
    async def test_geocode_with_limit(self):
        t = GeocodeTool()
        result = await t._execute(operation="geocode", address="Springfield", limit=3)
        assert result["success"] is True
        # all_results may be up to 3
        assert len(result["data"]["all_results"]) >= 1

    @pytest.mark.asyncio
    async def test_address_components_present(self):
        t = GeocodeTool()
        result = await t._execute(operation="geocode", address="Big Ben, London")
        assert result["success"] is True
        data = result["data"]
        assert "bounding_box" in data
        assert "address" in data
