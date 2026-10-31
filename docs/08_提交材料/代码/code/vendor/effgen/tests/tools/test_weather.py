"""Tests for WeatherTool — unit + integration (live Open-Meteo API)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from effgen.tools.builtin.weather import _WMO_CODES, WeatherTool

# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWeatherToolUnit:
    def test_instantiation(self):
        t = WeatherTool()
        assert t.metadata.name == "weather"

    def test_wmo_code_lookup(self):
        assert _WMO_CODES[0] == "Clear sky"
        assert _WMO_CODES[3] == "Overcast"
        assert _WMO_CODES[95] == "Thunderstorm"

    def test_units_params_metric(self):
        t = WeatherTool()
        p = t._units_params("metric")
        assert p["temperature_unit"] == "celsius"
        assert p["wind_speed_unit"] == "kmh"

    def test_units_params_imperial(self):
        t = WeatherTool()
        p = t._units_params("imperial")
        assert p["temperature_unit"] == "fahrenheit"
        assert p["wind_speed_unit"] == "mph"

    def test_cache_miss_returns_none(self):
        t = WeatherTool()
        assert t._get_cache("nonexistent_key") is None

    def test_cache_hit(self):
        t = WeatherTool(cache_ttl=60)
        t._set_cache("mykey", {"foo": "bar"})
        result = t._get_cache("mykey")
        assert result == {"foo": "bar"}

    def test_cache_expiry(self):
        import time
        t = WeatherTool(cache_ttl=0)
        t._set_cache("mykey", {"foo": "bar"})
        time.sleep(0.01)
        assert t._get_cache("mykey") is None

    def test_idx_helper(self):
        d = {"temps": [10.0, 20.0, 30.0]}
        assert WeatherTool._idx(d, "temps", 1) == 20.0
        assert WeatherTool._idx(d, "temps", 99) is None
        assert WeatherTool._idx(d, "missing", 0) is None

    @pytest.mark.asyncio
    async def test_missing_coordinates_returns_error(self):
        t = WeatherTool()
        result = await t._execute(operation="current")
        assert result["success"] is False
        assert "lat" in result["error"] or "location" in result["error"]

    @pytest.mark.asyncio
    async def test_historical_missing_dates(self):
        t = WeatherTool()
        result = await t._execute(operation="historical", lat=37.42, lon=-122.08)
        assert result["success"] is False
        assert "start_date" in result["error"]

    @pytest.mark.asyncio
    async def test_historical_rejects_future_dates(self):
        from datetime import date, timedelta

        tomorrow = date.today() + timedelta(days=1)
        t = WeatherTool()
        result = await t._execute(
            operation="historical",
            lat=37.42,
            lon=-122.08,
            start_date=str(tomorrow),
            end_date=str(tomorrow),
        )
        assert result["success"] is False
        assert "future" in result["error"]

    def test_recent_historical_uses_forecast_endpoint(self):
        from datetime import date, timedelta

        yesterday = date.today() - timedelta(days=1)
        fake_resp = {
            "daily": {
                "time": [str(yesterday)],
                "weather_code": [0],
                "temperature_2m_max": [20.0],
                "temperature_2m_min": [10.0],
                "precipitation_sum": [0.0],
                "wind_speed_10m_max": [8.0],
            },
            "timezone": "America/Los_Angeles",
        }
        seen_urls = []

        def fake_fetch(url):
            seen_urls.append(url)
            return fake_resp

        t = WeatherTool()
        with patch("effgen.tools.builtin.weather._fetch_json", side_effect=fake_fetch):
            result = t._historical(37.42, -122.08, str(yesterday), str(yesterday), "metric")

        assert result["source"] == "forecast"
        assert result["records"][0]["date"] == str(yesterday)
        assert "api.open-meteo.com/v1/forecast" in seen_urls[0]

    @pytest.mark.asyncio
    async def test_unknown_operation(self):
        t = WeatherTool()
        result = await t._execute(operation="invalid", lat=0.0, lon=0.0)
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_geocode_path_called(self):
        t = WeatherTool()
        # Mock _geocode so no network is needed for this unit test
        t._geocode = MagicMock(return_value=(37.42, -122.08, "Mountain View, US"))
        # Also mock the Open-Meteo fetch
        fake_resp = {
            "current": {
                "temperature_2m": 20.0,
                "relative_humidity_2m": 55,
                "apparent_temperature": 19.0,
                "precipitation": 0.0,
                "weather_code": 1,
                "wind_speed_10m": 10.0,
                "wind_direction_10m": 270,
                "surface_pressure": 1013.0,
                "cloud_cover": 10,
                "is_day": 1,
                "time": "2024-01-01T12:00",
            },
            "timezone": "America/Los_Angeles",
        }
        with patch("effgen.tools.builtin.weather._fetch_json", return_value=fake_resp):
            result = await t._execute(operation="current", location="Mountain View")
        assert result["success"] is True
        t._geocode.assert_called_once_with("Mountain View")


# ---------------------------------------------------------------------------
# Integration tests (live API — Open-Meteo is free, no key required)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.api
class TestWeatherToolIntegration:
    """Live calls to Open-Meteo. Mark integration so CI can skip if needed."""

    @pytest.mark.asyncio
    async def test_current_by_coords(self):
        t = WeatherTool()
        result = await t._execute(operation="current", lat=37.42, lon=-122.08)
        assert result["success"] is True, result.get("error")
        data = result["data"]
        assert "temperature" in data
        assert "conditions" in data
        assert data["coordinates"]["latitude"] == pytest.approx(37.42, abs=0.01)

    @pytest.mark.asyncio
    async def test_current_by_location(self):
        t = WeatherTool()
        result = await t._execute(operation="current", location="Berlin")
        assert result["success"] is True, result.get("error")
        data = result["data"]
        assert "temperature" in data
        assert "location" in data

    @pytest.mark.asyncio
    async def test_forecast(self):
        t = WeatherTool()
        result = await t._execute(operation="forecast", lat=48.8566, lon=2.3522, days=3)
        assert result["success"] is True, result.get("error")
        fc = result["data"]["forecast"]
        assert len(fc) == 3
        for day in fc:
            assert "date" in day
            assert "temp_max" in day

    @pytest.mark.asyncio
    async def test_historical(self):
        t = WeatherTool()
        from datetime import date, timedelta
        yesterday = date.today() - timedelta(days=1)
        result = await t._execute(
            operation="historical",
            lat=51.5074,
            lon=-0.1278,
            start_date=str(yesterday),
            end_date=str(yesterday),
        )
        assert result["success"] is True, result.get("error")
        records = result["data"]["records"]
        assert len(records) == 1
        assert records[0]["date"] == str(yesterday)
        assert records[0]["temp_max"] is not None

    @pytest.mark.asyncio
    async def test_imperial_units(self):
        t = WeatherTool()
        result = await t._execute(
            operation="current", lat=40.7128, lon=-74.006, units="imperial"
        )
        assert result["success"] is True
        assert result["data"]["temperature_unit"] == "°F"

    @pytest.mark.asyncio
    async def test_caching(self):
        t = WeatherTool(cache_ttl=60)
        r1 = await t._execute(operation="current", lat=35.6762, lon=139.6503)
        r2 = await t._execute(operation="current", lat=35.6762, lon=139.6503)
        assert r1["success"] is True
        assert r2.get("cached") is True
