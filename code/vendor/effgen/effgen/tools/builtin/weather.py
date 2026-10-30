"""
WeatherTool — free weather data via Open-Meteo.

Backend  : Open-Meteo (https://api.open-meteo.com/) — no API key required.
Optional : GeocodeTool integration (accept place name instead of lat/lon).

Operations
----------
- current(lat, lon)                          → current conditions
- forecast(lat, lon, days=7)                 → daily forecast
- historical(lat, lon, start_date, end_date) → historical daily data
- by_location(location, operation, **kw)     → geocode then weather
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import date, datetime, timedelta
from importlib.metadata import PackageNotFoundError, version
from typing import Any
from urllib.error import URLError
from urllib.parse import urlencode

from ..base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)
from ._net import safe_urlopen

logger = logging.getLogger(__name__)

_OPEN_METEO_BASE = "https://api.open-meteo.com/v1"
# Fixed API hosts (pinned; the shared SSRF guard also blocks internal targets).
# Open-Meteo serves forecast/geocoding/archive on subdomains of open-meteo.com.
_ALLOWED_HOSTS = frozenset({"*.open-meteo.com"})

# WMO Weather interpretation codes
_WMO_CODES: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Heavy freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def _user_agent() -> str:
    try:
        pkg_version = version("effgen")
    except PackageNotFoundError:
        pkg_version = "dev"
    return f"effGen/{pkg_version}"


def _fetch_json(url: str) -> dict:
    headers = {"User-Agent": _user_agent()}
    try:
        with safe_urlopen(
            url, headers=headers, timeout=15, allowed_hosts=_ALLOWED_HOSTS
        ) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except URLError as exc:
        raise ConnectionError(f"Network error fetching weather data: {exc}") from exc


class WeatherTool(BaseTool):
    """
    Weather data tool backed by the free Open-Meteo API.

    No API key required. Accepts lat/lon directly or a place name
    (resolved via the Open-Meteo geocoding endpoint).

    Operations (passed as the *operation* parameter):
        current     — real-time weather variables.
        forecast    — daily forecast for 1-16 days.
        historical  — daily historical data (requires start_date/end_date).
    """

    def __init__(self, cache_ttl: int = 600) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="weather",
                description=(
                    "Get weather data via the free Open-Meteo API (no key needed). "
                    "Supports current conditions, multi-day forecasts, and historical data. "
                    "Accepts latitude/longitude or a human-readable location name."
                ),
                category=ToolCategory.EXTERNAL_API,
                parameters=[
                    ParameterSpec(
                        name="operation",
                        type=ParameterType.STRING,
                        description=(
                            "Operation: 'current', 'forecast', or 'historical'. "
                            "Default: 'current'."
                        ),
                        required=False,
                        default="current",
                        enum=["current", "forecast", "historical"],
                    ),
                    ParameterSpec(
                        name="lat",
                        type=ParameterType.FLOAT,
                        description="Latitude (-90 to 90). Required unless location is given.",
                        required=False,
                    ),
                    ParameterSpec(
                        name="lon",
                        type=ParameterType.FLOAT,
                        description="Longitude (-180 to 180). Required unless location is given.",
                        required=False,
                    ),
                    ParameterSpec(
                        name="location",
                        type=ParameterType.STRING,
                        description=(
                            "Human-readable place name (e.g. 'Berlin', 'Tokyo'). "
                            "Used when lat/lon are omitted."
                        ),
                        required=False,
                    ),
                    ParameterSpec(
                        name="days",
                        type=ParameterType.INTEGER,
                        description="Forecast days (1-16). Only for operation='forecast'.",
                        required=False,
                        default=7,
                    ),
                    ParameterSpec(
                        name="start_date",
                        type=ParameterType.STRING,
                        description="Start date YYYY-MM-DD. Only for operation='historical'.",
                        required=False,
                    ),
                    ParameterSpec(
                        name="end_date",
                        type=ParameterType.STRING,
                        description="End date YYYY-MM-DD. Only for operation='historical'.",
                        required=False,
                    ),
                    ParameterSpec(
                        name="units",
                        type=ParameterType.STRING,
                        description="'metric' (°C, km/h) or 'imperial' (°F, mph). Default: 'metric'.",
                        required=False,
                        default="metric",
                        enum=["metric", "imperial"],
                    ),
                ],
                returns={
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "data": {"type": "object"},
                        "error": {"type": "string"},
                    },
                },
                timeout_seconds=20,
                tags=["weather", "open-meteo", "free", "geo"],
                examples=[
                    {
                        "operation": "current",
                        "lat": 37.42,
                        "lon": -122.08,
                    },
                    {
                        "operation": "forecast",
                        "location": "Berlin",
                        "days": 3,
                    },
                ],
            )
        )
        self.cache_ttl = cache_ttl
        self._cache: dict[str, tuple[float, dict]] = {}

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _get_cache(self, key: str) -> dict | None:
        entry = self._cache.get(key)
        if entry and time.time() - entry[0] < self.cache_ttl:
            return entry[1]
        return None

    def _set_cache(self, key: str, data: dict) -> None:
        self._cache[key] = (time.time(), data)

    # ------------------------------------------------------------------
    # Geocoding (Open-Meteo free geocoding endpoint)
    # ------------------------------------------------------------------

    def _geocode(self, location: str) -> tuple[float, float, str]:
        """Resolve a place name → (lat, lon, display_name)."""
        key = f"geo:{location.lower()}"
        cached = self._get_cache(key)
        if cached:
            return cached["lat"], cached["lon"], cached["name"]

        params = urlencode({"name": location, "count": 1, "language": "en", "format": "json"})
        url = f"https://geocoding-api.open-meteo.com/v1/search?{params}"
        data = _fetch_json(url)
        results = data.get("results") or []
        if not results:
            raise ValueError(f"Location not found: '{location}'")
        r = results[0]
        lat, lon = float(r["latitude"]), float(r["longitude"])
        name = r.get("name", location)
        country = r.get("country", "")
        display = f"{name}, {country}" if country else name
        self._set_cache(key, {"lat": lat, "lon": lon, "name": display})
        return lat, lon, display

    # ------------------------------------------------------------------
    # API calls
    # ------------------------------------------------------------------

    def _units_params(self, units: str) -> dict:
        if units == "imperial":
            return {"temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
                    "precipitation_unit": "inch"}
        return {"temperature_unit": "celsius", "wind_speed_unit": "kmh",
                "precipitation_unit": "mm"}

    def _current(self, lat: float, lon: float, units: str) -> dict:
        up = self._units_params(units)
        params = urlencode({
            "latitude": lat,
            "longitude": lon,
            "current": (
                "temperature_2m,relative_humidity_2m,apparent_temperature,"
                "precipitation,weather_code,wind_speed_10m,wind_direction_10m,"
                "surface_pressure,cloud_cover,is_day"
            ),
            **up,
            "timezone": "auto",
        })
        url = f"{_OPEN_METEO_BASE}/forecast?{params}"
        raw = _fetch_json(url)
        cur = raw.get("current", {})
        code = cur.get("weather_code", 0)
        temp_sym = "°F" if units == "imperial" else "°C"
        wind_unit_label = "mph" if units == "imperial" else "km/h"
        return {
            "temperature": cur.get("temperature_2m"),
            "apparent_temperature": cur.get("apparent_temperature"),
            "temperature_unit": temp_sym,
            "humidity": cur.get("relative_humidity_2m"),
            "precipitation": cur.get("precipitation"),
            "precipitation_unit": "inch" if units == "imperial" else "mm",
            "wind_speed": cur.get("wind_speed_10m"),
            "wind_direction": cur.get("wind_direction_10m"),
            "wind_unit": wind_unit_label,
            "cloud_cover": cur.get("cloud_cover"),
            "surface_pressure": cur.get("surface_pressure"),
            "weather_code": code,
            "conditions": _WMO_CODES.get(code, "Unknown"),
            "is_day": bool(cur.get("is_day")),
            "time": cur.get("time"),
            "timezone": raw.get("timezone"),
        }

    def _forecast(self, lat: float, lon: float, days: int, units: str) -> dict:
        days = max(1, min(days, 16))
        up = self._units_params(units)
        params = urlencode({
            "latitude": lat,
            "longitude": lon,
            "daily": (
                "weather_code,temperature_2m_max,temperature_2m_min,"
                "apparent_temperature_max,apparent_temperature_min,"
                "precipitation_sum,precipitation_probability_max,"
                "wind_speed_10m_max,wind_direction_10m_dominant"
            ),
            "forecast_days": days,
            **up,
            "timezone": "auto",
        })
        url = f"{_OPEN_METEO_BASE}/forecast?{params}"
        raw = _fetch_json(url)
        daily = raw.get("daily", {})
        dates = daily.get("time", [])
        temp_sym = "°F" if units == "imperial" else "°C"
        wind_unit_label = "mph" if units == "imperial" else "km/h"
        forecast_list = []
        for i, day in enumerate(dates):
            code = (daily.get("weather_code") or [0])[i] if daily.get("weather_code") else 0
            forecast_list.append({
                "date": day,
                "weather_code": code,
                "conditions": _WMO_CODES.get(int(code), "Unknown"),
                "temp_max": self._idx(daily, "temperature_2m_max", i),
                "temp_min": self._idx(daily, "temperature_2m_min", i),
                "temp_unit": temp_sym,
                "precipitation_sum": self._idx(daily, "precipitation_sum", i),
                "precipitation_prob": self._idx(daily, "precipitation_probability_max", i),
                "wind_speed_max": self._idx(daily, "wind_speed_10m_max", i),
                "wind_unit": wind_unit_label,
            })
        return {
            "days": days,
            "timezone": raw.get("timezone"),
            "forecast": forecast_list,
        }

    def _historical(
        self, lat: float, lon: float, start_date: str, end_date: str, units: str
    ) -> dict:
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d").date()
            end = datetime.strptime(end_date, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError("historical dates must use YYYY-MM-DD format.") from exc
        if start > end:
            raise ValueError("historical start_date must be on or before end_date.")
        if end > date.today():
            raise ValueError("historical end_date cannot be in the future.")

        up = self._units_params(units)
        params = urlencode({
            "latitude": lat,
            "longitude": lon,
            "daily": (
                "weather_code,temperature_2m_max,temperature_2m_min,"
                "precipitation_sum,wind_speed_10m_max"
            ),
            "start_date": start_date,
            "end_date": end_date,
            **up,
            "timezone": "auto",
        })
        # Open-Meteo's forecast endpoint explicitly supports recent past data
        # via date ranges/past_days; older data uses the archive endpoint.
        if start >= date.today() - timedelta(days=92):
            source = "forecast"
            url = f"{_OPEN_METEO_BASE}/forecast?{params}"
        else:
            source = "archive"
            url = f"https://archive-api.open-meteo.com/v1/archive?{params}"
        raw = _fetch_json(url)
        daily = raw.get("daily", {})
        dates = daily.get("time", [])
        temp_sym = "°F" if units == "imperial" else "°C"
        wind_unit_label = "mph" if units == "imperial" else "km/h"
        records = []
        for i, day in enumerate(dates):
            code = (daily.get("weather_code") or [0])[i] if daily.get("weather_code") else 0
            records.append({
                "date": day,
                "weather_code": code,
                "conditions": _WMO_CODES.get(int(code), "Unknown"),
                "temp_max": self._idx(daily, "temperature_2m_max", i),
                "temp_min": self._idx(daily, "temperature_2m_min", i),
                "temp_unit": temp_sym,
                "precipitation_sum": self._idx(daily, "precipitation_sum", i),
                "wind_speed_max": self._idx(daily, "wind_speed_10m_max", i),
                "wind_unit": wind_unit_label,
            })
        return {
            "start_date": start_date,
            "end_date": end_date,
            "timezone": raw.get("timezone"),
            "source": source,
            "records": records,
        }

    @staticmethod
    def _idx(d: dict, key: str, i: int) -> Any:
        arr = d.get(key)
        if arr and i < len(arr):
            return arr[i]
        return None

    # ------------------------------------------------------------------
    # _execute
    # ------------------------------------------------------------------

    async def _execute(
        self,
        operation: str = "current",
        lat: float | None = None,
        lon: float | None = None,
        location: str | None = None,
        days: int = 7,
        start_date: str | None = None,
        end_date: str | None = None,
        units: str = "metric",
        **_kwargs: Any,
    ) -> dict[str, Any]:
        try:
            # Resolve coordinates
            display_name = None
            if lat is None or lon is None:
                if not location:
                    return {
                        "success": False,
                        "data": {},
                        "error": (
                            "The call named no place to report on. Pass "
                            "location as a place name, or lat and lon as "
                            "decimal degrees."
                        ),
                    }
                lat, lon, display_name = await asyncio.get_event_loop().run_in_executor(
                    None, self._geocode, location
                )

            cache_key = f"{operation}:{lat:.4f}:{lon:.4f}:{units}:{days}:{start_date}:{end_date}"
            cached = self._get_cache(cache_key)
            if cached:
                return {"success": True, "data": cached, "error": None, "cached": True}

            if operation == "current":
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self._current, lat, lon, units
                )
            elif operation == "forecast":
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self._forecast, lat, lon, days, units
                )
            elif operation == "historical":
                if not start_date or not end_date:
                    return {
                        "success": False,
                        "data": {},
                        "error": "historical operation requires start_date and end_date.",
                    }
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self._historical, lat, lon, start_date, end_date, units
                )
            else:
                return {
                    "success": False,
                    "data": {},
                    "error": f"Unknown operation: {operation!r}",
                }

            result["coordinates"] = {"latitude": lat, "longitude": lon}
            if display_name:
                result["location"] = display_name
            self._set_cache(cache_key, result)
            return {"success": True, "data": result, "error": None}

        except (ConnectionError, ValueError, KeyError) as exc:
            logger.error("WeatherTool error: %s", exc)
            return {"success": False, "data": {}, "error": str(exc)}
