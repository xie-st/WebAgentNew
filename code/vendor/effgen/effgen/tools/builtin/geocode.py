"""
GeocodeTool — forward and reverse geocoding via Nominatim (OpenStreetMap).

API        : https://nominatim.openstreetmap.org/
License    : Nominatim ToS requires:
               1. User-Agent: "effGen/<version>"
               2. ≤ 1 request per second (enforced via a token-bucket throttle)
               3. No bulk / automated large-scale requests.

Operations
----------
- geocode(address)     → lat, lon, display name, bounding box, place type
- reverse(lat, lon)    → address components for a coordinate pair
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
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

_NOMINATIM_BASE = "https://nominatim.openstreetmap.org"
# Fixed API host (pinned; the shared SSRF guard also blocks internal targets).
_ALLOWED_HOSTS = frozenset({"nominatim.openstreetmap.org"})


def _user_agent() -> str:
    try:
        pkg_version = version("effgen")
    except PackageNotFoundError:
        pkg_version = "dev"
    return f"effGen/{pkg_version}"


# ---------------------------------------------------------------------------
# Token-bucket rate limiter (1 request / second per Nominatim ToS)
# ---------------------------------------------------------------------------

class _TokenBucket:
    """Thread-safe token bucket: 1 token/sec capacity, 1 token/sec refill."""

    def __init__(self, rate: float = 1.0) -> None:
        self._rate = rate          # tokens per second
        self._tokens: float = 1.0
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a token is available."""
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last
                self._tokens = min(1.0, self._tokens + elapsed * self._rate)
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
            time.sleep(0.05)


_BUCKET = _TokenBucket(rate=1.0)   # module-level singleton


def _fetch_json(url: str) -> dict | list:
    _BUCKET.acquire()
    headers = {"User-Agent": _user_agent(), "Accept-Language": "en"}
    try:
        with safe_urlopen(
            url, headers=headers, timeout=10, allowed_hosts=_ALLOWED_HOSTS
        ) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except URLError as exc:
        raise ConnectionError(f"Nominatim request failed: {exc}") from exc


class GeocodeTool(BaseTool):
    """
    Forward and reverse geocoding via Nominatim (OpenStreetMap).

    Rate limit: 1 request per second (enforced automatically).
    No API key required; User-Agent is set to 'effGen/<version>'.
    """

    def __init__(self) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="geocode",
                description=(
                    "Convert addresses to coordinates (geocode) or coordinates to addresses "
                    "(reverse geocode) using the free Nominatim / OpenStreetMap API. "
                    "No API key required. Rate limited to 1 request/sec per Nominatim ToS."
                ),
                category=ToolCategory.EXTERNAL_API,
                parameters=[
                    ParameterSpec(
                        name="operation",
                        type=ParameterType.STRING,
                        description="'geocode' (address → lat/lon) or 'reverse' (lat/lon → address).",
                        required=True,
                        enum=["geocode", "reverse"],
                    ),
                    ParameterSpec(
                        name="address",
                        type=ParameterType.STRING,
                        description="Full address or place name. Required for operation='geocode'.",
                        required=False,
                    ),
                    ParameterSpec(
                        name="lat",
                        type=ParameterType.FLOAT,
                        description="Latitude. Required for operation='reverse'.",
                        required=False,
                    ),
                    ParameterSpec(
                        name="lon",
                        type=ParameterType.FLOAT,
                        description="Longitude. Required for operation='reverse'.",
                        required=False,
                    ),
                    ParameterSpec(
                        name="limit",
                        type=ParameterType.INTEGER,
                        description="Max results for geocode (1-10, default 1).",
                        required=False,
                        default=1,
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
                timeout_seconds=15,
                tags=["geocode", "maps", "nominatim", "osm", "free"],
                examples=[
                    {"operation": "geocode", "address": "Eiffel Tower, Paris"},
                    {"operation": "reverse", "lat": 48.8584, "lon": 2.2945},
                ],
            )
        )

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _geocode(self, address: str, limit: int) -> list[dict]:
        params = urlencode({
            "q": address,
            "format": "jsonv2",
            "limit": max(1, min(limit, 10)),
            "addressdetails": 1,
        })
        url = f"{_NOMINATIM_BASE}/search?{params}"
        raw = _fetch_json(url)
        if not isinstance(raw, list):
            raise ValueError(f"Unexpected Nominatim response: {raw}")
        results = []
        for item in raw:
            results.append({
                "display_name": item.get("display_name", ""),
                "lat": float(item["lat"]),
                "lon": float(item["lon"]),
                "place_type": item.get("type", ""),
                "place_rank": item.get("place_rank"),
                "osm_id": item.get("osm_id"),
                "osm_type": item.get("osm_type"),
                "bounding_box": item.get("boundingbox"),
                "address": item.get("address", {}),
            })
        return results

    def _reverse(self, lat: float, lon: float) -> dict:
        params = urlencode({
            "lat": lat,
            "lon": lon,
            "format": "jsonv2",
            "addressdetails": 1,
        })
        url = f"{_NOMINATIM_BASE}/reverse?{params}"
        raw = _fetch_json(url)
        if isinstance(raw, dict) and "error" in raw:
            raise ValueError(f"Nominatim reverse error: {raw['error']}")
        return {
            "display_name": raw.get("display_name", ""),
            "lat": float(raw.get("lat", lat)),
            "lon": float(raw.get("lon", lon)),
            "place_type": raw.get("type", ""),
            "osm_id": raw.get("osm_id"),
            "osm_type": raw.get("osm_type"),
            "address": raw.get("address", {}),
        }

    # ------------------------------------------------------------------
    # _execute
    # ------------------------------------------------------------------

    async def _execute(
        self,
        operation: str,
        address: str | None = None,
        lat: float | None = None,
        lon: float | None = None,
        limit: int = 1,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        try:
            loop = asyncio.get_event_loop()
            if operation == "geocode":
                if not address:
                    return {
                        "success": False,
                        "data": {},
                        "error": (
                            "address is required for operation='geocode'. "
                            "Pass address as the place to look up, or use "
                            "operation='reverse' with lat and lon."
                        ),
                    }
                results = await loop.run_in_executor(None, self._geocode, address, limit)
                if not results:
                    return {
                        "success": False,
                        "data": {},
                        "error": f"No results found for address: '{address}'",
                    }
                primary = results[0]
                return {
                    "success": True,
                    "data": {
                        "query": address,
                        "lat": primary["lat"],
                        "lon": primary["lon"],
                        "display_name": primary["display_name"],
                        "place_type": primary["place_type"],
                        "bounding_box": primary["bounding_box"],
                        "address": primary["address"],
                        "all_results": results,
                    },
                    "error": None,
                }
            elif operation == "reverse":
                if lat is None or lon is None:
                    return {
                        "success": False,
                        "data": {},
                        "error": (
                            "lat and lon are required for operation='reverse'. "
                            "Pass both as decimal degrees, or use "
                            "operation='geocode' with an address."
                        ),
                    }
                result = await loop.run_in_executor(None, self._reverse, lat, lon)
                return {
                    "success": True,
                    "data": result,
                    "error": None,
                }
            else:
                return {
                    "success": False,
                    "data": {},
                    "error": f"Unknown operation: {operation!r}",
                }
        except (ConnectionError, ValueError) as exc:
            logger.error("GeocodeTool error: %s", exc)
            return {"success": False, "data": {}, "error": str(exc)}
