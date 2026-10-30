"""
MapsTool — render static map images backed by the `staticmap` library.

The library fetches OSM tiles over the network and composites them locally.
No API key required. Requires the `staticmap` Python package:

    pip install staticmap          # or: pip install 'effgen[geo]'

System note: No binary system dependencies; `staticmap` uses Pillow internally.

Operations
----------
- render(lat, lon, zoom, markers, size, dest)
      → PNG centred on (lat, lon).
- bounding_box(south, west, north, east, zoom, dest)
      → PNG covering a geographic rectangle.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from ..base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)

logger = logging.getLogger(__name__)


def _user_agent() -> str:
    try:
        pkg_version = version("effgen")
    except PackageNotFoundError:
        pkg_version = "dev"
    return f"effGen/{pkg_version}"


def _require_staticmap():
    try:
        import staticmap as _sm
        return _sm
    except ImportError as exc:
        raise ImportError(
            "staticmap is not installed. Install it with:\n"
            "    pip install staticmap\n"
            "or: pip install 'effgen[geo]'"
        ) from exc


#: `staticmap` reports a download failure by inlining every tile it could not
#: fetch, one URL each. At zoom 13 that is a dozen tiles and around 900
#: characters, and it grows with the render — but this string is what a caller
#: prints and what a log line carries, so it has to stay readable.
_TILE_FAILURE_RE = re.compile(
    r"could not download (\d+) tiles?:\s*\[(.*)", re.IGNORECASE | re.DOTALL
)
_TILE_URL_RE = re.compile(r"https?://([^/'\"\s]+)(/\S*?)['\"\)]")


def _restate(exc: Exception) -> str:
    """Return a bounded, actionable message for a rendering failure.

    Args:
        exc: Whatever the renderer raised.

    Returns:
        The tile-download failure restated as a count, the tile host and one
        example, or the original message when it is not that failure.
    """
    text = str(exc)
    match = _TILE_FAILURE_RE.search(text)
    if not match:
        return text

    count = match.group(1)
    tile = _TILE_URL_RE.search(match.group(2))
    where = f" from {tile.group(1)} (first: {tile.group(2).lstrip('/')})" if tile else ""
    return (
        f"could not download {count} map tile(s){where}. "
        f"Check network access to the tile host, or lower `zoom` to request "
        f"fewer tiles."
    )


class MapsTool(BaseTool):
    """
    Static map renderer backed by the `staticmap` Python library + OSM tiles.

    Operations:
        render       — render a map centred on (lat, lon).
        bounding_box — render a map covering a lat/lon bounding box.

    Output PNG is saved to *dest* path; if omitted a temp file is created.
    Returns file path + size metadata.
    """

    def __init__(self) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="maps",
                description=(
                    "Render static map images using OpenStreetMap tiles (no API key needed). "
                    "Supports point-centred maps with optional markers and bounding-box maps. "
                    "Saves the result as a PNG file and returns the file path."
                ),
                category=ToolCategory.EXTERNAL_API,
                parameters=[
                    ParameterSpec(
                        name="operation",
                        type=ParameterType.STRING,
                        description="'render' (centred map) or 'bounding_box' (area map).",
                        required=False,
                        default="render",
                        enum=["render", "bounding_box"],
                    ),
                    ParameterSpec(
                        name="lat",
                        type=ParameterType.FLOAT,
                        description="Centre latitude. Required for operation='render'.",
                        required=False,
                    ),
                    ParameterSpec(
                        name="lon",
                        type=ParameterType.FLOAT,
                        description="Centre longitude. Required for operation='render'.",
                        required=False,
                    ),
                    ParameterSpec(
                        name="zoom",
                        type=ParameterType.INTEGER,
                        description="Zoom level (0-19). Default: 12.",
                        required=False,
                        default=12,
                    ),
                    ParameterSpec(
                        name="width",
                        type=ParameterType.INTEGER,
                        description="Output image width in pixels. Default: 800.",
                        required=False,
                        default=800,
                    ),
                    ParameterSpec(
                        name="height",
                        type=ParameterType.INTEGER,
                        description="Output image height in pixels. Default: 600.",
                        required=False,
                        default=600,
                    ),
                    ParameterSpec(
                        name="markers",
                        type=ParameterType.ARRAY,
                        description=(
                            "List of marker dicts: {lat, lon, color='red', radius=8}. "
                            "Default: a single marker at the centre."
                        ),
                        required=False,
                    ),
                    ParameterSpec(
                        name="south",
                        type=ParameterType.FLOAT,
                        description="South latitude of bounding box.",
                        required=False,
                    ),
                    ParameterSpec(
                        name="west",
                        type=ParameterType.FLOAT,
                        description="West longitude of bounding box.",
                        required=False,
                    ),
                    ParameterSpec(
                        name="north",
                        type=ParameterType.FLOAT,
                        description="North latitude of bounding box.",
                        required=False,
                    ),
                    ParameterSpec(
                        name="east",
                        type=ParameterType.FLOAT,
                        description="East longitude of bounding box.",
                        required=False,
                    ),
                    ParameterSpec(
                        name="dest",
                        type=ParameterType.STRING,
                        description=(
                            "Output PNG file path. If omitted, a temporary file is created."
                        ),
                        required=False,
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
                timeout_seconds=60,
                tags=["maps", "staticmap", "osm", "png", "geo", "free"],
                examples=[
                    {
                        "operation": "render",
                        "lat": 48.8566,
                        "lon": 2.3522,
                        "zoom": 14,
                        "dest": "/tmp/paris.png",
                    },
                    {
                        "operation": "bounding_box",
                        "south": 48.8,
                        "west": 2.25,
                        "north": 48.9,
                        "east": 2.45,
                        "dest": "/tmp/paris_box.png",
                    },
                ],
            )
        )

    # ------------------------------------------------------------------
    # Core render helpers (run in executor to avoid blocking event loop)
    # ------------------------------------------------------------------

    def _render_map(
        self,
        lat: float,
        lon: float,
        zoom: int,
        width: int,
        height: int,
        markers: list[dict],
        dest: str,
    ) -> dict:
        sm = _require_staticmap()
        m = sm.StaticMap(
            width,
            height,
            headers={"User-Agent": _user_agent()},
        )
        for mk in markers:
            mlat = float(mk.get("lat", lat))
            mlon = float(mk.get("lon", lon))
            color = mk.get("color", "red")
            radius = int(mk.get("radius", 8))
            m.add_marker(sm.CircleMarker((mlon, mlat), color, radius))

        image = m.render(zoom=zoom, center=(lon, lat))
        image.save(dest)
        size = os.path.getsize(dest)
        return {
            "file": dest,
            "width": width,
            "height": height,
            "zoom": zoom,
            "center": {"lat": lat, "lon": lon},
            "markers": len(markers),
            "file_size_bytes": size,
        }

    def _render_bbox(
        self,
        south: float,
        west: float,
        north: float,
        east: float,
        zoom: int | None,
        width: int,
        height: int,
        dest: str,
    ) -> dict:
        sm = _require_staticmap()
        m = sm.StaticMap(
            width,
            height,
            headers={"User-Agent": _user_agent()},
        )
        # Add invisible corner markers so staticmap knows the extent
        for (clat, clon) in [(south, west), (north, east)]:
            m.add_marker(sm.CircleMarker((clon, clat), "#00000000", 1))

        image = m.render(zoom=zoom)
        image.save(dest)
        size = os.path.getsize(dest)
        return {
            "file": dest,
            "width": width,
            "height": height,
            "bounding_box": {
                "south": south,
                "west": west,
                "north": north,
                "east": east,
            },
            "file_size_bytes": size,
        }

    # ------------------------------------------------------------------
    # _execute
    # ------------------------------------------------------------------

    async def _execute(
        self,
        operation: str = "render",
        lat: float | None = None,
        lon: float | None = None,
        zoom: int = 12,
        width: int = 800,
        height: int = 600,
        markers: list[dict] | None = None,
        south: float | None = None,
        west: float | None = None,
        north: float | None = None,
        east: float | None = None,
        dest: str | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        try:
            _require_staticmap()  # raise early if not installed
        except ImportError as exc:
            return {"success": False, "data": {}, "error": str(exc)}

        try:
            if dest is None:
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                tmp.close()
                dest = tmp.name

            dest = str(Path(dest).expanduser().resolve())
            os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)

            loop = asyncio.get_event_loop()

            if operation == "render":
                if lat is None or lon is None:
                    return {
                        "success": False,
                        "data": {},
                        "error": (
                            "lat and lon are required for operation='render'. "
                            "Pass both as decimal degrees; the geocode tool "
                            "resolves a place name to a pair."
                        ),
                    }
                if markers is None:
                    markers = [{"lat": lat, "lon": lon}]
                result = await loop.run_in_executor(
                    None, self._render_map, lat, lon, zoom, width, height, markers, dest
                )

            elif operation == "bounding_box":
                if any(v is None for v in (south, west, north, east)):
                    return {
                        "success": False,
                        "data": {},
                        "error": "south, west, north, east are all required for 'bounding_box'.",
                    }
                result = await loop.run_in_executor(
                    None,
                    self._render_bbox,
                    south, west, north, east, zoom if zoom != 12 else None, width, height, dest,
                )

            else:
                return {
                    "success": False,
                    "data": {},
                    "error": f"Unknown operation: {operation!r}",
                }

            return {"success": True, "data": result, "error": None}

        except Exception as exc:
            logger.error("MapsTool error: %s", exc, exc_info=True)
            return {"success": False, "data": {}, "error": _restate(exc)}
