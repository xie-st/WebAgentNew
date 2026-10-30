# MapsTool

Render static PNG map images using [OpenStreetMap](https://www.openstreetmap.org/) tiles, powered by the [`staticmap`](https://github.com/komoot/staticmap) Python library.

**No API key required.** Network access is needed to fetch OSM tile images. No binary system dependencies.

## Installation

```bash
pip install staticmap
# or
pip install 'effgen[geo]'
```

## Operations

| `operation` | Description |
|---|---|
| `render` | Render a map centred on a lat/lon point with optional markers. |
| `bounding_box` | Render a map covering a geographic bounding box. |

## Parameters

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `operation` | string | No | `"render"` | `"render"` or `"bounding_box"`. |
| `lat` | float | For `render` | – | Centre latitude. |
| `lon` | float | For `render` | – | Centre longitude. |
| `zoom` | integer | No | `12` | Zoom level (0–19). |
| `width` | integer | No | `800` | Output width in pixels. |
| `height` | integer | No | `600` | Output height in pixels. |
| `markers` | array | No | Centre marker | List of `{lat, lon, color, radius}` dicts. |
| `south` | float | For `bounding_box` | – | South latitude of bounding box. |
| `west` | float | For `bounding_box` | – | West longitude of bounding box. |
| `north` | float | For `bounding_box` | – | North latitude of bounding box. |
| `east` | float | For `bounding_box` | – | East longitude of bounding box. |
| `dest` | string | No | auto temp file | Output PNG file path. |

## Output Schema

```json
{
  "success": true,
  "data": {
    "file": "/tmp/map_abc123.png",
    "width": 800,
    "height": 600,
    "zoom": 14,
    "center": {"lat": 37.422, "lon": -122.084},
    "markers": 1,
    "file_size_bytes": 625400
  },
  "error": null
}
```

## Minimal Working Example

```python
import asyncio
from effgen.tools.builtin import MapsTool

async def main():
    tool = MapsTool()

    # Point map with a marker
    result = await tool._execute(
        operation="render",
        lat=48.8566,
        lon=2.3522,
        zoom=13,
        dest="/tmp/paris.png",
    )
    print(result["data"]["file"])          # /tmp/paris.png
    print(result["data"]["file_size_bytes"])  # ~500000+

    # Bounding box map
    result = await tool._execute(
        operation="bounding_box",
        south=48.82,
        west=2.25,
        north=48.90,
        east=2.42,
        dest="/tmp/paris_box.png",
    )
    print(result["data"]["file"])

asyncio.run(main())
```

## Using with GeocodeTool + WeatherTool

```python
import asyncio
from effgen.tools.builtin import GeocodeTool, MapsTool, WeatherTool

async def main():
    geo = GeocodeTool()
    weather = WeatherTool()
    maps = MapsTool()

    # Geocode a place
    loc = await geo._execute(operation="geocode", address="Kyoto, Japan")
    lat, lon = loc["data"]["lat"], loc["data"]["lon"]

    # Get weather
    wx = await weather._execute(operation="current", lat=lat, lon=lon)
    print(f"{loc['data']['display_name']}: {wx['data']['conditions']}")

    # Render map
    result = await maps._execute(
        operation="render", lat=lat, lon=lon, zoom=12, dest="/tmp/kyoto.png"
    )
    print(f"Map saved: {result['data']['file']}")

asyncio.run(main())
```

## Notes

- If `dest` is omitted, a temporary file is created. The caller is responsible for deleting it.
- Tile fetching requires internet access to `tile.openstreetmap.org`.
- OSM tile usage policy: see [operations.osmfoundation.org/policies/tiles/](https://operations.osmfoundation.org/policies/tiles/).
- For high-volume or offline use, self-host OSM tiles and pass a custom `url_template` by sub-classing `MapsTool`.
