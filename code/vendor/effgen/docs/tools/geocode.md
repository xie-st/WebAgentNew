# GeocodeTool

Forward and reverse geocoding via the [Nominatim](https://nominatim.openstreetmap.org/) API (OpenStreetMap data).

**No API key required.** Rate limited to **1 request per second** per Nominatim's Terms of Service — this is enforced automatically by a built-in token bucket.

## Operations

| `operation` | Description |
|---|---|
| `geocode` | Convert an address / place name → latitude, longitude, address components. |
| `reverse` | Convert latitude + longitude → structured address. |

## Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `operation` | string | Yes | `"geocode"` or `"reverse"`. |
| `address` | string | For `geocode` | Full address or place name. |
| `lat` | float | For `reverse` | Latitude. |
| `lon` | float | For `reverse` | Longitude. |
| `limit` | integer | No (default 1) | Max results for `geocode` (1–10). |

## Output Schema

### geocode

```json
{
  "success": true,
  "data": {
    "query": "Eiffel Tower, Paris",
    "lat": 48.8584,
    "lon": 2.2945,
    "display_name": "Eiffel Tower, 5, Avenue Anatole France, ...",
    "place_type": "attraction",
    "bounding_box": ["48.857", "48.860", "2.291", "2.297"],
    "address": {
      "attraction": "Eiffel Tower",
      "city": "Paris",
      "country": "France"
    },
    "all_results": [...]
  },
  "error": null
}
```

### reverse

```json
{
  "success": true,
  "data": {
    "display_name": "Eiffel Tower, 5, Avenue Anatole France, ...",
    "lat": 48.8584,
    "lon": 2.2945,
    "place_type": "attraction",
    "address": {
      "attraction": "Eiffel Tower",
      "city": "Paris",
      "country": "France"
    }
  },
  "error": null
}
```

## Minimal Working Example

```python
import asyncio
from effgen.tools.builtin import GeocodeTool

async def main():
    tool = GeocodeTool()

    # Forward geocode
    result = await tool._execute(
        operation="geocode",
        address="1600 Amphitheatre Pkwy, Mountain View, CA",
    )
    lat, lon = result["data"]["lat"], result["data"]["lon"]
    print(f"lat={lat:.5f}  lon={lon:.5f}")

    # Reverse geocode
    rev = await tool._execute(operation="reverse", lat=lat, lon=lon)
    print(rev["data"]["display_name"])

asyncio.run(main())
```

## Nominatim ToS Compliance

- **User-Agent**: effGen sets `User-Agent: effGen/<version>` on every request (mandatory).
- **Rate limit**: 1 request per second, enforced by a thread-safe token bucket.
- **No bulk use**: Do not use this tool for large-scale batch geocoding. Contact Nominatim or self-host for high-volume needs.

See [nominatim.org/release-docs/develop/api/Overview/](https://nominatim.org/release-docs/develop/api/Overview/) for full API documentation.
