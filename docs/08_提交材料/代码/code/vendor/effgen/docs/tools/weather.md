# WeatherTool

Free weather data via the [Open-Meteo API](https://open-meteo.com/) — no API key required.

## Operations

| `operation` | Description |
|---|---|
| `current` | Real-time weather variables (temperature, humidity, wind, conditions). |
| `forecast` | Daily forecast for 1–16 days ahead. |
| `historical` | Historical daily data (requires `start_date` + `end_date`). |

## Parameters

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `operation` | string | No | `"current"` | One of `current`, `forecast`, `historical`. |
| `lat` | float | No* | – | Latitude (-90 to 90). |
| `lon` | float | No* | – | Longitude (-180 to 180). |
| `location` | string | No* | – | Place name resolved via Open-Meteo geocoding. Used when lat/lon are omitted. |
| `days` | integer | No | `7` | Forecast horizon (1–16). Only for `forecast`. |
| `start_date` | string | No | – | `YYYY-MM-DD`. Only for `historical`. |
| `end_date` | string | No | – | `YYYY-MM-DD`. Only for `historical`. |
| `units` | string | No | `"metric"` | `"metric"` (°C, km/h) or `"imperial"` (°F, mph). |

*Either `lat`+`lon` or `location` must be provided.

## Output Schema

```json
{
  "success": true,
  "data": {
    "temperature": 18.3,
    "temperature_unit": "°C",
    "apparent_temperature": 17.1,
    "humidity": 62,
    "precipitation": 0.0,
    "wind_speed": 14.2,
    "wind_unit": "km/h",
    "conditions": "Partly cloudy",
    "weather_code": 2,
    "is_day": true,
    "time": "2025-05-19T14:00",
    "timezone": "Europe/Berlin",
    "coordinates": {"latitude": 52.52, "longitude": 13.405},
    "location": "Berlin, Germany"
  },
  "error": null
}
```

## Minimal Working Example

```python
import asyncio
from effgen.tools.builtin import WeatherTool

async def main():
    tool = WeatherTool()

    # By coordinates
    result = await tool._execute(operation="current", lat=37.42, lon=-122.08)
    print(result["data"]["conditions"])   # e.g. "Clear sky"

    # By name
    result = await tool._execute(operation="forecast", location="Tokyo", days=3)
    for day in result["data"]["forecast"]:
        print(day["date"], day["conditions"])

asyncio.run(main())
```

## Rate Limits / Attribution

Open-Meteo is a free service with generous limits (~10 k requests/day for personal use). No attribution required for non-commercial use; see [open-meteo.com/en/terms](https://open-meteo.com/en/terms) for details.

## Notes

- Results are cached in-memory for 10 minutes (configurable via `cache_ttl`).
- Historical data uses the separate `archive-api.open-meteo.com` endpoint.
- WMO weather codes are translated to human-readable strings automatically.
