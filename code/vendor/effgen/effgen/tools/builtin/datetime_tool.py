"""
DateTime tool for date/time operations.

Provides current date/time, timezone conversions, date arithmetic,
and formatting — all using the Python standard library.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from ..base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)

logger = logging.getLogger(__name__)

# Common timezone offsets (UTC offset in hours)
TIMEZONE_OFFSETS = {
    "UTC": 0, "GMT": 0,
    "EST": -5, "EDT": -4, "CST": -6, "CDT": -5,
    "MST": -7, "MDT": -6, "PST": -8, "PDT": -7,
    "IST": 5.5, "JST": 9, "KST": 9, "CST_CN": 8,
    "AEST": 10, "AEDT": 11, "NZST": 12, "NZDT": 13,
    "CET": 1, "CEST": 2, "EET": 2, "EEST": 3,
    "BST": 1, "WET": 0, "WEST": 1,
    "HKT": 8, "SGT": 8, "BRT": -3, "ART": -3,
}


class DateTimeTool(BaseTool):
    """
    Date and time operations tool.

    Features:
    - Get current date/time in any timezone
    - Date arithmetic (add/subtract days, hours, etc.)
    - Date formatting
    - Day of week, week number
    - Time between two dates
    - No external dependencies
    """

    # Natural-name aliases for the ``operation`` selector.
    operation_aliases = {
        "current_time": "now",
        "current": "now",
        "current_date": "now",
        "today": "now",
        "time": "now",
        "timestamp": "now",
        "parse": "format",
        "strftime": "format",
        "difference": "diff",
        "delta": "diff",
        "between": "diff",
        "convert_timezone": "convert",
        "to_timezone": "convert",
        "tz": "convert",
    }

    def __init__(self) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="datetime",
                description=(
                    "Get current date/time, convert timezones, perform date arithmetic, "
                    "and format dates. No API key needed."
                ),
                category=ToolCategory.COMPUTATION,
                parameters=[
                    ParameterSpec(
                        name="operation",
                        type=ParameterType.STRING,
                        description="Operation: 'now', 'format', 'add', 'diff', 'convert'",
                        required=True,
                        enum=["now", "format", "add", "diff", "convert"],
                    ),
                    ParameterSpec(
                        name="timezone",
                        type=ParameterType.STRING,
                        description="Timezone abbreviation (e.g., 'UTC', 'EST', 'JST', 'IST')",
                        required=False,
                        default="UTC",
                    ),
                    ParameterSpec(
                        name="date",
                        type=ParameterType.STRING,
                        description="Date string (ISO format: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)",
                        required=False,
                    ),
                    ParameterSpec(
                        name="date2",
                        type=ParameterType.STRING,
                        description="Second date for 'diff' operation",
                        required=False,
                    ),
                    ParameterSpec(
                        name="days",
                        type=ParameterType.INTEGER,
                        description="Days to add (for 'add' operation, can be negative)",
                        required=False,
                        default=0,
                    ),
                    ParameterSpec(
                        name="hours",
                        type=ParameterType.INTEGER,
                        description="Hours to add (for 'add' operation, can be negative)",
                        required=False,
                        default=0,
                    ),
                    ParameterSpec(
                        name="to_timezone",
                        type=ParameterType.STRING,
                        description="Target timezone for 'convert' operation",
                        required=False,
                    ),
                ],
                returns={
                    "type": "object",
                    "properties": {
                        "datetime": {"type": "string"},
                        "date": {"type": "string"},
                        "time": {"type": "string"},
                        "timezone": {"type": "string"},
                    },
                },
                timeout_seconds=5,
                tags=["datetime", "time", "date", "timezone", "computation"],
                examples=[
                    {
                        "operation": "now",
                        "timezone": "UTC",
                        "output": {"datetime": "2026-03-01 12:00:00 UTC", "day_of_week": "Sunday"},
                    },
                    {
                        "operation": "add",
                        "date": "2026-03-01",
                        "days": 7,
                        "output": {"datetime": "2026-03-08 00:00:00 UTC"},
                    },
                ],
            )
        )

    def _get_tz(self, tz_name: str) -> timezone:
        """Get timezone object from name."""
        tz_upper = tz_name.upper().strip()
        offset_hours = TIMEZONE_OFFSETS.get(tz_upper)
        if offset_hours is None:
            # Try parsing as UTC+N or UTC-N
            import re
            m = re.match(r'UTC([+-])(\d+(?:\.\d+)?)', tz_upper)
            if m:
                sign = 1 if m.group(1) == '+' else -1
                offset_hours = sign * float(m.group(2))
            else:
                offset_hours = 0
                logger.warning(f"Unknown timezone '{tz_name}', defaulting to UTC")
        hours = int(offset_hours)
        minutes = int((offset_hours - hours) * 60)
        return timezone(timedelta(hours=hours, minutes=minutes))

    def _parse_date(self, date_str: str) -> datetime:
        """Parse a date string."""
        for fmt in [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
        ]:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        raise ValueError(f"Cannot parse date: '{date_str}'. Use YYYY-MM-DD format.")

    async def _execute(
        self,
        operation: str,
        timezone_param: str | None = None,
        date: str | None = None,
        date2: str | None = None,
        days: int = 0,
        hours: int = 0,
        to_timezone: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Execute datetime operation."""
        # Handle 'timezone' parameter name collision
        tz_name = kwargs.get("timezone", timezone_param) or "UTC"
        tz = self._get_tz(tz_name)

        if operation == "now":
            # If a date is provided, return info about that date instead of "now"
            if date:
                dt = self._parse_date(date).replace(tzinfo=tz)
                return {
                    "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "date": dt.strftime("%Y-%m-%d"),
                    "time": dt.strftime("%H:%M:%S"),
                    "timezone": tz_name.upper(),
                    "day_of_week": dt.strftime("%A"),
                    "week_number": dt.isocalendar()[1],
                    "timestamp": int(dt.timestamp()),
                }
            now = datetime.now(tz)
            return {
                "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "timezone": tz_name.upper(),
                "day_of_week": now.strftime("%A"),
                "week_number": now.isocalendar()[1],
                "timestamp": int(now.timestamp()),
            }

        elif operation == "format":
            if not date:
                raise ValueError("'date' parameter required for format operation")
            dt = self._parse_date(date).replace(tzinfo=tz)
            return {
                "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "date": dt.strftime("%Y-%m-%d"),
                "time": dt.strftime("%H:%M:%S"),
                "day_of_week": dt.strftime("%A"),
                "iso": dt.isoformat(),
                "timezone": tz_name.upper(),
            }

        elif operation == "add":
            if date:
                dt = self._parse_date(date).replace(tzinfo=tz)
            else:
                dt = datetime.now(tz)
            result = dt + timedelta(days=days, hours=hours)
            return {
                "original": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "result": result.strftime("%Y-%m-%d %H:%M:%S"),
                "date": result.strftime("%Y-%m-%d"),
                "day_of_week": result.strftime("%A"),
                "timezone": tz_name.upper(),
                "added": f"{days} days, {hours} hours",
            }

        elif operation == "diff":
            if not date or not date2:
                raise ValueError("Both 'date' and 'date2' required for diff operation")
            dt1 = self._parse_date(date)
            dt2 = self._parse_date(date2)
            diff = dt2 - dt1
            return {
                "date1": dt1.strftime("%Y-%m-%d %H:%M:%S"),
                "date2": dt2.strftime("%Y-%m-%d %H:%M:%S"),
                "days": diff.days,
                "total_seconds": int(diff.total_seconds()),
                "total_hours": round(diff.total_seconds() / 3600, 2),
            }

        elif operation == "convert":
            if not date and not to_timezone:
                raise ValueError("'date' and 'to_timezone' required for convert operation")
            if date:
                dt = self._parse_date(date).replace(tzinfo=tz)
            else:
                dt = datetime.now(tz)
            target_tz = self._get_tz(to_timezone or "UTC")
            converted = dt.astimezone(target_tz)
            return {
                "original": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "original_timezone": tz_name.upper(),
                "converted": converted.strftime("%Y-%m-%d %H:%M:%S"),
                "target_timezone": (to_timezone or "UTC").upper(),
            }

        raise ValueError(f"Unknown operation: {operation}")
