"""
Data science tools for the effGen framework.

Provides DataFrame manipulation (pandas), plotting (matplotlib),
and basic statistics (numpy/scipy). Heavy dependencies are optional —
tools report a clear error if the required library is missing.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any

from ..base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)
from ._fs import check_content_not_credentials, confine_path, normalize_allowed_dirs

logger = logging.getLogger(__name__)


def _require_pandas():
    try:
        import pandas as pd  # type: ignore
        return pd
    except ImportError as e:
        raise ImportError(
            "pandas is not installed. Install with: pip install pandas"
        ) from e


class DataFrameTool(BaseTool):
    """
    Load and manipulate tabular data using pandas.

    Supported operations:
      * ``head`` — return first N rows (default 5)
      * ``describe`` — summary statistics
      * ``info`` — column names, types, non-null counts
      * ``filter`` — rows where ``column <op> value``
      * ``aggregate`` — group by column, compute aggregation
    """

    def __init__(self, allowed_directories: list[str] | None = None) -> None:
        """Args:
            allowed_directories: Roots a data file may be read from. By default
                any path is allowed except protected system and credential
                locations (/etc, /proc, ~/.ssh, cloud creds, …), which are
                always refused. Pass a list to confine reads to those roots only.
        """
        self._allowed_dirs = normalize_allowed_dirs(allowed_directories)
        super().__init__(
            metadata=ToolMetadata(
                name="dataframe",
                description=(
                    "Load a CSV or JSON file and perform DataFrame operations: "
                    "head, describe, info, filter, aggregate. Requires pandas."
                ),
                category=ToolCategory.DATA_PROCESSING,
                parameters=[
                    ParameterSpec(
                        name="file_path",
                        type=ParameterType.STRING,
                        description="Path to CSV or JSON file",
                        required=True,
                    ),
                    ParameterSpec(
                        name="operation",
                        type=ParameterType.STRING,
                        description="Operation: head, describe, info, filter, aggregate",
                        required=False,
                        default="head",
                        enum=["head", "describe", "info", "filter", "aggregate"],
                    ),
                    ParameterSpec(
                        name="n",
                        type=ParameterType.INTEGER,
                        description="Number of rows (for head)",
                        required=False,
                        default=5,
                        min_value=1,
                        max_value=1000,
                    ),
                    ParameterSpec(
                        name="column",
                        type=ParameterType.STRING,
                        description="Column name (for filter/aggregate)",
                        required=False,
                    ),
                    ParameterSpec(
                        name="op",
                        type=ParameterType.STRING,
                        description="Filter op: ==, !=, <, <=, >, >=",
                        required=False,
                        enum=["==", "!=", "<", "<=", ">", ">="],
                    ),
                    ParameterSpec(
                        name="value",
                        type=ParameterType.ANY,
                        description="Filter value",
                        required=False,
                    ),
                    ParameterSpec(
                        name="agg",
                        type=ParameterType.STRING,
                        description="Aggregation: mean, sum, count, min, max",
                        required=False,
                        enum=["mean", "sum", "count", "min", "max"],
                    ),
                    ParameterSpec(
                        name="group_by",
                        type=ParameterType.STRING,
                        description="Column to group by (for aggregate)",
                        required=False,
                    ),
                ],
                timeout_seconds=30,
                tags=["data", "pandas", "dataframe"],
                examples=[
                    {"file_path": "data.csv", "operation": "head", "n": 5},
                ],
            )
        )

    def _load(self, file_path: str):
        pd = _require_pandas()
        safe_path = str(confine_path(file_path, self._allowed_dirs))
        # A credentials file renamed to .csv/.json still parses as a
        # (mostly nonsensical) one-column table under a naive reader, so the
        # extension alone does not prove this is safe to load — sniff the raw
        # text first, regardless of what a plain-text CSV/JSON reader can
        # technically get through.
        try:
            with open(safe_path, encoding="utf-8", errors="replace") as fh:
                head = fh.read(65536)
        except OSError as e:
            raise ValueError(f"Could not read '{file_path}': {e}")
        check_content_not_credentials(head, source=file_path)
        lower = safe_path.lower()
        if lower.endswith(".csv"):
            return pd.read_csv(safe_path)
        if lower.endswith(".json") or lower.endswith(".jsonl"):
            return pd.read_json(safe_path, lines=lower.endswith(".jsonl"))
        raise ValueError(f"Unsupported file format: {file_path}")

    async def _execute(
        self,
        file_path: str,
        operation: str = "head",
        n: int = 5,
        column: str | None = None,
        op: str | None = None,
        value: Any = None,
        agg: str | None = None,
        group_by: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        df = self._load(file_path)
        result: dict[str, Any] = {"operation": operation, "rows": len(df), "columns": list(df.columns)}

        if operation == "head":
            result["data"] = df.head(n).to_dict(orient="records")
        elif operation == "describe":
            result["data"] = json.loads(df.describe(include="all").to_json())
        elif operation == "info":
            result["data"] = {
                "dtypes": {c: str(t) for c, t in df.dtypes.items()},
                "non_null": {c: int(df[c].notna().sum()) for c in df.columns},
            }
        elif operation == "filter":
            if not column or not op:
                raise ValueError("filter requires 'column' and 'op'")
            ops = {
                "==": lambda a, b: a == b,
                "!=": lambda a, b: a != b,
                "<": lambda a, b: a < b,
                "<=": lambda a, b: a <= b,
                ">": lambda a, b: a > b,
                ">=": lambda a, b: a >= b,
            }
            filtered = df[ops[op](df[column], value)]
            result["data"] = filtered.head(50).to_dict(orient="records")
            result["matched_rows"] = len(filtered)
        elif operation == "aggregate":
            if not agg:
                raise ValueError("aggregate requires 'agg'")
            if group_by:
                grouped = df.groupby(group_by).agg(agg, numeric_only=True)
                result["data"] = json.loads(grouped.to_json())
            else:
                s = df.agg(agg, numeric_only=True)
                result["data"] = {k: (float(v) if v == v else None) for k, v in s.to_dict().items()}
        else:
            raise ValueError(f"Unknown operation: {operation}")

        return result


class PlotTool(BaseTool):
    """Generate simple charts using matplotlib and save to a temp file."""

    def __init__(self) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="plot",
                description=(
                    "Generate a chart (line, bar, scatter, histogram) from data and "
                    "save it as a PNG image. Returns the file path. Requires matplotlib."
                ),
                category=ToolCategory.DATA_PROCESSING,
                parameters=[
                    ParameterSpec(
                        name="chart_type",
                        type=ParameterType.STRING,
                        description="Chart type",
                        required=True,
                        enum=["line", "bar", "scatter", "hist"],
                    ),
                    ParameterSpec(
                        name="x",
                        type=ParameterType.ARRAY,
                        description="X values (omit for hist)",
                        required=False,
                    ),
                    ParameterSpec(
                        name="y",
                        type=ParameterType.ARRAY,
                        description="Y values (or values for hist)",
                        required=True,
                    ),
                    ParameterSpec(
                        name="title",
                        type=ParameterType.STRING,
                        description="Chart title",
                        required=False,
                        default="",
                    ),
                    ParameterSpec(
                        name="xlabel",
                        type=ParameterType.STRING,
                        description="X-axis label",
                        required=False,
                        default="",
                    ),
                    ParameterSpec(
                        name="ylabel",
                        type=ParameterType.STRING,
                        description="Y-axis label",
                        required=False,
                        default="",
                    ),
                    ParameterSpec(
                        name="output_path",
                        type=ParameterType.STRING,
                        description="Output file path (default: temp file)",
                        required=False,
                    ),
                ],
                timeout_seconds=30,
                tags=["data", "plot", "matplotlib", "chart"],
                examples=[
                    {"chart_type": "line", "x": [1, 2, 3], "y": [4, 5, 6], "title": "Demo"},
                ],
            )
        )

    async def _execute(
        self,
        chart_type: str,
        y: list,
        x: list | None = None,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        output_path: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        try:
            import matplotlib  # type: ignore
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # type: ignore
        except ImportError as e:
            raise ImportError(
                "matplotlib is not installed. Install with: pip install matplotlib"
            ) from e

        fig, ax = plt.subplots(figsize=(8, 5))
        if chart_type == "line":
            ax.plot(x if x is not None else list(range(len(y))), y)
        elif chart_type == "bar":
            ax.bar(x if x is not None else list(range(len(y))), y)
        elif chart_type == "scatter":
            if x is None:
                raise ValueError("scatter requires 'x'")
            ax.scatter(x, y)
        elif chart_type == "hist":
            ax.hist(y, bins=20)
        else:
            plt.close(fig)
            raise ValueError(f"Unknown chart_type: {chart_type}")

        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        fig.tight_layout()

        if output_path is None:
            fd, output_path = tempfile.mkstemp(prefix="effgen_plot_", suffix=".png")
            os.close(fd)
        fig.savefig(output_path, dpi=100)
        plt.close(fig)

        return {
            "chart_type": chart_type,
            "file_path": output_path,
            "data_points": len(y),
        }


class StatsTool(BaseTool):
    """Compute basic statistics: mean, median, std, correlation, regression."""

    def __init__(self) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="stats",
                description=(
                    "Compute statistics from a list of numbers: mean, median, std, "
                    "variance, min, max, correlation (requires y), or linear regression."
                ),
                category=ToolCategory.COMPUTATION,
                parameters=[
                    ParameterSpec(
                        name="operation",
                        type=ParameterType.STRING,
                        description="Statistic to compute",
                        required=True,
                        enum=[
                            "mean", "median", "std", "variance",
                            "min", "max", "summary", "correlation", "regression",
                        ],
                    ),
                    ParameterSpec(
                        name="data",
                        type=ParameterType.ARRAY,
                        description="List of numeric values (x values)",
                        required=True,
                    ),
                    ParameterSpec(
                        name="y",
                        type=ParameterType.ARRAY,
                        description="Second list (for correlation/regression)",
                        required=False,
                    ),
                ],
                timeout_seconds=10,
                tags=["data", "statistics", "numpy"],
                examples=[
                    {"operation": "mean", "data": [1, 2, 3, 4, 5]},
                ],
            )
        )

    async def _execute(
        self,
        operation: str,
        data: list,
        y: list | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        try:
            import numpy as np  # type: ignore
        except ImportError as e:
            raise ImportError(
                "numpy is not installed. Install with: pip install numpy"
            ) from e

        arr = np.asarray(data, dtype=float)
        if arr.size == 0:
            raise ValueError("data must not be empty")

        if operation == "mean":
            return {"operation": "mean", "result": float(np.mean(arr))}
        if operation == "median":
            return {"operation": "median", "result": float(np.median(arr))}
        if operation == "std":
            return {"operation": "std", "result": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0}
        if operation == "variance":
            return {"operation": "variance", "result": float(np.var(arr, ddof=1)) if arr.size > 1 else 0.0}
        if operation == "min":
            return {"operation": "min", "result": float(np.min(arr))}
        if operation == "max":
            return {"operation": "max", "result": float(np.max(arr))}
        if operation == "summary":
            return {
                "operation": "summary",
                "n": int(arr.size),
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }

        # Bivariate ops
        if y is None:
            raise ValueError(f"{operation} requires 'y'")
        yarr = np.asarray(y, dtype=float)
        if yarr.shape != arr.shape:
            raise ValueError("data and y must have the same length")

        if operation == "correlation":
            if arr.size < 2:
                raise ValueError("need at least 2 points for correlation")
            corr = float(np.corrcoef(arr, yarr)[0, 1])
            return {"operation": "correlation", "result": corr, "n": int(arr.size)}

        if operation == "regression":
            if arr.size < 2:
                raise ValueError("need at least 2 points for regression")
            # Prefer scipy if available for extra stats
            try:
                from scipy import stats as _sst  # type: ignore
                res = _sst.linregress(arr, yarr)
                return {
                    "operation": "regression",
                    "slope": float(res.slope),
                    "intercept": float(res.intercept),
                    "r_value": float(res.rvalue),
                    "p_value": float(res.pvalue),
                    "stderr": float(res.stderr),
                    "n": int(arr.size),
                }
            except ImportError:
                slope, intercept = np.polyfit(arr, yarr, 1)
                pred = slope * arr + intercept
                ss_res = float(np.sum((yarr - pred) ** 2))
                ss_tot = float(np.sum((yarr - np.mean(yarr)) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
                return {
                    "operation": "regression",
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "r_squared": r2,
                    "n": int(arr.size),
                }

        raise ValueError(f"Unknown operation: {operation}")
