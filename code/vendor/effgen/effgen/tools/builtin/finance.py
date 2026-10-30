"""
Finance tools for the effGen framework.

Provides stock prices, currency conversion, and cryptocurrency data using
free public APIs that do not require any API keys.

All financial data is for informational purposes only — not financial advice.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode

from ..base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)
from ._net import safe_urlopen

# Fixed API hosts (pinned; the shared SSRF guard also blocks internal targets).
# frankfurter.app currently 301-redirects to frankfurter.dev, so both are pinned
# (every redirect hop is re-validated against this list).
_ALLOWED_HOSTS = frozenset({
    "*.finance.yahoo.com",
    "*.frankfurter.app",
    "*.frankfurter.dev",
    "api.coingecko.com",
})

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "\u26a0\ufe0f Financial data for informational purposes only. Not financial advice."
)


def _user_agent() -> str:
    try:
        from effgen import __version__
    except ImportError:
        __version__ = "dev"
    return f"effGen/{__version__}"


def _fetch_json(url: str, timeout: int = 15, max_retries: int = 4) -> Any:
    """Fetch JSON data from a URL using urllib with exponential backoff for 429/503."""
    import time as _time
    headers = {"User-Agent": _user_agent(), "Accept": "application/json"}
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            with safe_urlopen(
                url, headers=headers, timeout=timeout, allowed_hosts=_ALLOWED_HOSTS
            ) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            if e.code in (429, 503, 502) and attempt < max_retries - 1:
                # Exponential backoff: 1s, 2s, 4s, 8s …
                delay = 2 ** attempt
                logger.warning(
                    "HTTP %d from %s (attempt %d/%d), retrying in %ds",
                    e.code, url, attempt + 1, max_retries, delay,
                )
                _time.sleep(delay)
                last_exc = ConnectionError(f"HTTP {e.code} error from {url}: {e.reason}")
                continue
            raise ConnectionError(f"HTTP {e.code} error from {url}: {e.reason}") from e
        except URLError as e:
            raise ConnectionError(f"Network error fetching {url}: {e.reason}") from e
    assert last_exc is not None
    raise last_exc


class StockPriceTool(BaseTool):
    """
    Fetch current stock price information.

    Primary backend: ``yfinance`` library (if installed).
    Fallback: Yahoo Finance v8 public chart API (free, no key).
    """

    def __init__(self) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="stock_price",
                description=(
                    "Get the current stock price and basic info for a given ticker "
                    "symbol (e.g., AAPL, MSFT, TSLA). Uses free Yahoo Finance data. "
                    + DISCLAIMER
                ),
                category=ToolCategory.EXTERNAL_API,
                parameters=[
                    ParameterSpec(
                        name="symbol",
                        type=ParameterType.STRING,
                        description="Stock ticker symbol (e.g., 'AAPL', 'MSFT')",
                        required=True,
                        min_length=1,
                        max_length=16,
                    ),
                ],
                returns={"type": "object"},
                timeout_seconds=20,
                tags=["finance", "stocks", "free", "yahoo"],
                examples=[
                    {"symbol": "AAPL", "output": {"symbol": "AAPL", "price": 195.12, "currency": "USD"}},
                ],
            )
        )

    def _fetch_yfinance(self, symbol: str) -> dict[str, Any] | None:
        try:
            import yfinance  # type: ignore
        except ImportError:
            return None
        try:
            t = yfinance.Ticker(symbol)
            info = getattr(t, "fast_info", None) or {}
            price = None
            currency = None
            if info:
                price = info.get("last_price") or info.get("lastPrice")
                currency = info.get("currency")
            if price is None:
                hist = t.history(period="1d")
                if not hist.empty:
                    price = float(hist["Close"].iloc[-1])
            return {
                "symbol": symbol.upper(),
                "price": float(price) if price is not None else None,
                "currency": currency or "USD",
                "source": "yfinance",
            }
        except Exception as e:
            logger.warning(f"yfinance failed for {symbol}: {e}")
            return None

    def _fetch_yahoo_api(self, symbol: str) -> dict[str, Any]:
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol)}"
            "?interval=1d&range=1d"
        )
        data = _fetch_json(url)
        chart = (data or {}).get("chart", {})
        if chart.get("error"):
            raise ValueError(f"Yahoo API error: {chart['error']}")
        results = chart.get("result") or []
        if not results:
            raise ValueError(f"No data returned for symbol '{symbol}'")
        meta = results[0].get("meta", {})
        price = meta.get("regularMarketPrice")
        return {
            "symbol": symbol.upper(),
            "price": float(price) if price is not None else None,
            "currency": meta.get("currency", "USD"),
            "previous_close": meta.get("chartPreviousClose"),
            "exchange": meta.get("exchangeName"),
            "market_state": meta.get("marketState"),
            "source": "yahoo_api",
        }

    async def _execute(self, symbol: str, **kwargs) -> dict[str, Any]:
        symbol = symbol.strip().upper()
        result = self._fetch_yfinance(symbol)
        if result is None:
            result = self._fetch_yahoo_api(symbol)
        result["disclaimer"] = DISCLAIMER
        return result


class CurrencyConverterTool(BaseTool):
    """Convert between currencies using the free frankfurter.app API (ECB data)."""

    def __init__(self) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="currency_converter",
                description=(
                    "Convert an amount from one currency to another using live "
                    "European Central Bank exchange rates (frankfurter.app, free, "
                    "no API key). " + DISCLAIMER
                ),
                category=ToolCategory.EXTERNAL_API,
                parameters=[
                    ParameterSpec(
                        name="amount",
                        type=ParameterType.FLOAT,
                        description="Amount to convert",
                        required=True,
                    ),
                    ParameterSpec(
                        name="from_currency",
                        type=ParameterType.STRING,
                        description="Source currency ISO code (e.g., USD)",
                        required=True,
                        min_length=3,
                        max_length=3,
                    ),
                    ParameterSpec(
                        name="to_currency",
                        type=ParameterType.STRING,
                        description="Target currency ISO code (e.g., EUR)",
                        required=True,
                        min_length=3,
                        max_length=3,
                    ),
                ],
                timeout_seconds=15,
                tags=["finance", "currency", "forex", "free"],
                examples=[
                    {
                        "amount": 100,
                        "from_currency": "USD",
                        "to_currency": "EUR",
                        "output": {"converted": 92.15, "rate": 0.9215},
                    },
                ],
            )
        )

    async def _execute(
        self,
        amount: float,
        from_currency: str,
        to_currency: str,
        **kwargs,
    ) -> dict[str, Any]:
        src = from_currency.strip().upper()
        dst = to_currency.strip().upper()
        if src == dst:
            return {
                "amount": amount,
                "from": src,
                "to": dst,
                "rate": 1.0,
                "converted": amount,
                "source": "identity",
                "disclaimer": DISCLAIMER,
            }
        params = urlencode({"amount": amount, "from": src, "to": dst})
        url = f"https://api.frankfurter.app/latest?{params}"
        data = _fetch_json(url)
        rates = data.get("rates") or {}
        if dst not in rates:
            raise ValueError(f"Currency not supported or invalid: {dst}")
        converted = float(rates[dst])
        rate = converted / amount if amount else 0.0
        return {
            "amount": amount,
            "from": src,
            "to": dst,
            "rate": rate,
            "converted": converted,
            "date": data.get("date"),
            "source": "frankfurter.app",
            "disclaimer": DISCLAIMER,
        }


class CryptoTool(BaseTool):
    """Fetch cryptocurrency prices via the free CoinGecko public API."""

    def __init__(self) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="crypto_price",
                description=(
                    "Get the current price of a cryptocurrency (e.g., bitcoin, "
                    "ethereum) in a fiat currency using the free CoinGecko API. "
                    + DISCLAIMER
                ),
                category=ToolCategory.EXTERNAL_API,
                parameters=[
                    ParameterSpec(
                        name="coin",
                        type=ParameterType.STRING,
                        description="CoinGecko coin id (e.g., 'bitcoin', 'ethereum', 'solana')",
                        required=True,
                        min_length=1,
                        max_length=64,
                    ),
                    ParameterSpec(
                        name="vs_currency",
                        type=ParameterType.STRING,
                        description="Target fiat currency code (default: 'usd')",
                        required=False,
                        default="usd",
                        min_length=2,
                        max_length=8,
                    ),
                ],
                timeout_seconds=15,
                tags=["finance", "crypto", "coingecko", "free"],
                examples=[
                    {"coin": "bitcoin", "vs_currency": "usd"},
                ],
            )
        )

    async def _execute(
        self,
        coin: str,
        vs_currency: str = "usd",
        **kwargs,
    ) -> dict[str, Any]:
        coin_id = coin.strip().lower()
        vs = vs_currency.strip().lower()
        params = urlencode({
            "ids": coin_id,
            "vs_currencies": vs,
            "include_24hr_change": "true",
            "include_market_cap": "true",
        })
        url = f"https://api.coingecko.com/api/v3/simple/price?{params}"
        data = _fetch_json(url)
        if coin_id not in data:
            raise ValueError(f"Unknown coin id '{coin_id}'. Use a valid CoinGecko id.")
        info = data[coin_id]
        price = info.get(vs)
        if price is None:
            raise ValueError(f"Currency '{vs}' not supported for {coin_id}")
        return {
            "coin": coin_id,
            "vs_currency": vs,
            "price": float(price),
            "market_cap": info.get(f"{vs}_market_cap"),
            "change_24h_pct": info.get(f"{vs}_24h_change"),
            "source": "coingecko",
            "disclaimer": DISCLAIMER,
        }
