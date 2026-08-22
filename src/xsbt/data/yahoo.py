"""Yahoo Finance daily bars via the public v8 chart endpoint.

Hand-rolled rather than yfinance so we own the retry policy, rate limit and failure
taxonomy. The parser is a plain function, so tests run off recorded JSON with no network.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from typing import Any

import pandas as pd
import requests

from xsbt.data.base import DATE_INDEX_NAME, PRICE_COLUMNS, FetchError, TickerNotFoundError

log = logging.getLogger(__name__)

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"

# A default python-requests UA gets 429'd almost immediately.
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


def parse_chart_payload(payload: dict[str, Any], ticker: str) -> pd.DataFrame:
    """Turn a chart response into a daily bar frame indexed by trading date."""
    chart = payload.get("chart") or {}
    error = chart.get("error")
    if error:
        raise FetchError(f"{ticker}: yahoo returned error {error}")

    results = chart.get("result")
    if not results:
        raise TickerNotFoundError(f"{ticker}: no result block in payload")

    result = results[0]
    timestamps = result.get("timestamp")
    if not timestamps:
        # Valid symbol, no bars in the window.
        return empty_frame()

    meta = result.get("meta", {})
    tz_name = meta.get("exchangeTimezoneName") or "UTC"

    indicators = result.get("indicators", {})
    quote_blocks = indicators.get("quote")
    if not quote_blocks:
        raise FetchError(f"{ticker}: timestamps present but no quote block")
    quote = quote_blocks[0]

    # adjclose is missing if the caller didn't ask for it; fall back so the shape holds.
    adj_blocks = indicators.get("adjclose") or [{}]
    adj_close = adj_blocks[0].get("adjclose", quote.get("close"))

    # Timestamps are the bar's open in exchange local time. Read as UTC, a 19:00 Sydney
    # open lands on the next calendar day, so convert before taking the date.
    index = (
        pd.to_datetime(pd.Series(timestamps), unit="s", utc=True)
        .dt.tz_convert(tz_name)
        .dt.normalize()
        .dt.tz_localize(None)
    )

    frame = pd.DataFrame(
        {
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "adj_close": adj_close,
            "volume": quote.get("volume"),
        },
        index=pd.DatetimeIndex(index, name=DATE_INDEX_NAME),
    ).astype("float64")

    # Nulls come back for halts and untraded sessions. Drop them rather than fill:
    # making up prices isn't the data layer's job.
    frame = frame.dropna(subset=["adj_close"])
    # The live bar sometimes repeats the previous one.
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame[list(PRICE_COLUMNS)]


def empty_frame() -> pd.DataFrame:
    """Correctly typed empty bar frame."""
    return pd.DataFrame(
        {col: pd.Series(dtype="float64") for col in PRICE_COLUMNS},
        index=pd.DatetimeIndex([], name=DATE_INDEX_NAME),
    )


class YahooFinanceSource:
    """Daily bars from Yahoo, one symbol per request.

    Backoff is ``backoff_base * 2 ** (attempt - 1)``, no jitter: one process pulling a
    few dozen symbols has no herd to spread out, and it keeps the retry path testable.
    A fixed ``min_interval`` between requests keeps us under the rate limit to begin with.
    """

    name = "yahoo"

    def __init__(
        self,
        session: requests.Session | None = None,
        *,
        max_attempts: int = 4,
        backoff_base: float = 0.5,
        min_interval: float = 0.2,
        timeout: float = 20.0,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        self._session = session or requests.Session()
        self._session.headers.setdefault("User-Agent", user_agent)
        self._max_attempts = max_attempts
        self._backoff_base = backoff_base
        self._min_interval = min_interval
        self._timeout = timeout
        self._last_request_at = 0.0

    def fetch(self, ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        if start > end:
            raise ValueError(f"start {start} is after end {end}")

        params = {
            "period1": to_epoch(start),
            # period2 is exclusive, so push past the end of the last day we want.
            "period2": to_epoch(end + dt.timedelta(days=1)),
            "interval": "1d",
            "events": "div,split",
            "includeAdjustedClose": "true",
        }
        frame = parse_chart_payload(self._get(ticker, params), ticker)
        # Yahoo rounds the window out to whole sessions; clip back.
        return frame.loc[str(start) : str(end)]

    def _get(self, ticker: str, params: dict[str, Any]) -> dict[str, Any]:
        url = CHART_URL.format(ticker=ticker)
        last_error: Exception | None = None

        for attempt in range(1, self._max_attempts + 1):
            self._throttle()
            try:
                response = self._session.get(url, params=params, timeout=self._timeout)
            except requests.RequestException as exc:
                last_error = exc
                log.warning("%s: request failed (attempt %d): %s", ticker, attempt, exc)
            else:
                if response.status_code == 404:
                    raise TickerNotFoundError(f"{ticker}: not found at yahoo (404)")
                if response.status_code in RETRYABLE_STATUS:
                    last_error = FetchError(f"{ticker}: HTTP {response.status_code}")
                    log.warning(
                        "%s: HTTP %d (attempt %d/%d)",
                        ticker,
                        response.status_code,
                        attempt,
                        self._max_attempts,
                    )
                elif not response.ok:
                    raise FetchError(f"{ticker}: HTTP {response.status_code}")
                else:
                    try:
                        payload: dict[str, Any] = response.json()
                    except ValueError as exc:
                        raise FetchError(f"{ticker}: response was not JSON") from exc
                    return payload

            if attempt < self._max_attempts:
                time.sleep(self._backoff_base * 2 ** (attempt - 1))

        raise FetchError(f"{ticker}: giving up after {self._max_attempts} attempts") from last_error

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_at = time.monotonic()


def to_epoch(day: dt.date) -> int:
    return int(dt.datetime(day.year, day.month, day.day, tzinfo=dt.UTC).timestamp())
