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

from xsbt.data.base import (
    BAR_COLUMNS,
    DATE_INDEX_NAME,
    FetchError,
    TickerNotFoundError,
)

log = logging.getLogger(__name__)

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"

# A default python-requests UA gets 429'd almost immediately.
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

TOO_MANY_REQUESTS = 429
RETRYABLE_STATUS = frozenset({TOO_MANY_REQUESTS, 500, 502, 503, 504})

# Ceiling on any single wait, so a server asking us back in an hour fails the run
# quickly instead of parking the process.
MAX_BACKOFF_SECONDS = 60.0


def retry_after_seconds(response: Any) -> float | None:
    """``Retry-After`` in its delta-seconds form, if the server sent one.

    Yahoo does not, but the CDNs and WAFs sitting in front of it do, and a server telling
    us when to come back beats guessing at it. The HTTP-date form is ignored on purpose:
    parsing it means trusting a clock skew we cannot measure.
    """
    raw = getattr(response, "headers", {}).get("Retry-After")
    if raw is None:
        return None
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return None


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
    index = _to_session_dates(timestamps, tz_name)

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

    events = result.get("events") or {}
    frame["dividend"] = _event_series(events.get("dividends"), "amount", tz_name, frame.index)
    frame["split_ratio"] = _event_series(events.get("splits"), "numerator", tz_name, frame.index)
    return frame[list(BAR_COLUMNS)]


def _to_session_dates(timestamps: Any, tz_name: str) -> pd.DatetimeIndex:
    """Epoch seconds at the exchange open, reduced to the local calendar date."""
    stamps = pd.to_datetime(pd.Series(timestamps, dtype="int64"), unit="s", utc=True)
    return pd.DatetimeIndex(stamps.dt.tz_convert(tz_name).dt.normalize().dt.tz_localize(None))


def _event_series(
    events: Any,
    field: str,
    tz_name: str,
    index: pd.DatetimeIndex,
) -> pd.Series:
    """One corporate action field aligned onto the bar index, NaN where nothing happened.

    Yahoo keys these by epoch second rather than by date, and the key is not always the
    same second as the bar it belongs to, so the timestamp inside each record is what gets
    converted. Actions that land outside the returned bars are dropped here; the
    adjustment audit notices them by a different route.
    """
    blank = pd.Series(float("nan"), index=index, dtype="float64")
    if not events:
        return blank

    records = list(events.values())
    dates = _to_session_dates([record["date"] for record in records], tz_name)
    values = pd.Series(
        [float(record[field]) for record in records], index=dates, dtype="float64"
    ).sort_index()
    # Two actions on one session would be a vendor error; keep the last and move on.
    values = values[~values.index.duplicated(keep="last")]
    return values.reindex(index)


def empty_frame() -> pd.DataFrame:
    """Correctly typed empty bar frame."""
    return pd.DataFrame(
        {col: pd.Series(dtype="float64") for col in BAR_COLUMNS},
        index=pd.DatetimeIndex([], name=DATE_INDEX_NAME),
    )


class YahooFinanceSource:
    """Daily bars from Yahoo, one symbol per request.

    Backoff is ``backoff_base * 2 ** (attempt - 1)``, no jitter: one process pulling a
    few dozen symbols has no herd to spread out, and it keeps the retry path testable.

    The request spacing is adaptive rather than fixed. It starts at ``min_interval`` and
    doubles towards ``max_interval`` each time Yahoo returns a 429, because a rate limit
    is a fact about the session and not about the symbol that happened to trip it.
    """

    name = "yahoo"

    def __init__(
        self,
        session: requests.Session | None = None,
        *,
        max_attempts: int = 5,
        backoff_base: float = 0.5,
        min_interval: float = 0.2,
        max_interval: float = 5.0,
        timeout: float = 20.0,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        self._session = session or requests.Session()
        # Assignment, not setdefault: requests.Session ships its own User-Agent, so
        # setdefault silently keeps python-requests and Yahoo 429s every call.
        self._session.headers["User-Agent"] = user_agent
        self._max_attempts = max_attempts
        self._backoff_base = backoff_base
        self._interval = min_interval
        self._max_interval = max(max_interval, min_interval)
        self._timeout = timeout
        self._last_request_at = 0.0

    @property
    def interval(self) -> float:
        """Current spacing between requests. Widens on rate limiting, never narrows."""
        return self._interval

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
            wait = self._backoff_base * 2 ** (attempt - 1)
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
                    if response.status_code == TOO_MANY_REQUESTS:
                        self._widen_interval(ticker)
                    # A server that tells us when to come back outranks our own guess.
                    wait = retry_after_seconds(response) or wait
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
                time.sleep(min(wait, MAX_BACKOFF_SECONDS))

        raise FetchError(f"{ticker}: giving up after {self._max_attempts} attempts") from last_error

    def _widen_interval(self, ticker: str) -> None:
        """Slow the whole session down after a rate limit, not just this one symbol.

        Retrying the failed request with backoff and then going straight back to full
        speed for the next symbol is how a single 429 turns into forty of them: the
        limiter is counting requests per session, so the next symbol walks into the same
        wall. Doubling the floor is what actually stops the cascade, and it holds for the
        rest of the run because the limit does not go away just because one call got
        through.
        """
        if self._interval >= self._max_interval:
            return
        self._interval = min(self._interval * 2, self._max_interval)
        log.warning("%s: rate limited, slowing to %.2fs between requests", ticker, self._interval)

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self._interval:
            time.sleep(self._interval - elapsed)
        self._last_request_at = time.monotonic()


def to_epoch(day: dt.date) -> int:
    return int(dt.datetime(day.year, day.month, day.day, tzinfo=dt.UTC).timestamp())
