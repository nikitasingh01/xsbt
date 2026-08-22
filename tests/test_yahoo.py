from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd
import pytest
import requests

from tests.helpers import FakeResponse, FakeSession
from xsbt.data.base import PRICE_COLUMNS, FetchError, TickerNotFoundError
from xsbt.data.yahoo import YahooFinanceSource, parse_chart_payload


def test_parses_recorded_payload(aapl_payload: dict[str, Any]) -> None:
    frame = parse_chart_payload(aapl_payload, "AAPL")

    assert list(frame.columns) == list(PRICE_COLUMNS)
    assert frame.index.name == "date"
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert frame.index.tz is None
    assert frame.index.is_monotonic_increasing
    assert not frame.index.has_duplicates
    # 2023-01-01 and -02 were a weekend and New Year's Day.
    assert frame.index[0] == pd.Timestamp("2023-01-03")
    assert (frame["adj_close"] > 0).all()
    assert frame.notna().all().all()


def test_adjusted_close_differs_from_close(aapl_payload: dict[str, Any]) -> None:
    # AAPL paid a dividend in Feb 2023, so the two series must not be identical.
    frame = parse_chart_payload(aapl_payload, "AAPL")
    assert not frame["adj_close"].equals(frame["close"])


def test_trading_date_uses_exchange_timezone() -> None:
    """A 10:00 Sydney open is 23:00 UTC the day before. Read as UTC it lands on the
    wrong date, which silently shifts every signal by a day."""
    opens = pd.DatetimeIndex(
        [
            pd.Timestamp("2023-01-03 10:00", tz="Australia/Sydney"),
            pd.Timestamp("2023-01-04 10:00", tz="Australia/Sydney"),
        ]
    )
    payload = _chart_payload(
        [int(ts.timestamp()) for ts in opens],
        closes=[10.0, 11.0],
        tz="Australia/Sydney",
    )

    frame = parse_chart_payload(payload, "BHP.AX")

    assert list(frame.index) == [pd.Timestamp("2023-01-03"), pd.Timestamp("2023-01-04")]


def test_null_bars_are_dropped_not_filled() -> None:
    payload = _chart_payload(
        [_epoch("2023-01-03"), _epoch("2023-01-04"), _epoch("2023-01-05")],
        closes=[10.0, None, 12.0],
    )

    frame = parse_chart_payload(payload, "X")

    assert len(frame) == 2
    assert list(frame["adj_close"]) == [10.0, 12.0]


def test_repeated_final_bar_keeps_last() -> None:
    stamp = _epoch("2023-01-03")
    payload = _chart_payload([stamp, stamp], closes=[10.0, 10.5])

    frame = parse_chart_payload(payload, "X")

    assert len(frame) == 1
    assert frame["adj_close"].iloc[0] == 10.5


def test_valid_symbol_with_no_bars_returns_empty_frame() -> None:
    payload = {
        "chart": {"result": [{"meta": {}, "timestamp": [], "indicators": {}}], "error": None}
    }

    frame = parse_chart_payload(payload, "X")

    assert frame.empty
    assert list(frame.columns) == list(PRICE_COLUMNS)


def test_error_block_raises_fetch_error(not_found_payload: dict[str, Any]) -> None:
    with pytest.raises(FetchError, match="Not Found"):
        parse_chart_payload(not_found_payload, "ZZZZNOTREAL")


def test_an_empty_result_block_is_a_missing_ticker() -> None:
    """No error and no result either. Yahoo does this for some delisted symbols."""
    with pytest.raises(TickerNotFoundError, match="no result block"):
        parse_chart_payload({"chart": {"result": [], "error": None}}, "DEAD")


def test_bars_with_no_quote_block_are_rejected_rather_than_guessed() -> None:
    payload = {
        "chart": {
            "result": [{"meta": {}, "timestamp": [_epoch("2023-01-03")], "indicators": {}}],
            "error": None,
        }
    }

    with pytest.raises(FetchError, match="no quote block"):
        parse_chart_payload(payload, "X")


def test_a_missing_adjclose_block_falls_back_to_close() -> None:
    """Only happens if includeAdjustedClose is dropped, but the frame shape has to hold.

    Falling back to close means no adjustment rather than no data, which the caller can
    at least see; a missing column would break the panel join much further downstream.
    """
    payload = _chart_payload([_epoch("2023-01-03")], closes=[10.0])
    del payload["chart"]["result"][0]["indicators"]["adjclose"]

    frame = parse_chart_payload(payload, "X")

    assert frame["adj_close"].tolist() == [10.0]
    assert frame["adj_close"].equals(frame["close"])


def test_a_permanent_http_error_is_not_retried(no_sleep: None) -> None:
    """403 is a blocked client, not a busy one. Retrying just burns the rate limit."""
    session = FakeSession([FakeResponse(403)])
    source = YahooFinanceSource(session=session)  # type: ignore[arg-type]

    with pytest.raises(FetchError, match="HTTP 403"):
        source.fetch("AAPL", dt.date(2023, 1, 1), dt.date(2023, 3, 31))

    assert len(session.calls) == 1


def test_http_404_raises_ticker_not_found(no_sleep: None) -> None:
    session = FakeSession([FakeResponse(404)])
    source = YahooFinanceSource(session=session)  # type: ignore[arg-type]

    with pytest.raises(TickerNotFoundError):
        source.fetch("ZZZZNOTREAL", dt.date(2023, 1, 1), dt.date(2023, 3, 31))

    # Permanent failure: one attempt, no retry burning the rate limit.
    assert len(session.calls) == 1


def test_retries_transient_status_then_succeeds(
    aapl_payload: dict[str, Any], no_sleep: None
) -> None:
    session = FakeSession([FakeResponse(503), FakeResponse(429), FakeResponse(200, aapl_payload)])
    source = YahooFinanceSource(session=session)  # type: ignore[arg-type]

    frame = source.fetch("AAPL", dt.date(2023, 1, 1), dt.date(2023, 3, 31))

    assert len(session.calls) == 3
    assert not frame.empty


def test_retries_connection_errors(aapl_payload: dict[str, Any], no_sleep: None) -> None:
    session = FakeSession([requests.ConnectionError("reset"), FakeResponse(200, aapl_payload)])
    source = YahooFinanceSource(session=session)  # type: ignore[arg-type]

    assert not source.fetch("AAPL", dt.date(2023, 1, 1), dt.date(2023, 3, 31)).empty


def test_gives_up_after_max_attempts(no_sleep: None) -> None:
    session = FakeSession([FakeResponse(503)] * 4)
    source = YahooFinanceSource(session=session, max_attempts=4)  # type: ignore[arg-type]

    with pytest.raises(FetchError, match="giving up after 4"):
        source.fetch("AAPL", dt.date(2023, 1, 1), dt.date(2023, 3, 31))

    assert len(session.calls) == 4


def test_non_json_response_raises(no_sleep: None) -> None:
    session = FakeSession([FakeResponse(200, payload=None, text="<html>maintenance</html>")])
    source = YahooFinanceSource(session=session)  # type: ignore[arg-type]

    with pytest.raises(FetchError, match="not JSON"):
        source.fetch("AAPL", dt.date(2023, 1, 1), dt.date(2023, 3, 31))


def test_requested_window_is_clipped(aapl_payload: dict[str, Any], no_sleep: None) -> None:
    session = FakeSession([FakeResponse(200, aapl_payload)])
    source = YahooFinanceSource(session=session)  # type: ignore[arg-type]

    frame = source.fetch("AAPL", dt.date(2023, 2, 1), dt.date(2023, 2, 28))

    assert frame.index.min() >= pd.Timestamp("2023-02-01")
    assert frame.index.max() <= pd.Timestamp("2023-02-28")


def test_start_after_end_rejected() -> None:
    source = YahooFinanceSource(session=FakeSession([]))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="after end"):
        source.fetch("AAPL", dt.date(2023, 3, 31), dt.date(2023, 1, 1))


def _epoch(day: str) -> int:
    return int(pd.Timestamp(f"{day} 09:30", tz="America/New_York").timestamp())


def _chart_payload(
    timestamps: list[int],
    closes: list[float | None],
    tz: str = "America/New_York",
) -> dict[str, Any]:
    return {
        "chart": {
            "result": [
                {
                    "meta": {"symbol": "X", "exchangeTimezoneName": tz},
                    "timestamp": timestamps,
                    "indicators": {
                        "quote": [
                            {
                                "open": closes,
                                "high": closes,
                                "low": closes,
                                "close": closes,
                                "volume": [1_000_000] * len(closes),
                            }
                        ],
                        "adjclose": [{"adjclose": closes}],
                    },
                }
            ],
            "error": None,
        }
    }
