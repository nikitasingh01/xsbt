from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd
import pytest
import requests

from tests.helpers import FakeResponse, FakeSession
from xsbt.data import yahoo
from xsbt.data.base import BAR_COLUMNS, PRICE_COLUMNS, FetchError, TickerNotFoundError
from xsbt.data.yahoo import YahooFinanceSource, parse_chart_payload


@pytest.fixture
def slept(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Records what the retry path would have waited instead of waiting it.

    Pair with ``min_interval=0.0`` so the inter-request throttle stays out of the list
    and what is left is purely the backoff decision under test.
    """
    waits: list[float] = []
    monkeypatch.setattr(yahoo.time, "sleep", waits.append)
    return waits


def test_parses_recorded_payload(aapl_payload: dict[str, Any]) -> None:
    frame = parse_chart_payload(aapl_payload, "AAPL")

    assert list(frame.columns) == list(BAR_COLUMNS)
    assert frame.index.name == "date"
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert frame.index.tz is None
    assert frame.index.is_monotonic_increasing
    assert not frame.index.has_duplicates
    # 2023-01-01 and -02 were a weekend and New Year's Day.
    assert frame.index[0] == pd.Timestamp("2023-01-03")
    assert (frame["adj_close"] > 0).all()
    assert frame[list(PRICE_COLUMNS)].notna().all().all()


def test_dividend_lands_on_its_own_session(aapl_payload: dict[str, Any]) -> None:
    """AAPL went ex on 2023-02-10 for 23c.

    Yahoo keys events by epoch second, and the key is not always the same second as the
    bar, so this pins the conversion rather than the lookup.
    """
    frame = parse_chart_payload(aapl_payload, "AAPL")

    paid = frame["dividend"].dropna()
    assert paid.index.tolist() == [pd.Timestamp("2023-02-10")]
    assert paid.iloc[0] == pytest.approx(0.23)
    # Nothing split in the quarter, so that column stays empty.
    assert frame["split_ratio"].isna().all()


def test_a_payload_with_no_events_block_still_parses() -> None:
    """Most quarters have no action at all. The columns still have to be there."""
    payload = _chart_payload([_epoch("2023-01-03"), _epoch("2023-01-04")], closes=[10.0, 11.0])
    assert "events" not in payload["chart"]["result"][0]

    frame = parse_chart_payload(payload, "X")

    assert frame["dividend"].isna().all()
    assert frame["split_ratio"].isna().all()


def test_an_action_dated_outside_the_bars_is_dropped() -> None:
    """The window is clipped after the fact, so events can fall off either end.

    Reindexing keeps them out. Left in, they would land as extra rows with no prices.
    """
    payload = _chart_payload([_epoch("2023-01-04")], closes=[10.0])
    payload["chart"]["result"][0]["events"] = {
        "dividends": {"1": {"amount": 0.5, "date": _epoch("2022-12-15")}}
    }

    frame = parse_chart_payload(payload, "X")

    assert len(frame) == 1
    assert frame["dividend"].isna().all()


def test_a_split_is_carried_through_as_its_numerator() -> None:
    """4:1 is stored as 4.0. The denominator is 1 for every split we have ever seen."""
    payload = _chart_payload([_epoch("2023-01-03"), _epoch("2023-01-04")], closes=[40.0, 10.0])
    payload["chart"]["result"][0]["events"] = {
        "splits": {"1": {"numerator": 4, "denominator": 1, "date": _epoch("2023-01-04")}}
    }

    frame = parse_chart_payload(payload, "X")

    assert frame["split_ratio"].dropna().to_dict() == {pd.Timestamp("2023-01-04"): 4.0}


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
    assert list(frame.columns) == list(BAR_COLUMNS)


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


def test_the_browser_user_agent_replaces_the_library_default() -> None:
    """Yahoo 429s a python-requests User-Agent on essentially every call.

    Worth pinning because the failure is so well disguised: every symbol comes back 429,
    which reads as being rate limited for going too fast, and no amount of backoff or
    throttling fixes it. It has to be asserted against a real requests.Session, since a
    test double starts with empty headers and hides the whole problem.
    """
    session = requests.Session()
    assert "python-requests" in session.headers["User-Agent"], "the trap this test guards"

    YahooFinanceSource(session=session)

    assert session.headers["User-Agent"] == yahoo.DEFAULT_USER_AGENT


def test_a_rate_limit_slows_the_rest_of_the_session(
    aapl_payload: dict[str, Any], no_sleep: None
) -> None:
    """The limiter counts requests per session, so the next symbol hits the same wall.

    Backing off only the failed request and then resuming at full speed is how one 429
    becomes forty. The widened interval has to outlive the call that earned it.
    """
    session = FakeSession([FakeResponse(429), FakeResponse(200, aapl_payload)])
    source = YahooFinanceSource(session=session, min_interval=0.25)  # type: ignore[arg-type]

    source.fetch("AAPL", dt.date(2023, 1, 1), dt.date(2023, 3, 31))

    # Still widened after the retry succeeded: getting one call through does not mean
    # the limit lifted.
    assert source.interval == 0.5


def test_the_slowdown_stops_at_the_ceiling(aapl_payload: dict[str, Any], no_sleep: None) -> None:
    session = FakeSession([FakeResponse(429)] * 3 + [FakeResponse(200, aapl_payload)])
    source = YahooFinanceSource(  # type: ignore[arg-type]
        session=session, min_interval=0.25, max_interval=0.75, max_attempts=5
    )

    source.fetch("AAPL", dt.date(2023, 1, 1), dt.date(2023, 3, 31))

    # 0.25 -> 0.5 -> capped, rather than doubling to 2.0 on the third.
    assert source.interval == 0.75


def test_a_server_that_says_when_to_come_back_is_believed(
    aapl_payload: dict[str, Any], slept: list[float]
) -> None:
    session = FakeSession(
        [FakeResponse(429, headers={"Retry-After": "7"}), FakeResponse(200, aapl_payload)]
    )
    source = YahooFinanceSource(session=session, min_interval=0.0)  # type: ignore[arg-type]

    source.fetch("AAPL", dt.date(2023, 1, 1), dt.date(2023, 3, 31))

    # 7 from the header, not the 0.5 our own backoff would have picked.
    assert slept == [7.0]


def test_an_absurd_retry_after_is_capped(no_sleep: None, slept: list[float]) -> None:
    """An hour-long wait should fail the run, not park the process until it times out."""
    session = FakeSession([FakeResponse(429, headers={"Retry-After": "3600"})] * 2)
    source = YahooFinanceSource(  # type: ignore[arg-type]
        session=session, min_interval=0.0, max_attempts=2
    )

    with pytest.raises(FetchError, match="giving up"):
        source.fetch("AAPL", dt.date(2023, 1, 1), dt.date(2023, 3, 31))

    assert slept == [yahoo.MAX_BACKOFF_SECONDS]


def test_a_dated_retry_after_falls_back_to_our_own_backoff(
    aapl_payload: dict[str, Any], slept: list[float]
) -> None:
    """Retry-After also has an HTTP-date form, which we deliberately do not parse."""
    header = {"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}
    session = FakeSession([FakeResponse(429, headers=header), FakeResponse(200, aapl_payload)])
    source = YahooFinanceSource(  # type: ignore[arg-type]
        session=session, min_interval=0.0, backoff_base=0.5
    )

    source.fetch("AAPL", dt.date(2023, 1, 1), dt.date(2023, 3, 31))

    assert slept == [0.5]


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
