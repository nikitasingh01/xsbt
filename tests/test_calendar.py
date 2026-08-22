from __future__ import annotations

import pandas as pd
import pytest

from xsbt.engine.calendar import apply_execution_lag, rebalance_dates


@pytest.fixture
def sessions() -> pd.DatetimeIndex:
    """Two years of weekdays, minus a few US holidays so month ends are not tidy."""
    days = pd.bdate_range("2020-01-01", "2021-12-31", name="date")
    holidays = pd.to_datetime(
        ["2020-12-25", "2020-01-01", "2021-01-01", "2021-12-24", "2020-07-03"]
    )
    return days.difference(holidays)


def test_month_end_uses_last_open_session(sessions: pd.DatetimeIndex) -> None:
    dates = rebalance_dates(sessions, "M")

    assert len(dates) == 24
    # 2020-12-31 was a Thursday and open; the 25th (a holiday) must not be picked.
    assert pd.Timestamp("2020-12-31") in dates
    # 2021-12-31 was a Friday and open even though the 24th was closed.
    assert dates[-1] == pd.Timestamp("2021-12-31")
    assert dates.is_monotonic_increasing
    assert dates.isin(sessions).all()


def test_month_end_when_the_last_weekday_is_shut() -> None:
    """A month whose final session is a holiday should rebalance on the day before."""
    days = pd.bdate_range("2021-05-01", "2021-06-15", name="date")
    open_days = days.difference(pd.to_datetime(["2021-05-31"]))

    dates = rebalance_dates(open_days, "M")

    assert dates[0] == pd.Timestamp("2021-05-28")


def test_quarter_and_week_ends(sessions: pd.DatetimeIndex) -> None:
    quarters = rebalance_dates(sessions, "Q")
    weeks = rebalance_dates(sessions, "W")

    assert len(quarters) == 8
    assert quarters[0] == pd.Timestamp("2020-03-31")
    assert 100 < len(weeks) < 110
    assert weeks.isin(sessions).all()


def test_every_n_sessions(sessions: pd.DatetimeIndex) -> None:
    dates = rebalance_dates(sessions, "21D")

    assert dates[0] == sessions[0]
    assert dates[1] == sessions[21]
    assert len(dates) == (len(sessions) + 20) // 21


def test_empty_index_is_not_an_error() -> None:
    empty = pd.DatetimeIndex([], name="date")

    assert len(rebalance_dates(empty, "M")) == 0


def test_execution_lag_moves_to_the_next_session(sessions: pd.DatetimeIndex) -> None:
    signals = rebalance_dates(sessions, "M")

    mapped = apply_execution_lag(sessions, signals, lag=1)

    # 2020-01-31 was a Friday; the next open session is Monday the 3rd.
    assert mapped[pd.Timestamp("2020-01-31")] == pd.Timestamp("2020-02-03")
    assert (mapped.to_numpy() > mapped.index.to_numpy()).all()


def test_zero_lag_is_same_session(sessions: pd.DatetimeIndex) -> None:
    signals = rebalance_dates(sessions, "M")

    mapped = apply_execution_lag(sessions, signals, lag=0)

    assert (mapped.to_numpy() == mapped.index.to_numpy()).all()


def test_signals_we_could_never_trade_are_dropped(sessions: pd.DatetimeIndex) -> None:
    """The final month end has no later session to trade into, so it earns nothing."""
    signals = rebalance_dates(sessions, "M")

    mapped = apply_execution_lag(sessions, signals, lag=1)

    assert signals[-1] == pd.Timestamp("2021-12-31")
    assert pd.Timestamp("2021-12-31") not in mapped.index
    assert len(mapped) == len(signals) - 1


def test_unknown_signal_date_is_rejected(sessions: pd.DatetimeIndex) -> None:
    with pytest.raises(KeyError, match="not in the session index"):
        apply_execution_lag(sessions, pd.DatetimeIndex([pd.Timestamp("2020-01-04")]), lag=1)


def test_negative_lag_rejected(sessions: pd.DatetimeIndex) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        apply_execution_lag(sessions, sessions[:2], lag=-1)
