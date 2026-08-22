"""Rebalance dates and the gap between forming a signal and holding the position."""

from __future__ import annotations

import pandas as pd

PERIOD_ALIAS = {"M": "M", "W": "W", "Q": "Q"}


def rebalance_dates(sessions: pd.DatetimeIndex, frequency: str) -> pd.DatetimeIndex:
    """Sessions on which a new signal is formed.

    'M', 'W' and 'Q' pick the last session in each calendar period, so a month ending on
    a holiday still rebalances on the last day the market was actually open. 'nD' takes
    every n-th session from the start of the sample.
    """
    if len(sessions) == 0:
        return sessions

    frequency = frequency.strip().upper()

    if frequency in PERIOD_ALIAS:
        periods = sessions.to_period(PERIOD_ALIAS[frequency])
        last = pd.Series(sessions, index=periods).groupby(level=0).last()
        return pd.DatetimeIndex(last.to_numpy(), name=sessions.name)

    if frequency.endswith("D"):
        step = int(frequency[:-1])
        if step <= 0:
            raise ValueError(f"rebalance step must be positive, got {frequency!r}")
        return sessions[::step]

    raise ValueError(f"unsupported rebalance frequency {frequency!r}")


def apply_execution_lag(
    sessions: pd.DatetimeIndex,
    signal_dates: pd.DatetimeIndex,
    lag: int,
) -> pd.Series:
    """Map each signal date to the session the resulting book is first held.

    Signal dates whose implementation would land past the end of the sample are dropped:
    a trade we never got to put on contributes no P&L.
    """
    if lag < 0:
        raise ValueError(f"execution lag must be non-negative, got {lag}")

    positions = sessions.get_indexer(signal_dates)
    if (positions < 0).any():
        missing = signal_dates[positions < 0]
        raise KeyError(f"signal dates not in the session index: {list(missing[:5])}")

    implemented = positions + lag
    keep = implemented < len(sessions)
    return pd.Series(
        sessions[implemented[keep]],
        index=signal_dates[keep],
        name="implementation_date",
    )
