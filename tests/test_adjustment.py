"""Rebuilding adjusted close from the events the vendor reports alongside it.

The worked cases are small enough to check by hand, which is the point: the arithmetic is
easy to get subtly wrong (off-by-one on the ex-date, factor applied on the wrong side) and
a real snapshot agrees to 1e-6 whether or not the shift is right.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.helpers import make_bars
from xsbt.data.adjustment import audit_adjustment, implied_adjustment

SESSIONS = pd.bdate_range("2023-01-02", periods=6, name="date")


def test_no_dividends_means_no_adjustment() -> None:
    bars = make_bars(SESSIONS, np.full(6, 100.0))

    assert implied_adjustment(bars).tolist() == [1.0] * 6


def test_one_dividend_scales_only_the_history_before_it() -> None:
    """1.00 off a 100.00 close is a 1% factor, and it applies to sessions strictly before
    the ex-date. The ex-date bar and everything after it stay at 1.0."""
    bars = make_bars(SESSIONS[:5], np.full(5, 100.0), dividends={"2023-01-04": 1.0})

    factors = implied_adjustment(bars)

    assert factors.tolist() == [0.99, 0.99, 1.0, 1.0, 1.0]


def test_two_dividends_compound() -> None:
    """1.00 then 2.00, both off a 100.00 close: 0.99 * 0.98 = 0.9702 at the front."""
    bars = make_bars(
        SESSIONS,
        np.full(6, 100.0),
        dividends={"2023-01-04": 1.0, "2023-01-06": 2.0},
    )

    factors = implied_adjustment(bars)

    assert factors.tolist() == pytest.approx([0.9702, 0.9702, 0.98, 0.98, 1.0, 1.0])


def test_a_dividend_on_the_first_bar_is_skipped() -> None:
    """There is no prior close to divide by, so that one cannot be reconstructed.

    Skipping is right: the alternative is a NaN that poisons the whole cumulative
    product and turns one unknowable factor into a completely unusable series.
    """
    bars = make_bars(SESSIONS, np.full(6, 100.0), dividends={"2023-01-02": 1.0})

    assert implied_adjustment(bars).tolist() == [1.0] * 6


def test_a_consistent_pair_reconciles() -> None:
    closes = np.full(5, 100.0)
    bars = make_bars(
        SESSIONS[:5],
        closes,
        adj_close=closes * np.array([0.99, 0.99, 1.0, 1.0, 1.0]),
        dividends={"2023-01-04": 1.0},
    )

    audit = audit_adjustment("X", bars)

    assert audit.ok
    assert audit.complaint() is None
    assert audit.max_error == pytest.approx(0.0)
    assert (audit.sessions, audit.dividends, audit.splits) == (5, 1, 0)


def test_a_missing_dividend_fails_and_names_the_session() -> None:
    """The vendor adjusted for nothing; we know a dividend went ex. That is the whole
    failure mode this check exists for, and 1% is well past the 1e-4 tolerance."""
    closes = np.full(5, 100.0)
    bars = make_bars(SESSIONS[:5], closes, adj_close=closes, dividends={"2023-01-04": 1.0})

    audit = audit_adjustment("X", bars)

    assert not audit.ok
    assert audit.max_error == pytest.approx(0.01)
    assert audit.worst_date == "2023-01-02"
    complaint = audit.complaint()
    assert complaint is not None
    assert "1.00%" in complaint and "2023-01-02" in complaint


def test_a_step_no_event_explains_fails() -> None:
    """The other direction: an adjustment we cannot account for. Usually a split folded
    into one series and not the other, which is why splits need no separate check."""
    closes = np.full(6, 100.0)
    bars = make_bars(SESSIONS, closes, adj_close=closes * np.array([0.25] * 3 + [1.0] * 3))

    audit = audit_adjustment("X", bars)

    assert not audit.ok
    # Gaps are relative to the vendor's own factor, so a 4x step reads as 1/0.25 - 1.
    assert audit.max_error == pytest.approx(3.0)
    assert audit.worst_date == "2023-01-02"


def test_a_level_offset_is_reported_rather_than_failed() -> None:
    """A dividend going ex after the last bar we hold leaves the whole series scaled.

    Real and common: on the live snapshot most large caps carry one. It is also
    harmless, because a constant on every price cancels out of every return, so it gets
    reported and not failed.
    """
    closes = np.full(5, 100.0)
    shape = np.array([0.99, 0.99, 1.0, 1.0, 1.0])
    bars = make_bars(
        SESSIONS[:5],
        closes,
        adj_close=closes * shape * 0.5,
        dividends={"2023-01-04": 1.0},
    )

    audit = audit_adjustment("X", bars)

    assert audit.ok
    assert audit.unexplained_level == pytest.approx(0.5)


def test_a_split_is_counted_but_contributes_no_factor() -> None:
    """Yahoo's close is already split-adjusted and it restates dividends into the same
    units, so a split moves both sides of adj_close / close and cancels."""
    closes = np.array([25.0, 25.0, 25.0, 25.0])
    bars = make_bars(SESSIONS[:4], closes, adj_close=closes, splits={"2023-01-04": 4.0})

    audit = audit_adjustment("X", bars)

    assert audit.ok
    assert audit.splits == 1


def test_a_frame_too_short_to_check_is_not_an_error() -> None:
    """One bar has no prior close anywhere, so there is nothing to reconcile. A newly
    listed name should not turn the whole audit red."""
    bars = make_bars(SESSIONS[:1], np.array([100.0]))

    audit = audit_adjustment("X", bars)

    assert audit.ok
    assert (audit.sessions, audit.worst_date) == (1, None)


def test_an_unusable_adjusted_close_is_the_worst_outcome_not_a_pass() -> None:
    """All zeroes cannot be divided out. Reporting NaN keeps it failing rather than
    letting a degenerate series slip through as agreeing with everything."""
    bars = make_bars(SESSIONS, np.full(6, 100.0), adj_close=np.zeros(6))

    audit = audit_adjustment("X", bars)

    assert not audit.ok
    assert np.isnan(audit.max_error)
    assert audit.complaint() == "adjusted close could not be reconstructed from the reported events"


def test_the_tolerance_is_the_dial() -> None:
    closes = np.full(5, 100.0)
    bars = make_bars(SESSIONS[:5], closes, adj_close=closes, dividends={"2023-01-04": 1.0})

    assert not audit_adjustment("X", bars, tolerance=1e-4).ok
    assert audit_adjustment("X", bars, tolerance=0.02).ok
