"""Leg attribution and the market fit.

Both are small enough to check by hand, so the expected numbers below are worked out in
the docstrings rather than lifted from a previous run of the code.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tests.helpers import make_backtest_config, random_walk_panel
from xsbt.analytics.attribution import attribute_legs, fit_market
from xsbt.analytics.metrics import TRADING_DAYS
from xsbt.engine.backtest import run_backtest
from xsbt.strategies import Momentum


def legs_frame(long: list[float], short: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range("2021-01-04", periods=len(long), name="date")
    return pd.DataFrame({"long": long, "short": short}, index=dates)


def series(values: list[float], start: str = "2021-01-04") -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values), name="date"))


def test_leg_shares_add_up() -> None:
    """Long sums to +2%, short to +1%, so the long book is two thirds of the P&L."""
    legs = attribute_legs(legs_frame([0.01, 0.02, -0.01], [0.005, -0.005, 0.01]))

    assert legs.long_share_of_pnl == pytest.approx(2.0 / 3.0)
    assert legs.long_ann_return == pytest.approx(0.02 / 3 * TRADING_DAYS)
    assert legs.short_ann_return == pytest.approx(0.01 / 3 * TRADING_DAYS)
    assert legs.long_hit_rate == pytest.approx(2.0 / 3.0)
    assert legs.short_hit_rate == pytest.approx(2.0 / 3.0)


def test_a_losing_leg_pushes_the_share_outside_zero_to_one() -> None:
    """Long makes 3%, short loses 1%, so the long book carried more than all of it."""
    legs = attribute_legs(legs_frame([0.02, 0.005, 0.005], [-0.005, -0.005, 0.0]))

    assert legs.long_share_of_pnl == pytest.approx(0.03 / 0.02)
    assert legs.long_share_of_pnl > 1.0


def test_share_is_undefined_when_the_book_made_nothing() -> None:
    legs = attribute_legs(legs_frame([0.01, -0.01], [0.02, -0.02]))

    assert math.isnan(legs.long_share_of_pnl)
    assert legs.legs_offset


def test_shares_are_suppressed_when_the_legs_nearly_cancel() -> None:
    """The real momentum run nets +2.2% out of +12.2% long and -10.0% short.

    Dividing through gives 562% and -462%, which is arithmetically correct and reads as
    a broken report. Outside the readable range the split is dropped instead.
    """
    legs = attribute_legs(legs_frame([0.07, 0.05], [-0.04, -0.06]))

    assert math.isnan(legs.long_share_of_pnl)
    assert legs.legs_offset
    # The contribution row is the one that still answers the question.
    assert legs.long_ann_return > 0.0 > legs.short_ann_return


def test_shares_are_suppressed_when_a_losing_book_cancels_too() -> None:
    """Same failure on the other side: reversal loses money and printed -268% / +368%."""
    legs = attribute_legs(legs_frame([0.07, 0.05], [-0.10, -0.06]))

    assert legs.legs_offset


def test_shares_survive_when_both_legs_pull_the_same_way() -> None:
    """The case the row exists for: nothing cancels, so the split means something."""
    legs = attribute_legs(legs_frame([0.04, 0.02], [0.015, 0.005]))

    assert legs.long_share_of_pnl == pytest.approx(0.75)
    assert not legs.legs_offset


def test_opposed_legs_are_perfectly_anticorrelated() -> None:
    legs = attribute_legs(legs_frame([0.01, -0.02, 0.03], [-0.01, 0.02, -0.03]))

    assert legs.correlation == pytest.approx(-1.0)


def test_missing_columns_are_rejected_by_name() -> None:
    frame = legs_frame([0.01], [0.01]).rename(columns={"short": "short_return"})

    with pytest.raises(KeyError, match="expected long and short columns"):
        attribute_legs(frame)


def test_an_empty_frame_is_an_error_not_a_row_of_nans() -> None:
    with pytest.raises(ValueError, match="no leg returns"):
        attribute_legs(legs_frame([], []))


def test_attribution_serialises_without_nan_literals() -> None:
    payload = attribute_legs(legs_frame([0.01, -0.01], [0.02, -0.02])).as_dict()

    assert payload["long_share_of_pnl"] is None
    assert payload["long_hit_rate"] == pytest.approx(0.5)


def test_the_legs_of_a_real_run_add_up_to_its_gross_return() -> None:
    """The contributions are of one portfolio, so summing them has to give it back."""
    panel = random_walk_panel()
    config = make_backtest_config(panel)
    result = run_backtest(panel, Momentum(config.strategy), config)

    combined = result.legs["long"] + result.legs["short"]

    pd.testing.assert_series_equal(combined, result.gross_returns, check_names=False)
    attribute_legs(result.legs)  # the view the report asks for is the shape this wants


def test_beta_of_a_pure_multiple_of_the_market() -> None:
    market = series([0.01, -0.01, 0.02, -0.02, 0.005])
    fit = fit_market(2.0 * market, market)

    assert fit.beta == pytest.approx(2.0)
    assert fit.r_squared == pytest.approx(1.0)
    assert fit.alpha == pytest.approx(0.0, abs=1e-12)
    assert fit.tracking_error == pytest.approx(0.0, abs=1e-15)
    # Nothing is left over to divide by, so the appraisal ratio has no meaning here.
    assert math.isnan(fit.residual_sharpe)


def test_beta_alpha_and_tracking_error_against_hand_arithmetic() -> None:
    """x = [1, -1, 1, -1]bp-ish, y = [2, -2, 0, 0].

    Both series are mean zero, so cov(y, x) = 0.0004/3 and var(x) = 0.0004/3, giving
    beta = 1. The residual is then [1, -1, -1, 1] * 1e-2 with mean zero, so alpha is
    zero and the tracking error is sqrt(0.0004/3) * sqrt(252) = 0.1833.
    """
    x = series([0.01, -0.01, 0.01, -0.01])
    y = series([0.02, -0.02, 0.0, 0.0])

    fit = fit_market(y, x)

    assert fit.beta == pytest.approx(1.0)
    assert fit.alpha == pytest.approx(0.0, abs=1e-12)
    assert fit.tracking_error == pytest.approx(math.sqrt(0.0004 / 3) * math.sqrt(TRADING_DAYS))
    assert fit.correlation == pytest.approx(1.0 / math.sqrt(2.0))
    assert fit.r_squared == pytest.approx(0.5)
    assert fit.observations == 4


def test_alpha_is_the_annualised_intercept() -> None:
    """A constant 2bp a day on top of half the market has to come back as 2bp x 252."""
    market = series([0.01, -0.01, 0.02, -0.02, 0.005, -0.005])
    fit = fit_market(0.5 * market + 0.0002, market)

    assert fit.beta == pytest.approx(0.5)
    assert fit.alpha == pytest.approx(0.0002 * TRADING_DAYS)


def test_a_market_neutral_book_has_no_beta() -> None:
    """The market alternates sign; the book moves in pairs, so the two never covary."""
    market = series([0.01, -0.01, 0.01, -0.01, 0.01, -0.01])
    paired = series([0.002, 0.002, -0.001, -0.001, -0.001, -0.001])

    fit = fit_market(paired, market)

    assert fit.beta == pytest.approx(0.0, abs=1e-12)
    assert fit.r_squared == pytest.approx(0.0, abs=1e-12)
    assert fit.alpha == pytest.approx(0.0, abs=1e-12)


def test_only_overlapping_sessions_are_used() -> None:
    strategy = series([0.01, -0.01, 0.02, -0.02, 0.01], start="2021-01-04")
    market = series([0.01, -0.01, 0.02], start="2021-01-06")

    fit = fit_market(strategy, market)

    assert fit.observations == 3


def test_too_little_overlap_is_refused_rather_than_fitted() -> None:
    strategy = series([0.01, -0.01, 0.02, -0.02])
    market = series([0.01, -0.01], start="2021-01-04")

    with pytest.raises(ValueError, match="at least 3 overlapping"):
        fit_market(strategy, market)


def test_a_flat_benchmark_has_no_beta_to_find() -> None:
    market = series([0.0, 0.0, 0.0, 0.0])
    strategy = series([0.01, -0.01, 0.02, -0.02])

    with pytest.raises(ValueError, match="no variance"):
        fit_market(strategy, market)


def test_the_hurdle_comes_off_both_sides() -> None:
    """Subtracting a constant from x and y leaves the slope alone but moves the intercept."""
    market = series([0.01, -0.01, 0.02, -0.02, 0.005])
    strategy = 0.5 * market + 0.0002

    plain = fit_market(strategy, market)
    hurdled = fit_market(strategy, market, risk_free_rate=0.03)

    assert hurdled.beta == pytest.approx(plain.beta)
    assert hurdled.alpha < plain.alpha


def test_nan_sessions_are_dropped_from_the_pair() -> None:
    market = series([0.01, np.nan, 0.02, -0.02, 0.01])
    strategy = series([0.02, 0.03, 0.04, -0.04, 0.02])

    assert fit_market(strategy, market).observations == 4
