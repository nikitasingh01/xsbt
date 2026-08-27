"""Metrics checked against cases with a closed-form answer.

Where a metric has an exact value that can be worked out on paper, that is what gets
asserted. Re-deriving the implementation in the test would prove nothing.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from xsbt.analytics.attribution import attribute_legs, attribute_names, fit_market
from xsbt.analytics.metrics import (
    TRADING_DAYS,
    annualised_volatility,
    cagr,
    calmar_ratio,
    conditional_value_at_risk,
    default_hac_lags,
    deflate,
    equity_curve,
    expected_max_sharpe,
    hit_rate,
    max_drawdown,
    monthly_returns,
    newey_west_factor,
    probabilistic_sharpe_ratio,
    rolling_sharpe,
    sharpe_ratio,
    sharpe_standard_error,
    sharpe_tstat,
    sortino_ratio,
    summarise,
    total_return,
    value_at_risk,
    yearly_returns,
)


def series(values: list[float] | np.ndarray, start: str = "2020-01-01") -> pd.Series:
    index = pd.bdate_range(start, periods=len(values), name="date")
    return pd.Series(np.asarray(values, dtype="float64"), index=index, name="net_return")


def test_doubling_over_one_year_is_a_hundred_percent_cagr() -> None:
    daily = 2 ** (1 / TRADING_DAYS) - 1

    returns = series([daily] * TRADING_DAYS)

    assert total_return(returns) == pytest.approx(1.0)
    assert cagr(returns) == pytest.approx(1.0)


def test_cagr_over_two_years_annualises_down() -> None:
    daily = 2 ** (1 / TRADING_DAYS) - 1

    returns = series([daily] * (2 * TRADING_DAYS))

    # Quadrupled over two years, so 100% a year.
    assert total_return(returns) == pytest.approx(3.0)
    assert cagr(returns) == pytest.approx(1.0)


def test_annualised_volatility_of_a_square_wave() -> None:
    """Alternating +-1%: mean 0, so the sample std is 0.01 * sqrt(n / (n - 1))."""
    returns = series([0.01, -0.01] * 126)

    assert annualised_volatility(returns) == pytest.approx(0.01 * TRADING_DAYS / math.sqrt(251))


def test_flat_returns_have_no_volatility_and_no_sharpe() -> None:
    returns = series([0.001] * 100)

    assert annualised_volatility(returns) == pytest.approx(0.0)
    assert math.isnan(sharpe_ratio(returns))


def test_sharpe_of_a_known_series() -> None:
    """Returns of 2% and 0% alternating: mean 1%, deviations +-1%, so the annualised
    Sharpe collapses to sqrt(n - 1)."""
    returns = series([0.02, 0.0] * 126)

    assert sharpe_ratio(returns) == pytest.approx(math.sqrt(251))


def test_risk_free_rate_is_a_hurdle() -> None:
    returns = series([0.02, 0.0] * 126)

    assert sharpe_ratio(returns, risk_free_rate=0.05) < sharpe_ratio(returns)


def test_sortino_only_counts_the_downside() -> None:
    """+2% / -1% alternating: mean 0.5%, downside deviation 1% / sqrt(2)."""
    returns = series([0.02, -0.01] * 126)

    assert sortino_ratio(returns) == pytest.approx(math.sqrt(126))


def test_sortino_is_undefined_without_losses() -> None:
    returns = series([0.01, 0.02] * 50)

    assert math.isnan(sortino_ratio(returns))


def test_tstat_grows_with_the_square_root_of_the_sample() -> None:
    short = series([0.02, 0.0] * 126)
    long = series([0.02, 0.0] * 252)

    assert sharpe_tstat(long) / sharpe_tstat(short) == pytest.approx(math.sqrt(2), rel=0.01)


def test_standard_error_shrinks_as_the_sample_grows() -> None:
    short = series([0.01, -0.005] * 126)
    long = series([0.01, -0.005] * 1260)

    assert sharpe_standard_error(long) < sharpe_standard_error(short)


def test_a_drifting_series_is_far_more_significant_than_a_flat_one() -> None:
    """Same volatility, one with a mean and one without. A single random sample can look
    significant by chance, so the assertion is on the gap, not on an absolute threshold."""
    rng = np.random.default_rng(7)
    noise = rng.normal(0.0, 0.01, 4 * TRADING_DAYS)
    noise -= noise.mean()  # pin the sample mean at zero so the test cannot get lucky

    flat = series(noise)
    drifting = series(noise + 0.001)

    assert sharpe_tstat(drifting) > 2.0
    assert sharpe_tstat(drifting) > abs(sharpe_tstat(flat)) + 2.0


def test_a_series_with_no_memory_needs_no_correction() -> None:
    """White noise has nothing for the Bartlett weights to pick up, so f sits near 1."""
    rng = np.random.default_rng(11)

    factor = newey_west_factor(series(rng.normal(0.0, 0.01, 4000)), lags=21)

    assert factor == pytest.approx(1.0, abs=0.15)


def test_a_repeating_series_widens_the_error_bar() -> None:
    """Each value held for a stretch, so neighbouring days repeat each other.

    That is what monthly rebalancing does to a daily series, and the point of the whole
    correction: the effective sample is smaller than the session count claims.
    """
    rng = np.random.default_rng(3)
    held = series(np.repeat(rng.normal(0.001, 0.01, 200), 21))

    assert newey_west_factor(held, lags=21) > 2.0
    assert sharpe_standard_error(held, hac_lags=21) > sharpe_standard_error(held)
    assert abs(sharpe_tstat(held, hac_lags=21)) < abs(sharpe_tstat(held))


def test_an_alternating_series_tightens_it_instead() -> None:
    """The correction is not a one-way safety margin. Negative autocorrelation means the
    iid bar was the conservative one, and the estimator has to be allowed to say so."""
    zigzag = series([0.01, -0.01] * 500)

    assert newey_west_factor(zigzag, lags=4) < 1.0
    assert sharpe_standard_error(zigzag, hac_lags=4) < sharpe_standard_error(zigzag)


def test_the_correction_is_worked_out_by_hand_at_one_lag() -> None:
    """Four returns, one lag. rho_1 and the Bartlett weight are small enough to check."""
    sample = series([0.02, -0.01, 0.03, 0.00])
    centred = np.array([0.02, -0.01, 0.03, 0.00]) - 0.01

    variance = float(centred @ centred) / 4
    autocovariance = float(centred[1:] @ centred[:-1]) / 4
    expected = 1.0 + 2.0 * (1.0 - 1.0 / 2.0) * autocovariance / variance

    assert newey_west_factor(sample, lags=1) == pytest.approx(expected)


def test_zero_lags_leaves_the_iid_error_bar_alone() -> None:
    sample = series([0.01, -0.004, 0.02, 0.0, -0.01] * 40)

    assert newey_west_factor(sample, lags=0) == 1.0
    assert sharpe_standard_error(sample, hac_lags=0) == pytest.approx(sharpe_standard_error(sample))


def test_the_automatic_bandwidth_grows_slowly_with_the_sample() -> None:
    """Newey-West (1994): floor(4 * (n / 100) ** (2/9)). Slow growth is the point, so a
    long sample does not spend its degrees of freedom estimating autocovariances."""
    assert default_hac_lags(100) == 4
    assert default_hac_lags(4000) == 9
    assert default_hac_lags(2) == 0


def test_summary_reports_both_error_bars() -> None:
    summary = summarise(series([0.01, -0.004, 0.02, 0.0, -0.01] * 60), hac_lags=5)

    assert summary.hac_lags == 5
    assert math.isfinite(summary.sharpe_se)
    assert math.isfinite(summary.sharpe_se_hac)
    assert summary.sharpe_tstat_hac == pytest.approx(summary.sharpe / summary.sharpe_se_hac)


def test_the_probabilistic_sharpe_matches_the_iid_bar_on_a_normal_series() -> None:
    """A consistency check between the two ways this repo scores a Sharpe.

    Set skew to zero and kurtosis to 3 and the PSR denominator collapses to Lo's
    sqrt(1 + SR^2 / 2), so the z-score it computes should be the same t-statistic the
    error bar gives, up to sqrt(n - 1) against sqrt(n). If these two ever drift apart,
    one of them has picked up a units bug, which on this formula means a stray sqrt(252).
    """
    rng = np.random.default_rng(11)
    sample = series(rng.normal(0.0004, 0.01, 3000))

    z = NormalDist().inv_cdf(probabilistic_sharpe_ratio(sample))

    assert z == pytest.approx(sharpe_tstat(sample), rel=0.02)


def test_a_left_tail_is_charged_for_where_the_plain_t_stat_ignores_it() -> None:
    """Same mean and roughly the same volatility, but one of them crashes occasionally.

    The t-stat cannot tell them apart, because it only reads the first two moments. The
    PSR should prefer the one without the left tail, which is the whole reason for using
    it: a mean earned between crashes is worth less than the same mean earned steadily.
    """
    steady = np.full(1200, 0.0006)
    steady[::3] = -0.0004
    crashy = steady.copy()
    crashy[::200] = -0.05
    crashy[1::200] = 0.05 + float(steady[1])

    assert probabilistic_sharpe_ratio(series(crashy)) < probabilistic_sharpe_ratio(series(steady))


def test_the_expected_best_of_many_tries_grows_with_the_number_of_tries() -> None:
    """Search wider and the winner looks better even when nothing has any edge."""
    assert expected_max_sharpe(5, 0.4) < expected_max_sharpe(25, 0.4)
    assert expected_max_sharpe(25, 0.4) < expected_max_sharpe(500, 0.4)


def test_the_expected_best_scales_with_how_far_apart_the_tries_landed() -> None:
    """Linear in the spread, which is what lets an annualised spread go straight in.

    A grid whose cells all come out in the same place has nothing to flatter its winner
    with, so the hurdle it produces is zero.
    """
    assert expected_max_sharpe(25, 0.8) == pytest.approx(2.0 * expected_max_sharpe(25, 0.4))
    assert expected_max_sharpe(25, 0.0) == 0.0
    assert expected_max_sharpe(1, 0.4) == 0.0


def test_charging_for_the_search_can_only_lower_the_odds() -> None:
    """The hurdle is never below zero, so the deflated number is never the kinder one."""
    sample = series(np.random.default_rng(3).normal(0.0006, 0.01, 2000))

    deflation = deflate(sample, [0.9, 0.4, 0.1, -0.2, 0.55])

    assert deflation is not None
    assert deflation.trials == 5
    assert deflation.expected_max_sharpe > 0.0
    assert deflation.deflated < deflation.probabilistic


def test_a_grid_that_disagrees_with_itself_is_charged_more() -> None:
    """Two searches of the same width over the same returns, different spreads.

    The scattered grid is the one where a good cell is more easily luck, so it should
    take the bigger haircut. This is the property that makes the correction specific to
    the search actually run rather than to searches in general.
    """
    sample = series(np.random.default_rng(5).normal(0.0006, 0.01, 2000))

    tight = deflate(sample, [0.30, 0.32, 0.34, 0.36, 0.38])
    scattered = deflate(sample, [-0.4, 0.0, 0.34, 0.7, 1.2])

    assert tight is not None
    assert scattered is not None
    assert scattered.expected_max_sharpe > tight.expected_max_sharpe
    assert scattered.deflated < tight.deflated


def test_cells_that_never_ran_are_dropped_rather_than_counted() -> None:
    """A grid cell that raised is not a trial, and NaN would poison the spread anyway."""
    sample = series(np.random.default_rng(7).normal(0.0006, 0.01, 800))

    deflation = deflate(sample, [0.4, float("nan"), 0.1, float("nan"), 0.25])

    assert deflation is not None
    assert deflation.trials == 3


def test_there_is_no_search_to_charge_for_below_two_cells() -> None:
    sample = series(np.random.default_rng(9).normal(0.0006, 0.01, 800))

    assert deflate(sample, []) is None
    assert deflate(sample, [0.4]) is None
    assert deflate(sample, [0.4, float("nan")]) is None


def test_max_drawdown_peak_trough_and_recovery() -> None:
    # 1.10, 0.88, 0.924, 1.1088 -> worst is 0.88 / 1.10 - 1.
    returns = series([0.10, -0.20, 0.05, 0.20])

    drawdown = max_drawdown(returns)

    assert drawdown.depth == pytest.approx(-0.20)
    assert drawdown.peak == returns.index[0]
    assert drawdown.trough == returns.index[1]
    assert drawdown.recovered == returns.index[3]
    assert drawdown.length_sessions == 3


def test_drawdown_that_never_recovers() -> None:
    # 1.10, 0.55, 0.5555 - the bounce is nowhere near the old high.
    returns = series([0.10, -0.50, 0.01])

    drawdown = max_drawdown(returns)

    assert drawdown.depth == pytest.approx(-0.50)
    assert drawdown.recovered is None
    assert drawdown.length_sessions == 2


def test_a_line_going_up_has_no_drawdown() -> None:
    returns = series([0.01] * 50)

    assert max_drawdown(returns).depth == pytest.approx(0.0)
    assert math.isnan(calmar_ratio(returns))


def test_calmar_is_growth_over_pain() -> None:
    returns = series([0.10, -0.20, 0.05, 0.20])

    assert calmar_ratio(returns) == pytest.approx(cagr(returns) / 0.20)


def test_var_and_cvar_on_a_uniform_grid() -> None:
    """101 evenly spaced returns from -10% to +10%, so the 5% quantile lands on a point."""
    returns = series(np.linspace(-0.10, 0.10, 101))

    assert value_at_risk(returns) == pytest.approx(-0.09)
    assert conditional_value_at_risk(returns) == pytest.approx(-0.095)


def test_cvar_is_at_least_as_bad_as_var() -> None:
    rng = np.random.default_rng(11)
    returns = series(rng.normal(0.0, 0.01, 1000))

    assert conditional_value_at_risk(returns) <= value_at_risk(returns)


def test_hit_rate_ignores_flat_days() -> None:
    returns = series([0.01, -0.01, 0.0, 0.01])

    assert hit_rate(returns) == pytest.approx(2 / 3)


def test_equity_curve_treats_gaps_as_flat() -> None:
    returns = series([0.10, np.nan, 0.10])

    np.testing.assert_allclose(equity_curve(returns).to_numpy(), [1.10, 1.10, 1.21])


def test_monthly_returns_compound_within_the_month() -> None:
    returns = series([0.10] * 3, start="2020-01-02")

    monthly = monthly_returns(returns)

    assert len(monthly) == 1
    assert monthly.iloc[0] == pytest.approx(1.10**3 - 1)


def test_yearly_returns_split_on_the_calendar() -> None:
    index = pd.to_datetime(["2020-12-30", "2020-12-31", "2021-01-04"])
    returns = pd.Series([0.10, 0.10, 0.10], index=index)

    yearly = yearly_returns(returns)

    assert len(yearly) == 2
    assert yearly.iloc[0] == pytest.approx(0.21)
    assert yearly.iloc[1] == pytest.approx(0.10)


def test_rolling_sharpe_needs_a_full_window() -> None:
    returns = series([0.02, 0.0] * 200)

    rolled = rolling_sharpe(returns, window=TRADING_DAYS)

    assert len(rolled) == len(returns) - TRADING_DAYS + 1
    assert rolled.iloc[0] == pytest.approx(math.sqrt(251), rel=1e-9)


def test_rolling_sharpe_on_a_short_sample_is_empty() -> None:
    assert rolling_sharpe(series([0.01] * 10), window=TRADING_DAYS).empty


def test_summary_carries_the_costs_through() -> None:
    returns = series([0.001] * 500)
    turnover = series([0.0] * 500)
    turnover.iloc[::100] = 1.0
    costs = turnover * 0.001

    summary = summarise(returns, turnover=turnover, costs=costs)

    assert summary.sessions == 500
    assert summary.trades == 5
    assert summary.avg_turnover_per_trade == pytest.approx(1.0)
    assert summary.total_cost == pytest.approx(0.005)
    assert summary.ann_turnover == pytest.approx(5 / (500 / TRADING_DAYS))


def test_summary_is_json_safe() -> None:
    """A monotone series has no Sharpe and no drawdown. NaN is not valid JSON, so those
    have to come out as None rather than as literals no parser will take."""
    summary = summarise(series([0.001] * 100))

    payload = summary.as_dict()

    assert payload["sharpe"] is None
    assert payload["max_drawdown_trough"] is None
    assert payload["max_drawdown_recovered"] is None
    assert payload["cagr"] is not None


def test_summarising_nothing_is_an_error() -> None:
    with pytest.raises(ValueError, match="no returns"):
        summarise(pd.Series(dtype="float64"))


def test_leg_attribution_splits_the_pnl() -> None:
    index = pd.bdate_range("2020-01-01", periods=TRADING_DAYS, name="date")
    legs = pd.DataFrame({"long": 0.001, "short": -0.0005}, index=index)

    split = attribute_legs(legs)

    assert split.long_ann_return == pytest.approx(0.252)
    assert split.short_ann_return == pytest.approx(-0.126)
    # Long made twice the total, because the short gave half of it back.
    assert split.long_share_of_pnl == pytest.approx(2.0)


def test_leg_attribution_needs_both_legs() -> None:
    frame = pd.DataFrame({"long": [0.01]}, index=pd.bdate_range("2020-01-01", periods=1))

    with pytest.raises(KeyError, match="expected long and short"):
        attribute_legs(frame)


def name_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Three names over two sessions, small enough to total in your head.

    AAA: 0.5 * 0.10 + 0.5 * -0.02  =  0.040
    BBB: -0.5 * 0.04 + -0.5 * 0.06 = -0.050
    CCC: never held                =  0.000
    """
    index = pd.bdate_range("2020-01-01", periods=2, name="date")
    weights = pd.DataFrame({"AAA": [0.5, 0.5], "BBB": [-0.5, -0.5], "CCC": [0.0, 0.0]}, index=index)
    returns = pd.DataFrame(
        {"AAA": [0.10, -0.02], "BBB": [0.04, 0.06], "CCC": [0.20, 0.20]}, index=index
    )
    return weights, returns


def test_name_attribution_totals_each_name_by_hand() -> None:
    split = attribute_names(*name_panels())

    assert split.table.loc["AAA", "contribution"] == pytest.approx(0.04)
    assert split.table.loc["BBB", "contribution"] == pytest.approx(-0.05)
    assert split.table.loc["AAA", "sessions_held"] == 2


def test_a_name_never_held_is_left_out_however_well_it_did() -> None:
    """CCC returned 20% a day throughout. We did not own it, so it contributed nothing
    and does not belong in a table about where our P&L came from."""
    split = attribute_names(*name_panels())

    assert "CCC" not in split.table.index
    assert split.names_held == 2


def test_name_contributions_add_up_to_gross_pnl() -> None:
    """The decomposition has to be exhaustive, or it is decoration rather than attribution."""
    weights, returns = name_panels()

    split = attribute_names(weights, returns)

    assert split.table["contribution"].sum() == pytest.approx(
        float((weights * returns).sum().sum())
    )


def test_concentration_stays_readable_when_the_legs_offset() -> None:
    """Measured on absolute contributions on purpose. Against the net, this book nets
    -0.01 out of 0.09 traded either way, and the share would read as several hundred
    percent: the same trap that made the leg split unreadable."""
    split = attribute_names(*name_panels())

    assert 0.0 <= split.concentration <= 1.0
    assert split.concentration == pytest.approx(1.0)
    assert split.table["share_of_gross"].sum() == pytest.approx(1.0)


def test_name_attribution_on_a_book_that_never_traded() -> None:
    index = pd.bdate_range("2020-01-01", periods=2, name="date")
    flat = pd.DataFrame({"AAA": [0.0, 0.0]}, index=index)

    with pytest.raises(ValueError, match="no name ever carried a position"):
        attribute_names(flat, flat)


def test_market_fit_recovers_an_exact_relationship() -> None:
    rng = np.random.default_rng(3)
    market = series(rng.normal(0.0003, 0.011, 750))
    strategy = 0.0002 + 0.5 * market

    fit = fit_market(strategy, market)

    assert fit.beta == pytest.approx(0.5)
    assert fit.alpha == pytest.approx(0.0002 * TRADING_DAYS)
    assert fit.r_squared == pytest.approx(1.0)
    assert fit.observations == 750


def test_market_fit_of_an_uncorrelated_book() -> None:
    rng = np.random.default_rng(5)
    market = series(rng.normal(0.0003, 0.011, 2000))
    strategy = series(rng.normal(0.0002, 0.006, 2000))

    fit = fit_market(strategy, market)

    assert abs(fit.beta) < 0.1
    assert fit.r_squared < 0.01
    # With no market exposure to strip out, residual risk is the whole of the risk.
    assert fit.tracking_error == pytest.approx(annualised_volatility(strategy), rel=0.01)


def test_market_fit_aligns_on_overlapping_dates() -> None:
    rng = np.random.default_rng(13)
    market = series(rng.normal(0.0, 0.01, 100))
    strategy = market.iloc[20:] * 0.4

    fit = fit_market(strategy, market)

    assert fit.observations == 80
    assert fit.beta == pytest.approx(0.4, rel=1e-6)


def test_market_fit_needs_something_to_fit() -> None:
    market = series([0.01, 0.02])

    with pytest.raises(ValueError, match="at least 3"):
        fit_market(market, market)


def test_market_fit_rejects_a_constant_benchmark() -> None:
    flat = series([0.001] * 50)
    rng = np.random.default_rng(17)
    strategy = series(rng.normal(0.0, 0.01, 50))

    with pytest.raises(ValueError, match="no variance"):
        fit_market(strategy, flat)
