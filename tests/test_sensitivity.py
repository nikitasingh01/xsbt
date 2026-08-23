"""Cost and parameter sensitivity.

The cost sweep is checked against the engine's own output rather than against a formula,
because the whole point of doing it analytically is that it must agree with a real run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.helpers import make_backtest_config, random_walk_panel
from xsbt.analytics.metrics import cagr, sharpe_ratio
from xsbt.analytics.sensitivity import (
    breakeven_cost_bps,
    cost_sweep,
    net_returns_at,
    parameter_grid,
)
from xsbt.config import BacktestConfig
from xsbt.engine.backtest import BacktestResult, run_backtest
from xsbt.strategies import Momentum


@pytest.fixture
def panel() -> pd.DataFrame:
    return random_walk_panel()


@pytest.fixture
def config(panel: pd.DataFrame) -> BacktestConfig:
    return make_backtest_config(panel, cost_bps=10.0)


@pytest.fixture
def result(panel: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
    return run_backtest(panel, Momentum(config.strategy), config)


def test_zero_cost_returns_are_the_gross_returns(result: BacktestResult) -> None:
    pd.testing.assert_series_equal(
        net_returns_at(result, 0.0), result.gross_returns, check_names=False
    )


def test_the_analytic_sweep_reproduces_a_real_run(
    panel: pd.DataFrame, result: BacktestResult
) -> None:
    """Repricing the run at 25bps must match actually running it at 25bps."""
    rerun = run_backtest(
        panel,
        Momentum(result.config.strategy),
        make_backtest_config(panel, cost_bps=25.0),
    )

    pd.testing.assert_series_equal(
        net_returns_at(result, 25.0), rerun.returns, check_names=False, check_exact=True
    )


def test_the_sweep_agrees_with_the_configured_cost(result: BacktestResult) -> None:
    pd.testing.assert_series_equal(
        net_returns_at(result, result.config.portfolio.cost_bps),
        result.returns,
        check_names=False,
        check_exact=True,
    )


def test_cost_sweep_is_monotone(result: BacktestResult) -> None:
    sweep = cost_sweep(result, levels=(0.0, 5.0, 10.0, 20.0, 50.0))

    assert list(sweep.index) == [0.0, 5.0, 10.0, 20.0, 50.0]
    assert sweep["cagr"].is_monotonic_decreasing
    assert sweep["sharpe"].is_monotonic_decreasing
    # Only the drag changes, so the volatility barely moves.
    assert sweep["ann_volatility"].std() < 0.01


def test_cost_sweep_matches_the_metrics_it_reports(result: BacktestResult) -> None:
    sweep = cost_sweep(result, levels=(20.0,))

    net = net_returns_at(result, 20.0)
    assert sweep.loc[20.0, "cagr"] == pytest.approx(cagr(net))
    assert sweep.loc[20.0, "sharpe"] == pytest.approx(sharpe_ratio(net))


def synthetic_result(
    config: BacktestConfig, *, gross: float, trade_every: int, sessions: int
) -> BacktestResult:
    """A run with a flat gross return and a trade every n sessions.

    Breakeven has a closed form on a path like this, which a random walk does not.
    """
    index = pd.bdate_range("2018-01-01", periods=sessions, name="date")
    daily = pd.DataFrame({"gross_return": gross, "turnover": 0.0}, index=index)
    daily.iloc[::trade_every, daily.columns.get_loc("turnover")] = 1.0
    daily["cost"] = daily["turnover"] * config.portfolio.cost_bps / 1e4
    daily["net_return"] = daily["gross_return"] - daily["cost"]

    empty = pd.DataFrame(index=index)
    return BacktestResult(
        config=config, daily=daily, weights=empty, target_weights=empty, metadata={}
    )


def test_breakeven_cost_is_where_growth_stops(config: BacktestConfig) -> None:
    """504 sessions at +4bps a day, trading 24 times. Breaking even needs

        (1.0004) ** 480 * (1.0004 - r) ** 24 == 1

    which solves to r = 83.66bps.
    """
    run = synthetic_result(config, gross=0.0004, trade_every=21, sessions=504)

    breakeven = breakeven_cost_bps(run)

    assert breakeven == pytest.approx(83.66, rel=1e-3)
    assert cagr(net_returns_at(run, breakeven)) == pytest.approx(0.0, abs=1e-9)
    assert cagr(net_returns_at(run, breakeven * 1.2)) < 0.0


def test_breakeven_is_zero_for_a_strategy_that_loses_before_costs(
    config: BacktestConfig,
) -> None:
    run = synthetic_result(config, gross=-0.0001, trade_every=21, sessions=504)

    assert breakeven_cost_bps(run) == 0.0


def test_breakeven_is_undefined_without_trading(config: BacktestConfig) -> None:
    run = synthetic_result(config, gross=0.0004, trade_every=21, sessions=504)
    run.daily["turnover"] = 0.0

    assert np.isnan(breakeven_cost_bps(run))


def test_breakeven_gives_up_at_the_ceiling(config: BacktestConfig) -> None:
    """A book that trades almost nothing can absorb any cost we would ever assume."""
    run = synthetic_result(config, gross=0.0004, trade_every=500, sessions=504)

    assert breakeven_cost_bps(run, ceiling=100.0) == float("inf")


def test_parameter_grid_shape_and_labels(panel: pd.DataFrame, config: BacktestConfig) -> None:
    grid = parameter_grid(panel, config, lookbacks=[21, 63, 126], top_fractions=[0.2, 0.25, 0.5])

    assert grid.shape == (3, 3)
    assert grid.index.name == "lookback_days"
    assert grid.columns.name == "top_fraction"
    assert grid.notna().all().all()


def test_grid_cells_that_cannot_run_come_back_empty(
    panel: pd.DataFrame, config: BacktestConfig
) -> None:
    """A lookback longer than the sample should blank one cell, not kill the grid."""
    grid = parameter_grid(panel, config, lookbacks=[63, 5000], top_fractions=[0.25])

    assert not np.isnan(grid.loc[63, 0.25])
    assert np.isnan(grid.loc[5000, 0.25])


def test_grid_can_report_other_metrics(panel: pd.DataFrame, config: BacktestConfig) -> None:
    drawdowns = parameter_grid(
        panel, config, lookbacks=[63], top_fractions=[0.25], metric="max_drawdown"
    )

    assert drawdowns.loc[63, 0.25] <= 0.0


def test_unknown_grid_metric_is_rejected(panel: pd.DataFrame, config: BacktestConfig) -> None:
    with pytest.raises(ValueError, match="unsupported grid metric"):
        parameter_grid(panel, config, lookbacks=[63], top_fractions=[0.25], metric="vibes")
