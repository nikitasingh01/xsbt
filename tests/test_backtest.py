"""End-to-end engine tests: the wiring, the accounting and the round trip through disk.

The panel here is a seeded random walk rather than a hand-built toy. Hand-computed
accounting is covered in test_portfolio.py, and the lookahead guarantee has a file of its
own in test_no_lookahead.py.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.helpers import make_backtest_config, random_walk_panel
from xsbt.config import BacktestConfig
from xsbt.engine.backtest import BacktestResult, run_backtest
from xsbt.engine.calendar import apply_execution_lag, rebalance_dates
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


def test_daily_frame_covers_every_session(panel: pd.DataFrame, result: BacktestResult) -> None:
    pd.testing.assert_index_equal(result.daily.index, panel.index)
    assert result.daily.notna().all().all()
    assert set(result.daily.columns) >= {
        "gross_return",
        "net_return",
        "cost",
        "turnover",
        "long_return",
        "short_return",
        "gross_exposure",
        "net_exposure",
    }


def test_nothing_is_held_before_the_first_trade(result: BacktestResult) -> None:
    first_trade = result.target_weights.index[0]

    assert result.weights.loc[:first_trade].abs().to_numpy().sum() == 0.0
    assert result.daily["gross_return"].loc[:first_trade].eq(0.0).all()


def test_a_book_traded_at_a_close_earns_from_the_next_session(
    result: BacktestResult,
) -> None:
    first_trade = result.target_weights.index[0]
    sessions = result.daily.index
    next_session = sessions[sessions.get_loc(first_trade) + 1]

    pd.testing.assert_series_equal(
        result.weights.loc[next_session],
        result.target_weights.loc[first_trade],
        check_names=False,
    )


def test_trades_land_one_session_after_the_signal(
    result: BacktestResult, config: BacktestConfig
) -> None:
    sessions = result.daily.index
    expected = apply_execution_lag(sessions, rebalance_dates(sessions, "M"), lag=1)

    traded = result.target_weights.index
    assert traded.isin(expected.to_numpy()).all()
    # Every trade date sits strictly after the month end that produced it.
    signals = expected.index[expected.isin(traded)]
    assert (signals.to_numpy() < traded.to_numpy()).all()


def test_costs_only_ever_subtract(result: BacktestResult) -> None:
    assert (result.costs >= 0.0).all()
    assert result.costs.sum() > 0.0
    assert (result.returns <= result.gross_returns + 1e-15).all()


def test_zero_cost_leaves_gross_untouched(panel: pd.DataFrame, config: BacktestConfig) -> None:
    free = run_backtest(panel, Momentum(config.strategy), make_backtest_config(panel, cost_bps=0.0))
    charged = run_backtest(panel, Momentum(config.strategy), config)

    pd.testing.assert_series_equal(free.returns, free.gross_returns, check_names=False)
    pd.testing.assert_series_equal(free.gross_returns, charged.gross_returns, check_names=False)
    assert charged.equity_curve().iloc[-1] < free.equity_curve().iloc[-1]


def test_gross_leverage_scales_the_target_book(panel: pd.DataFrame, config: BacktestConfig) -> None:
    levered = run_backtest(
        panel, Momentum(config.strategy), make_backtest_config(panel, gross_leverage=2.0)
    )

    assert levered.target_weights.abs().sum(axis=1).round(10).eq(2.0).all()


def test_book_is_neutral_and_fully_invested_at_each_trade(result: BacktestResult) -> None:
    gross = result.target_weights.abs().sum(axis=1)
    net = result.target_weights.sum(axis=1)

    np.testing.assert_allclose(gross.to_numpy(), 1.0, atol=1e-12)
    np.testing.assert_allclose(net.to_numpy(), 0.0, atol=1e-12)


def test_metadata_records_what_produced_the_run(
    panel: pd.DataFrame, result: BacktestResult, config: BacktestConfig
) -> None:
    meta = result.metadata

    assert meta["config_fingerprint"] == config.fingerprint()
    assert meta["strategy"] == "Momentum"
    assert meta["universe_size"] == panel.shape[1]
    assert meta["rebalances_traded"] > 0
    assert meta["rebalances_traded"] + meta["rebalances_skipped"] == meta["rebalances_scheduled"]
    assert meta["first_session"] == "2018-01-01"


def test_snapshot_id_is_carried_into_the_result(
    panel: pd.DataFrame, config: BacktestConfig
) -> None:
    run = run_backtest(panel, Momentum(config.strategy), config, snapshot_id="deadbeef")

    assert run.metadata["data_snapshot_id"] == "deadbeef"


def test_benchmark_is_aligned_onto_the_daily_frame(
    panel: pd.DataFrame, config: BacktestConfig
) -> None:
    spy = pd.Series(0.001, index=panel.index, name="benchmark_return")

    run = run_backtest(panel, Momentum(config.strategy), config, benchmark=spy)

    assert run.benchmark is not None
    assert run.benchmark.eq(0.001).all()


def test_no_benchmark_means_no_column(result: BacktestResult) -> None:
    assert result.benchmark is None


def test_two_runs_of_the_same_config_agree(panel: pd.DataFrame, config: BacktestConfig) -> None:
    first = run_backtest(panel, Momentum(config.strategy), config)
    second = run_backtest(panel, Momentum(config.strategy), config)

    pd.testing.assert_frame_equal(first.daily, second.daily, check_exact=True)


def test_result_round_trips_through_disk(result: BacktestResult, tmp_path: Path) -> None:
    result.save(tmp_path / "run")
    loaded = BacktestResult.load(tmp_path / "run")

    pd.testing.assert_frame_equal(result.daily, loaded.daily, check_freq=False)
    pd.testing.assert_frame_equal(result.weights, loaded.weights, check_freq=False)
    pd.testing.assert_frame_equal(result.target_weights, loaded.target_weights, check_freq=False)
    assert loaded.config == result.config
    assert loaded.metadata == result.metadata


def test_loading_a_directory_that_is_not_a_run(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not a run directory"):
        BacktestResult.load(tmp_path)


def test_lookback_longer_than_the_sample_is_an_error(panel: pd.DataFrame) -> None:
    config = make_backtest_config(panel)
    long_lookback = config.model_copy(
        update={"strategy": config.strategy.model_copy(update={"lookback_days": 5000})}
    )

    with pytest.raises(ValueError, match="no rebalance produced a book"):
        run_backtest(panel, Momentum(long_lookback.strategy), long_lookback)


def test_config_window_outside_the_data_is_an_error(panel: pd.DataFrame) -> None:
    config = make_backtest_config(panel)
    elsewhere = config.model_copy(
        update={
            "data": config.data.model_copy(
                update={"start": dt.date(1990, 1, 1), "end": dt.date(1991, 1, 1)}
            )
        }
    )

    with pytest.raises(ValueError, match="no price data"):
        run_backtest(panel, Momentum(elsewhere.strategy), elsewhere)
