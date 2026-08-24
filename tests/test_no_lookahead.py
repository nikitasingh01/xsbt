"""The test the whole design is arranged around.

Prices go to the strategy whole, future included, rather than pre-sliced by the engine.
That is deliberate and argued in docs/DESIGN.md, but it means the guarantee has to be
earned by experiment: rewrite the future, re-run, and the past must not move. A failure
here makes every other number in the report meaningless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.helpers import make_backtest_config, random_walk_panel
from xsbt.config import BacktestConfig
from xsbt.engine.backtest import BacktestResult, run_backtest
from xsbt.engine.calendar import apply_execution_lag, rebalance_dates
from xsbt.strategies import Momentum

#: Far enough in to have traded a couple of dozen rebalances before the cut.
CUT_SESSION = 400


@pytest.fixture
def panel() -> pd.DataFrame:
    return random_walk_panel()


@pytest.fixture
def config(panel: pd.DataFrame) -> BacktestConfig:
    return make_backtest_config(panel, cost_bps=10.0)


@pytest.fixture
def result(panel: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
    return run_backtest(panel, Momentum(config.strategy), config)


def rewrite_the_future(panel: pd.DataFrame, cut: pd.Timestamp, seed: int) -> pd.DataFrame:
    """Replace everything after ``cut`` with a different random walk off the same level.

    Scaling the future by some huge factor would be a blunter corruption, but it also
    leaves the cross-sectional ranking intact and blows the book up on the join day. A
    fresh walk reorders every name after the cut while keeping the returns plausible.
    """
    rng = np.random.default_rng(seed)
    future = panel.index > cut
    steps = rng.normal(0.0, 0.02, size=(int(future.sum()), panel.shape[1]))

    poisoned = panel.copy()
    poisoned.loc[future] = panel.loc[cut].to_numpy() * np.exp(steps.cumsum(axis=0))
    return poisoned


def test_future_prices_cannot_change_past_returns(
    panel: pd.DataFrame, config: BacktestConfig
) -> None:
    """Rewrite everything after a cut date and the history before it must not move.

    Bit-identical, not approximately equal: a tolerance here would let a small leak
    through, and there is no such thing as a small amount of lookahead.
    """
    cut = panel.index[CUT_SESSION]
    poisoned = rewrite_the_future(panel, cut, seed=31337)

    clean = run_backtest(panel, Momentum(config.strategy), config)
    dirty = run_backtest(poisoned, Momentum(config.strategy), config)

    pd.testing.assert_frame_equal(clean.daily.loc[:cut], dirty.daily.loc[:cut], check_exact=True)
    pd.testing.assert_frame_equal(
        clean.weights.loc[:cut], dirty.weights.loc[:cut], check_exact=True
    )


def test_the_lookahead_test_can_actually_fail(panel: pd.DataFrame, config: BacktestConfig) -> None:
    """Positive control. Without this, the test above would pass on an engine that
    ignored prices entirely."""
    cut = panel.index[CUT_SESSION]
    poisoned = rewrite_the_future(panel, cut, seed=31337)

    clean = run_backtest(panel, Momentum(config.strategy), config)
    dirty = run_backtest(poisoned, Momentum(config.strategy), config)

    after = clean.daily.index > cut
    assert not clean.daily.loc[after].equals(dirty.daily.loc[after])
    assert not clean.target_weights.loc[cut:].equals(dirty.target_weights.loc[cut:])


def test_a_book_only_uses_prices_up_to_its_signal_date(
    panel: pd.DataFrame, config: BacktestConfig, result: BacktestResult
) -> None:
    """Recompute two books from a panel truncated at the signal date. Same answer.

    The corruption test above catches a leak once it changes an outcome. This one goes at
    the strategy directly: given strictly less data, it has to produce the same book.
    """
    sessions = panel.index
    mapped = apply_execution_lag(sessions, rebalance_dates(sessions, "M"), lag=1)
    strategy = Momentum(config.strategy)

    for signal_date in [mapped.index[3], mapped.index[-2]]:
        truncated = strategy.target_weights(panel.loc[:signal_date], signal_date)
        traded = result.target_weights.loc[mapped[signal_date]]
        pd.testing.assert_series_equal(
            truncated.reindex(traded.index).fillna(0.0), traded, check_names=False
        )
