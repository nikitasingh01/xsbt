"""Property tests: the invariants the rest of the code quietly relies on.

The tests elsewhere pin one worked example each, computed by hand, which is the right way
to check arithmetic. These check the statements that have to hold for *every* input, and
those are the ones a refactor breaks quietly: the book is dollar neutral, costs are linear
in the rate, net return falls as costs rise.

Deliberately small. Hypothesis is building panels and running the simulator, so the example
counts are tuned to stay inside a few seconds rather than to sweep the space.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from xsbt.config import StrategyConfig
from xsbt.engine.costs import LinearCostModel
from xsbt.engine.portfolio import simulate
from xsbt.strategies import build
from xsbt.strategies.base import CrossSectionalRankStrategy

# No per-example deadline: Windows timing is noisy enough that one produces flaky failures
# on work that is genuinely fast. The example count is the budget instead.
QUICK = settings(max_examples=60, deadline=None)
PANEL = settings(max_examples=25, deadline=None)

# Drawn off a grid rather than from the full float range. Two floats a single ULP apart are
# distinct until something rescales them, and then they tie; that is a property of floats,
# not a bug in the ranking, and it would show up here as noise.
ticks = st.integers(min_value=-10_000, max_value=10_000).map(lambda n: n / 1000.0)
score_lists = st.lists(ticks, min_size=2, max_size=30, unique=True)
top_fractions = st.floats(min_value=0.05, max_value=0.5)
# Returns in basis points, capped at 5% a day. Books here are gross 1.0, so the worst
# session moves NAV by 5% and the simulator's wipeout guard is never the thing under test.
daily_returns = st.integers(min_value=-500, max_value=500).map(lambda n: n / 10_000.0)


def rank_strategy(top_fraction: float, name: str = "momentum") -> CrossSectionalRankStrategy:
    strategy = build(StrategyConfig(name=name, lookback_days=20, top_fraction=top_fraction))
    assert isinstance(strategy, CrossSectionalRankStrategy)
    return strategy


def score_series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=[f"T{i:02d}" for i in range(len(values))], dtype="float64")


# --------------------------------------------------------------------------------------
# The book a rank strategy produces
# --------------------------------------------------------------------------------------


@given(values=score_lists, top_fraction=top_fractions)
@QUICK
def test_the_book_is_dollar_neutral_at_gross_one(values: list[float], top_fraction: float) -> None:
    """The two things the engine assumes when it scales the book to its leverage.

    Break either and every reported number becomes a number about a different portfolio,
    with nothing downstream noticing.
    """
    weights = rank_strategy(top_fraction).weights_from_scores(score_series(values))

    assert weights.sum() == pytest.approx(0.0, abs=1e-12)
    assert weights.abs().sum() == pytest.approx(1.0)


@given(values=score_lists, top_fraction=top_fractions)
@QUICK
def test_the_legs_are_the_same_size_and_equally_weighted(
    values: list[float], top_fraction: float
) -> None:
    weights = rank_strategy(top_fraction).weights_from_scores(score_series(values))

    longs, shorts = weights[weights > 0], weights[weights < 0]
    assert len(longs) == len(shorts) >= 1
    # Equal weighted means one position size per side, not merely a similar one.
    assert longs.nunique() == shorts.nunique() == 1
    assert longs.iloc[0] == pytest.approx(-shorts.iloc[0])


@given(
    values=score_lists,
    top_fraction=top_fractions,
    scale=st.floats(min_value=0.1, max_value=5.0),
    shift=st.floats(min_value=-5.0, max_value=5.0),
)
@QUICK
def test_a_rank_strategy_reads_the_ordering_and_nothing_else(
    values: list[float], top_fraction: float, scale: float, shift: float
) -> None:
    """Rescaling the scores must not move a single position.

    This is the assumption the no-lookahead test leans on: it corrupts future prices and
    expects past books unchanged, which only means anything if the book is a function of
    the ranking rather than of the levels.
    """
    strategy = rank_strategy(top_fraction)
    scores = score_series(values)

    plain = strategy.weights_from_scores(scores)
    rescaled = strategy.weights_from_scores(scores * scale + shift)

    pd.testing.assert_series_equal(plain, rescaled)


@given(values=score_lists, top_fraction=top_fractions)
@QUICK
def test_flipping_the_scores_swaps_the_legs(values: list[float], top_fraction: float) -> None:
    """Momentum and reversal differ by a minus sign, and test_strategies.py holds them to
    it on one panel. This is the same claim over arbitrary scores."""
    scores = score_series(values)
    strategy = rank_strategy(top_fraction)

    pd.testing.assert_series_equal(
        strategy.weights_from_scores(scores),
        -strategy.weights_from_scores(-scores),
    )


# --------------------------------------------------------------------------------------
# Portfolio accounting
# --------------------------------------------------------------------------------------


@st.composite
def books(draw: st.DrawFn) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A return panel, plus target books on some of its sessions.

    Targets come out of the strategy rather than being drawn freely, so every book the
    simulator sees is one the engine could actually have handed it.
    """
    n_names = draw(st.integers(min_value=2, max_value=6))
    n_days = draw(st.integers(min_value=3, max_value=12))
    tickers = [f"T{i:02d}" for i in range(n_names)]
    dates = pd.bdate_range("2021-01-04", periods=n_days, name="date")

    cells = draw(st.lists(daily_returns, min_size=n_days * n_names, max_size=n_days * n_names))
    returns = pd.DataFrame(np.asarray(cells).reshape(n_days, n_names), index=dates, columns=tickers)

    strategy = rank_strategy(draw(top_fractions))
    sessions = draw(
        st.lists(st.integers(min_value=0, max_value=n_days - 1), min_size=1, unique=True)
    )
    rows = {}
    for day in sorted(sessions):
        ordering = draw(st.lists(ticks, min_size=n_names, max_size=n_names, unique=True))
        rows[dates[day]] = strategy.weights_from_scores(pd.Series(ordering, index=tickers))

    return returns, pd.DataFrame(rows).T.rename_axis("date")


@given(book=books(), cost_bps=st.floats(min_value=0.0, max_value=100.0))
@PANEL
def test_costs_are_the_only_gap_between_gross_and_net(
    book: tuple[pd.DataFrame, pd.DataFrame], cost_bps: float
) -> None:
    returns, targets = book
    path = simulate(returns, targets, LinearCostModel(cost_bps))

    pd.testing.assert_series_equal(
        path.gross_returns - path.costs, path.net_returns, check_names=False
    )
    assert (path.costs >= 0.0).all()
    assert (path.turnover >= 0.0).all()
    # Costs are charged on trades, so a session that did not trade cannot carry one.
    assert (path.costs[path.turnover == 0.0] == 0.0).all()


@given(book=books())
@PANEL
def test_free_trading_leaves_net_equal_to_gross(book: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    returns, targets = book
    path = simulate(returns, targets, LinearCostModel(0.0))

    pd.testing.assert_series_equal(path.gross_returns, path.net_returns, check_names=False)


@given(
    book=books(),
    cheap=st.floats(min_value=0.0, max_value=40.0),
    extra=st.floats(min_value=1.0, max_value=60.0),
)
@PANEL
def test_net_return_falls_as_costs_rise(
    book: tuple[pd.DataFrame, pd.DataFrame], cheap: float, extra: float
) -> None:
    """The assumption the breakeven number rests on.

    analytics/sensitivity.py bisects for the cost level where CAGR crosses zero, which is
    only a valid search if net return is monotonic in the rate. Nothing else tests that,
    and a spread-aware cost model added later could break it without touching sensitivity.py.
    """
    returns, targets = book
    cheap_path = simulate(returns, targets, LinearCostModel(cheap))
    dear_path = simulate(returns, targets, LinearCostModel(cheap + extra))

    assert (dear_path.net_returns <= cheap_path.net_returns).all()
    # And strictly worse wherever something actually traded.
    traded = cheap_path.turnover > 0.0
    assume(traded.any())
    assert (dear_path.net_returns[traded] < cheap_path.net_returns[traded]).all()


@given(book=books())
@PANEL
def test_the_day_earns_on_the_book_held_into_it(
    book: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """``weights`` is documented as the book held *into* each session.

    If it ever became the book held out of it, every return would land a day early. That
    is lookahead wearing an accounting disguise, and it would not look wrong anywhere else.
    """
    returns, targets = book
    path = simulate(returns, targets, LinearCostModel(5.0))

    earned = (path.weights * returns).sum(axis=1)

    pd.testing.assert_series_equal(earned, path.gross_returns, check_names=False)
    # Nothing is held into the first session, so it earns nothing whatever prices did.
    assert path.weights.iloc[0].eq(0.0).all()
    assert path.gross_returns.iloc[0] == 0.0


@given(book=books())
@PANEL
def test_a_target_is_held_in_full_on_the_next_session(
    book: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Trading on date t means the target is what earns on t+1, not on t.

    Also the cleanest statement of where drift starts: the book is exactly on target for
    one session, and from the session after that it is drifting.
    """
    returns, targets = book
    path = simulate(returns, targets, LinearCostModel(0.0))

    sessions = returns.index
    for date, target in targets.iterrows():
        following = sessions.get_loc(date) + 1
        if following >= len(sessions):
            continue  # Traded on the last session, so nothing ever held it.
        pd.testing.assert_series_equal(path.weights.iloc[following], target, check_names=False)


@given(book=books())
@PANEL
def test_the_first_rebalance_trades_the_whole_book(
    book: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Coming off an empty book, turnover is the gross exposure being put on: 1.0 here.

    Worth pinning because turnover is reported annualised, and an engine that skipped the
    opening trade would understate the cost of running the strategy for its whole life.
    """
    returns, targets = book
    path = simulate(returns, targets, LinearCostModel(0.0))

    first = targets.index[0]
    assert path.turnover.loc[first] == pytest.approx(targets.iloc[0].abs().sum())
