"""Portfolio accounting, checked against numbers worked out by hand.

The worked example runs two names over four sessions:

              A       B
    day 0   +10%     0%     book goes on at the close: A +0.5, B -0.5
    day 1     0%   +10%
    day 2   -10%     0%
    day 3     0%   -10%

Holding into day 1 the book is (+0.5, -0.5), so it earns -0.5 * 0.10 = -5.0%. NAV is
0.95 and the weights drift to (0.5/0.95, -0.55/0.95) = (10/19, -11/19). Day 2 earns
-(10/19) * 0.10 = -1/19, NAV falls to 0.95 * 18/19 = 0.90 and the weights drift to
(0.5, -11/18). Day 3 earns (11/18) * 0.10 = 11/180, so gross equity finishes at
0.90 * 191/180 = 0.955.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xsbt.engine.costs import LinearCostModel
from xsbt.engine.portfolio import exposures, leg_returns, simulate

DATES = pd.bdate_range("2021-01-04", periods=4, name="date")


@pytest.fixture
def returns() -> pd.DataFrame:
    return pd.DataFrame(
        {"A": [0.10, 0.00, -0.10, 0.00], "B": [0.00, 0.10, 0.00, -0.10]},
        index=DATES,
    )


@pytest.fixture
def targets() -> pd.DataFrame:
    return pd.DataFrame({"A": [0.5], "B": [-0.5]}, index=DATES[:1])


def test_hand_computed_gross_path(returns: pd.DataFrame, targets: pd.DataFrame) -> None:
    path = simulate(returns, targets, LinearCostModel(0.0))

    expected = [0.0, -0.05, -1 / 19, 11 / 180]
    np.testing.assert_allclose(path.gross_returns.to_numpy(), expected, rtol=1e-12)
    assert path.gross_returns.add(1.0).prod() == pytest.approx(0.955, rel=1e-12)


def test_hand_computed_weight_drift(returns: pd.DataFrame, targets: pd.DataFrame) -> None:
    path = simulate(returns, targets, LinearCostModel(0.0))

    # Weights are the ones held *into* each session.
    np.testing.assert_allclose(path.weights.loc[DATES[0]].to_numpy(), [0.0, 0.0])
    np.testing.assert_allclose(path.weights.loc[DATES[1]].to_numpy(), [0.5, -0.5])
    np.testing.assert_allclose(path.weights.loc[DATES[2]].to_numpy(), [10 / 19, -11 / 19])
    np.testing.assert_allclose(path.weights.loc[DATES[3]].to_numpy(), [0.5, -11 / 18])


def test_gross_exposure_drifts_away_from_target(
    returns: pd.DataFrame, targets: pd.DataFrame
) -> None:
    """A short that moves against you grows. If gross stayed pinned at 1.0 the engine
    would be silently rebalancing every day."""
    path = simulate(returns, targets, LinearCostModel(0.0))

    gross = exposures(path.weights)["gross"]

    assert gross.loc[DATES[1]] == pytest.approx(1.0)
    assert gross.loc[DATES[2]] == pytest.approx(21 / 19)
    assert gross.loc[DATES[2]] > 1.0


def test_first_trade_turns_over_the_whole_book(
    returns: pd.DataFrame, targets: pd.DataFrame
) -> None:
    path = simulate(returns, targets, LinearCostModel(0.0))

    assert path.turnover.loc[DATES[0]] == pytest.approx(1.0)
    assert path.turnover.iloc[1:].eq(0.0).all()


def test_zero_cost_means_net_equals_gross(returns: pd.DataFrame, targets: pd.DataFrame) -> None:
    path = simulate(returns, targets, LinearCostModel(0.0))

    pd.testing.assert_series_equal(path.net_returns, path.gross_returns, check_names=False)


def test_cost_is_charged_on_notional_traded(returns: pd.DataFrame, targets: pd.DataFrame) -> None:
    path = simulate(returns, targets, LinearCostModel(10.0))

    # 1.0 of notional at 10bps.
    assert path.costs.loc[DATES[0]] == pytest.approx(0.001)
    assert path.costs.sum() == pytest.approx(0.001)
    assert path.net_returns.loc[DATES[0]] == pytest.approx(-0.001)


def test_rebalance_charges_only_the_drift(returns: pd.DataFrame) -> None:
    """Re-stating the same target after one day should cost the drift, not the book."""
    targets = pd.DataFrame({"A": [0.5, 0.5], "B": [-0.5, -0.5]}, index=[DATES[0], DATES[1]])

    path = simulate(returns, targets, LinearCostModel(0.0))

    # Drift after day 1 is (10/19, -11/19); coming back to (0.5, -0.5) trades the gap.
    expected = abs(0.5 - 10 / 19) + abs(-0.5 + 11 / 19)
    assert path.turnover.loc[DATES[1]] == pytest.approx(expected)


def test_missing_returns_do_not_move_the_book() -> None:
    """A halted name has a NaN return. It must not poison the whole day's P&L."""
    returns = pd.DataFrame({"A": [0.0, 0.10, 0.0, 0.0], "B": [0.0, np.nan, 0.0, 0.0]}, index=DATES)
    targets = pd.DataFrame({"A": [0.5], "B": [-0.5]}, index=DATES[:1])

    path = simulate(returns, targets, LinearCostModel(0.0))

    assert path.gross_returns.loc[DATES[1]] == pytest.approx(0.05)
    assert path.gross_returns.notna().all()


def test_leg_attribution_splits_the_day(returns: pd.DataFrame, targets: pd.DataFrame) -> None:
    path = simulate(returns, targets, LinearCostModel(0.0))

    legs = leg_returns(path.weights, returns)

    np.testing.assert_allclose(
        legs.sum(axis=1).to_numpy(), path.gross_returns.to_numpy(), atol=1e-15
    )
    assert legs.loc[DATES[1], "long"] == pytest.approx(0.0)
    assert legs.loc[DATES[1], "short"] == pytest.approx(-0.05)


def test_exposures_report_a_neutral_book(returns: pd.DataFrame, targets: pd.DataFrame) -> None:
    path = simulate(returns, targets, LinearCostModel(0.0))

    book = exposures(path.weights)

    assert book.loc[DATES[1], "net"] == pytest.approx(0.0)
    assert book.loc[DATES[1], "long_names"] == 1
    assert book.loc[DATES[1], "short_names"] == 1


def test_target_outside_the_return_index_is_rejected(returns: pd.DataFrame) -> None:
    targets = pd.DataFrame({"A": [0.5], "B": [-0.5]}, index=[pd.Timestamp("2019-01-02")])

    with pytest.raises(KeyError, match="outside the return index"):
        simulate(returns, targets, LinearCostModel(0.0))


def test_no_targets_is_rejected(returns: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="nothing to simulate"):
        simulate(returns, pd.DataFrame(), LinearCostModel(0.0))


def test_wipeout_raises_rather_than_returning_nonsense() -> None:
    returns = pd.DataFrame({"A": [0.0, -1.5]}, index=DATES[:2])
    targets = pd.DataFrame({"A": [1.0]}, index=DATES[:1])

    with pytest.raises(ValueError, match="wiped out"):
        simulate(returns, targets, LinearCostModel(0.0))


def test_negative_cost_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        LinearCostModel(-1.0)
