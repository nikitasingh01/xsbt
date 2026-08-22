"""Portfolio accounting: what the book earns, what it costs to move it.

The one thing worth reading carefully is the drift. Between rebalances the weights are
not held constant, they move with prices:

    r_p,d   = sum_i w_i,d-1 * r_i,d
    w_i,d   = w_i,d-1 * (1 + r_i,d) / (1 + r_p,d)

Pinning weights instead would quietly assume a daily rebalance back to target, which
both understates turnover and hands the book a free short-horizon mean-reversion bet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from xsbt.engine.costs import CostModel


@dataclass(frozen=True)
class PortfolioPath:
    """Day-by-day result of holding a sequence of target books."""

    gross_returns: pd.Series
    net_returns: pd.Series
    costs: pd.Series
    #: Notional traded as a fraction of NAV. Buying 50% and selling 50% reads as 1.0.
    #: The commonly quoted "one-way turnover" is half of this.
    turnover: pd.Series
    #: Weights held *into* each session, i.e. the ones that earned that day's return.
    weights: pd.DataFrame


def simulate(
    returns: pd.DataFrame,
    targets: pd.DataFrame,
    cost_model: CostModel,
) -> PortfolioPath:
    """Walk the book forward one session at a time.

    Args:
        returns: daily simple returns, date x ticker.
        targets: desired weights, indexed by the session the book is first held. Rows
            need not cover every session; on other days the book just drifts.
        cost_model: charges on notional traded.
    """
    if targets.empty:
        raise ValueError("no target weights: nothing to simulate")

    unknown = targets.index.difference(returns.index)
    if len(unknown):
        raise KeyError(f"target dates outside the return index: {list(unknown[:5])}")

    tickers = returns.columns
    dates = returns.index

    # A halted or not-yet-listed name has a NaN return. It cannot move a book we do not
    # hold, and eligibility keeps us out of names without prices, so zero is right here.
    daily = np.nan_to_num(returns.to_numpy(dtype=float), nan=0.0)
    target_rows = targets.reindex(columns=tickers).fillna(0.0).to_numpy(dtype=float)
    target_at = dict(zip(dates.get_indexer(targets.index), range(len(targets)), strict=True))

    n_days = len(dates)
    held = np.zeros(len(tickers))
    weights = np.zeros((n_days, len(tickers)))
    gross = np.zeros(n_days)
    costs = np.zeros(n_days)
    turnover = np.zeros(n_days)

    for i in range(n_days):
        weights[i] = held
        day = daily[i]

        gross[i] = float(held @ day)
        nav_multiple = 1.0 + gross[i]
        if nav_multiple <= 0.0:
            raise ValueError(
                f"portfolio wiped out on {dates[i].date()} "
                f"(gross return {gross[i]:.4f}); check leverage and inputs"
            )
        drifted = held * (1.0 + day) / nav_multiple

        row = target_at.get(i)
        if row is None:
            held = drifted
        else:
            traded = float(np.abs(target_rows[row] - drifted).sum())
            turnover[i] = traded
            costs[i] = cost_model.charge(traded)
            held = target_rows[row]

    return PortfolioPath(
        gross_returns=pd.Series(gross, index=dates, name="gross_return"),
        net_returns=pd.Series(gross - costs, index=dates, name="net_return"),
        costs=pd.Series(costs, index=dates, name="cost"),
        turnover=pd.Series(turnover, index=dates, name="turnover"),
        weights=pd.DataFrame(weights, index=dates, columns=tickers),
    )


def leg_returns(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Split gross P&L into its long and short halves.

    A PM needs this: an edge that lives entirely in the short book is a different
    proposition to one that does not, because shorts carry borrow and are the first
    thing to become uncoverable.
    """
    aligned = returns.reindex_like(weights).fillna(0.0)
    contribution = weights * aligned
    return pd.DataFrame(
        {
            "long": contribution.where(weights > 0, 0.0).sum(axis=1),
            "short": contribution.where(weights < 0, 0.0).sum(axis=1),
        }
    )


def exposures(weights: pd.DataFrame) -> pd.DataFrame:
    """Gross, net and name count over time. The sanity check on a 'neutral' book."""
    return pd.DataFrame(
        {
            "gross": weights.abs().sum(axis=1),
            "net": weights.sum(axis=1),
            "long_names": (weights > 0).sum(axis=1),
            "short_names": (weights < 0).sum(axis=1),
        }
    )
