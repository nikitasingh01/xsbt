"""Where the P&L came from: the two legs, and how much of it is just the market."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from xsbt.analytics.metrics import (
    FLAT_VOL,
    TRADING_DAYS,
    annualised_volatility,
    daily_risk_free,
    hit_rate,
    sharpe_ratio,
)

#: Range over which a leg's share of P&L still describes the book. Above 1 is common and
#: just means the other leg lost money. Past 2 the two legs are mostly cancelling out.
SHARE_RANGE = (-1.0, 2.0)

#: Names shown at each end of the contributor table.
CONTRIBUTOR_ROWS = 5

#: How many names the concentration figure is measured over.
CONCENTRATION_NAMES = 3


@dataclass(frozen=True)
class LegAttribution:
    """Long and short contributions to the same daily portfolio return.

    These are contributions, not standalone leg returns: at gross 1.0 each leg carries
    half the book, so the two numbers add up to the portfolio's gross return.

    The annual figures are arithmetic, ``mean * 252``, because contributions are additive
    day by day and compounding each leg on its own would break the decomposition. The
    headline CAGR is geometric, so the two legs will not sum to it. The gap is roughly
    half the variance plus the cost drag, and the report says as much.
    """

    long_ann_return: float
    short_ann_return: float
    long_ann_volatility: float
    short_ann_volatility: float
    long_sharpe: float
    short_sharpe: float
    long_hit_rate: float
    short_hit_rate: float
    #: Long P&L over total P&L. Outside [0, 1] when one leg lost money, and NaN when the
    #: legs offset so heavily that the ratio stops carrying information.
    long_share_of_pnl: float
    correlation: float

    @property
    def legs_offset(self) -> bool:
        """True when the share of P&L was suppressed because the legs nearly cancel."""
        return math.isnan(self.long_share_of_pnl)

    def as_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


def attribute_legs(legs: pd.DataFrame) -> LegAttribution:
    """Split a run's daily P&L across its long and short books.

    Args:
        legs: two columns, ``long`` and ``short``, of daily contributions.
    """
    if not {"long", "short"} <= set(legs.columns):
        raise KeyError(f"expected long and short columns, got {list(legs.columns)}")

    long_leg = legs["long"].dropna()
    short_leg = legs["short"].dropna()
    if long_leg.empty:
        raise ValueError("no leg returns to attribute")

    long_total = float(long_leg.sum())
    short_total = float(short_leg.sum())
    combined = long_total + short_total

    # A long/short book nets a small number out of two large offsetting ones, so dividing
    # by that remainder gives shares like 562% and -462%: correct, and unreadable. Outside
    # the range the split is dropped and the report says why. NaN fails the comparison, so
    # a book that made exactly nothing lands here too.
    share = long_total / combined if combined != 0.0 else float("nan")
    low, high = SHARE_RANGE
    if not low <= share <= high:
        share = float("nan")

    return LegAttribution(
        long_ann_return=float(long_leg.mean() * TRADING_DAYS),
        short_ann_return=float(short_leg.mean() * TRADING_DAYS),
        long_ann_volatility=annualised_volatility(long_leg),
        short_ann_volatility=annualised_volatility(short_leg),
        long_sharpe=sharpe_ratio(long_leg),
        short_sharpe=sharpe_ratio(short_leg),
        long_hit_rate=hit_rate(long_leg),
        short_hit_rate=hit_rate(short_leg),
        long_share_of_pnl=share,
        correlation=float(long_leg.corr(short_leg)),
    )


@dataclass(frozen=True)
class NameAttribution:
    """Which names the P&L actually came from.

    At eight names a side, one position is an eighth of a leg, and the first thing a PM
    asks about a book that thin is whether the result is a handful of names or the whole
    cross-section. This answers that.
    """

    #: One row per name that ever carried a position, largest contribution first.
    #: Columns: contribution, share_of_gross, sessions_held, avg_abs_weight.
    table: pd.DataFrame
    #: Share of gross P&L from the largest few names, measured on absolute
    #: contributions so it stays inside [0, 1] on a book whose legs offset.
    concentration: float
    concentration_names: int

    @property
    def names_held(self) -> int:
        return len(self.table)

    def best(self, rows: int = CONTRIBUTOR_ROWS) -> pd.DataFrame:
        return self.table.head(rows)

    def worst(self, rows: int = CONTRIBUTOR_ROWS) -> pd.DataFrame:
        """Worst first, so the table reads outward from zero in both directions."""
        return self.table.tail(rows).iloc[::-1]

    def as_dict(self) -> dict[str, Any]:
        return {
            "concentration": _json_safe({"v": self.concentration})["v"],
            "concentration_names": self.concentration_names,
            "names_held": self.names_held,
            "contributions": [
                {"ticker": ticker, **_json_safe(row)}
                for ticker, row in self.table.to_dict("index").items()
            ],
        }


def attribute_names(weights: pd.DataFrame, returns: pd.DataFrame) -> NameAttribution:
    """Total each name's contribution to gross P&L over the sample.

    Args:
        weights: weights held into each session, date x ticker.
        returns: daily simple returns on the same panel.
    """
    aligned = returns.reindex_like(weights).fillna(0.0)
    held = weights != 0.0
    live = held.any(axis=0)
    if not bool(live.any()):
        raise ValueError("no name ever carried a position")

    contribution = (weights * aligned).sum(axis=0)[live]
    magnitude = contribution.abs()
    total = float(magnitude.sum())

    table = pd.DataFrame(
        {
            "contribution": contribution,
            # Share of the absolute pool, not of the net. Dividing by a net that two
            # offsetting legs have already cancelled is what produced the unreadable
            # 562% leg split, and the same trap is waiting one level down here.
            "share_of_gross": magnitude / total if total > 0.0 else magnitude * float("nan"),
            "sessions_held": held.sum(axis=0)[live].astype(int),
            "avg_abs_weight": weights.abs().where(held).mean(axis=0)[live],
        }
    ).sort_values("contribution", ascending=False, kind="stable")
    table.index.name = "ticker"

    top = min(CONCENTRATION_NAMES, len(table))
    return NameAttribution(
        table=table,
        concentration=(
            float(magnitude.nlargest(top).sum() / total) if total > 0.0 else float("nan")
        ),
        concentration_names=top,
    )


@dataclass(frozen=True)
class MarketFit:
    """Regression of strategy excess returns on benchmark excess returns."""

    beta: float
    #: Annualised intercept: the part of the return the market does not explain.
    alpha: float
    r_squared: float
    correlation: float
    #: Annualised volatility of the return left over after taking out beta * market.
    tracking_error: float
    #: alpha / tracking_error. The Sharpe of what the market cannot explain.
    residual_sharpe: float
    observations: int

    def as_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


def fit_market(
    returns: pd.Series,
    benchmark: pd.Series,
    *,
    risk_free_rate: float = 0.0,
) -> MarketFit:
    """Ordinary least squares of the strategy on the benchmark, both over cash.

    A dollar-neutral book should come out near zero beta. If it does not, the ranking is
    picking up something directional and the Sharpe is partly a market call.
    """
    rf = daily_risk_free(risk_free_rate)
    paired = pd.concat({"y": returns, "x": benchmark}, axis=1).dropna()
    if len(paired) < 3:
        raise ValueError(f"need at least 3 overlapping observations, got {len(paired)}")

    y = paired["y"].to_numpy() - rf
    x = paired["x"].to_numpy() - rf

    variance = float(np.var(x, ddof=1))
    if variance == 0.0:
        raise ValueError("benchmark has no variance; cannot fit a beta")

    beta = float(np.cov(y, x, ddof=1)[0, 1] / variance)
    # Residual keeps the intercept in it, so its mean is the daily alpha.
    residual = y - beta * x
    alpha_daily = float(np.mean(residual))
    residual_vol = float(np.std(residual, ddof=1)) * math.sqrt(TRADING_DAYS)
    correlation = float(np.corrcoef(y, x)[0, 1])

    return MarketFit(
        beta=beta,
        alpha=alpha_daily * TRADING_DAYS,
        r_squared=correlation**2,
        correlation=correlation,
        tracking_error=residual_vol,
        residual_sharpe=(
            alpha_daily * TRADING_DAYS / residual_vol if residual_vol > FLAT_VOL else float("nan")
        ),
        observations=len(paired),
    )


def _json_safe(values: dict[str, Any]) -> dict[str, Any]:
    """NaN and infinity are not JSON literals, and numpy scalars are not JSON types."""
    out: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, float | np.floating):
            out[key] = float(value) if math.isfinite(float(value)) else None
        elif isinstance(value, np.integer):
            out[key] = int(value)
        else:
            out[key] = value
    return out
