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

#: A leg's share of P&L is worth printing while it stays inside this range. A share above
#: 1 is meaningful and common: it says the other leg lost money. Past 2, the two legs are
#: mostly cancelling and the ratio has stopped describing the book.
SHARE_RANGE = (-1.0, 2.0)


@dataclass(frozen=True)
class LegAttribution:
    """Long and short contributions to the same daily portfolio return.

    These are contributions, not standalone leg returns: at gross 1.0 each leg carries
    half the book, so the two numbers add up to the portfolio's gross return.
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

    # A long/short book routinely nets a small number out of two large offsetting ones,
    # and dividing by that remainder gives shares like 562% and -462%: arithmetically
    # right, unreadable, and the first thing a PM will call a bug. Outside the readable
    # range the split is dropped, and the report says why. NaN fails the comparison, so
    # a book that made exactly nothing falls through here too.
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
class MarketFit:
    """Regression of strategy excess returns on benchmark excess returns."""

    beta: float
    #: Annualised intercept. This is the number that has to survive.
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
    return {
        key: None if isinstance(v, float) and not math.isfinite(v) else v
        for key, v in values.items()
    }
