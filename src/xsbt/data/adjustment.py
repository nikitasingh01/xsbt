"""Checking the vendor's adjusted close against the events the vendor itself reports.

``adj_close`` is a derived field. Everything downstream keys off it, and if the vendor gets
it wrong the backtest inherits a fake return on every ex-dividend date without anything
looking broken. Since the payload also carries the dividends and splits the adjustment was
built from, the two can be checked against each other, which is cheaper than trusting it.

The arithmetic is the standard back-adjustment. On the session a dividend goes ex, the
price drops by roughly the payout, so history before that session is scaled by

    f = 1 - amount / close on the session before the ex-date

and the factor at any date is the product of every such factor still ahead of it::

    adj_close_t / close_t = prod over ex-dates e > t of f_e

Splits do not appear in that product, and that is not an oversight. Yahoo's ``close`` is
already split-adjusted, and it restates dividend amounts into the same post-split units, so
both sides of the division move together and the split cancels. It still gets checked, just
not separately: a split folded into one series and not the other would put a step into
``adj_close / close`` that no dividend explains, which is exactly what this compares.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

#: Relative agreement demanded between the vendor's factor path and the reconstructed one.
#: Measured rather than guessed. Across the 41 names in the sample universe, 2700-odd
#: dividends and 26 splits, the worst disagreement is 1e-6, which is Yahoo rounding
#: adj_close to about seven significant figures. The smallest real error worth catching is
#: one missed dividend, around 5e-3 on a large cap. 1e-4 sits between the two with room on
#: both sides.
DEFAULT_TOLERANCE = 1e-4


def implied_adjustment(bars: pd.DataFrame) -> pd.Series:
    """Back-adjustment factor path implied by the dividends on the frame.

    Normalised to 1.0 on the last session, which is the convention the vendor uses too.
    """
    close = bars["close"]
    dividends = bars["dividend"]

    factors = pd.Series(1.0, index=bars.index, dtype="float64")
    # The session before the ex-date, because that is the last close still carrying the
    # dividend. On the first bar there is no such close, so that one cannot be checked.
    previous_close = close.shift(1)
    ex_dates = dividends.notna() & previous_close.notna() & (previous_close > 0.0)
    factors[ex_dates] = 1.0 - dividends[ex_dates] / previous_close[ex_dates]

    # Product of everything strictly ahead of each date: cumulative product taken
    # backwards, then shifted off the date itself.
    ahead = factors[::-1].cumprod()[::-1].shift(-1)
    ahead.iloc[-1] = 1.0
    return ahead


@dataclass(frozen=True)
class AdjustmentAudit:
    """Whether one ticker's adjusted close is consistent with its own corporate actions."""

    ticker: str
    sessions: int
    dividends: int
    splits: int
    #: Worst relative gap between the vendor's factor path and the reconstructed one.
    max_error: float
    #: Session the worst gap sits on, which is where to start looking if it fails.
    worst_date: str | None
    #: Adjustment the vendor applied that no event in this window accounts for. Normally
    #: 1.0. Anything else usually means a dividend went ex after the last bar we hold,
    #: which is harmless: a constant scale on every price leaves returns untouched.
    unexplained_level: float
    tolerance: float

    @property
    def ok(self) -> bool:
        return math.isfinite(self.max_error) and self.max_error <= self.tolerance

    def complaint(self) -> str | None:
        """One line naming what disagrees, or None when nothing does."""
        if self.ok:
            return None
        if not math.isfinite(self.max_error):
            return "adjusted close could not be reconstructed from the reported events"
        return (
            f"adjusted close is {self.max_error:.2%} away from its own events "
            f"on {self.worst_date} ({self.dividends} dividends, {self.splits} splits)"
        )


def audit_adjustment(
    ticker: str,
    bars: pd.DataFrame,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> AdjustmentAudit:
    """Rebuild ``adj_close / close`` from the events and compare it against the vendor's.

    Levels are divided out before comparing. The vendor's series carries every adjustment
    it knows about, including ones with ex-dates past the last bar in the window, and we
    only hold the events inside the window. Comparing levels would flag those as errors
    when they are just a constant scale, so the level is reported separately as
    ``unexplained_level`` and the comparison is left to test the shape, which is where a
    genuine mistake would show up.
    """
    if len(bars) < 2:
        return AdjustmentAudit(
            ticker=ticker,
            sessions=len(bars),
            dividends=0,
            splits=0,
            max_error=0.0,
            worst_date=None,
            unexplained_level=1.0,
            tolerance=tolerance,
        )

    vendor = bars["adj_close"] / bars["close"]
    level = float(vendor.iloc[-1])
    implied = implied_adjustment(bars)

    comparable = vendor.notna() & implied.notna() & (vendor != 0.0)
    if not comparable.any() or not math.isfinite(level) or level == 0.0:
        error, worst = float("nan"), None
    else:
        shape = vendor[comparable] / level
        gap = ((implied[comparable] - shape) / shape).abs()
        error = float(gap.max())
        worst = str(pd.Timestamp(gap.idxmax()).date())

    return AdjustmentAudit(
        ticker=ticker,
        sessions=len(bars),
        dividends=int(bars["dividend"].notna().sum()),
        splits=int(bars["split_ratio"].notna().sum()),
        max_error=error,
        worst_date=worst,
        unexplained_level=level if math.isfinite(level) else float("nan"),
        tolerance=tolerance,
    )
