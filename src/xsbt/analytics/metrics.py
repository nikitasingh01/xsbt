"""Performance statistics.

Everything here takes a daily simple-return series and is independent of how that
series was produced, so the same functions work on a strategy, a single leg or a
benchmark. Annualisation uses 252 sessions throughout.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS = 252

#: A daily standard deviation below this is cancellation noise around a constant series
#: rather than risk. pandas returns ~2e-19 for a flat series instead of a clean zero, and
#: dividing by that hands a PM a Sharpe ratio of 7e16.
FLAT_VOL = 1e-15


def daily_risk_free(annual_rate: float) -> float:
    """Compound an annual rate down to a per-session rate."""
    return float((1.0 + annual_rate) ** (1.0 / TRADING_DAYS) - 1.0)


def equity_curve(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()


def total_return(returns: pd.Series) -> float:
    if returns.empty:
        return float("nan")
    return float(equity_curve(returns).iloc[-1] - 1.0)


def years(returns: pd.Series) -> float:
    return len(returns) / TRADING_DAYS


def cagr(returns: pd.Series) -> float:
    """Geometric annual growth rate, counted in sessions rather than calendar days.

    Calendar dating would make a backtest that happens to end on a Monday look different
    from one that ends on a Friday.
    """
    span = years(returns)
    if span <= 0:
        return float("nan")
    ending = 1.0 + total_return(returns)
    if ending <= 0.0:
        return -1.0
    return float(ending ** (1.0 / span) - 1.0)


def annualised_volatility(returns: pd.Series) -> float:
    if len(returns) < 2:
        return float("nan")
    return float(returns.std(ddof=1) * math.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    excess = returns.dropna() - daily_risk_free(risk_free_rate)
    if len(excess) < 2:
        return float("nan")
    sigma = excess.std(ddof=1)
    if not sigma > FLAT_VOL:
        return float("nan")
    return float(excess.mean() / sigma * math.sqrt(TRADING_DAYS))


def default_hac_lags(n: int) -> int:
    """Newey-West (1994) automatic bandwidth, ``floor(4 * (n / 100) ** (2/9))``.

    The fallback when the caller has nothing better. A caller who knows how long the book
    is held for does have something better, and should pass it.
    """
    if n < 3:
        return 0
    return max(1, min(int(4.0 * (n / 100.0) ** (2.0 / 9.0)), n - 2))


def newey_west_factor(returns: pd.Series, lags: int) -> float:
    """How much autocorrelation inflates the variance of the sample mean.

        f = 1 + 2 * sum_k (1 - k / (lags + 1)) * rho_k

    The Bartlett weights are what keep the estimate from going negative. Above 1 means
    the series repeats itself, so the effective sample is smaller than the session count
    suggests and every error bar built on that count is too tight. Below 1 means it
    alternates, and the iid bar was the conservative one.
    """
    sample = returns.dropna()
    n = len(sample)
    if lags < 1 or n < 3:
        return 1.0

    centred = sample.to_numpy(dtype=float) - float(sample.mean())
    variance = float(centred @ centred) / n
    if not variance > FLAT_VOL**2:
        return 1.0

    factor = 1.0
    for k in range(1, min(lags, n - 1) + 1):
        autocovariance = float(centred[k:] @ centred[:-k]) / n
        factor += 2.0 * (1.0 - k / (lags + 1.0)) * autocovariance / variance
    return max(factor, 0.0)


def sharpe_standard_error(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    *,
    hac_lags: int | None = None,
) -> float:
    """Standard error of the annualised Sharpe, following Lo (2002).

    The base result assumes returns are iid. They are not: holding one book across a
    whole rebalance period leaves the daily returns correlated with each other. Pass
    ``hac_lags`` to rescale the bar by the Newey-West factor over that many lags. Usually
    that widens it, but a series that alternates day to day gets a tighter bar, and the
    estimator is allowed to say so. A Sharpe quoted without an error bar is a number with
    no scale on it, and one quoted with the wrong error bar is worse.
    """
    sample = returns.dropna()
    n = len(sample)
    if n < 2:
        return float("nan")
    sharpe = sharpe_ratio(sample, risk_free_rate)
    if not math.isfinite(sharpe):
        return float("nan")

    per_period = sharpe / math.sqrt(TRADING_DAYS)
    iid = math.sqrt(TRADING_DAYS * (1.0 + 0.5 * per_period**2) / n)
    if hac_lags is None:
        return float(iid)
    return float(iid * math.sqrt(newey_west_factor(sample, hac_lags)))


def sharpe_tstat(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    *,
    hac_lags: int | None = None,
) -> float:
    """How many standard errors the Sharpe sits above zero. Below ~2, treat it as luck."""
    sharpe = sharpe_ratio(returns, risk_free_rate)
    error = sharpe_standard_error(returns, risk_free_rate, hac_lags=hac_lags)
    if not math.isfinite(sharpe) or not math.isfinite(error) or error == 0.0:
        return float("nan")
    return float(sharpe / error)


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Sharpe with only the downside counted in the denominator.

    The downside deviation divides by the full sample, not by the number of losing days,
    so a strategy that loses rarely is not punished for the small sample.
    """
    excess = returns.dropna() - daily_risk_free(risk_free_rate)
    if len(excess) < 2:
        return float("nan")
    downside = np.minimum(excess.to_numpy(), 0.0)
    deviation = math.sqrt(float(np.mean(downside**2)))
    if not deviation > FLAT_VOL:
        return float("nan")
    return float(excess.mean() / deviation * math.sqrt(TRADING_DAYS))


@dataclass(frozen=True)
class Drawdown:
    """The worst peak-to-trough loss, and how long it took to get out of it."""

    depth: float
    peak: pd.Timestamp | None
    trough: pd.Timestamp | None
    #: None if the sample ends before the previous peak is regained.
    recovered: pd.Timestamp | None
    #: Sessions from the peak to recovery, or to the end of the sample.
    length_sessions: int


def max_drawdown(returns: pd.Series) -> Drawdown:
    if returns.empty:
        return Drawdown(float("nan"), None, None, None, 0)

    equity = equity_curve(returns)
    underwater = equity / equity.cummax() - 1.0
    if underwater.min() == 0.0:
        # Never below a previous high. Reporting a peak and a recovery date for that
        # would just be the first session twice.
        return Drawdown(0.0, None, None, None, 0)

    trough = underwater.idxmin()
    peak = equity.loc[:trough].idxmax()
    tail = equity.loc[trough:]
    back_above = tail.index[tail >= equity.loc[peak]]
    recovered = back_above[0] if len(back_above) else None

    end = recovered if recovered is not None else equity.index[-1]
    length = int(equity.index.get_loc(end) - equity.index.get_loc(peak))

    return Drawdown(float(underwater.min()), peak, trough, recovered, length)


def time_underwater(returns: pd.Series) -> float:
    """Fraction of sessions spent below a previous high."""
    if returns.empty:
        return float("nan")
    equity = equity_curve(returns)
    return float((equity < equity.cummax()).mean())


def calmar_ratio(returns: pd.Series) -> float:
    depth = max_drawdown(returns).depth
    if not math.isfinite(depth) or depth == 0.0:
        return float("nan")
    return float(cagr(returns) / abs(depth))


def value_at_risk(returns: pd.Series, level: float = 0.95) -> float:
    """Daily loss threshold breached (1 - level) of the time. Reported as a negative."""
    sample = returns.dropna()
    if sample.empty:
        return float("nan")
    return float(sample.quantile(1.0 - level))


def conditional_value_at_risk(returns: pd.Series, level: float = 0.95) -> float:
    """Average loss on the days that breach the VaR threshold."""
    sample = returns.dropna()
    if sample.empty:
        return float("nan")
    cutoff = value_at_risk(sample, level)
    tail = sample[sample <= cutoff]
    return float(tail.mean()) if len(tail) else float("nan")


def hit_rate(returns: pd.Series) -> float:
    """Share of non-flat periods that were up. Flat days are excluded rather than
    counted as losses, otherwise a book that is not yet on drags the number down."""
    sample = returns.dropna()
    active = sample[sample != 0.0]
    if active.empty:
        return float("nan")
    return float((active > 0.0).mean())


def monthly_returns(returns: pd.Series) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype="float64")
    return (1.0 + returns.fillna(0.0)).resample("ME").prod() - 1.0


def yearly_returns(returns: pd.Series) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype="float64")
    return (1.0 + returns.fillna(0.0)).resample("YE").prod() - 1.0


def rolling_sharpe(returns: pd.Series, window: int = TRADING_DAYS) -> pd.Series:
    """Trailing Sharpe, for looking at whether the edge is one regime or many.

    Windows with no variation at all (before the first trade goes on, typically) are
    dropped rather than plotted at some enormous value.
    """
    if len(returns) < window:
        return pd.Series(dtype="float64", index=returns.index[:0])
    rolled = returns.rolling(window)
    sigma = rolled.std(ddof=1)
    return (rolled.mean() / sigma.where(sigma > FLAT_VOL) * math.sqrt(TRADING_DAYS)).dropna()


@dataclass(frozen=True)
class PerformanceSummary:
    """The block of numbers a PM reads first."""

    start: str
    end: str
    sessions: int
    years: float

    total_return: float
    cagr: float
    ann_volatility: float

    sharpe: float
    sharpe_se: float
    sharpe_tstat: float
    #: The same Sharpe with the error bar rescaled for autocorrelation. This is the pair
    #: the verdict is keyed to; the iid pair above is kept so the gap is visible.
    sharpe_se_hac: float
    sharpe_tstat_hac: float
    hac_lags: int
    sortino: float
    calmar: float

    max_drawdown: float
    max_drawdown_peak: str | None
    max_drawdown_trough: str | None
    max_drawdown_recovered: str | None
    max_drawdown_sessions: int
    time_underwater: float

    var_95: float
    cvar_95: float
    skew: float
    excess_kurtosis: float
    best_day: float
    worst_day: float

    hit_rate_daily: float
    hit_rate_monthly: float
    best_month: float
    worst_month: float

    ann_turnover: float
    avg_turnover_per_trade: float
    trades: int
    total_cost: float
    ann_cost_drag: float

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe: NaN and inf become None rather than invalid JSON literals."""
        out: dict[str, Any] = {}
        for key, value in asdict(self).items():
            if isinstance(value, float) and not math.isfinite(value):
                out[key] = None
            else:
                out[key] = value
        return out


def summarise(
    returns: pd.Series,
    *,
    risk_free_rate: float = 0.0,
    turnover: pd.Series | None = None,
    costs: pd.Series | None = None,
    hac_lags: int | None = None,
) -> PerformanceSummary:
    """Roll the whole metric set up for one return series.

    Args:
        returns: daily simple returns, net of costs if that is what you want measured.
        risk_free_rate: annualised hurdle for Sharpe and Sortino. Not credited to P&L.
        turnover: daily notional traded as a fraction of NAV, if you have it.
        costs: daily cost drag as a fraction of NAV, if you have it.
        hac_lags: bandwidth for the autocorrelation-adjusted Sharpe error bar. Defaults
            to the Newey-West automatic rule; pass the holding horizon if you know it.
    """
    if returns.empty:
        raise ValueError("no returns to summarise")

    sample = returns.dropna()
    span = max(years(sample), 1e-9)
    drawdown = max_drawdown(sample)
    monthly = monthly_returns(sample)
    lags = default_hac_lags(len(sample)) if hac_lags is None else max(0, int(hac_lags))

    traded = turnover[turnover > 0.0] if turnover is not None else pd.Series(dtype="float64")

    return PerformanceSummary(
        start=str(sample.index[0].date()),
        end=str(sample.index[-1].date()),
        sessions=len(sample),
        years=round(span, 2),
        total_return=total_return(sample),
        cagr=cagr(sample),
        ann_volatility=annualised_volatility(sample),
        sharpe=sharpe_ratio(sample, risk_free_rate),
        sharpe_se=sharpe_standard_error(sample, risk_free_rate),
        sharpe_tstat=sharpe_tstat(sample, risk_free_rate),
        sharpe_se_hac=sharpe_standard_error(sample, risk_free_rate, hac_lags=lags),
        sharpe_tstat_hac=sharpe_tstat(sample, risk_free_rate, hac_lags=lags),
        hac_lags=lags,
        sortino=sortino_ratio(sample, risk_free_rate),
        calmar=calmar_ratio(sample),
        max_drawdown=drawdown.depth,
        max_drawdown_peak=str(drawdown.peak.date()) if drawdown.peak is not None else None,
        max_drawdown_trough=(str(drawdown.trough.date()) if drawdown.trough is not None else None),
        max_drawdown_recovered=(
            str(drawdown.recovered.date()) if drawdown.recovered is not None else None
        ),
        max_drawdown_sessions=drawdown.length_sessions,
        time_underwater=time_underwater(sample),
        var_95=value_at_risk(sample),
        cvar_95=conditional_value_at_risk(sample),
        skew=float(sample.skew()),
        excess_kurtosis=float(sample.kurtosis()),
        best_day=float(sample.max()),
        worst_day=float(sample.min()),
        hit_rate_daily=hit_rate(sample),
        hit_rate_monthly=hit_rate(monthly),
        best_month=float(monthly.max()) if len(monthly) else float("nan"),
        worst_month=float(monthly.min()) if len(monthly) else float("nan"),
        ann_turnover=float(traded.sum() / span) if len(traded) else 0.0,
        avg_turnover_per_trade=float(traded.mean()) if len(traded) else 0.0,
        trades=len(traded),
        total_cost=float(costs.sum()) if costs is not None else 0.0,
        ann_cost_drag=float(costs.sum() / span) if costs is not None else 0.0,
    )
