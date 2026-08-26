"""Sensitivity checks: does the result survive costs, and is the config a lucky peak?"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pandas as pd

from xsbt.analytics.metrics import (
    annualised_volatility,
    cagr,
    max_drawdown,
    sharpe_ratio,
    sharpe_tstat,
)
from xsbt.config import BacktestConfig, StrategyConfig
from xsbt.engine.backtest import BacktestResult, run_backtest
from xsbt.strategies import build

log = logging.getLogger(__name__)

DEFAULT_COST_LEVELS: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0, 50.0)


def net_returns_at(result: BacktestResult, cost_bps: float) -> pd.Series:
    """Rebuild the net return series for a different cost level.

    No re-run is needed. Target weights depend only on prices, and the drift between
    rebalances is computed on gross return, so cost enters the engine purely as a drag
    on the daily return. Changing the rate is therefore exact, not an approximation.
    """
    return result.gross_returns - (cost_bps / 1e4) * result.turnover


def cost_sweep(
    result: BacktestResult,
    levels: Sequence[float] = DEFAULT_COST_LEVELS,
    *,
    risk_free_rate: float | None = None,
    hac_lags: int | None = None,
) -> pd.DataFrame:
    """Headline numbers across a range of assumed transaction costs.

    The question this answers is not 'what is the Sharpe' but 'how wrong does my cost
    assumption have to be before this stops being worth trading'.

    ``hac_lags`` has to match whatever the headline block used, or the row at the
    configured cost level quietly disagrees with the card at the top of the page.
    """
    hurdle = result.config.portfolio.risk_free_rate if risk_free_rate is None else risk_free_rate

    rows = []
    for level in levels:
        net = net_returns_at(result, level)
        rows.append(
            {
                "cost_bps": float(level),
                "cagr": cagr(net),
                "ann_volatility": annualised_volatility(net),
                "sharpe": sharpe_ratio(net, hurdle),
                "sharpe_tstat": sharpe_tstat(net, hurdle, hac_lags=hac_lags),
                "max_drawdown": max_drawdown(net).depth,
            }
        )
    return pd.DataFrame(rows).set_index("cost_bps")


def breakeven_cost_bps(result: BacktestResult, *, ceiling: float = 1000.0) -> float:
    """Cost level at which the strategy stops making money.

    Bisected on CAGR rather than solved in closed form, because CAGR compounds and the
    arithmetic answer is optimistic by a few basis points.
    """
    if result.turnover.sum() <= 0.0:
        return float("nan")
    if cagr(net_returns_at(result, 0.0)) <= 0.0:
        return 0.0
    if cagr(net_returns_at(result, ceiling)) > 0.0:
        return float("inf")

    low, high = 0.0, ceiling
    for _ in range(60):
        mid = 0.5 * (low + high)
        if cagr(net_returns_at(result, mid)) > 0.0:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def parameter_grid(
    prices: pd.DataFrame,
    config: BacktestConfig,
    *,
    lookbacks: Sequence[int],
    top_fractions: Sequence[float],
    metric: str = "sharpe",
    dollar_volume: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Re-run the backtest across a lookback x top-fraction grid.

    A single good cell surrounded by bad ones is a fitted parameter, not a signal. Cells
    that cannot run at all (lookback longer than the sample, say) come back as NaN rather
    than taking the whole grid down.
    """
    if metric not in {"sharpe", "cagr", "max_drawdown"}:
        raise ValueError(f"unsupported grid metric {metric!r}")

    grid = pd.DataFrame(
        index=pd.Index(list(lookbacks), name="lookback_days"),
        columns=pd.Index([float(f) for f in top_fractions], name="top_fraction"),
        dtype="float64",
    )

    for lookback in lookbacks:
        for fraction in top_fractions:
            try:
                # The strategy block is rebuilt with model_validate rather than copied:
                # model_copy skips the validators, so a cell whose lookback no longer
                # clears skip_days would reach the strategy as a backwards window instead
                # of being rejected here.
                cell = config.model_copy(
                    update={
                        "strategy": StrategyConfig.model_validate(
                            config.strategy.model_dump()
                            | {
                                "lookback_days": int(lookback),
                                "top_fraction": float(fraction),
                            }
                        )
                    }
                )
                run = run_backtest(prices, build(cell.strategy), cell, dollar_volume=dollar_volume)
            except (ValueError, KeyError) as exc:
                log.warning("grid cell lookback=%s top=%s skipped: %s", lookback, fraction, exc)
                continue

            grid.loc[lookback, float(fraction)] = _score(run, metric, cell)

    return grid


def _score(run: BacktestResult, metric: str, config: BacktestConfig) -> float:
    if metric == "sharpe":
        return sharpe_ratio(run.returns, config.portfolio.risk_free_rate)
    if metric == "cagr":
        return cagr(run.returns)
    return max_drawdown(run.returns).depth
