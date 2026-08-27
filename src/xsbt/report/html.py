"""Turns a saved run into a single self-contained HTML page and a metrics.json twin.

The page is built to answer one question a PM actually asks: is this edge real, and can
it be traded after costs. Every section is there because it changes that answer.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from xsbt import __version__
from xsbt.analytics.attribution import (
    LegAttribution,
    MarketFit,
    NameAttribution,
    attribute_legs,
    attribute_names,
    fit_market,
)
from xsbt.analytics.metrics import (
    PerformanceSummary,
    SearchDeflation,
    deflate,
    summarise,
    yearly_returns,
)
from xsbt.analytics.sensitivity import (
    DEFAULT_COST_LEVELS,
    breakeven_cost_bps,
    cost_sweep,
    parameter_grid,
)
from xsbt.config import StrategyConfig
from xsbt.data.cache import write_json_atomic
from xsbt.engine.backtest import BacktestResult
from xsbt.report import plots

log = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent
TEMPLATE_NAME = "template.html"

#: Shown in the report footer. The long version lives in docs/ASSUMPTIONS.md.
CAVEATS: tuple[tuple[str, str], ...] = (
    (
        "Survivorship",
        "The universe is a fixed list of names that are liquid today, so it quietly "
        "excludes anything that was delisted or acquired during the sample. This flatters "
        "the long book and is the largest single bias in these numbers.",
    ),
    (
        "Restated history",
        "Yahoo rewrites adjusted close every time a dividend or split lands, so the same "
        "query run a month apart gives different history. The run is pinned to a cache "
        "snapshot id so at least it is reproducible against the bytes we actually saw.",
    ),
    (
        "Costs",
        "A flat charge per unit of notional traded, applied as a drag on the daily return. "
        "There is no spread model, no market impact and no capacity limit, so the true "
        "cost rises with size in a way this does not capture.",
    ),
    (
        "Shorting",
        "Borrow is assumed available and free, and the short rebate is not credited. Both "
        "assumptions are generous; the hardest names to rank short are usually the hardest "
        "to borrow.",
    ),
    (
        "Execution",
        "Signals are formed on the rebalance close and the book is put on at a later close "
        "at that close's price, with no slippage between the decision and the fill.",
    ),
    (
        "Significance",
        "The Sharpe standard error starts from Lo (2002) and is then rescaled by a "
        "Newey-West factor over one holding period, because carrying the same book for a "
        "month leaves the daily returns correlated. Both bars are shown, so the size of "
        "that adjustment is visible. The deflated Sharpe then charges for the parameter "
        "grid on top of that, though it treats neighbouring cells as separate tries when "
        "they are close to the same strategy, so read it as a floor on the penalty.",
    ),
)


@dataclass(frozen=True)
class ReportData:
    """Everything the page renders and everything metrics.json holds."""

    result: BacktestResult
    net: PerformanceSummary
    gross: PerformanceSummary
    legs: LegAttribution
    #: None when the report was generated without the price panel to total names over.
    names: NameAttribution | None
    #: None when the run has no benchmark, or too little overlap to fit one.
    market: MarketFit | None
    cost_sensitivity: pd.DataFrame
    breakeven_bps: float
    #: None when the report was generated without prices to re-run on.
    grid: pd.DataFrame | None
    #: None when there is no grid, so nothing to charge the search against.
    deflation: SearchDeflation | None
    yearly: pd.DataFrame

    def as_dict(self) -> dict[str, Any]:
        """The machine-readable twin of the page.

        run_utc is deliberately dropped: this file is the numbers, and two runs off the
        same snapshot should diff clean. The wall clock lives in metadata.json beside it.
        """
        provenance = {k: v for k, v in self.result.metadata.items() if k != "run_utc"}
        return {
            "run": provenance,
            "config": self.result.config.as_dict(),
            "net": self.net.as_dict(),
            "gross": self.gross.as_dict(),
            "legs": self.legs.as_dict(),
            "names": self.names.as_dict() if self.names is not None else None,
            "market": self.market.as_dict() if self.market is not None else None,
            "cost_sensitivity": _records(self.cost_sensitivity),
            "breakeven_cost_bps": _json_value(self.breakeven_bps),
            "parameter_grid": _grid_records(self.grid),
            "search_deflation": (self.deflation.as_dict() if self.deflation is not None else None),
            "yearly_returns": _records(self.yearly),
        }


def analyse(
    result: BacktestResult,
    *,
    prices: pd.DataFrame | None = None,
    dollar_volume: pd.DataFrame | None = None,
    cost_levels: Sequence[float] | None = None,
    include_grid: bool = True,
) -> ReportData:
    """Run every diagnostic the report shows.

    Args:
        result: a finished backtest.
        prices: the panel the run used. Supply it for the two sections that need the
            underlying names: per-name attribution and the parameter grid. Without it
            they are dropped rather than faked.
        dollar_volume: matching volume panel, so grid cells apply the same liquidity floor.
        cost_levels: bps levels for the cost sweep. Defaults to a standard ladder plus
            whatever this run assumed.
        include_grid: the grid re-runs the backtest a few dozen times, which is by far
            the slowest thing here. Turning it off leaves the rest of the page intact.
    """
    portfolio = result.config.portfolio
    hurdle = portfolio.risk_free_rate
    lags = holding_horizon(result)

    net = summarise(
        result.returns,
        risk_free_rate=hurdle,
        turnover=result.turnover,
        costs=result.costs,
        hac_lags=lags,
    )
    gross = summarise(
        result.gross_returns,
        risk_free_rate=hurdle,
        turnover=result.turnover,
        hac_lags=lags,
    )

    levels = cost_ladder(portfolio.cost_bps) if cost_levels is None else tuple(cost_levels)

    market = None
    benchmark = result.benchmark
    if benchmark is not None:
        try:
            market = fit_market(result.returns, benchmark, risk_free_rate=hurdle)
        except ValueError as exc:
            log.warning("skipping the market fit: %s", exc)

    grid = None
    names = None
    if prices is not None:
        names = attribute_names(
            result.weights, prices.pct_change(fill_method=None).reindex_like(result.weights)
        )
        if include_grid:
            strategy = result.config.strategy
            grid = parameter_grid(
                prices,
                result.config,
                lookbacks=grid_lookbacks(strategy),
                top_fractions=grid_fractions(strategy),
                dollar_volume=dollar_volume,
            )

    # Only meaningful with a grid, because the grid is where the count of configurations
    # tried and the spread between them comes from. No grid, no search to charge for.
    deflation = None
    if grid is not None:
        deflation = deflate(
            result.returns, grid.to_numpy(dtype=float).ravel(), risk_free_rate=hurdle
        )

    return ReportData(
        result=result,
        net=net,
        gross=gross,
        legs=attribute_legs(result.legs),
        names=names,
        market=market,
        cost_sensitivity=cost_sweep(result, levels, hac_lags=lags),
        breakeven_bps=breakeven_cost_bps(result),
        grid=grid,
        deflation=deflation,
        yearly=_yearly_table(result),
    )


def holding_horizon(result: BacktestResult) -> int | None:
    """Sessions the book is held for between rebalances, taken from the run itself.

    This is the bandwidth the Sharpe error bar is widened over. The autocorrelation in a
    daily series like this one comes from carrying the same positions across a whole
    rebalance period, so the window that matters is that horizon, not a rule of thumb
    that knows nothing about how the book is traded. None falls back to the rule.
    """
    scheduled = int(result.metadata.get("rebalances_scheduled") or 0)
    if scheduled < 1 or result.daily.empty:
        return None
    return max(1, round(len(result.daily) / scheduled))


def cost_ladder(configured: float) -> tuple[float, ...]:
    """The standard ladder, plus whatever this run actually assumed."""
    return tuple(sorted({*DEFAULT_COST_LEVELS, float(configured)}))


def grid_lookbacks(strategy: StrategyConfig) -> tuple[int, ...]:
    """Lookbacks either side of the configured one.

    Scaled rather than fixed, so the grid brackets the live setting whether it is a
    one-month reversal or a twelve-month momentum, and the configured value is always
    one of the rows. Anything too short to clear the skip is dropped.
    """
    scaled = {max(2, round(strategy.lookback_days * m)) for m in (0.5, 0.75, 1.0, 1.5, 2.0)}
    return tuple(sorted(n for n in scaled if n > strategy.skip_days))


def grid_fractions(strategy: StrategyConfig) -> tuple[float, ...]:
    """Quintile through to a two-way split, always including the configured fraction."""
    return tuple(sorted({0.1, 0.2, 0.3, 0.4, round(strategy.top_fraction, 4)}))


def significance_note(
    summary: PerformanceSummary,
    deflation: SearchDeflation | None = None,
) -> str:
    """One honest sentence about whether the Sharpe is distinguishable from luck."""
    t = summary.sharpe_tstat_hac
    if t is None or not math.isfinite(t):
        return "There is not enough data here to test the Sharpe against zero."

    if abs(t) < 1.0:
        verdict = "is indistinguishable from zero over this sample"
    elif abs(t) < 2.0:
        verdict = "is suggestive but short of the usual two-standard-error bar"
    elif t > 0:
        verdict = "clears the usual two-standard-error bar"
    else:
        verdict = "is significantly negative, which is a result but not a tradeable one"

    note = (
        f"Net Sharpe of {summary.sharpe:.2f} with a standard error of "
        f"{summary.sharpe_se_hac:.2f} over {summary.years:.1f} years gives t = {t:.2f}, so the "
        f"edge {verdict}. That error bar is adjusted for autocorrelation over "
        f"{summary.hac_lags} sessions"
    )
    if deflation is None or not math.isfinite(deflation.deflated):
        return f"{note}, and it still charges nothing for the parameter combinations tried."
    return (
        f"{note}. Charging for the {deflation.trials} configurations searched on the way "
        f"here leaves a {deflation.deflated:.0%} chance the true Sharpe is above zero."
    )


def render(data: ReportData) -> str:
    """Render the page to a string. No files touched, which keeps it easy to test."""
    result = data.result
    figures = {
        "equity": plots.equity(result),
        "drawdown": plots.drawdown(result),
        "rolling_sharpe": plots.rolling_sharpe_plot(result),
        "yearly": plots.yearly_bars(result),
        "monthly": plots.monthly_histogram(result),
        "legs": plots.leg_contribution(result),
        "exposures": plots.exposures(result),
        "cost": plots.cost_curve(
            data.cost_sensitivity,
            configured=result.config.portfolio.cost_bps,
            breakeven=data.breakeven_bps,
        ),
    }
    if data.grid is not None:
        figures["grid"] = plots.parameter_heatmap(data.grid)

    template = _environment().get_template(TEMPLATE_NAME)
    return template.render(
        config=result.config,
        meta=result.metadata,
        version=__version__,
        net=data.net,
        gross=data.gross,
        legs=data.legs,
        names=data.names,
        best_names=_records(data.names.best()) if data.names is not None else [],
        worst_names=_records(data.names.worst()) if data.names is not None else [],
        market=data.market,
        costs=_records(data.cost_sensitivity),
        breakeven=data.breakeven_bps,
        yearly=_records(data.yearly),
        has_benchmark=result.benchmark is not None,
        figures=figures,
        deflation=data.deflation,
        verdict=significance_note(data.net, data.deflation),
        caveats=CAVEATS,
    )


def write_html(data: ReportData, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(data), encoding="utf-8")
    return path


def write_metrics(data: ReportData, path: Path | str) -> Path:
    path = Path(path)
    write_json_atomic(path, data.as_dict())
    return path


def write_returns(result: BacktestResult, path: Path | str) -> Path:
    """The daily frame as CSV, for anyone who would rather use their own tools.

    Fixed float formatting so that two runs off one snapshot produce identical bytes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.daily.to_csv(path, float_format="%.10g")
    return path


def _environment() -> Environment:
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["pct"] = _pct
    env.filters["num"] = _num
    env.filters["signed"] = _signed_pct
    return env


def _yearly_table(result: BacktestResult) -> pd.DataFrame:
    frame = pd.DataFrame({"strategy": yearly_returns(result.returns)})
    benchmark = result.benchmark
    if benchmark is not None:
        frame["benchmark"] = yearly_returns(benchmark)
    frame.index = pd.Index([stamp.year for stamp in frame.index], name="year")
    return frame


def _pct(value: float | None, digits: int = 2) -> str:
    number = _finite(value)
    return "n/a" if number is None else f"{number * 100:.{digits}f}%"


def _signed_pct(value: float | None, digits: int = 2) -> str:
    number = _finite(value)
    return "n/a" if number is None else f"{number * 100:+.{digits}f}%"


def _num(value: float | None, digits: int = 2) -> str:
    number = _finite(value)
    if number is None:
        return "n/a"
    if math.isinf(number):
        return "unbounded"
    return f"{number:,.{digits}f}"


def _finite(value: Any) -> float | None:
    """None for anything that would render as 'nan', infinity kept so it can be labelled."""
    if value is None:
        return None
    number = float(value)
    return None if math.isnan(number) else number


def _json_value(value: Any) -> Any:
    """NaN and infinity are Python floats but not valid JSON literals."""
    if isinstance(value, float | np.floating):
        return float(value) if math.isfinite(value) else None
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    reset = frame.reset_index()
    return [{k: _json_value(v) for k, v in row.items()} for row in reset.to_dict("records")]


def _grid_records(grid: pd.DataFrame | None) -> list[dict[str, Any]] | None:
    if grid is None:
        return None
    return [
        {
            "lookback_days": int(lookback),
            "top_fraction": float(fraction),
            "sharpe": _json_value(grid.loc[lookback, fraction]),
        }
        for lookback in grid.index
        for fraction in grid.columns
    ]
