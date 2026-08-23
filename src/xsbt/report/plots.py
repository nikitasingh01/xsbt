"""Matplotlib figures, returned as base64 PNGs so the report stays a single file."""

from __future__ import annotations

import base64
import io
import math

import matplotlib

# Chosen before pyplot is imported. There is no display in Docker or CI.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from xsbt.analytics.metrics import (
    equity_curve,
    monthly_returns,
    rolling_sharpe,
    yearly_returns,
)
from xsbt.engine.backtest import BacktestResult

WIDE = (9.5, 3.6)
SQUARE = (5.6, 4.2)
DPI = 110

NET = "#1f4e79"
GROSS = "#93b7d8"
BENCH = "#8a8a8a"
SHORT = "#d9822b"
LOSS = "#c0392b"
GAIN = "#2e8b57"
AXIS = "#333333"
GRID = "#dddddd"


def to_data_uri(fig: Figure) -> str:
    """Serialise a figure and close it. Leaking figures blows up memory over a grid."""
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=DPI)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _style(ax: Axes, title: str, ylabel: str = "") -> None:
    ax.set_title(title, fontsize=10, loc="left", color=AXIS)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8)


def _nothing_to_plot(ax: Axes, message: str) -> None:
    """Say so on the figure rather than emitting a blank pane."""
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=9, color=AXIS)
    ax.set_xticks([])
    ax.set_yticks([])


def equity(result: BacktestResult) -> str:
    net = equity_curve(result.returns)
    gross = equity_curve(result.gross_returns)

    fig, ax = plt.subplots(figsize=WIDE)
    ax.plot(gross.index, gross, color=GROSS, lw=1.0, label="gross")
    ax.plot(net.index, net, color=NET, lw=1.5, label="net of costs")

    floor = min(float(net.min()), float(gross.min()))
    benchmark = result.benchmark
    if benchmark is not None:
        market = equity_curve(benchmark)
        ax.plot(market.index, market, color=BENCH, lw=1.0, ls="--", label="benchmark")
        floor = min(floor, float(market.min()))

    label = "growth of 1"
    if floor > 0.0:
        # Log only when every line is positive, which it is unless something is wrong.
        ax.set_yscale("log")
        label = "growth of 1 (log)"

    _style(ax, "Cumulative growth of 1", label)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    return to_data_uri(fig)


def drawdown(result: BacktestResult) -> str:
    curve = equity_curve(result.returns)
    underwater = (curve / curve.cummax() - 1.0) * 100.0

    fig, ax = plt.subplots(figsize=WIDE)
    ax.fill_between(underwater.index, underwater, 0.0, color=LOSS, alpha=0.25)
    ax.plot(underwater.index, underwater, color=LOSS, lw=1.0)

    _style(ax, "Drawdown from previous high, net of costs", "%")
    return to_data_uri(fig)


def rolling_sharpe_plot(result: BacktestResult, window: int = 252) -> str:
    rolled = rolling_sharpe(result.returns, window=window)

    fig, ax = plt.subplots(figsize=WIDE)
    if rolled.empty:
        _nothing_to_plot(ax, f"sample is shorter than {window} sessions")
    else:
        ax.plot(rolled.index, rolled, color=NET, lw=1.2)
        ax.axhline(0.0, color=AXIS, lw=0.8)
        ax.axhline(1.0, color=GAIN, lw=0.8, ls="--")

    _style(ax, f"Rolling {window}-session Sharpe", "Sharpe")
    return to_data_uri(fig)


def yearly_bars(result: BacktestResult) -> str:
    yearly = yearly_returns(result.returns) * 100.0

    fig, ax = plt.subplots(figsize=WIDE)
    if yearly.empty:
        _nothing_to_plot(ax, "no completed periods")
    else:
        ax.bar(
            [str(d.year) for d in yearly.index],
            yearly.to_numpy(),
            color=[GAIN if v >= 0 else LOSS for v in yearly],
            width=0.65,
        )
        ax.axhline(0.0, color=AXIS, lw=0.8)
        ax.tick_params(axis="x", rotation=45)

    _style(ax, "Return by calendar year, net of costs", "%")
    return to_data_uri(fig)


def monthly_histogram(result: BacktestResult) -> str:
    monthly = monthly_returns(result.returns) * 100.0

    fig, ax = plt.subplots(figsize=SQUARE)
    if monthly.empty:
        _nothing_to_plot(ax, "no completed months")
    else:
        ax.hist(monthly.to_numpy(), bins=25, color=NET, alpha=0.8)
        ax.axvline(0.0, color=AXIS, lw=0.8)
        mean = float(monthly.mean())
        if math.isfinite(mean):
            ax.axvline(mean, color=SHORT, lw=1.2, ls="--", label="mean")
            ax.legend(frameon=False, fontsize=8)
        ax.set_xlabel("%", fontsize=8)

    _style(ax, "Monthly return distribution", "months")
    return to_data_uri(fig)


def leg_contribution(result: BacktestResult) -> str:
    """Contributions are additive by construction, so they cumulate by summing."""
    legs = result.legs.cumsum() * 100.0

    fig, ax = plt.subplots(figsize=WIDE)
    ax.plot(legs.index, legs["long"], color=NET, lw=1.2, label="long book")
    ax.plot(legs.index, legs["short"], color=SHORT, lw=1.2, label="short book")
    ax.plot(legs.index, legs.sum(axis=1), color=AXIS, lw=1.2, ls="--", label="combined")
    ax.axhline(0.0, color=AXIS, lw=0.8)

    _style(ax, "Cumulative contribution by leg, before costs", "%")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    return to_data_uri(fig)


def exposures(result: BacktestResult) -> str:
    book = result.exposures

    fig, ax = plt.subplots(figsize=WIDE)
    ax.plot(book.index, book["gross_exposure"], color=NET, lw=1.0, label="gross")
    ax.plot(book.index, book["net_exposure"], color=SHORT, lw=1.0, label="net")
    ax.axhline(0.0, color=AXIS, lw=0.8)

    _style(ax, "Exposure between rebalances, with weights left to drift", "x NAV")
    ax.legend(frameon=False, fontsize=8, loc="center right")
    return to_data_uri(fig)


def cost_curve(sweep: pd.DataFrame, *, configured: float, breakeven: float) -> str:
    fig, ax = plt.subplots(figsize=SQUARE)
    ax.plot(sweep.index, sweep["sharpe"], color=NET, lw=1.4, marker="o", ms=4)
    ax.axhline(0.0, color=AXIS, lw=0.8)

    if configured in sweep.index:
        ax.plot(
            [configured],
            [sweep.loc[configured, "sharpe"]],
            marker="o",
            ms=10,
            mfc="none",
            mec=LOSS,
            mew=1.6,
            label="as configured",
        )
    if math.isfinite(breakeven) and breakeven <= float(sweep.index.max()):
        ax.axvline(breakeven, color=LOSS, lw=1.0, ls="--", label="breakeven")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8)

    _style(ax, "Sharpe against assumed cost", "Sharpe")
    ax.set_xlabel("bps per unit of notional traded", fontsize=8)
    return to_data_uri(fig)


def parameter_heatmap(grid: pd.DataFrame, *, label: str = "Sharpe") -> str:
    fig, ax = plt.subplots(figsize=SQUARE, layout="constrained")
    values = grid.to_numpy(dtype="float64")

    if not np.isfinite(values).any():
        _nothing_to_plot(ax, "no grid cell could be run")
        ax.set_title(f"{label} across the parameter grid", fontsize=10, loc="left", color=AXIS)
        return to_data_uri(fig)

    # Symmetric limits so that zero sits in the middle of the colour map and a negative
    # cell is unmistakably a different colour from a positive one.
    limit = float(np.nanmax(np.abs(values))) or 1.0
    image = ax.imshow(values, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")

    ax.set_xticks(range(grid.shape[1]), [f"{c:g}" for c in grid.columns], fontsize=8)
    ax.set_yticks(range(grid.shape[0]), [str(i) for i in grid.index], fontsize=8)
    ax.set_xlabel("top fraction", fontsize=8)
    ax.set_ylabel("lookback, sessions", fontsize=8)
    ax.set_title(f"{label} across the parameter grid", fontsize=10, loc="left", color=AXIS)

    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            value = values[row, col]
            text = f"{value:.2f}" if math.isfinite(value) else "n/a"
            ax.text(col, row, text, ha="center", va="center", fontsize=8, color="#222222")

    fig.colorbar(image, ax=ax, shrink=0.85)
    return to_data_uri(fig)
