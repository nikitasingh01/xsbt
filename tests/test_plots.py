"""Figure rendering, including the cases where there is nothing worth drawing.

A PNG cannot be asserted on pixel by pixel without a golden file, and a golden file
fails on every matplotlib point release while telling you nothing. So these check the
two things that do break in practice: the figure comes back as a decodable PNG, and
nothing is left open on the pyplot stack afterwards.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Iterator, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from tests.helpers import make_backtest_config, random_walk_panel
from xsbt.engine.backtest import BacktestResult
from xsbt.report import plots

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
DATA_URI = "data:image/png;base64,"

#: The plots take a whole result, so one config is built once and shared. None of them
#: read it, but a BacktestResult without one would be a lie.
CONFIG = make_backtest_config(random_walk_panel(names=4, sessions=40))

BOOK = ("A", "B", "C", "D")


@pytest.fixture(autouse=True)
def no_leaked_figures() -> Iterator[None]:
    """to_data_uri promises to close what it opens. Over a parameter grid a leak is fatal."""
    plt.close("all")
    yield
    assert plt.get_fignums() == [], "a figure was left open on the pyplot stack"


def stub_result(
    returns: Sequence[float], *, benchmark: Sequence[float] | None = None
) -> BacktestResult:
    """A result assembled by hand, so a degenerate path can be aimed at directly."""
    index = pd.bdate_range("2020-01-02", periods=len(returns), name="date")
    net = pd.Series(list(returns), index=index, dtype="float64")

    daily = pd.DataFrame(
        {
            "gross_return": net,
            "net_return": net,
            "cost": 0.0,
            "turnover": 0.0,
            "long_return": net / 2.0,
            "short_return": net / 2.0,
            "gross_exposure": 1.0,
            "net_exposure": 0.0,
            "long_names": 2.0,
            "short_names": 2.0,
        },
        index=index,
    )
    if benchmark is not None:
        daily["benchmark_return"] = pd.Series(list(benchmark), index=index, dtype="float64")

    weights = pd.DataFrame(0.25, index=index, columns=list(BOOK))
    return BacktestResult(
        config=CONFIG,
        daily=daily,
        weights=weights,
        target_weights=weights.iloc[:1],
        metadata={},
    )


def png_bytes(uri: str) -> bytes:
    assert uri.startswith(DATA_URI), f"not a png data uri: {uri[:40]!r}"
    raw = base64.b64decode(uri.removeprefix(DATA_URI), validate=True)
    assert raw.startswith(PNG_MAGIC), "payload is not a PNG"
    return raw


def sweep(levels: Sequence[float], sharpes: Sequence[float]) -> pd.DataFrame:
    return pd.DataFrame({"sharpe": list(sharpes)}, index=pd.Index(list(levels), name="cost_bps"))


@pytest.fixture
def result() -> BacktestResult:
    """Three years of a mildly profitable book, which is enough for every panel."""
    rng = np.random.default_rng(7)
    return stub_result(
        rng.normal(0.0004, 0.006, size=760).tolist(),
        benchmark=rng.normal(0.0003, 0.009, size=760).tolist(),
    )


@pytest.mark.parametrize(
    "draw",
    [
        plots.equity,
        plots.drawdown,
        plots.rolling_sharpe_plot,
        plots.yearly_bars,
        plots.monthly_histogram,
        plots.leg_contribution,
        plots.exposures,
    ],
    ids=lambda f: f.__name__,
)
def test_every_panel_renders_to_a_real_png(
    draw: Callable[[BacktestResult], str], result: BacktestResult
) -> None:
    raw = png_bytes(draw(result))

    # A blank canvas at this size is well under a kilobyte, so this catches an empty axis.
    assert len(raw) > 5_000


def test_the_benchmark_line_is_only_drawn_when_there_is_one(result: BacktestResult) -> None:
    without = stub_result(result.returns.tolist())

    assert plots.equity(result) != plots.equity(without)


def test_the_equity_chart_gives_up_the_log_scale_when_the_book_is_wiped_out() -> None:
    """A -100% day puts the curve on zero, and log of zero is not a chart."""
    wiped = stub_result([0.01, -1.0, 0.01, 0.01])

    assert png_bytes(plots.equity(wiped))


def test_rolling_sharpe_says_so_rather_than_drawing_an_empty_pane() -> None:
    short = stub_result([0.001] * 60)

    assert png_bytes(plots.rolling_sharpe_plot(short, window=252))


def test_the_yearly_and_monthly_panels_survive_an_empty_run() -> None:
    """analyse() would refuse this, but the plot layer should not be the thing that raises."""
    nothing = stub_result([])

    assert png_bytes(plots.yearly_bars(nothing))
    assert png_bytes(plots.monthly_histogram(nothing))


def test_the_cost_curve_marks_the_configured_level_and_the_breakeven() -> None:
    curve = sweep([0.0, 5.0, 10.0, 20.0], [1.2, 0.8, 0.4, -0.4])

    assert png_bytes(plots.cost_curve(curve, configured=10.0, breakeven=15.0))


def test_the_cost_curve_leaves_off_markers_it_has_no_value_for() -> None:
    """An off-ladder cost and a breakeven past the end of the sweep: no legend at all."""
    curve = sweep([0.0, 5.0, 10.0], [1.2, 1.1, 1.0])

    assert png_bytes(plots.cost_curve(curve, configured=7.5, breakeven=float("inf")))


def test_the_heatmap_labels_the_cells_that_could_not_be_run() -> None:
    grid = pd.DataFrame(
        [[0.8, float("nan")], [-0.3, 0.5]],
        index=pd.Index([60, 120], name="lookback_days"),
        columns=pd.Index([0.2, 0.3], name="top_fraction"),
    )

    assert png_bytes(plots.parameter_heatmap(grid))


def test_the_heatmap_says_so_when_no_cell_could_be_run() -> None:
    empty = pd.DataFrame(
        float("nan"),
        index=pd.Index([60, 120], name="lookback_days"),
        columns=pd.Index([0.2, 0.3], name="top_fraction"),
    )

    assert png_bytes(plots.parameter_heatmap(empty))


def test_an_all_zero_grid_does_not_divide_by_its_own_range() -> None:
    """The colour limits come off the largest absolute cell, which here is zero."""
    flat = pd.DataFrame(
        0.0,
        index=pd.Index([60, 120], name="lookback_days"),
        columns=pd.Index([0.2, 0.3], name="top_fraction"),
    )

    assert png_bytes(plots.parameter_heatmap(flat))
