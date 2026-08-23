"""The report layer: what gets computed, what gets written, and what the page says.

The HTML is checked for the things a reader would notice if they broke, rather than
against a golden file, which would fail on every wording change and tell us nothing.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
from pathlib import Path

import pandas as pd
import pytest

from tests.helpers import make_backtest_config, random_walk_panel
from xsbt.analytics.metrics import PerformanceSummary
from xsbt.config import StrategyConfig
from xsbt.engine.backtest import BacktestResult, run_backtest
from xsbt.report.html import (
    ReportData,
    analyse,
    cost_ladder,
    grid_fractions,
    grid_lookbacks,
    render,
    significance_note,
    write_html,
    write_metrics,
    write_returns,
)
from xsbt.strategies import Momentum

#: Every figure the template asks for, before the optional parameter grid.
BASE_FIGURES = 8


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return random_walk_panel()


@pytest.fixture(scope="module")
def benchmark(panel: pd.DataFrame) -> pd.Series:
    market = panel.mean(axis=1).pct_change(fill_method=None)
    market.name = "SPY"
    return market


@pytest.fixture(scope="module")
def result(panel: pd.DataFrame, benchmark: pd.Series) -> BacktestResult:
    config = make_backtest_config(panel, cost_bps=10.0)
    return run_backtest(panel, Momentum(config.strategy), config, benchmark=benchmark)


@pytest.fixture(scope="module")
def report(result: BacktestResult) -> ReportData:
    return analyse(result)


@pytest.fixture(scope="module")
def report_with_grid(panel: pd.DataFrame, result: BacktestResult) -> ReportData:
    """The expensive one: the grid re-runs the backtest once per cell."""
    return analyse(result, prices=panel)


def summary_with_tstat(base: PerformanceSummary, t: float) -> PerformanceSummary:
    return dataclasses.replace(base, sharpe_tstat=t)


def test_analyse_fills_in_every_section(report: ReportData) -> None:
    assert report.net.sessions == report.gross.sessions
    assert report.market is not None
    assert not report.cost_sensitivity.empty
    assert not report.yearly.empty
    assert math.isfinite(report.breakeven_bps) or math.isinf(report.breakeven_bps)


def test_costs_only_ever_reduce_the_headline(report: ReportData) -> None:
    assert report.net.cagr <= report.gross.cagr
    assert report.net.sharpe <= report.gross.sharpe


def test_the_grid_is_dropped_rather_than_faked_without_prices(report: ReportData) -> None:
    assert report.grid is None


def test_the_grid_brackets_the_configured_setting(
    report_with_grid: ReportData, result: BacktestResult
) -> None:
    grid = report_with_grid.grid
    strategy = result.config.strategy

    assert grid is not None
    assert strategy.lookback_days in grid.index
    assert strategy.top_fraction in grid.columns
    assert grid.notna().any().any()


def test_the_market_fit_is_skipped_when_there_is_nothing_to_regress_on(
    panel: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    """Two overlapping sessions is not a regression, and it should not stop the report."""
    config = make_backtest_config(panel)
    stub = pd.Series(0.001, index=panel.index[:2], name="SPY")
    run = run_backtest(panel, Momentum(config.strategy), config, benchmark=stub)

    with caplog.at_level("WARNING"):
        data = analyse(run)

    assert data.market is None
    assert "market fit" in caplog.text


def test_a_run_without_a_benchmark_has_no_market_section(panel: pd.DataFrame) -> None:
    config = make_backtest_config(panel)
    run = run_backtest(panel, Momentum(config.strategy), config)

    assert analyse(run).market is None


def test_the_cost_ladder_always_contains_the_rate_that_was_run() -> None:
    assert 7.5 in cost_ladder(7.5)
    assert list(cost_ladder(7.5)) == sorted(cost_ladder(7.5))
    # Already on the standard ladder, so it must not appear twice.
    assert list(cost_ladder(10.0)).count(10.0) == 1


def test_grid_lookbacks_scale_around_the_configured_one() -> None:
    strategy = StrategyConfig(name="momentum", lookback_days=126)

    lookbacks = grid_lookbacks(strategy)

    assert 126 in lookbacks
    assert min(lookbacks) < 126 < max(lookbacks)
    assert list(lookbacks) == sorted(set(lookbacks))


def test_grid_lookbacks_drop_windows_the_skip_would_swallow() -> None:
    """A 40-session skip leaves nothing inside a 30-session window, so don't offer it."""
    strategy = StrategyConfig(name="momentum", lookback_days=60, skip_days=40)

    assert all(n > 40 for n in grid_lookbacks(strategy))
    assert 60 in grid_lookbacks(strategy)


def test_grid_fractions_include_an_unusual_configured_value() -> None:
    strategy = StrategyConfig(name="momentum", top_fraction=0.15)

    assert 0.15 in grid_fractions(strategy)
    assert list(grid_fractions(strategy)) == sorted(set(grid_fractions(strategy)))


@pytest.mark.parametrize(
    ("tstat", "phrase"),
    [
        (0.4, "indistinguishable from zero"),
        (1.5, "short of the usual two-standard-error bar"),
        (3.0, "clears the usual two-standard-error bar"),
        (-2.5, "significantly negative"),
    ],
)
def test_the_verdict_says_what_the_t_stat_actually_supports(
    report: ReportData, tstat: float, phrase: str
) -> None:
    assert phrase in significance_note(summary_with_tstat(report.net, tstat))


def test_the_verdict_admits_when_there_is_nothing_to_test(report: ReportData) -> None:
    note = significance_note(summary_with_tstat(report.net, float("nan")))

    assert "not enough data" in note


def test_the_page_is_self_contained(report: ReportData) -> None:
    """It gets emailed around, so it must not reach out for anything to render."""
    html = render(report)

    assert html.lstrip().startswith("<!DOCTYPE html>")
    assert html.count("data:image/png;base64,") == BASE_FIGURES
    assert 'src="http' not in html
    assert "<script" not in html


def test_the_page_names_the_run_and_its_window(report: ReportData) -> None:
    html = render(report)

    assert report.result.config.name in html
    assert report.net.start in html
    assert report.net.end in html


def test_the_page_carries_the_caveats(report: ReportData) -> None:
    html = render(report)

    for heading in ("Survivorship", "Restated history", "Shorting", "Significance"):
        assert heading in html


def test_the_grid_section_appears_only_when_there_is_a_grid(
    report: ReportData, report_with_grid: ReportData
) -> None:
    assert "Re-run" in render(report)
    assert render(report_with_grid).count("data:image/png;base64,") == BASE_FIGURES + 1


def test_the_page_renders_without_a_benchmark(panel: pd.DataFrame) -> None:
    config = make_backtest_config(panel)
    run = run_backtest(panel, Momentum(config.strategy), config)

    html = render(analyse(run))

    assert "No benchmark was available" in html


def test_the_page_renders_from_a_run_with_no_provenance(report: ReportData) -> None:
    """metadata.json is best-effort. A missing key should not take the page down."""
    stripped = dataclasses.replace(report, result=dataclasses.replace(report.result, metadata={}))

    html = render(stripped)

    assert "not recorded" in html


def test_a_reader_never_sees_a_raw_nan_or_inf(report: ReportData) -> None:
    """Whatever the arithmetic did, the page shows something a reader can act on.

    An unreachable breakeven reads as unbounded and a metric that could not be computed
    reads as n/a. A bare nan or inf on the page just reads as a broken report.
    """
    broken = dataclasses.replace(
        report,
        net=dataclasses.replace(report.net, calmar=float("nan"), sortino=float("nan")),
        breakeven_bps=float("inf"),
    )

    # Stripped of the base64 payloads, which are not prose and contain every letter.
    # Whole words only, or "provenance" counts as a hit.
    prose = re.sub(r'src="data:image[^"]*"', "", render(broken)).lower()

    assert "unbounded" in prose
    assert "n/a" in prose
    assert not re.search(r"\bnan\b", prose)
    assert not re.search(r"\binf\b", prose)


def test_the_cost_section_explains_a_breakeven_of_zero(report: ReportData) -> None:
    """A book that loses money at zero cost has a breakeven of 0, which is correct.

    Printing "growth stops at roughly 0 bps" around it is not: the reader is left
    thinking the cost assumption is the problem, when nothing about the costs would
    change the answer. Reversal hits this on the real data.
    """
    doomed = dataclasses.replace(report, breakeven_bps=0.0)

    prose = re.sub(r"<[^>]+>", " ", render(doomed))

    assert "no cost level that rescues this" in prose
    assert "growth stops" not in prose.lower()


def test_the_cost_section_copes_with_a_book_that_never_traded(report: ReportData) -> None:
    """Zero turnover leaves nothing to solve the breakeven against."""
    idle = dataclasses.replace(report, breakeven_bps=float("nan"))

    prose = re.sub(r"<[^>]+>", " ", render(idle))

    assert "no cost level to solve for" in prose


def test_metrics_json_holds_no_invalid_literals(report: ReportData, tmp_path: Path) -> None:
    """NaN and Infinity are Python's, not JSON's. Anything reading this file will choke."""
    path = write_metrics(report, tmp_path / "metrics.json")

    def refuse(literal: str) -> float:
        raise AssertionError(f"metrics.json contains a bare {literal}")

    payload = json.loads(path.read_text(), parse_constant=refuse)

    assert payload["config"]["name"] == report.result.config.name
    assert payload["net"]["sessions"] == report.net.sessions
    assert payload["parameter_grid"] is None


def test_metrics_json_is_the_same_bytes_for_the_same_snapshot(
    panel: pd.DataFrame, benchmark: pd.Series, tmp_path: Path
) -> None:
    """Reproducibility is the point of the whole data layer; this is where it shows."""
    config = make_backtest_config(panel, cost_bps=10.0)
    paths = []
    for n in (1, 2):
        run = run_backtest(panel, Momentum(config.strategy), config, benchmark=benchmark)
        paths.append(write_metrics(analyse(run), tmp_path / f"metrics_{n}.json"))

    assert paths[0].read_bytes() == paths[1].read_bytes()


def test_the_wall_clock_is_kept_out_of_the_metrics(report: ReportData, tmp_path: Path) -> None:
    """It lives in metadata.json instead, so two identical runs still diff clean."""
    payload = json.loads(write_metrics(report, tmp_path / "metrics.json").read_text())

    assert "run_utc" not in payload["run"]
    assert payload["run"]["config_fingerprint"] == report.result.config.fingerprint()


def test_the_grid_is_flattened_into_labelled_records(
    report_with_grid: ReportData, tmp_path: Path
) -> None:
    payload = json.loads(write_metrics(report_with_grid, tmp_path / "m.json").read_text())
    grid = report_with_grid.grid

    assert grid is not None
    assert len(payload["parameter_grid"]) == grid.size
    assert set(payload["parameter_grid"][0]) == {"lookback_days", "top_fraction", "sharpe"}


def test_write_html_creates_the_directory_it_was_pointed_at(
    report: ReportData, tmp_path: Path
) -> None:
    path = write_html(report, tmp_path / "nested" / "deeper" / "report.html")

    assert path.exists()
    assert "xsbt backtest report" in path.read_text(encoding="utf-8")


def test_returns_csv_round_trips_the_daily_frame(report: ReportData, tmp_path: Path) -> None:
    path = write_returns(report.result, tmp_path / "returns.csv")

    reloaded = pd.read_csv(path, index_col=0, parse_dates=True)

    assert list(reloaded.columns) == list(report.result.daily.columns)
    assert len(reloaded) == len(report.result.daily)
    pd.testing.assert_series_equal(
        reloaded["net_return"], report.result.returns, check_freq=False, rtol=1e-9
    )


def test_returns_csv_is_stable_across_writes(report: ReportData, tmp_path: Path) -> None:
    first = write_returns(report.result, tmp_path / "a.csv").read_bytes()
    second = write_returns(report.result, tmp_path / "b.csv").read_bytes()

    assert first == second
