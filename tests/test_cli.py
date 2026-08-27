"""End to end through the command line, offline.

Every test here runs from a temporary working directory, which is also what pins down
the documented rule that paths in a config are relative to where you invoke it from.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from tests.helpers import StubSource, make_bars, random_walk_panel, seed_cache, write_universe
from xsbt.cli import app
from xsbt.data.cache import PriceCache

runner = CliRunner()


@dataclass(frozen=True)
class Project:
    """A working directory with a universe, a config and a populated cache in it."""

    root: Path
    panel: pd.DataFrame
    config: Path
    cache_dir: Path


@pytest.fixture
def panel() -> pd.DataFrame:
    return random_walk_panel(names=8, sessions=320)


@pytest.fixture
def project(tmp_path: Path, panel: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> Project:
    monkeypatch.chdir(tmp_path)
    write_universe(tmp_path / "universe.csv", list(panel.columns))

    benchmark = pd.Series(
        100.0 * (1.0 + panel.pct_change(fill_method=None).mean(axis=1).fillna(0.0)).cumprod(),
        name="SPY",
    )
    seed_cache(tmp_path / "data" / "cache", panel, benchmark=benchmark)

    config = tmp_path / "configs" / "test.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        "name: test\n"
        "description: a unit test strategy\n"
        "data:\n"
        "  universe: universe.csv\n"
        f"  start: {panel.index[0].date()}\n"
        f"  end: {panel.index[-1].date()}\n"
        "  cache_dir: data/cache\n"
        "  benchmark: SPY\n"
        "strategy:\n"
        "  name: momentum\n"
        "  lookback_days: 60\n"
        "  top_fraction: 0.25\n"
        "  min_names: 4\n"
        "portfolio:\n"
        "  rebalance: M\n"
        "  cost_bps: 10.0\n"
    )
    return Project(root=tmp_path, panel=panel, config=config, cache_dir=tmp_path / "data/cache")


def test_help_lists_every_stage() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for verb in ("fetch", "run", "report", "verify"):
        assert verb in result.output


def test_version_is_reported_and_exits_clean() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert "xsbt" in result.output


def test_the_package_also_runs_as_a_module() -> None:
    """The Makefile drives `python -m xsbt`, which does not go through the console script."""
    proc = subprocess.run(
        [sys.executable, "-m", "xsbt", "--version"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "xsbt" in proc.stdout


def test_a_run_writes_every_artifact(project: Project) -> None:
    result = runner.invoke(app, ["run", "-c", str(project.config), "--offline", "--no-grid"])

    assert result.exit_code == 0, result.output
    written = project.root / "runs" / "test"
    for name in (
        "daily.parquet",
        "weights.parquet",
        "target_weights.parquet",
        "metadata.json",
        "config.json",
        "metrics.json",
        "returns.csv",
        "report.html",
    ):
        assert (written / name).exists(), f"{name} was not written"


def test_the_run_directory_can_be_pointed_somewhere_else(project: Project) -> None:
    result = runner.invoke(
        app, ["run", "-c", str(project.config), "--offline", "--no-grid", "--out", "elsewhere"]
    )

    assert result.exit_code == 0, result.output
    assert (project.root / "elsewhere" / "report.html").exists()


def test_the_summary_gives_a_pm_the_headline_numbers(project: Project) -> None:
    result = runner.invoke(app, ["run", "-c", str(project.config), "--offline", "--no-grid"])

    assert result.exit_code == 0, result.output
    for label in ("CAGR", "Sharpe", "max drawdown", "turnover", "breakeven"):
        assert label in result.output


def test_no_report_leaves_the_machine_readable_artifacts_behind(project: Project) -> None:
    """Useful in a sweep, where nobody is going to open a hundred HTML files."""
    result = runner.invoke(
        app, ["run", "-c", str(project.config), "--offline", "--no-grid", "--no-report"]
    )

    assert result.exit_code == 0, result.output
    written = project.root / "runs" / "test"
    assert not (written / "report.html").exists()
    assert (written / "metrics.json").exists()


def test_the_report_is_a_single_file_that_needs_nothing_else(project: Project) -> None:
    runner.invoke(app, ["run", "-c", str(project.config), "--offline", "--no-grid"])

    html = (project.root / "runs" / "test" / "report.html").read_text(encoding="utf-8")

    assert "data:image/png;base64," in html
    assert "a unit test strategy" in html


def test_two_runs_on_one_snapshot_produce_the_same_metrics(project: Project) -> None:
    """The acceptance test for the whole reproducibility story."""
    for out in ("first", "second"):
        result = runner.invoke(
            app,
            [
                "run",
                "-c",
                str(project.config),
                "--offline",
                "--no-grid",
                "--no-report",
                "--out",
                out,
            ],
        )
        assert result.exit_code == 0, result.output

    first = (project.root / "first" / "metrics.json").read_bytes()
    second = (project.root / "second" / "metrics.json").read_bytes()

    assert first == second


def test_the_run_records_the_snapshot_it_was_priced_off(project: Project) -> None:
    runner.invoke(app, ["run", "-c", str(project.config), "--offline", "--no-grid"])

    metadata = json.loads((project.root / "runs" / "test" / "metadata.json").read_text())

    assert metadata["data_snapshot_id"] == PriceCache(project.cache_dir).snapshot_id
    assert metadata["universe_size"] == project.panel.shape[1]
    assert metadata["rebalances_traded"] > 0


def test_report_rebuilds_from_a_saved_run_without_rerunning_it(project: Project) -> None:
    runner.invoke(app, ["run", "-c", str(project.config), "--offline", "--no-grid"])
    target = project.root / "runs" / "test" / "report.html"
    target.unlink()

    result = runner.invoke(app, ["report", "--run", "runs/test", "--no-grid"])

    assert result.exit_code == 0, result.output
    assert target.exists()


def test_report_can_write_the_page_anywhere(project: Project) -> None:
    runner.invoke(app, ["run", "-c", str(project.config), "--offline", "--no-grid"])

    result = runner.invoke(
        app, ["report", "--run", "runs/test", "--no-grid", "--out", "share/test.html"]
    )

    assert result.exit_code == 0, result.output
    assert (project.root / "share" / "test.html").exists()


def test_report_drops_what_needs_prices_rather_than_reaching_for_the_network(
    project: Project,
) -> None:
    """A report on a past run has no business fetching anything new."""
    runner.invoke(app, ["run", "-c", str(project.config), "--offline", "--no-grid"])
    for parquet in (project.cache_dir / "prices").glob("*.parquet"):
        parquet.unlink()

    result = runner.invoke(app, ["report", "--run", "runs/test"])

    assert result.exit_code == 0, result.output
    assert "per-name and parameter sections skipped" in result.output
    assert (project.root / "runs" / "test" / "report.html").exists()


def test_report_on_something_that_is_not_a_run_directory(project: Project) -> None:
    result = runner.invoke(app, ["report", "--run", "runs/nothing-here"])

    assert result.exit_code == 1
    assert "not a run directory" in result.output


def test_verify_passes_on_an_untouched_snapshot(project: Project) -> None:
    result = runner.invoke(app, ["verify", "--cache-dir", "data/cache"])

    assert result.exit_code == 0, result.output
    assert "verified" in result.output


def test_verify_fails_when_a_cached_file_has_been_edited(project: Project) -> None:
    """This is the whole reason the manifest carries a hash."""
    cache = PriceCache(project.cache_dir)
    ticker = next(iter(cache.manifest.entries))
    tampered = cache.read(ticker)
    tampered["close"] *= 2.0
    tampered.to_parquet(cache.path_for(ticker))

    result = runner.invoke(app, ["verify", "--cache-dir", "data/cache"])

    assert result.exit_code == 1
    assert ticker in result.output


def test_verify_says_so_rather_than_passing_an_empty_cache(project: Project) -> None:
    result = runner.invoke(app, ["verify", "--cache-dir", "nothing/here"])

    assert result.exit_code == 1
    assert "no cached tickers" in result.output


def test_verify_also_reconciles_adjusted_close_against_the_events(project: Project) -> None:
    """Hashing proves the bytes have not moved. It says nothing about whether the vendor
    got the adjustment right, which is what this second half of the command is for."""
    result = runner.invoke(app, ["verify", "--cache-dir", "data/cache"])

    assert result.exit_code == 0, result.output
    assert "adjusted closes reconciled" in result.output


def test_verify_fails_when_a_dividend_went_unadjusted(project: Project) -> None:
    cache = PriceCache(project.cache_dir)
    ticker = next(iter(cache.manifest.entries))
    bars = cache.read(ticker)
    # A 5% payout the adjusted series knows nothing about.
    bars.iloc[5, bars.columns.get_loc("dividend")] = bars["close"].iloc[4] * 0.05
    cache.write(
        ticker,
        bars,
        source="stub",
        requested_start=bars.index[0].date(),
        requested_end=bars.index[-1].date(),
    )

    result = runner.invoke(app, ["verify", "--cache-dir", "data/cache"])

    assert result.exit_code == 1
    assert "away from its own events" in result.output


def test_verify_can_skip_the_adjustment_check(project: Project) -> None:
    """The hash check is fast and the reconstruction is not, so it can be left out of a
    tight loop."""
    result = runner.invoke(app, ["verify", "--cache-dir", "data/cache", "--no-adjustments"])

    assert result.exit_code == 0, result.output
    assert "adjusted closes reconciled" not in result.output


def test_fetch_populates_the_cache_from_the_source(
    tmp_path: Path, panel: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    write_universe(tmp_path / "universe.csv", list(panel.columns))
    frames = {t: make_bars(panel.index, panel[t].to_numpy()) for t in panel.columns}
    frames["SPY"] = make_bars(panel.index, panel.mean(axis=1).to_numpy())
    monkeypatch.setattr("xsbt.data.market.YahooFinanceSource", lambda: StubSource(frames))

    result = runner.invoke(
        app,
        [
            "fetch",
            "--universe",
            "universe.csv",
            "--start",
            str(panel.index[0].date()),
            "--end",
            str(panel.index[-1].date()),
            "--cache-dir",
            "data/cache",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "cached 8 of 8 tickers" in result.output
    cache = PriceCache(tmp_path / "data" / "cache")
    assert set(cache.manifest.entries) == {*panel.columns, "SPY"}
    assert cache.verify() == {}
    # Reconciled on the way in, not only when someone remembers to run verify.
    assert "adjusted closes reconciled" in result.output


def test_fetch_reports_the_names_it_could_not_get(
    tmp_path: Path, panel: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    write_universe(tmp_path / "universe.csv", [*panel.columns, "DEADCO"])
    frames = {t: make_bars(panel.index, panel[t].to_numpy()) for t in panel.columns}
    monkeypatch.setattr("xsbt.data.market.YahooFinanceSource", lambda: StubSource(frames))

    result = runner.invoke(
        app,
        [
            "fetch",
            "--universe",
            "universe.csv",
            "--start",
            str(panel.index[0].date()),
            "--end",
            str(panel.index[-1].date()),
            "--cache-dir",
            "data/cache",
            "--benchmark",
            "",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "cached 8 of 9 tickers" in result.output
    assert "DEADCO" in result.output


def test_fetch_needs_either_a_config_or_all_three_flags(tmp_path: Path) -> None:
    result = runner.invoke(app, ["fetch", "--universe", str(tmp_path / "u.csv")])

    assert result.exit_code == 1
    assert "--start" in result.output and "--end" in result.output


def test_a_date_that_is_not_a_date_is_caught_before_anything_runs(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["fetch", "--universe", "u.csv", "--start", "last tuesday", "--end", "2020-01-01"],
    )

    assert result.exit_code == 1
    assert "ISO date" in result.output


def test_a_missing_config_is_one_line_not_a_traceback(tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", "-c", str(tmp_path / "nope.yaml"), "--offline"])

    assert result.exit_code == 1
    assert "config not found" in result.output
    assert "Traceback" not in result.output


def test_offline_with_nothing_cached_tells_you_to_fetch(
    tmp_path: Path, panel: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    write_universe(tmp_path / "universe.csv", list(panel.columns))
    config = tmp_path / "empty.yaml"
    config.write_text(
        "name: empty\n"
        "data:\n"
        "  universe: universe.csv\n"
        f"  start: {panel.index[0].date()}\n"
        f"  end: {panel.index[-1].date()}\n"
        "  cache_dir: data/cache\n"
        "  benchmark: null\n"
        "strategy:\n"
        "  name: momentum\n"
    )

    result = runner.invoke(app, ["run", "-c", "empty.yaml", "--offline"])

    assert result.exit_code == 1
    assert "xsbt fetch" in result.output


def test_an_unknown_strategy_names_the_ones_that_exist(project: Project) -> None:
    text = project.config.read_text().replace("name: momentum", "name: mean_reversal_2")
    project.config.write_text(text)

    result = runner.invoke(app, ["run", "-c", str(project.config), "--offline", "--no-grid"])

    assert result.exit_code == 1
    assert "unknown strategy" in result.output


def test_a_window_shorter_than_the_lookback_fails_with_a_reason(
    project: Project, panel: pd.DataFrame
) -> None:
    """Better a clear refusal than a backtest with two rebalances in it."""
    short_end = panel.index[20].date()
    text = project.config.read_text().replace(f"end: {panel.index[-1].date()}", f"end: {short_end}")
    project.config.write_text(text)

    result = runner.invoke(app, ["run", "-c", str(project.config), "--offline", "--no-grid"])

    assert result.exit_code == 1
    assert "lookback_days" in result.output


def test_the_grid_run_costs_more_but_lands_in_the_report(project: Project) -> None:
    result = runner.invoke(app, ["run", "-c", str(project.config), "--offline"])

    assert result.exit_code == 0, result.output
    metrics = json.loads((project.root / "runs" / "test" / "metrics.json").read_text())

    assert metrics["parameter_grid"]
    assert {"lookback_days", "top_fraction", "sharpe"} == set(metrics["parameter_grid"][0])


def test_dates_outside_the_cached_window_are_a_cache_miss_not_a_silent_fetch(
    project: Project, panel: pd.DataFrame
) -> None:
    later = dt.date(2099, 1, 1)
    text = project.config.read_text().replace(f"end: {panel.index[-1].date()}", f"end: {later}")
    project.config.write_text(text)

    result = runner.invoke(app, ["run", "-c", str(project.config), "--offline", "--no-grid"])

    assert result.exit_code == 1
    assert "offline" in result.output or "xsbt fetch" in result.output
