"""Command line front end.

One verb per stage, so the artifacts are inspectable in between:

    xsbt fetch  --config configs/momentum.yaml
    xsbt run    --config configs/momentum.yaml --out runs/momentum
    xsbt report --run runs/momentum --out reports/momentum.html
    xsbt verify --cache-dir data/cache
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from xsbt import __version__
from xsbt.config import BacktestConfig, DataConfig
from xsbt.data.base import DataError
from xsbt.data.cache import PriceCache
from xsbt.data.market import MarketData, load_market_data, open_repository
from xsbt.engine.backtest import BacktestResult, run_backtest
from xsbt.report.html import ReportData, analyse, write_html, write_metrics, write_returns
from xsbt.strategies import build

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Cross-sectional long/short equity backtester.",
)
console = Console()

REPORT_FILE = "report.html"
METRICS_FILE = "metrics.json"
RETURNS_FILE = "returns.csv"


def _show_version(shown: bool) -> None:
    if shown:
        console.print(f"xsbt {__version__}")
        raise typer.Exit()


@app.callback()
def cli(
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show what the data layer is doing.")
    ] = False,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_show_version,
            is_eager=True,
            help="Print the version and exit.",
        ),
    ] = False,
) -> None:
    # Third-party libraries are noisy at INFO, so only our own logger is turned up.
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False, show_time=False, markup=False)],
    )
    logging.getLogger("xsbt").setLevel(logging.DEBUG if verbose else logging.INFO)


@app.command()
def fetch(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Take the universe and window from a run config."),
    ] = None,
    universe: Annotated[
        Path | None, typer.Option(help="Universe CSV, when not using --config.")
    ] = None,
    start: Annotated[str | None, typer.Option(help="First session, ISO date.")] = None,
    end: Annotated[str | None, typer.Option(help="Last session, ISO date.")] = None,
    cache_dir: Annotated[Path, typer.Option(help="Snapshot directory.")] = Path("data/cache"),
    benchmark: Annotated[
        str, typer.Option(help="Extra symbol pulled alongside the universe. Blank to skip.")
    ] = "SPY",
    refresh: Annotated[
        bool, typer.Option(help="Refetch even where the cache already covers the window.")
    ] = False,
) -> None:
    """Populate the local price snapshot. This is the only command that uses the network."""
    with friendly_errors():
        data = _resolve_data_config(config, universe, start, end, cache_dir, benchmark)
        repository = open_repository(data, offline=False)
        market = load_market_data(data, repository, refresh=refresh)

        wanted = market.prices.shape[1] + len(market.missing)
        console.print(f"cached {market.prices.shape[1]} of {wanted} tickers in {data.cache_dir}")
        console.print(
            f"  {market.prices.index.min().date()} to {market.prices.index.max().date()}, "
            f"{len(market.prices)} sessions"
        )
        if data.benchmark:
            state = "ok" if market.benchmark is not None else "unavailable"
            console.print(f"  benchmark {data.benchmark}: {state}")
        for ticker, reason in sorted(market.missing.items()):
            console.print(f"  [yellow]missing[/] {ticker}: {reason}")
        console.print(f"  snapshot {market.snapshot_id[:16]}")


@app.command()
def run(
    config: Annotated[Path, typer.Option("--config", "-c", help="Run config YAML.")],
    out: Annotated[
        Path | None, typer.Option(help="Run directory. Defaults to runs/<config name>.")
    ] = None,
    offline: Annotated[
        bool, typer.Option(help="Never touch the network; replay the cached snapshot.")
    ] = False,
    refresh: Annotated[bool, typer.Option(help="Refetch prices before running.")] = False,
    report: Annotated[
        bool, typer.Option("--report/--no-report", help="Also write report.html.")
    ] = True,
    grid: Annotated[
        bool,
        typer.Option(
            "--grid/--no-grid",
            help="Include the parameter sweep, which re-runs the backtest a few dozen times.",
        ),
    ] = True,
) -> None:
    """Run a backtest and write the run directory, metrics and report."""
    with friendly_errors():
        settings = BacktestConfig.from_yaml(config)
        repository = open_repository(settings.data, offline=offline)
        market = load_market_data(settings.data, repository, refresh=refresh)

        result = run_backtest(
            market.prices,
            build(settings.strategy),
            settings,
            dollar_volume=market.dollar_volume,
            benchmark=market.benchmark,
            snapshot_id=market.snapshot_id,
        )

        directory = out or Path("runs") / settings.name
        result.save(directory)
        data = _write_artifacts(result, directory, market=market, grid=grid, report=report)

        _print_summary(data)
        console.print(f"\nwritten to {directory}")


@app.command()
def report(
    run_dir: Annotated[Path, typer.Option("--run", help="Run directory written by `xsbt run`.")],
    out: Annotated[
        Path | None, typer.Option(help="HTML path. Defaults to <run>/report.html.")
    ] = None,
    grid: Annotated[
        bool, typer.Option("--grid/--no-grid", help="Include the parameter sweep.")
    ] = True,
) -> None:
    """Rebuild the report from a saved run, without re-running the backtest.

    Prices are re-read from the cache offline, because a report on a past run has no
    business fetching anything new. If the cache has moved on, the sections that need the
    panel are dropped and the rest of the page is still produced.
    """
    with friendly_errors():
        result = BacktestResult.load(run_dir)

        market = None
        try:
            repository = open_repository(result.config.data, offline=True)
            market = load_market_data(result.config.data, repository)
        except (DataError, FileNotFoundError) as exc:
            console.print(f"[yellow]note[/] per-name and parameter sections skipped: {exc}")

        data = analyse(
            result,
            prices=market.prices if market else None,
            dollar_volume=market.dollar_volume if market else None,
            include_grid=grid,
        )
        target = write_html(data, out or run_dir / REPORT_FILE)
        write_metrics(data, run_dir / METRICS_FILE)

        _print_summary(data)
        console.print(f"\nwritten to {target}")


@app.command()
def verify(
    cache_dir: Annotated[Path, typer.Option(help="Snapshot directory.")] = Path("data/cache"),
) -> None:
    """Re-hash every cached file against the manifest.

    Non-zero exit if anything drifted, so this can sit in a scheduled job.
    """
    with friendly_errors():
        cache = PriceCache(cache_dir)
        entries = cache.manifest.entries
        if not entries:
            console.print(f"[yellow]note[/] no cached tickers under {cache_dir}")
            raise typer.Exit(code=1)

        problems = cache.verify()
        for ticker, complaint in sorted(problems.items()):
            console.print(f"[red]{ticker}[/] {complaint}")

        console.print(f"{len(entries) - len(problems)} of {len(entries)} tickers verified")
        console.print(f"snapshot {cache.snapshot_id[:16]}")
        if problems:
            raise typer.Exit(code=1)


@contextmanager
def friendly_errors() -> Iterator[None]:
    """Turn an expected failure into one line and a non-zero exit, not a traceback."""
    try:
        yield
    except (DataError, FileNotFoundError, ValueError, KeyError) as exc:
        console.print(f"[bold red]error[/] {exc}")
        raise typer.Exit(code=1) from None


def _resolve_data_config(
    config: Path | None,
    universe: Path | None,
    start: str | None,
    end: str | None,
    cache_dir: Path,
    benchmark: str,
) -> DataConfig:
    """A config file or the individual flags, not a mix of the two."""
    if config is not None:
        return BacktestConfig.from_yaml(config).data

    missing = [
        flag
        for flag, value in (("--universe", universe), ("--start", start), ("--end", end))
        if value is None
    ]
    if missing:
        raise ValueError(f"either pass --config, or all of {', '.join(missing)}")

    assert universe is not None and start is not None and end is not None  # checked above
    return DataConfig(
        universe=universe,
        start=_iso_date(start, "--start"),
        end=_iso_date(end, "--end"),
        cache_dir=cache_dir,
        benchmark=benchmark or None,
    )


def _iso_date(value: str, flag: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{flag} must be an ISO date such as 2010-01-01, got {value!r}") from None


def _write_artifacts(
    result: BacktestResult,
    directory: Path,
    *,
    market: MarketData,
    grid: bool,
    report: bool,
) -> ReportData:
    data = analyse(
        result,
        prices=market.prices,
        dollar_volume=market.dollar_volume,
        include_grid=grid,
    )
    write_metrics(data, directory / METRICS_FILE)
    write_returns(result, directory / RETURNS_FILE)
    if report:
        write_html(data, directory / REPORT_FILE)
    return data


def _print_summary(data: ReportData) -> None:
    net = data.net
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim")
    table.add_column(justify="right")

    rows = (
        ("period", f"{net.start} to {net.end} ({net.years:.1f}y)"),
        ("CAGR, net", f"{net.cagr:.2%}"),
        ("volatility", f"{net.ann_volatility:.2%}"),
        ("Sharpe, net", f"{net.sharpe:.2f} (t = {net.sharpe_tstat_hac:.2f})"),
        ("max drawdown", f"{net.max_drawdown:.2%}"),
        ("turnover p.a.", f"{net.ann_turnover:.2f}x over {net.trades} rebalances"),
        ("cost drag p.a.", f"{net.ann_cost_drag:.2%}"),
        ("breakeven cost", f"{data.breakeven_bps:.0f}bps"),
    )
    for label, value in rows:
        table.add_row(label, value)
    if data.deflation is not None:
        table.add_row(
            "deflated Sharpe",
            f"{data.deflation.deflated:.0%} over {data.deflation.trials} configs",
        )
    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
