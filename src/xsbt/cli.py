"""Command line front end.

One verb per stage, so the artifacts are inspectable in between:

    xsbt fetch  --config configs/momentum.yaml
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

from xsbt import __version__
from xsbt.config import BacktestConfig, DataConfig
from xsbt.data.base import DataError
from xsbt.data.cache import PriceCache
from xsbt.data.market import load_market_data, open_repository

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Cross-sectional long/short equity backtester.",
)
console = Console()


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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
