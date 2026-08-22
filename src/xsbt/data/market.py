"""Turns a config into the panels the engine wants.

Everything below the engine deals in per-ticker bar frames; the engine deals in wide
date x ticker panels. This is the one place that crosses over, so the CLI and the report
ask for market data the same way.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from xsbt.config import DataConfig
from xsbt.data.base import DATE_INDEX_NAME, DataError, to_panel
from xsbt.data.cache import PriceCache
from xsbt.data.repository import PriceRepository
from xsbt.data.universe import load_universe, tickers_of
from xsbt.data.yahoo import YahooFinanceSource

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarketData:
    """One snapshot of everything a run needs."""

    #: date x ticker, the field named in the config (adjusted close by default).
    prices: pd.DataFrame
    #: date x ticker, close * volume. Only used by the eligibility floor.
    dollar_volume: pd.DataFrame
    #: Daily benchmark returns, or None if the config asks for no benchmark.
    benchmark: pd.Series | None
    #: Hash of the cache the panels were read out of.
    snapshot_id: str
    #: Ticker -> why it is not in the panel. Empty is the happy path.
    missing: dict[str, str]


def open_repository(
    data: DataConfig,
    *,
    offline: bool = False,
    cache_dir: Path | None = None,
) -> PriceRepository:
    """Repository pointed at the config's cache. Offline forbids the network entirely."""
    cache = PriceCache(cache_dir or data.cache_dir)
    source = None if offline else YahooFinanceSource()
    return PriceRepository(cache, source, offline=offline)


def load_market_data(
    data: DataConfig,
    repository: PriceRepository,
    *,
    refresh: bool = False,
) -> MarketData:
    """Read the universe named in ``data`` and pivot it into panels.

    The benchmark is deliberately fetched outside the universe. It is a yardstick for the
    report, not a name we are allowed to rank and trade.
    """
    members = load_universe(data.universe)
    tickers = tickers_of(members)
    start, end = data.start, data.end

    frames, missing = repository.get_many(tickers, start, end, refresh=refresh)
    if not frames:
        raise DataError(
            f"no prices for any of {len(tickers)} tickers in {start}..{end}. "
            "If you are offline, run `xsbt fetch` first."
        )
    if missing:
        log.warning("%d of %d tickers unavailable: %s", len(missing), len(tickers), sorted(missing))

    prices = to_panel(frames, field=data.field)
    dollar_volume = _dollar_volume(frames).reindex(index=prices.index, columns=prices.columns)

    benchmark = None
    if data.benchmark:
        benchmark = _benchmark_returns(data, repository, refresh=refresh)

    return MarketData(
        prices=prices,
        dollar_volume=dollar_volume,
        benchmark=benchmark,
        snapshot_id=repository.cache.snapshot_id,
        missing=missing,
    )


def _dollar_volume(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Traded notional per name per session.

    Close times volume, not VWAP times volume, because we only have daily bars. It is
    good enough for a liquidity floor and wrong for anything finer.
    """
    columns = {ticker: frame["close"] * frame["volume"] for ticker, frame in frames.items()}
    panel = pd.DataFrame(columns).sort_index().sort_index(axis=1)
    panel.index.name = DATE_INDEX_NAME
    panel.columns.name = "ticker"
    return panel


def _benchmark_returns(
    data: DataConfig,
    repository: PriceRepository,
    *,
    refresh: bool,
) -> pd.Series | None:
    """Benchmark daily returns, or None if we could not get them.

    A missing benchmark costs the report its beta section. It is not worth failing a
    whole run over, so this warns and carries on.
    """
    ticker = data.benchmark
    assert ticker is not None  # only called when the config names one
    try:
        bars = repository.get(ticker, data.start, data.end, refresh=refresh)
    except DataError as exc:
        log.warning("benchmark %s unavailable, skipping beta analysis: %s", ticker, exc)
        return None

    returns = bars[data.field].pct_change(fill_method=None)
    returns.name = ticker
    return returns
