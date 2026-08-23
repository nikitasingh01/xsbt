"""Test doubles and small builders shared across the suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from xsbt.config import BacktestConfig, DataConfig, PortfolioConfig, StrategyConfig
from xsbt.data.cache import PriceCache


class FakeResponse:
    def __init__(self, status_code: int, payload: Any = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    """Replays a scripted list of responses (or exceptions) in order."""

    def __init__(self, responses: list[Any]) -> None:
        self.headers: dict[str, str] = {}
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, params: dict[str, Any] | None = None, timeout: float = 0) -> Any:
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        if not self.responses:
            raise AssertionError("FakeSession ran out of scripted responses")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class StubSource:
    """A PriceSource backed by an in-memory dict, so cache tests need no network."""

    name = "stub"

    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames = frames
        self.fetch_calls: list[str] = []

    def fetch(self, ticker: str, start: Any, end: Any) -> pd.DataFrame:
        from xsbt.data.base import TickerNotFoundError

        self.fetch_calls.append(ticker)
        if ticker not in self.frames:
            raise TickerNotFoundError(f"{ticker}: not in stub")
        return self.frames[ticker].loc[str(start) : str(end)]


def make_bars(dates: pd.DatetimeIndex, closes: np.ndarray, volume: float = 1e7) -> pd.DataFrame:
    """Bar frame in the shape a PriceSource returns."""
    closes = np.asarray(closes, dtype="float64")
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "adj_close": closes,
            "volume": np.full(len(closes), volume),
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )


def random_walk_panel(
    names: int = 8, sessions: int = 520, seed: int = 20240117, start: str = "2018-01-01"
) -> pd.DataFrame:
    """A seeded lognormal walk, for tests that need a plausible panel rather than a toy."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=sessions, name="date")
    steps = rng.normal(0.0004, 0.012, size=(sessions, names))
    return pd.DataFrame(
        100.0 * np.exp(steps.cumsum(axis=0)),
        index=dates,
        columns=[f"T{i:02d}" for i in range(names)],
    )


def make_backtest_config(panel: pd.DataFrame, **portfolio: Any) -> BacktestConfig:
    """A runnable config pinned to whatever window ``panel`` covers."""
    return BacktestConfig(
        name="unit_test",
        data=DataConfig(
            universe=Path("configs/universe_us_liquid.csv"),
            start=panel.index[0].date(),
            end=panel.index[-1].date(),
        ),
        strategy=StrategyConfig(
            name="momentum", lookback_days=60, skip_days=0, top_fraction=0.25, min_names=4
        ),
        portfolio=PortfolioConfig.model_validate(portfolio),
    )


def seed_cache(
    root: Path, panel: pd.DataFrame, *, benchmark: pd.Series | None = None
) -> PriceCache:
    """Write a panel into a price cache so offline commands have something to read.

    The requested window is recorded as the panel's own span, which is what
    PriceRepository checks before it decides a ticker needs refetching.
    """
    cache = PriceCache(root)
    start, end = panel.index[0].date(), panel.index[-1].date()

    columns: dict[str, pd.Series] = {t: panel[t] for t in panel.columns}
    if benchmark is not None:
        columns[str(benchmark.name)] = benchmark

    for ticker, closes in columns.items():
        cache.write(
            ticker,
            make_bars(pd.DatetimeIndex(closes.index), closes.to_numpy()),
            source="stub",
            requested_start=start,
            requested_end=end,
        )
    return cache


def write_universe(path: Path, tickers: list[str]) -> Path:
    """Minimal universe CSV in the shape load_universe expects."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(f"{t},{t} Inc,Test" for t in tickers)
    path.write_text(f"ticker,name,sector\n{rows}\n")
    return path
