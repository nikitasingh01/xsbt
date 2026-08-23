"""Wires a strategy, a calendar and a cost model into a P&L series."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from xsbt import __version__
from xsbt.config import BacktestConfig
from xsbt.data.cache import utc_now_iso, write_json_atomic
from xsbt.engine.calendar import apply_execution_lag, rebalance_dates
from xsbt.engine.costs import CostModel, LinearCostModel
from xsbt.engine.portfolio import exposures, leg_returns, simulate
from xsbt.strategies.base import Strategy

log = logging.getLogger(__name__)

DAILY_FILE = "daily.parquet"
WEIGHTS_FILE = "weights.parquet"
TARGETS_FILE = "target_weights.parquet"
METADATA_FILE = "metadata.json"
CONFIG_FILE = "config.json"


@dataclass
class BacktestResult:
    """Everything a report needs, and nothing that can't be written to disk."""

    config: BacktestConfig
    #: One row per session: returns, costs, turnover, leg P&L, exposures.
    daily: pd.DataFrame
    #: Weights held into each session.
    weights: pd.DataFrame
    #: Weights asked for, indexed by the session they were first held.
    target_weights: pd.DataFrame
    metadata: dict[str, Any]

    @property
    def returns(self) -> pd.Series:
        return self.daily["net_return"]

    @property
    def gross_returns(self) -> pd.Series:
        return self.daily["gross_return"]

    @property
    def costs(self) -> pd.Series:
        return self.daily["cost"]

    @property
    def turnover(self) -> pd.Series:
        return self.daily["turnover"]

    @property
    def legs(self) -> pd.DataFrame:
        """Daily contribution from each side, named the way analytics.attribute_legs wants."""
        return self.daily[["long_return", "short_return"]].rename(
            columns={"long_return": "long", "short_return": "short"}
        )

    @property
    def exposures(self) -> pd.DataFrame:
        return self.daily[["gross_exposure", "net_exposure", "long_names", "short_names"]]

    @property
    def benchmark(self) -> pd.Series | None:
        if "benchmark_return" not in self.daily.columns:
            return None
        return self.daily["benchmark_return"]

    def equity_curve(self, net: bool = True) -> pd.Series:
        returns = self.returns if net else self.gross_returns
        return (1.0 + returns).cumprod()

    def save(self, directory: Path | str) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.daily.to_parquet(directory / DAILY_FILE)
        self.weights.to_parquet(directory / WEIGHTS_FILE)
        self.target_weights.to_parquet(directory / TARGETS_FILE)
        write_json_atomic(directory / METADATA_FILE, self.metadata)
        write_json_atomic(directory / CONFIG_FILE, self.config.as_dict())
        return directory

    @classmethod
    def load(cls, directory: Path | str) -> BacktestResult:
        directory = Path(directory)
        expected = (DAILY_FILE, METADATA_FILE, CONFIG_FILE)
        missing = [f for f in expected if not (directory / f).exists()]
        if missing:
            raise FileNotFoundError(f"{directory}: not a run directory (missing {missing})")
        config = json.loads((directory / CONFIG_FILE).read_text())
        return cls(
            config=BacktestConfig.model_validate(config),
            daily=pd.read_parquet(directory / DAILY_FILE),
            weights=pd.read_parquet(directory / WEIGHTS_FILE),
            target_weights=pd.read_parquet(directory / TARGETS_FILE),
            metadata=json.loads((directory / METADATA_FILE).read_text()),
        )


def run_backtest(
    prices: pd.DataFrame,
    strategy: Strategy,
    config: BacktestConfig,
    *,
    dollar_volume: pd.DataFrame | None = None,
    benchmark: pd.Series | None = None,
    cost_model: CostModel | None = None,
    snapshot_id: str | None = None,
) -> BacktestResult:
    """Run ``strategy`` over ``prices`` and return the full path.

    ``prices`` is handed over whole, including the future. The strategy is trusted to
    look only backwards from the date it is asked about, and tests/test_no_lookahead.py
    is what enforces that trust.
    """
    portfolio = config.portfolio
    prices = prices.loc[str(config.data.start) : str(config.data.end)].sort_index()
    if prices.empty:
        raise ValueError(f"no price data in {config.data.start}..{config.data.end}")

    sessions = prices.index
    # fill_method=None: a gap should stay a gap rather than become a fabricated 0% day.
    returns = prices.pct_change(fill_method=None)

    signal_dates = rebalance_dates(sessions, portfolio.rebalance)
    implementation = apply_execution_lag(sessions, signal_dates, portfolio.execution_lag_days)

    books: dict[pd.Timestamp, pd.Series] = {}
    skipped = 0
    for signal_date, hold_from in implementation.items():
        weights = strategy.target_weights(prices, signal_date, dollar_volume)
        if weights.empty:
            skipped += 1
            continue
        books[hold_from] = weights * portfolio.gross_leverage

    if not books:
        raise ValueError(
            "no rebalance produced a book. Check that lookback_days fits inside the "
            "sample and that min_names is not above the universe size."
        )

    targets = pd.DataFrame(books).T.reindex(columns=prices.columns).fillna(0.0)
    targets.index.name = sessions.name

    path = simulate(returns, targets, cost_model or LinearCostModel(portfolio.cost_bps))
    legs = leg_returns(path.weights, returns)
    book = exposures(path.weights)

    daily = pd.DataFrame(
        {
            "gross_return": path.gross_returns,
            "net_return": path.net_returns,
            "cost": path.costs,
            "turnover": path.turnover,
            "long_return": legs["long"],
            "short_return": legs["short"],
            "gross_exposure": book["gross"],
            "net_exposure": book["net"],
            "long_names": book["long_names"].astype("float64"),
            "short_names": book["short_names"].astype("float64"),
        }
    )
    if benchmark is not None:
        daily["benchmark_return"] = benchmark.reindex(daily.index)

    metadata = {
        "xsbt_version": __version__,
        "run_utc": utc_now_iso(),
        "git_commit": git_commit(),
        "config_fingerprint": config.fingerprint(),
        "data_snapshot_id": snapshot_id,
        "strategy": type(strategy).__name__,
        "universe_size": int(prices.shape[1]),
        "sessions": len(sessions),
        "first_session": str(sessions.min().date()),
        "last_session": str(sessions.max().date()),
        "rebalances_scheduled": len(implementation),
        "rebalances_traded": len(books),
        "rebalances_skipped": int(skipped),
    }
    if skipped:
        log.info(
            "%d of %d rebalances skipped (too few eligible names)", skipped, len(implementation)
        )

    return BacktestResult(
        config=config,
        daily=daily,
        weights=path.weights,
        target_weights=targets,
        metadata=metadata,
    )


def git_commit() -> str | None:
    """Best-effort commit id, so a report can be traced back to the code that made it."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            cwd=Path(__file__).resolve().parent,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None
