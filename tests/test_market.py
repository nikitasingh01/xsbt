"""Assembling a config into the panels the engine wants."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from tests.helpers import seed_cache, write_universe
from xsbt.config import DataConfig
from xsbt.data.base import CacheMissError, DataError
from xsbt.data.market import load_market_data, open_repository


@pytest.fixture
def panel() -> pd.DataFrame:
    dates = pd.bdate_range("2021-01-04", periods=30, name="date")
    return pd.DataFrame(
        {t: 100.0 + i + pd.Series(range(len(dates)), index=dates) for i, t in enumerate("ABCD")}
    )


@pytest.fixture
def benchmark(panel: pd.DataFrame) -> pd.Series:
    return pd.Series(200.0 + 0.5 * pd.Series(range(len(panel)), index=panel.index), name="SPY")


@pytest.fixture
def cache_dir(tmp_path: Path, panel: pd.DataFrame, benchmark: pd.Series) -> Path:
    root = tmp_path / "cache"
    seed_cache(root, panel, benchmark=benchmark)
    return root


def make_data_config(
    tmp_path: Path,
    cache_dir: Path,
    panel: pd.DataFrame,
    *,
    tickers: list[str] | None = None,
    benchmark: str | None = "SPY",
) -> DataConfig:
    universe = write_universe(tmp_path / "universe.csv", tickers or list(panel.columns))
    return DataConfig(
        universe=universe,
        start=panel.index[0].date(),
        end=panel.index[-1].date(),
        cache_dir=cache_dir,
        benchmark=benchmark,
    )


def test_panels_are_aligned_and_carry_the_whole_universe(
    tmp_path: Path, cache_dir: Path, panel: pd.DataFrame
) -> None:
    data = make_data_config(tmp_path, cache_dir, panel)
    market = load_market_data(data, open_repository(data, offline=True))

    assert list(market.prices.columns) == ["A", "B", "C", "D"]
    assert market.prices.shape == market.dollar_volume.shape
    assert market.prices.index.equals(market.dollar_volume.index)
    assert market.missing == {}


def test_dollar_volume_is_close_times_volume(
    tmp_path: Path, cache_dir: Path, panel: pd.DataFrame
) -> None:
    """seed_cache writes a flat 1e7 share volume, so the notional is just the price."""
    data = make_data_config(tmp_path, cache_dir, panel)
    market = load_market_data(data, open_repository(data, offline=True))

    pd.testing.assert_frame_equal(market.dollar_volume, market.prices * 1e7, check_names=False)


def test_the_benchmark_is_returns_and_stays_out_of_the_tradeable_universe(
    tmp_path: Path, cache_dir: Path, panel: pd.DataFrame
) -> None:
    data = make_data_config(tmp_path, cache_dir, panel)
    market = load_market_data(data, open_repository(data, offline=True))

    assert market.benchmark is not None
    assert "SPY" not in market.prices.columns
    assert market.benchmark.name == "SPY"
    # First session has nothing to difference against.
    assert pd.isna(market.benchmark.iloc[0])
    assert market.benchmark.iloc[1] == pytest.approx(0.5 / 200.0)


def test_no_benchmark_configured_means_no_benchmark_series(
    tmp_path: Path, cache_dir: Path, panel: pd.DataFrame
) -> None:
    data = make_data_config(tmp_path, cache_dir, panel, benchmark=None)
    market = load_market_data(data, open_repository(data, offline=True))

    assert market.benchmark is None


def test_an_unavailable_benchmark_is_a_warning_not_a_failed_run(
    tmp_path: Path, cache_dir: Path, panel: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    """Losing the beta section is worth less than losing the whole backtest."""
    data = make_data_config(tmp_path, cache_dir, panel, benchmark="NOTCACHED")

    with caplog.at_level("WARNING"):
        market = load_market_data(data, open_repository(data, offline=True))

    assert market.benchmark is None
    assert market.prices.shape[1] == 4
    assert "NOTCACHED" in caplog.text


def test_one_dead_ticker_is_reported_and_the_rest_still_load(
    tmp_path: Path, cache_dir: Path, panel: pd.DataFrame
) -> None:
    data = make_data_config(tmp_path, cache_dir, panel, tickers=["A", "B", "ZZZZ"])
    market = load_market_data(data, open_repository(data, offline=True))

    assert list(market.prices.columns) == ["A", "B"]
    assert list(market.missing) == ["ZZZZ"]


def test_an_empty_cache_says_to_fetch_rather_than_returning_nothing(
    tmp_path: Path, panel: pd.DataFrame
) -> None:
    data = make_data_config(tmp_path, tmp_path / "empty", panel)

    with pytest.raises(DataError, match="xsbt fetch"):
        load_market_data(data, open_repository(data, offline=True))


def test_the_snapshot_id_is_the_cache_it_was_read_from(
    tmp_path: Path, cache_dir: Path, panel: pd.DataFrame
) -> None:
    data = make_data_config(tmp_path, cache_dir, panel)
    repository = open_repository(data, offline=True)

    market = load_market_data(data, repository)

    assert market.snapshot_id == repository.cache.snapshot_id
    assert len(market.snapshot_id) == 64


def test_offline_repositories_have_no_source_to_reach_out_with(
    tmp_path: Path, cache_dir: Path, panel: pd.DataFrame
) -> None:
    data = make_data_config(tmp_path, cache_dir, panel)
    repository = open_repository(data, offline=True)

    assert repository.source is None
    with pytest.raises(CacheMissError):
        repository.get("NEVERSEEN", dt.date(2021, 1, 4), dt.date(2021, 2, 12))


def test_the_cache_directory_can_be_overridden(
    tmp_path: Path, cache_dir: Path, panel: pd.DataFrame
) -> None:
    data = make_data_config(tmp_path, tmp_path / "elsewhere", panel)

    repository = open_repository(data, offline=True, cache_dir=cache_dir)

    assert repository.cache.root == cache_dir
