from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from tests.helpers import StubSource
from xsbt.data.base import CacheMissError
from xsbt.data.cache import PriceCache
from xsbt.data.repository import PriceRepository

START = dt.date(2020, 1, 1)
END = dt.date(2020, 1, 14)


@pytest.fixture
def repo(tmp_path: Path, toy_frames: dict[str, pd.DataFrame]) -> PriceRepository:
    return PriceRepository(PriceCache(tmp_path), StubSource(toy_frames))


def test_second_read_comes_from_cache(repo: PriceRepository) -> None:
    repo.get("AAA", START, END)
    repo.get("AAA", START, END)

    source = repo.source
    assert isinstance(source, StubSource)
    assert source.fetch_calls == ["AAA"]


def test_refresh_forces_a_refetch(repo: PriceRepository) -> None:
    repo.get("AAA", START, END)
    repo.get("AAA", START, END, refresh=True)

    assert isinstance(repo.source, StubSource)
    assert repo.source.fetch_calls == ["AAA", "AAA"]


def test_widening_the_window_refetches(repo: PriceRepository) -> None:
    repo.get("AAA", dt.date(2020, 1, 6), END)
    repo.get("AAA", START, END)

    assert isinstance(repo.source, StubSource)
    assert repo.source.fetch_calls == ["AAA", "AAA"]


def test_offline_miss_is_an_error(tmp_path: Path) -> None:
    repo = PriceRepository(PriceCache(tmp_path), offline=True)

    with pytest.raises(CacheMissError, match="xsbt fetch"):
        repo.get("AAA", START, END)


def test_offline_hit_works(tmp_path: Path, toy_frames: dict[str, pd.DataFrame]) -> None:
    cache = PriceCache(tmp_path)
    PriceRepository(cache, StubSource(toy_frames)).get("AAA", START, END)

    offline = PriceRepository(PriceCache(tmp_path), offline=True)

    assert len(offline.get("AAA", START, END)) == 10


def test_offline_without_source_is_allowed_but_online_is_not(tmp_path: Path) -> None:
    PriceRepository(PriceCache(tmp_path), offline=True)

    with pytest.raises(ValueError, match="source is required"):
        PriceRepository(PriceCache(tmp_path))


def test_get_many_collects_failures_without_aborting(repo: PriceRepository) -> None:
    frames, failures = repo.get_many(["AAA", "NOPE", "BBB"], START, END)

    assert sorted(frames) == ["AAA", "BBB"]
    assert list(failures) == ["NOPE"]


def test_panel_is_wide_and_sorted(repo: PriceRepository) -> None:
    panel, failures = repo.panel(["BBB", "AAA"], START, END)

    assert failures == {}
    assert list(panel.columns) == ["AAA", "BBB"]
    assert panel.index.name == "date"
    assert panel.index.is_monotonic_increasing
    assert panel.shape == (10, 2)
