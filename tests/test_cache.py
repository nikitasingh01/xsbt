from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.helpers import make_bars
from xsbt.data.cache import PriceCache, digest_frame

START = dt.date(2020, 1, 1)
END = dt.date(2020, 1, 14)


@pytest.fixture
def bars() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=10, name="date")
    return make_bars(dates, np.linspace(100.0, 109.0, 10))


def test_round_trip_preserves_values_and_index(tmp_path: Path, bars: pd.DataFrame) -> None:
    cache = PriceCache(tmp_path)
    cache.write("AAA", bars, source="stub", requested_start=START, requested_end=END)

    # check_freq=False: parquet doesn't carry the inferred DatetimeIndex freq, and real
    # price data never has one anyway (holidays).
    pd.testing.assert_frame_equal(cache.read("AAA"), bars, check_freq=False)


def test_manifest_records_what_was_fetched(tmp_path: Path, bars: pd.DataFrame) -> None:
    cache = PriceCache(tmp_path)
    entry = cache.write("AAA", bars, source="stub", requested_start=START, requested_end=END)

    assert entry.rows == 10
    assert entry.first_date == "2020-01-01"
    assert entry.last_date == "2020-01-14"
    assert entry.requested_start == "2020-01-01"
    assert entry.requested_end == "2020-01-14"
    assert entry.source == "stub"
    assert entry.sha256 == digest_frame(bars)
    assert entry.fetched_utc.endswith("+00:00")


def test_manifest_survives_reload(tmp_path: Path, bars: pd.DataFrame) -> None:
    PriceCache(tmp_path).write("AAA", bars, source="stub", requested_start=START, requested_end=END)

    reopened = PriceCache(tmp_path)

    assert reopened.has("AAA")
    assert reopened.manifest.entries["AAA"].rows == 10


def test_snapshot_id_is_stable_and_content_addressed(tmp_path: Path, bars: pd.DataFrame) -> None:
    cache = PriceCache(tmp_path)
    cache.write("AAA", bars, source="stub", requested_start=START, requested_end=END)
    first = cache.snapshot_id

    # Rewriting identical data must not move the id, or every run looks like a new one.
    cache.write("AAA", bars, source="stub", requested_start=START, requested_end=END)
    assert cache.snapshot_id == first

    # Yahoo restating a price is exactly what this is meant to catch.
    restated = bars.copy()
    restated.iloc[0, restated.columns.get_loc("adj_close")] = 999.0
    cache.write("AAA", restated, source="stub", requested_start=START, requested_end=END)
    assert cache.snapshot_id != first


def test_verify_flags_tampering(tmp_path: Path, bars: pd.DataFrame) -> None:
    cache = PriceCache(tmp_path)
    cache.write("AAA", bars, source="stub", requested_start=START, requested_end=END)
    assert cache.verify() == {}

    tampered = bars.copy()
    tampered.iloc[3, tampered.columns.get_loc("close")] = 0.0
    tampered.to_parquet(cache.path_for("AAA"))

    problems = cache.verify()
    assert "AAA" in problems
    assert "hash mismatch" in problems["AAA"]


def test_verify_flags_missing_file(tmp_path: Path, bars: pd.DataFrame) -> None:
    cache = PriceCache(tmp_path)
    cache.write("AAA", bars, source="stub", requested_start=START, requested_end=END)
    cache.path_for("AAA").unlink()

    assert cache.verify() == {"AAA": "file missing"}


def test_covers_uses_requested_window_not_returned_data(tmp_path: Path, bars: pd.DataFrame) -> None:
    """A name that listed in 2015 has no 2010 bars. If coverage were judged on the data
    that came back we would refetch it forever."""
    cache = PriceCache(tmp_path)
    cache.write(
        "AAA",
        bars,
        source="stub",
        requested_start=dt.date(2010, 1, 1),
        requested_end=dt.date(2020, 1, 14),
    )

    assert cache.covers("AAA", dt.date(2010, 1, 1), dt.date(2020, 1, 14))
    assert cache.covers("AAA", dt.date(2015, 6, 1), dt.date(2019, 1, 1))
    assert not cache.covers("AAA", dt.date(2009, 1, 1), dt.date(2020, 1, 14))
    assert not cache.covers("AAA", dt.date(2010, 1, 1), dt.date(2021, 1, 1))
    assert not cache.covers("BBB", dt.date(2010, 1, 1), dt.date(2020, 1, 14))


def test_unknown_schema_version_refuses_to_load(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({"schema_version": 99, "entries": {}}))

    with pytest.raises(ValueError, match="schema"):
        PriceCache(tmp_path)


def test_missing_manifest_starts_empty(tmp_path: Path) -> None:
    cache = PriceCache(tmp_path)

    assert cache.manifest.entries == {}
    assert not cache.has("AAA")
