from __future__ import annotations

from pathlib import Path

import pytest

from xsbt.data.universe import load_universe, sector_map, tickers_of

UNIVERSE = Path(__file__).resolve().parents[1] / "configs" / "universe_us_liquid.csv"


def test_loads_shipped_universe() -> None:
    members = load_universe(UNIVERSE)

    assert len(members) == 40
    assert len(set(tickers_of(members))) == 40
    assert all(m.ticker.isupper() for m in members)
    assert all(m.name and m.sector for m in members)


def test_shipped_universe_spans_sectors() -> None:
    sectors = set(sector_map(load_universe(UNIVERSE)).values())

    # A single-sector universe makes a cross-sectional rank a sector bet in disguise.
    assert len(sectors) >= 8


def test_comments_and_whitespace_are_handled(tmp_path: Path) -> None:
    path = tmp_path / "u.csv"
    path.write_text("# a note\nticker,name,sector\n aapl ,Apple, Technology \n")

    members = load_universe(path)

    assert members[0].ticker == "AAPL"
    assert members[0].sector == "Technology"


def test_duplicate_tickers_rejected(tmp_path: Path) -> None:
    path = tmp_path / "u.csv"
    path.write_text("ticker,name,sector\nAAPL,Apple,Tech\nAAPL,Apple Again,Tech\n")

    with pytest.raises(ValueError, match="duplicate tickers"):
        load_universe(path)


def test_missing_column_rejected(tmp_path: Path) -> None:
    path = tmp_path / "u.csv"
    path.write_text("ticker,name\nAAPL,Apple\n")

    with pytest.raises(ValueError, match="missing column"):
        load_universe(path)


def test_missing_file_is_explicit(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="universe file not found"):
        load_universe(tmp_path / "nope.csv")
