"""Universe definitions, loaded from CSV."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("ticker", "name", "sector")


@dataclass(frozen=True)
class UniverseMember:
    ticker: str
    name: str
    sector: str


def load_universe(path: Path | str) -> list[UniverseMember]:
    """Read a universe CSV with columns ticker, name, sector."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"universe file not found: {path}")

    frame = pd.read_csv(path, comment="#")
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing column(s) {missing}")

    frame["ticker"] = frame["ticker"].str.strip().str.upper()
    duplicated = frame.loc[frame["ticker"].duplicated(), "ticker"].tolist()
    if duplicated:
        raise ValueError(f"{path}: duplicate tickers {sorted(set(duplicated))}")

    return [
        UniverseMember(
            ticker=row["ticker"],
            name=str(row["name"]).strip(),
            sector=str(row["sector"]).strip(),
        )
        for _, row in frame.iterrows()
    ]


def tickers_of(members: list[UniverseMember]) -> list[str]:
    return [m.ticker for m in members]


def sector_map(members: list[UniverseMember]) -> dict[str, str]:
    return {m.ticker: m.sector for m in members}
