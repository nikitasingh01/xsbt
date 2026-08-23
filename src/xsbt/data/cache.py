"""On-disk price snapshot: one parquet per ticker plus a manifest.

The manifest is what makes a run reproducible. Yahoo restates adj_close every time a
dividend or split lands, so "the same query" run a month apart gives different history.
Recording when each ticker was pulled and hashing its contents turns the cache into a
dated snapshot we can point at, rather than a moving target.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from xsbt.data.base import PRICE_COLUMNS

log = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 1


def digest_frame(frame: pd.DataFrame) -> str:
    """Content hash of a bar frame.

    Hashes a canonical CSV rather than the parquet bytes, which are not stable across
    pyarrow versions.
    """
    payload = frame.to_csv(float_format="%.10g").encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CacheEntry:
    """What we know about one cached ticker."""

    ticker: str
    source: str
    fetched_utc: str
    rows: int
    first_date: str | None
    last_date: str | None
    # The window that was asked for, which is not the window that came back: a name that
    # listed in 2015 has no 2010 bars. Without this we would refetch it on every run.
    requested_start: str
    requested_end: str
    sha256: str

    def covers(self, start: dt.date, end: dt.date) -> bool:
        return _parse_date(self.requested_start) <= start and _parse_date(self.requested_end) >= end


@dataclass
class Manifest:
    schema_version: int
    created_utc: str
    updated_utc: str
    entries: dict[str, CacheEntry]

    @classmethod
    def empty(cls) -> Manifest:
        now = utc_now_iso()
        return cls(schema_version=SCHEMA_VERSION, created_utc=now, updated_utc=now, entries={})

    @classmethod
    def load(cls, path: Path) -> Manifest:
        if not path.exists():
            return cls.empty()
        raw: dict[str, Any] = json.loads(path.read_text())
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"{path}: manifest schema {raw.get('schema_version')} != {SCHEMA_VERSION}. "
                "Delete the cache directory and refetch."
            )
        entries = {t: CacheEntry(**e) for t, e in raw.get("entries", {}).items()}
        return cls(
            schema_version=raw["schema_version"],
            created_utc=raw["created_utc"],
            updated_utc=raw["updated_utc"],
            entries=entries,
        )

    def save(self, path: Path) -> None:
        self.updated_utc = utc_now_iso()
        body = {
            "schema_version": self.schema_version,
            "created_utc": self.created_utc,
            "updated_utc": self.updated_utc,
            "snapshot_id": self.snapshot_id,
            "entries": {t: asdict(e) for t, e in sorted(self.entries.items())},
        }
        write_json_atomic(path, body)

    @property
    def snapshot_id(self) -> str:
        """One id for the state of the whole cache. Goes into every backtest result."""
        joined = "\n".join(f"{t}:{e.sha256}" for t, e in sorted(self.entries.items()))
        return hashlib.sha256(joined.encode()).hexdigest()


class PriceCache:
    """Parquet-backed store of raw vendor bars."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.prices_dir = self.root / "prices"
        self.manifest_path = self.root / MANIFEST_NAME
        self.manifest = Manifest.load(self.manifest_path)

    def path_for(self, ticker: str) -> Path:
        # Yahoo writes class B shares as BRK-B, but other vendors write BRK/B, and that
        # would silently become a subdirectory. Normalise before it reaches the filesystem.
        return self.prices_dir / f"{ticker.replace('/', '_')}.parquet"

    def has(self, ticker: str) -> bool:
        return ticker in self.manifest.entries and self.path_for(ticker).exists()

    def covers(self, ticker: str, start: dt.date, end: dt.date) -> bool:
        entry = self.manifest.entries.get(ticker)
        return bool(entry and self.path_for(ticker).exists() and entry.covers(start, end))

    def read(self, ticker: str) -> pd.DataFrame:
        frame = pd.read_parquet(self.path_for(ticker))
        return frame[list(PRICE_COLUMNS)].sort_index()

    def write(
        self,
        ticker: str,
        frame: pd.DataFrame,
        *,
        source: str,
        requested_start: dt.date,
        requested_end: dt.date,
    ) -> CacheEntry:
        self.prices_dir.mkdir(parents=True, exist_ok=True)
        frame = frame[list(PRICE_COLUMNS)].sort_index()

        path = self.path_for(ticker)
        tmp = path.with_suffix(".parquet.tmp")
        frame.to_parquet(tmp, engine="pyarrow", index=True)
        tmp.replace(path)

        entry = CacheEntry(
            ticker=ticker,
            source=source,
            fetched_utc=utc_now_iso(),
            rows=len(frame),
            first_date=_iso_or_none(frame.index.min()),
            last_date=_iso_or_none(frame.index.max()),
            requested_start=requested_start.isoformat(),
            requested_end=requested_end.isoformat(),
            sha256=digest_frame(frame),
        )
        self.manifest.entries[ticker] = entry
        self.manifest.save(self.manifest_path)
        return entry

    def verify(self) -> dict[str, str]:
        """Re-hash every cached file. Returns ticker -> complaint for anything that drifted."""
        problems: dict[str, str] = {}
        for ticker, entry in self.manifest.entries.items():
            path = self.path_for(ticker)
            if not path.exists():
                problems[ticker] = "file missing"
                continue
            actual = digest_frame(self.read(ticker))
            if actual != entry.sha256:
                problems[ticker] = (
                    f"hash mismatch (manifest {entry.sha256[:12]}, file {actual[:12]})"
                )
        return problems

    @property
    def snapshot_id(self) -> str:
        return self.manifest.snapshot_id


def utc_now_iso() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def write_json_atomic(path: Path, body: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(body, indent=2, sort_keys=False) + "\n")
    tmp.replace(path)


def _iso_or_none(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return str(pd.Timestamp(value).date())


def _parse_date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)
