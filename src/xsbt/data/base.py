"""Interfaces shared by every price source."""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import pandas as pd

PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "adj_close", "volume")

#: Corporate actions carried alongside the bars, NaN on every session without one. Kept
#: because ``adj_close`` is the vendor's own derived field and these are what it is derived
#: from, so holding both lets us check one against the other (see data/adjustment.py).
EVENT_COLUMNS: tuple[str, ...] = ("dividend", "split_ratio")

#: The full on-disk and in-memory schema for one ticker.
BAR_COLUMNS: tuple[str, ...] = PRICE_COLUMNS + EVENT_COLUMNS

DATE_INDEX_NAME = "date"


class DataError(Exception):
    """Base class for price retrieval failures."""


class TickerNotFoundError(DataError):
    """Symbol is unknown to the source. Permanent, so don't retry."""


class FetchError(DataError):
    """Transient failure: network, rate limit, or a malformed payload."""


class CacheMissError(DataError):
    """Requested in offline mode but not in the local snapshot."""


@runtime_checkable
class PriceSource(Protocol):
    """Daily bars for one symbol.

    Implement this to swap Yahoo for a vendor feed, a CSV drop or a database. Nothing
    downstream of the repository knows which one it is talking to.
    """

    name: str

    def fetch(self, ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Bars over ``[start, end]`` inclusive, indexed by date, columns BAR_COLUMNS."""
        ...


def to_panel(frames: Mapping[str, pd.DataFrame], field: str = "adj_close") -> pd.DataFrame:
    """Pivot per-ticker frames into a wide ``date x ticker`` panel of one field.

    Uses the union of dates, so a name that wasn't trading shows up as NaN. We don't
    forward-fill here; what to do about gaps is the strategy's call, not the data
    layer's (see CrossSectionalRankStrategy.eligible).
    """
    if not frames:
        return pd.DataFrame(index=pd.DatetimeIndex([], name=DATE_INDEX_NAME))

    missing = sorted(t for t, f in frames.items() if field not in f.columns)
    if missing:
        raise KeyError(f"field {field!r} not present for: {missing}")

    panel = pd.DataFrame({ticker: frame[field] for ticker, frame in frames.items()})
    panel = panel.sort_index().sort_index(axis=1)
    panel.index.name = DATE_INDEX_NAME
    panel.columns.name = "ticker"
    return panel
