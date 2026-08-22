"""Cache-first access to prices. This is what the rest of the system talks to."""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Iterable

import pandas as pd

from xsbt.data.base import CacheMissError, DataError, PriceSource, to_panel
from xsbt.data.cache import PriceCache

log = logging.getLogger(__name__)


class PriceRepository:
    """Reads from the local snapshot, falling back to the source on a miss.

    In offline mode a miss is an error rather than a fetch, which is what makes CI and
    any re-run of an old result honest: nothing can quietly reach out to the internet
    and change the answer.
    """

    def __init__(
        self,
        cache: PriceCache,
        source: PriceSource | None = None,
        *,
        offline: bool = False,
    ) -> None:
        if source is None and not offline:
            raise ValueError("a source is required unless offline=True")
        self.cache = cache
        self.source = source
        self.offline = offline

    def get(
        self,
        ticker: str,
        start: dt.date,
        end: dt.date,
        *,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Bars for one ticker over ``[start, end]``, fetching only if we have to."""
        if not refresh and self.cache.covers(ticker, start, end):
            return self._slice(self.cache.read(ticker), start, end)

        if self.offline:
            raise CacheMissError(
                f"{ticker}: {start}..{end} not in snapshot at {self.cache.root} "
                f"(offline mode). Run `xsbt fetch` first."
            )

        assert self.source is not None  # guarded in __init__
        log.info("fetching %s %s..%s from %s", ticker, start, end, self.source.name)
        frame = self.source.fetch(ticker, start, end)
        self.cache.write(
            ticker,
            frame,
            source=self.source.name,
            requested_start=start,
            requested_end=end,
        )
        return self._slice(frame, start, end)

    def get_many(
        self,
        tickers: Iterable[str],
        start: dt.date,
        end: dt.date,
        *,
        refresh: bool = False,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
        """Fetch a list of tickers.

        Returns ``(frames, failures)``. One dead symbol shouldn't sink a 40-name pull, so
        failures are collected and handed back for the caller to report on.
        """
        frames: dict[str, pd.DataFrame] = {}
        failures: dict[str, str] = {}
        for ticker in tickers:
            try:
                frames[ticker] = self.get(ticker, start, end, refresh=refresh)
            except DataError as exc:
                log.warning("%s: %s", ticker, exc)
                failures[ticker] = str(exc)
        return frames, failures

    def panel(
        self,
        tickers: Iterable[str],
        start: dt.date,
        end: dt.date,
        *,
        field: str = "adj_close",
        refresh: bool = False,
    ) -> tuple[pd.DataFrame, dict[str, str]]:
        """Wide ``date x ticker`` panel of one field, plus whatever failed."""
        frames, failures = self.get_many(tickers, start, end, refresh=refresh)
        return to_panel(frames, field=field), failures

    @staticmethod
    def _slice(frame: pd.DataFrame, start: dt.date, end: dt.date) -> pd.DataFrame:
        return frame.loc[str(start) : str(end)]
