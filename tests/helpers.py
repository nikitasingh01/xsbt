"""Test doubles and small builders shared across the suite."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


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
