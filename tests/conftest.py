from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.helpers import make_bars

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def aapl_payload() -> dict[str, Any]:
    return json.loads((FIXTURES / "yahoo_aapl_2023q1.json").read_text())


@pytest.fixture
def not_found_payload() -> dict[str, Any]:
    return json.loads((FIXTURES / "yahoo_not_found.json").read_text())


@pytest.fixture
def toy_panel() -> pd.DataFrame:
    """Four names over ten business days, prices chosen so the ranking is unambiguous.

    Daily rates: A +2%, B +1%, C -1%, D -2%. Over any trailing window momentum ranks
    A > B > C > D and reversal ranks the reverse, with no ties to break.
    """
    dates = pd.bdate_range("2020-01-01", periods=10, name="date")
    rates = {"AAA": 0.02, "BBB": 0.01, "CCC": -0.01, "DDD": -0.02}
    data = {t: 100.0 * (1.0 + r) ** np.arange(len(dates)) for t, r in rates.items()}
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def toy_frames(toy_panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {t: make_bars(toy_panel.index, toy_panel[t].to_numpy()) for t in toy_panel.columns}


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collapse backoff and throttling so retry tests run instantly."""
    import xsbt.data.yahoo as yahoo

    monkeypatch.setattr(yahoo.time, "sleep", lambda _seconds: None)
