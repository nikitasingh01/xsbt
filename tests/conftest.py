from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def aapl_payload() -> dict[str, Any]:
    return json.loads((FIXTURES / "yahoo_aapl_2023q1.json").read_text())


@pytest.fixture
def not_found_payload() -> dict[str, Any]:
    return json.loads((FIXTURES / "yahoo_not_found.json").read_text())


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collapse backoff and throttling so retry tests run instantly."""
    import xsbt.data.yahoo as yahoo

    monkeypatch.setattr(yahoo.time, "sleep", lambda _seconds: None)
