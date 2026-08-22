from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
import yaml

from xsbt.config import BacktestConfig

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

MINIMAL = {
    "name": "test",
    "data": {"universe": "u.csv", "start": "2020-01-01", "end": "2021-01-01"},
    "strategy": {"name": "momentum"},
}


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.yaml")))
def test_shipped_configs_are_valid(path: Path) -> None:
    config = BacktestConfig.from_yaml(path)

    assert config.name
    assert config.strategy.name in {"momentum", "reversal"}
    assert config.data.universe.exists()


def test_defaults_are_conservative() -> None:
    config = BacktestConfig.model_validate(MINIMAL)

    # The one that matters: you cannot trade the close that produced your signal.
    assert config.portfolio.execution_lag_days == 1
    assert config.portfolio.cost_bps > 0
    assert config.data.field == "adj_close"


def test_fingerprint_tracks_content_not_object_identity() -> None:
    a = BacktestConfig.model_validate(MINIMAL)
    b = BacktestConfig.model_validate(MINIMAL)
    assert a.fingerprint() == b.fingerprint()

    changed = BacktestConfig.model_validate(
        {**MINIMAL, "strategy": {"name": "momentum", "lookback_days": 63}}
    )
    assert changed.fingerprint() != a.fingerprint()


@pytest.mark.parametrize(
    ("patch", "match"),
    [
        ({"data": {**MINIMAL["data"], "end": "2019-01-01"}}, "must be before"),
        ({"strategy": {"name": "momentum", "top_fraction": 0.9}}, "top_fraction"),
        ({"strategy": {"name": "momentum", "top_fraction": 0.0}}, "top_fraction"),
        ({"strategy": {"name": "momentum", "lookback_days": 0}}, "lookback_days"),
        (
            {"strategy": {"name": "momentum", "lookback_days": 10, "skip_days": 10}},
            "no window",
        ),
        ({"portfolio": {"rebalance": "fortnightly"}}, "rebalance must be"),
        ({"portfolio": {"execution_lag_days": -1}}, "execution_lag_days"),
        ({"portfolio": {"gross_leverage": 0.0}}, "gross_leverage"),
        ({"portfolio": {"cost_bps": -1.0}}, "cost_bps"),
    ],
)
def test_rejects_bad_input(patch: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        BacktestConfig.model_validate({**MINIMAL, **patch})


def test_typos_are_rejected_rather_than_ignored() -> None:
    """extra='forbid' turns a silently-ignored misspelling into a loud failure."""
    with pytest.raises(ValueError, match="lookback_dayz"):
        BacktestConfig.model_validate(
            {**MINIMAL, "strategy": {"name": "momentum", "lookback_dayz": 63}}
        )


@pytest.mark.parametrize("value", ["M", "W", "Q", "5D", "21d", " m "])
def test_accepted_rebalance_frequencies(value: str) -> None:
    config = BacktestConfig.model_validate({**MINIMAL, "portfolio": {"rebalance": value}})

    assert config.portfolio.rebalance == value.strip().upper()


def test_from_yaml_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(MINIMAL))

    config = BacktestConfig.from_yaml(path)

    assert config.data.start == dt.date(2020, 1, 1)
    assert config.as_dict()["strategy"]["name"] == "momentum"


def test_missing_config_file_is_explicit(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="config not found"):
        BacktestConfig.from_yaml(tmp_path / "nope.yaml")
