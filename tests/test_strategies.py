from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xsbt.config import StrategyConfig
from xsbt.strategies import REGISTRY, Momentum, Reversal, build

BASE = StrategyConfig(name="momentum", lookback_days=5, skip_days=0, top_fraction=0.25, min_names=4)


def config(**overrides: object) -> StrategyConfig:
    return StrategyConfig.model_validate({**BASE.model_dump(), **overrides})


def test_momentum_buys_winners(toy_panel: pd.DataFrame) -> None:
    weights = Momentum(config()).target_weights(toy_panel, toy_panel.index[7])

    assert weights["AAA"] == pytest.approx(0.5)
    assert weights["DDD"] == pytest.approx(-0.5)
    assert weights["BBB"] == 0.0


def test_reversal_is_momentum_with_the_sign_flipped(toy_panel: pd.DataFrame) -> None:
    asof = toy_panel.index[7]
    settings = config()

    momentum = Momentum(settings).target_weights(toy_panel, asof)
    reversal = Reversal(config(name="reversal")).target_weights(toy_panel, asof)

    pd.testing.assert_series_equal(reversal, -momentum)


def test_book_is_dollar_neutral_at_unit_gross(toy_panel: pd.DataFrame) -> None:
    weights = Momentum(config()).target_weights(toy_panel, toy_panel.index[7])

    assert weights.sum() == pytest.approx(0.0)
    assert weights.abs().sum() == pytest.approx(1.0)


def test_skip_days_moves_the_measurement_window() -> None:
    """AAA rallies then crashes in the last two sessions; DDD does the reverse. Whether
    momentum is long AAA or long DDD is entirely down to whether the skip is applied."""
    dates = pd.bdate_range("2020-01-01", periods=8, name="date")
    panel = pd.DataFrame(
        {
            "AAA": [100, 100, 100, 110, 120, 130, 80, 70],
            "BBB": [100] * 8,
            "CCC": [100] * 8,
            "DDD": [100, 100, 100, 95, 90, 85, 120, 130],
        },
        index=dates,
        dtype="float64",
    )
    asof = dates[7]

    no_skip = Momentum(config(lookback_days=5, skip_days=0)).target_weights(panel, asof)
    skipped = Momentum(config(lookback_days=5, skip_days=2)).target_weights(panel, asof)

    assert no_skip["DDD"] == pytest.approx(0.5)
    assert no_skip["AAA"] == pytest.approx(-0.5)
    assert skipped["AAA"] == pytest.approx(0.5)
    assert skipped["DDD"] == pytest.approx(-0.5)


def test_too_little_history_produces_no_book(toy_panel: pd.DataFrame) -> None:
    weights = Momentum(config(lookback_days=5)).target_weights(toy_panel, toy_panel.index[2])

    assert weights.empty


def test_names_missing_at_either_end_of_the_window_are_dropped(
    toy_panel: pd.DataFrame,
) -> None:
    panel = toy_panel.copy()
    panel.loc[panel.index[2], "AAA"] = np.nan  # start of the window for asof = index[7]

    weights = Momentum(config(min_names=2)).target_weights(panel, panel.index[7])

    assert "AAA" not in weights.index
    assert weights.abs().sum() == pytest.approx(1.0)


def test_name_not_priced_on_the_rebalance_date_is_dropped(toy_panel: pd.DataFrame) -> None:
    panel = toy_panel.copy()
    panel.loc[panel.index[7], "AAA"] = np.nan

    weights = Momentum(config(min_names=2)).target_weights(panel, panel.index[7])

    assert "AAA" not in weights.index


def test_thin_universe_is_skipped_rather_than_guessed(toy_panel: pd.DataFrame) -> None:
    panel = toy_panel.copy()
    panel.loc[panel.index[2], ["AAA", "BBB"]] = np.nan

    weights = Momentum(config(min_names=4)).target_weights(panel, panel.index[7])

    assert weights.empty


def test_dollar_volume_floor_excludes_illiquid_names(toy_panel: pd.DataFrame) -> None:
    volume = pd.DataFrame(1e9, index=toy_panel.index, columns=toy_panel.columns)
    volume["AAA"] = 1.0

    weights = Momentum(config(min_names=2, min_dollar_volume=1e6)).target_weights(
        toy_panel, toy_panel.index[7], dollar_volume=volume
    )

    assert "AAA" not in weights.index


def test_leg_size_follows_top_fraction() -> None:
    dates = pd.bdate_range("2020-01-01", periods=10, name="date")
    panel = pd.DataFrame(
        {f"T{i:02d}": 100.0 * (1.0 + i / 100) ** np.arange(10) for i in range(10)},
        index=dates,
    )

    weights = Momentum(config(top_fraction=0.2, min_names=4)).target_weights(panel, dates[7])

    assert (weights > 0).sum() == 2
    assert (weights < 0).sum() == 2
    assert weights.max() == pytest.approx(0.25)


def test_ties_are_broken_deterministically() -> None:
    dates = pd.bdate_range("2020-01-01", periods=10, name="date")
    panel = pd.DataFrame({t: [100.0] * 10 for t in ("AAA", "BBB", "CCC", "DDD")}, index=dates)
    strategy = Momentum(config())

    first = strategy.target_weights(panel, dates[7])
    second = strategy.target_weights(panel, dates[7])

    pd.testing.assert_series_equal(first, second)
    assert first.abs().sum() == pytest.approx(1.0)


def test_registry_resolves_by_name() -> None:
    assert set(REGISTRY) == {"momentum", "reversal"}
    assert isinstance(build(config(name="momentum")), Momentum)
    assert isinstance(build(config(name="reversal")), Reversal)


def test_unknown_strategy_lists_what_is_available() -> None:
    with pytest.raises(KeyError, match="registered"):
        build(config(name="wishful"))
