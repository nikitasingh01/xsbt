"""Strategy interface and the cross-sectional rank base class.

Adding a strategy means writing a ``score`` method. Everything else (windowing,
eligibility, ranking, weighting, dollar-neutrality) is handled here, so momentum and
reversal differ by a minus sign and nothing else.
"""

from __future__ import annotations

import abc
from typing import ClassVar, Protocol, TypeVar, runtime_checkable

import pandas as pd

from xsbt.config import StrategyConfig


@runtime_checkable
class Strategy(Protocol):
    """Turns a price history into a book, as at one date."""

    # A property rather than a plain attribute so that subclasses are free to pin the
    # name as a ClassVar, which is what the rank strategies below do.
    @property
    def name(self) -> str: ...

    def target_weights(
        self,
        prices: pd.DataFrame,
        asof: pd.Timestamp,
        dollar_volume: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Weights to hold, summing to 1.0 gross. Empty means 'no view, hold what you have'."""
        ...


@runtime_checkable
class StrategyFactory(Protocol):
    """What the registry holds: something named, that a config can construct.

    Deliberately keyed to ``Strategy`` and not to the rank base below. A strategy that
    satisfies the protocol without inheriting from anything is still nameable in a YAML
    config, which is the whole point of making the seam structural.
    """

    name: str

    def __call__(self, config: StrategyConfig) -> Strategy: ...


REGISTRY: dict[str, StrategyFactory] = {}

F = TypeVar("F", bound=StrategyFactory)


def register(cls: F) -> F:
    REGISTRY[cls.name] = cls
    return cls


def build(config: StrategyConfig) -> Strategy:
    """Instantiate the strategy named in a config."""
    try:
        factory = REGISTRY[config.name]
    except KeyError:
        raise KeyError(
            f"unknown strategy {config.name!r}; registered: {sorted(REGISTRY)}"
        ) from None
    return factory(config)


class CrossSectionalRankStrategy(abc.ABC):
    """Rank the universe, go long the top slice and short the bottom, equal weighted.

    Weights are produced at gross 1.0 and dollar neutral. Scaling to the book's actual
    leverage is the engine's job, so the strategy never needs to know about it.
    """

    name: ClassVar[str]

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    def __repr__(self) -> str:
        c = self.config
        return (
            f"{type(self).__name__}(lookback={c.lookback_days}, skip={c.skip_days}, "
            f"top={c.top_fraction:g})"
        )

    @abc.abstractmethod
    def score(self, window: pd.DataFrame) -> pd.Series:
        """Rank score per name; higher means we want to be long it.

        ``window`` is the measurement window of prices with the skip already applied, so
        the first row is ``asof - lookback_days`` and the last is ``asof - skip_days``.
        """

    def target_weights(
        self,
        prices: pd.DataFrame,
        asof: pd.Timestamp,
        dollar_volume: pd.DataFrame | None = None,
    ) -> pd.Series:
        window = self.measurement_window(prices, asof)
        if window is None:
            return pd.Series(dtype="float64")

        names = self.eligible(prices, window, asof, dollar_volume)
        if len(names) < self.config.min_names:
            # Ranking three names does not produce a signal, it produces noise.
            return pd.Series(dtype="float64")

        scores = self.score(window[names]).dropna()
        if len(scores) < self.config.min_names:
            return pd.Series(dtype="float64")

        return self.weights_from_scores(scores)

    def measurement_window(self, prices: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame | None:
        """Rows ``[asof - lookback, asof - skip]``, or None if history is too short.

        Only data at or before ``asof`` is ever touched. That is the whole lookahead
        story for this class, and tests/test_no_lookahead.py holds it to it.
        """
        position = prices.index.get_loc(asof)
        if not isinstance(position, int):
            raise KeyError(f"{asof} is not a unique session in the price index")

        start = position - self.config.lookback_days
        end = position - self.config.skip_days
        if start < 0:
            return None
        return prices.iloc[start : end + 1]

    def eligible(
        self,
        prices: pd.DataFrame,
        window: pd.DataFrame,
        asof: pd.Timestamp,
        dollar_volume: pd.DataFrame | None = None,
    ) -> pd.Index:
        """Names we are willing to rank on this date.

        Both ends of the measurement window must be present, or the trailing return is
        computed off a stale or missing price. The name must also be priced on ``asof``
        itself, since that is when we would be trading it.
        """
        ok = window.iloc[0].notna() & window.iloc[-1].notna() & prices.loc[asof].notna()

        floor = self.config.min_dollar_volume
        if floor > 0 and dollar_volume is not None:
            traded = dollar_volume.reindex(index=window.index, columns=window.columns)
            ok &= traded.mean(skipna=True) >= floor

        return window.columns[ok.fillna(False)]

    def weights_from_scores(self, scores: pd.Series) -> pd.Series:
        """Equal-weight the top and bottom slices, dollar neutral, gross 1.0.

        Ties fall back to the order of the incoming index (alphabetical, since the price
        panel is sorted), which keeps a rerun reproducible.
        """
        n = len(scores)
        k = max(1, min(int(n * self.config.top_fraction), n // 2))

        ordered = scores.sort_values(ascending=False, kind="stable")
        weights = pd.Series(0.0, index=scores.index, name="weight")
        weights[ordered.index[:k]] = 0.5 / k
        weights[ordered.index[n - k :]] = -0.5 / k
        return weights
