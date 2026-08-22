"""Cross-sectional momentum: buy what has been going up."""

from __future__ import annotations

import pandas as pd

from xsbt.strategies.base import CrossSectionalRankStrategy, register


@register
class Momentum(CrossSectionalRankStrategy):
    name = "momentum"

    def score(self, window: pd.DataFrame) -> pd.Series:
        return window.iloc[-1] / window.iloc[0] - 1.0
