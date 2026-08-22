"""Short-term reversal: buy what has been going down.

Same construction as momentum with the score negated, which is the point of the base
class. Pair it with a short lookback and no skip.
"""

from __future__ import annotations

import pandas as pd

from xsbt.strategies.base import CrossSectionalRankStrategy, register


@register
class Reversal(CrossSectionalRankStrategy):
    name = "reversal"

    def score(self, window: pd.DataFrame) -> pd.Series:
        return -(window.iloc[-1] / window.iloc[0] - 1.0)
