"""Transaction costs.

One linear model today. It is a separate seam because it is the first thing a researcher
will want to replace: per-name spreads, a square-root impact term, borrow on the short
leg. Those all fit behind the same call.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class CostModel(Protocol):
    name: str

    def charge(self, traded_notional: float) -> float:
        """Cost as a fraction of NAV, given notional traded as a fraction of NAV."""
        ...


class LinearCostModel:
    """Flat charge per unit of notional traded.

    ``cost_bps`` is charged on each unit of notional that changes hands, so a full round
    trip (into a name and back out) costs ``2 * cost_bps``. For liquid US large caps
    10bps is a deliberately unkind assumption: half-spread is closer to 1-2bps. Erring
    high keeps a strategy from looking good only because the cost line was optimistic.
    """

    def __init__(self, cost_bps: float) -> None:
        if cost_bps < 0:
            raise ValueError(f"cost_bps must be non-negative, got {cost_bps}")
        self.cost_bps = cost_bps
        self.name = f"linear_{cost_bps:g}bps"

    @property
    def rate(self) -> float:
        return self.cost_bps / 1e4

    def charge(self, traded_notional: float) -> float:
        return float(np.abs(traded_notional) * self.rate)
