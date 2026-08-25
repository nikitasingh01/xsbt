"""Strategies. Importing this package is what populates the registry."""

from xsbt.strategies.base import (
    REGISTRY,
    CrossSectionalRankStrategy,
    Strategy,
    StrategyFactory,
    build,
    register,
)
from xsbt.strategies.momentum import Momentum
from xsbt.strategies.reversal import Reversal

__all__ = [
    "REGISTRY",
    "CrossSectionalRankStrategy",
    "Momentum",
    "Reversal",
    "Strategy",
    "StrategyFactory",
    "build",
    "register",
]
