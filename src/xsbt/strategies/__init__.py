"""Strategies. Importing this package is what populates the registry."""

from xsbt.strategies.base import (
    REGISTRY,
    CrossSectionalRankStrategy,
    Strategy,
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
    "build",
    "register",
]
