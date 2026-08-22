"""Strategies. Importing this package is what populates the registry."""

from xsbt.strategies.base import (
    REGISTRY,
    CrossSectionalRankStrategy,
    Strategy,
    build,
    register,
)

__all__ = [
    "REGISTRY",
    "CrossSectionalRankStrategy",
    "Strategy",
    "build",
    "register",
]
