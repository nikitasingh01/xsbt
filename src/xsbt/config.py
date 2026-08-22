"""Run configuration: validated, hashable, loaded from YAML.

Paths in a config are relative to the working directory, not the config file. One rule,
and it matches how the CLI, the Makefile and the container all invoke things.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DataConfig(Frozen):
    universe: Path
    start: dt.date
    end: dt.date
    cache_dir: Path = Path("data/cache")
    # Priced off adjusted close. Anything else needs a view on corporate actions that
    # this system does not have.
    field: str = "adj_close"
    # Fetched alongside the universe, used only for beta/alpha in the report.
    benchmark: str | None = "SPY"

    @model_validator(mode="after")
    def _check_window(self) -> Self:
        if self.start >= self.end:
            raise ValueError(f"start {self.start} must be before end {self.end}")
        return self


class StrategyConfig(Frozen):
    #: Key into xsbt.strategies.REGISTRY.
    name: str
    lookback_days: int = Field(default=126, gt=0)
    #: Sessions to skip at the near end of the window. Raw 12-month momentum is polluted
    #: by short-term reversal in the most recent month; the literature skips it.
    skip_days: int = Field(default=0, ge=0)
    top_fraction: float = Field(default=0.2, gt=0.0, le=0.5)
    #: Below this many eligible names, ranking is noise. Skip the rebalance and hold.
    min_names: int = Field(default=4, ge=2)
    #: Eligibility floor on trailing average dollar volume. 0 disables the check.
    min_dollar_volume: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _skip_shorter_than_lookback(self) -> Self:
        if self.skip_days >= self.lookback_days:
            raise ValueError(
                f"skip_days {self.skip_days} leaves no window inside "
                f"lookback_days {self.lookback_days}"
            )
        return self


class PortfolioConfig(Frozen):
    #: 'M' month end, 'W' week end, 'Q' quarter end, or 'nD' every n sessions.
    rebalance: str = "M"
    #: Sessions between the close that produces a signal and the close we trade on. A book
    #: put on at the close of day d earns from d+1, so lag 1 means a signal from close t is
    #: traded at close t+1 and earns from t+2. Lag 0 is the market-on-close convention and
    #: assumes you can compute the signal and get the order in before the same bell.
    execution_lag_days: int = Field(default=1, ge=0)
    #: Sum of absolute weights. 1.0 = 0.5 long, 0.5 short.
    gross_leverage: float = Field(default=1.0, gt=0.0)
    #: Cost per unit of traded notional, charged on the way in and again on the way out.
    #: See docs/ASSUMPTIONS.md.
    cost_bps: float = Field(default=10.0, ge=0.0)
    #: Annualised hurdle for Sharpe and Sortino only. It is not credited to the P&L: a
    #: dollar-neutral book earns something like the short rebate, and pretending to know
    #: that number without borrow data would be worse than leaving it out.
    risk_free_rate: float = Field(default=0.0, ge=0.0)

    @field_validator("rebalance")
    @classmethod
    def _known_frequency(cls, value: str) -> str:
        text = value.strip().upper()
        if text in {"M", "W", "Q"}:
            return text
        if text.endswith("D") and text[:-1].isdigit() and int(text[:-1]) > 0:
            return text
        raise ValueError(f"rebalance must be M, W, Q or nD, got {value!r}")


class BacktestConfig(Frozen):
    name: str
    data: DataConfig
    strategy: StrategyConfig
    portfolio: PortfolioConfig = PortfolioConfig()
    description: str = ""

    @classmethod
    def from_yaml(cls, path: Path | str) -> BacktestConfig:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"config not found: {path}")
        raw = yaml.safe_load(path.read_text()) or {}
        return cls.model_validate(raw)

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def fingerprint(self) -> str:
        """Hash of the resolved config. Two runs with the same fingerprint and the same
        data snapshot must produce the same numbers."""
        canonical = json.dumps(self.as_dict(), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()
