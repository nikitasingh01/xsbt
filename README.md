# xsbt

A cross-sectional equity backtester. It fetches prices reproducibly, runs long/short
rank-based strategies over history, and produces a report that answers the only question
worth asking of a new strategy: **is this edge real, and can it be traded after costs?**

Two strategies ship with it, cross-sectional momentum and short-term reversal, but the
point of the repo is the machinery underneath. Adding the third strategy is one file and
one method.

```bash
make setup                                   # editable install + dev deps
make test                                    # full suite, offline, no network
make demo                                    # fetch, run both strategies, build both reports
open runs/momentum/report.html
```

---

## What you get

| Stage | Command | Output |
|---|---|---|
| Data | `xsbt fetch` | parquet snapshot + `manifest.json` with a SHA-256 per ticker |
| Backtest | `xsbt run` | `daily.parquet`, `weights.parquet`, `metrics.json`, `returns.csv` |
| Report | `xsbt report` | one self-contained `report.html`, no assets, no kernel |
| Integrity | `xsbt verify` | re-hashes the cache, non-zero exit on drift |

Each stage is a separate verb on purpose. The artifacts in between are inspectable, the
report can be rebuilt without re-running the backtest, and only `fetch` is allowed to
touch the network.

---

## Architecture

```
 configs/universe_us_liquid.csv
             |
             v
     PriceRepository  <---->  PriceCache                     data layer
     (YahooFinanceSource)     parquet + manifest.json
             |                snapshot_id = SHA-256 over the manifest
             v
     prices: DataFrame[date x ticker]   dollar_volume: DataFrame[date x ticker]
             |
             v
     Strategy.target_weights(prices, asof, dollar_volume)     strategy layer
     |  CrossSectionalRankStrategy: window -> eligible -> rank -> top/bottom slice
     |  Momentum.score() / Reversal.score()  <- all a new strategy writes
             |
             v
     run_backtest()                                           engine
     |  rebalance_dates -> apply_execution_lag -> simulate() with weight drift
     |  LinearCostModel charges turnover on the way in and again on the way out
             |
             v
     BacktestResult -> save(runs/<name>/)
             |
             v
     analyse() -> ReportData -> render() -> report.html       reporting
        metrics, leg attribution, market fit, cost sweep, parameter grid
```

The seams between those boxes are `Protocol`s (`PriceSource`, `Strategy`, `CostModel`),
so any one of them can be swapped without the others noticing. A different vendor is a
new class with a `fetch` method. A spread-aware cost model is a new class with a `charge`
method.

---

## The three layers

### 1. Data: reproducible, not just cached

Yahoo restates adjusted close every time a dividend or split lands, so the same query run
a month apart returns different history. A backtest that re-fetches on every run is not
reproducible, and it fails quietly rather than loudly.

So the cache here is a point-in-time snapshot, not a performance optimisation:

* one parquet per ticker, holding the raw payload as fetched (OHLCV plus adjusted close),
* `manifest.json` records the fetch timestamp, row count, first and last session, the
  requested window and a SHA-256 of the file contents,
* the `snapshot_id` is a SHA-256 over the whole manifest, and every run embeds it,
* `--offline` forbids network access entirely and replays from the cache. The test suite
  and CI run offline only.

`xsbt verify` re-hashes every file against the manifest and exits non-zero if anything
drifted, which makes it something you can put in a scheduled job.

### 2. Strategy and engine: the strategy is small because the engine is not

The engine owns everything that is easy to get subtly wrong:

* **Rebalance calendar.** `M`, `W`, `Q` or `nD`, resolved against the actual session
  index, so month end lands on the last trading day rather than on the 31st.
* **Execution lag.** A signal formed at the close of `t` is traded at the close of
  `t + execution_lag_days` and starts earning the session after that. Default 1.
* **Weight drift.** Between rebalances the book is not reset. Weights drift with realised
  returns, which is what actually happens to a portfolio nobody is touching:

  ```
  r_p,d = sum_i w_i,d-1 * r_i,d
  w_i,d = w_i,d-1 * (1 + r_i,d) / (1 + r_p,d)
  ```

* **Costs.** Turnover is notional traded over NAV, charged linearly at `cost_bps` on the
  way in and again on the way out.

The strategy sees none of that. It gets a price window and returns a score.

### 3. Reporting: written for the reader, not the author

`report.html` is a single file with the matplotlib figures inlined as base64. It emails
cleanly and opens on any machine. Alongside it, `metrics.json` and `returns.csv` for
anyone who would rather use their own tools.

Beyond the standard block (CAGR, vol, Sharpe, Sortino, Calmar, drawdown depth and
duration, hit rate, skew, VaR and CVaR), the sections that actually move a decision:

| Section | The question it answers |
|---|---|
| Sharpe standard error and t-stat | Is this distinguishable from luck over this sample? |
| Cost sweep and breakeven bps | At what cost level does the edge die? |
| Long leg vs short leg | Is the alpha in the hard-to-borrow short book? |
| Beta and alpha vs SPY | Is this repackaged market exposure? |
| Turnover and cost drag | What does this cost to run? |
| Parameter grid, lookback x top fraction | Is the config a cherry-picked peak? |
| Yearly returns and rolling Sharpe | Does it work outside one lucky regime? |
| Assumptions and caveats | What does this backtest not capture? |

The breakeven number comes from bisection on CAGR rather than from extrapolating the
arithmetic mean, because the arithmetic answer is systematically optimistic.

---

## Adding a strategy

The whole file. This is the extension case the design was built around.

```python
# src/xsbt/strategies/lowvol.py
"""Cross-sectional low volatility: long the calm names, short the jumpy ones."""

from __future__ import annotations

import pandas as pd

from xsbt.strategies.base import CrossSectionalRankStrategy, register


@register
class LowVolatility(CrossSectionalRankStrategy):
    name = "low_volatility"

    def score(self, window: pd.DataFrame) -> pd.Series:
        return -window.pct_change().std()
```

Import it in `strategies/__init__.py` so the registry sees it, then point a config at it:

```yaml
strategy:
  name: low_volatility
  lookback_days: 63
```

Windowing, the eligibility filter, ranking, the top and bottom split, equal weighting and
dollar neutrality are all inherited. `score` receives the measurement window with the
skip already applied, and higher means you want to be long.

Momentum and reversal differ by exactly one minus sign, and `tests/test_strategies.py`
asserts that their weights come out as exact negatives of each other on the same panel.

---

## No lookahead, enforced by a test

This is the failure mode that makes every other number in the report meaningless, so it
is not left to code review.

`tests/test_no_lookahead.py` runs a backtest, rewrites **all** price data strictly after
some date `T`, re-runs, and asserts the return series up to `T` is bit-identical. Nothing
about the future can leak backwards without that test going red.

The mechanism it guards: `CrossSectionalRankStrategy.measurement_window` slices
`prices.iloc[position - lookback : position - skip + 1]`, where `position` is the location
of `asof`. Prices are handed to the strategy whole, including the future, deliberately.
Trusting the strategy and then testing the trust is more robust than pre-slicing the
panel, because pre-slicing hides a bug instead of catching it.

---

## Reproducibility

Two runs against the same cache produce byte-identical `metrics.json`. That is asserted
twice, once at the library level and once end to end through the CLI.

To make it hold:

* the resolved config hashes to a `config_fingerprint`, a SHA-256 of canonical JSON,
* the data snapshot hashes to a `snapshot_id`,
* both, plus the git commit and the xsbt version, land in `metadata.json` and in the
  report's provenance block,
* the wall clock is deliberately kept **out** of `metrics.json` and lives in
  `metadata.json` instead, so two identical runs diff clean,
* `returns.csv` uses fixed float formatting for the same reason.

If the fingerprint and the snapshot id both match, the numbers match. If either moves,
the report is allowed to move with it, and you can see which one it was.

---

## Configuration

One YAML per strategy, validated by pydantic with `extra="forbid"`, so a typo is an error
rather than a silently ignored key. Paths are relative to your working directory, not to
the config file: one rule, and it matches how the CLI, the Makefile and the container all
invoke things.

```yaml
name: momentum_6m_skip1m
description: >
  Cross-sectional 6-month momentum, skipping the most recent month.

data:
  universe: configs/universe_us_liquid.csv
  start: 2010-01-01
  end: 2026-08-01
  cache_dir: data/cache
  benchmark: SPY

strategy:
  name: momentum
  lookback_days: 126     # roughly six months of sessions
  skip_days: 21          # keep short-term reversal out of the signal
  top_fraction: 0.2      # long the top quintile, short the bottom
  min_names: 10          # below this, ranking is noise, so skip the rebalance

portfolio:
  rebalance: M
  execution_lag_days: 1
  gross_leverage: 1.0
  cost_bps: 10.0
```

---

## Command line

```bash
# The only command that uses the network.
xsbt fetch --config configs/momentum.yaml
xsbt fetch --universe configs/universe_us_liquid.csv --start 2010-01-01 --end 2026-08-01

# Run a backtest. --offline refuses to fetch anything not already cached.
xsbt run --config configs/momentum.yaml --out runs/momentum
xsbt run --config configs/reversal.yaml --offline --no-grid    # skip the parameter sweep

# Rebuild a report from a saved run, without re-running the backtest.
xsbt report --run runs/momentum --out reports/momentum.html

# Re-hash the cache against its manifest. Non-zero exit if anything drifted.
xsbt verify --cache-dir data/cache
```

`--verbose` turns up the xsbt logger without making third-party libraries noisy.

---

## Development

```bash
make lint         # ruff check src tests
make format       # ruff format + autofix
make typecheck    # mypy, strict, on src/
make test         # pytest, offline
make test-cov     # with a coverage report
make docker       # build the container image
```

CI runs lint, typecheck and the full suite on Python 3.11, 3.12 and 3.13. Every test is
offline and deterministic: the Yahoo client is tested against recorded JSON fixtures, and
the panels used elsewhere are seeded random walks.

Layout:

```
src/xsbt/
  cli.py            typer app: fetch | run | report | verify
  config.py         pydantic models, YAML loading, config fingerprint
  data/             base.py (protocols) yahoo.py cache.py repository.py
                    universe.py market.py
  strategies/       base.py (protocol, registry, rank base) momentum.py reversal.py
  engine/           calendar.py portfolio.py costs.py backtest.py
  analytics/        metrics.py attribution.py sensitivity.py
  report/           html.py plots.py template.html
tests/              fixtures/ holds the recorded Yahoo JSON
docs/               DESIGN.md, ASSUMPTIONS.md
```

---

## Scope

This is a focused slice, not a platform. What it deliberately does not do, and why, is
written up in [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md). The short version:

* the universe is a fixed list of names that are liquid **today**, so it is survivorship
  biased, and that is the single largest bias in the numbers,
* borrow cost and short financing are not modelled,
* there is no market impact or capacity model, so the cost sweep is linear in turnover
  and understates the cost of size,
* execution is at the close with no slippage between decision and fill,
* no walk-forward fitting, so the parameter grid is a robustness check rather than a
  selection procedure,
* corporate actions are whatever Yahoo's adjusted close says they are.

Every one of these is repeated in the report footer, because a report that hides its
assumptions is worse than no report.

The decisions behind the design, and the alternatives I rejected, are in
[`docs/DESIGN.md`](docs/DESIGN.md).
