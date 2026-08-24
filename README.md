# xsbt

A cross-sectional equity backtester. It fetches prices reproducibly, runs long/short
rank-based strategies over history, and writes a report that answers the question a PM
actually asks: **is this edge real, and can it be traded after costs?**

Momentum and reversal both ship with it, but the point of the repo is the machinery
underneath. Adding a third strategy is one file and one method.

```bash
make setup                 # editable install and dev deps
make test                  # full suite, offline
make demo                  # fetch, run both strategies, build both reports
open runs/momentum/report.html
```

Python 3.11 or newer. `make test` needs no network and takes about a minute. `make demo` is
the only step that does: it pulls 41 symbols from Yahoo, which takes a couple of minutes,
and everything after the fetch runs `--offline` against what it wrote.

---

## What you get

| Stage | Command | Output |
|---|---|---|
| Data | `xsbt fetch` | parquet snapshot and `manifest.json`, one SHA-256 per ticker |
| Backtest | `xsbt run` | `daily.parquet`, `weights.parquet`, `metrics.json`, `returns.csv` |
| Report | `xsbt report` | one self-contained `report.html` |
| Integrity | `xsbt verify` | re-hashes the cache, non-zero exit on drift |

Each stage is a separate verb. The files in between are inspectable, the report can be
rebuilt without re-running the backtest, and only `fetch` touches the network.

---

## What it found

Both strategies over 2010-01-04 to 2026-07-31, 4169 sessions, 40 US large caps, monthly
rebalance, long the top quintile and short the bottom, costed at 10bps per unit traded.

| | Momentum 6m skip 1m | Reversal 1m |
|---|---|---|
| CAGR, net | 0.66% | -5.53% |
| Volatility | 10.99% | 9.78% |
| Sharpe, net | 0.12 | -0.53 |
| Sharpe t-stat | 0.47 | -2.17 |
| Max drawdown | -34.9% | -64.6% |
| Turnover p.a. | 9.0x | 18.7x |
| Breakeven cost | 17bps | none, it loses at zero cost |

Neither is tradeable. Momentum cannot be told apart from zero over this sample, and a 17bps
breakeven leaves nothing over a realistic 10bps once size is involved. Reversal is
significantly negative: at 18.7x turnover it pays 1.87% a year in costs, and it was losing
before those costs were charged.

Forty survivors give an 8-long, 8-short book, which is thin. These should be read as a
check that the machinery works, not as an estimate of the effect. The report says so on its
own front page.

A fresh fetch will land near these numbers rather than on them, because Yahoo restates
adjusted close whenever a dividend or split settles. Two runs against one snapshot are
byte-identical, and the suite asserts that.

---

## Architecture

```
 configs/universe_us_liquid.csv
             |
             v
     PriceRepository  <---->  PriceCache                     data
     (YahooFinanceSource)     parquet + manifest.json
             |                snapshot_id = SHA-256 over the manifest
             v
     prices: DataFrame[date x ticker]   dollar_volume: DataFrame[date x ticker]
             |
             v
     Strategy.target_weights(prices, asof, dollar_volume)     strategy
     |  CrossSectionalRankStrategy: window -> eligible -> rank -> top/bottom slice
     |  Momentum.score() / Reversal.score()  <- all a new strategy writes
             |
             v
     run_backtest()                                           engine
     |  rebalance_dates -> apply_execution_lag -> simulate() with weight drift
     |  LinearCostModel charges turnover in and again out
             |
             v
     BacktestResult -> save(runs/<name>/)
             |
             v
     analyse() -> ReportData -> render() -> report.html       reporting
```

The seams are `Protocol`s: `PriceSource`, `Strategy`, `CostModel`. Any one of them can be
swapped without the others noticing. A different vendor is a new class with a `fetch`
method. A spread-aware cost model is a new class with a `charge` method.

---

## The three layers

### 1. Data: a snapshot, not a cache

Yahoo restates adjusted close every time a dividend or split lands, so the same query run a
month apart returns different history. A backtest that re-fetches on every run is not
reproducible, and it fails quietly rather than loudly. So:

* one parquet per ticker, holding the payload as fetched (OHLCV plus adjusted close),
* `manifest.json` records the fetch time, row count, first and last session, the requested
  window and a SHA-256 of the file,
* `snapshot_id` is a SHA-256 over the whole manifest, and every run embeds it,
* `--offline` forbids the network and replays from the cache. Tests and CI are offline only.

Two runs against the same cache produce byte-identical `metrics.json`, asserted at the
library level and again end to end through the CLI. The resolved config hashes to a
`config_fingerprint` and the snapshot to a `snapshot_id`, both go into `metadata.json` and
into the report, and the wall clock is kept out of `metrics.json`. Same fingerprint plus
same snapshot means the same numbers; if either moves, you can see which one it was.

`xsbt verify` re-hashes every file against the manifest and exits non-zero on drift, so it
can go in a scheduled job.

### 2. Strategy and engine: the strategy is small because the engine is not

The engine owns everything that is easy to get subtly wrong:

* **Rebalance calendar.** `M`, `W`, `Q` or `nD`, resolved against the real session index,
  so month end lands on the last trading day rather than on the 31st.
* **Execution lag.** A signal formed at the close of `t` is traded at the close of
  `t + execution_lag_days` and starts earning the session after. Default 1.
* **Weight drift.** Between rebalances the book is left alone and weights drift with
  realised returns, which is what happens to a portfolio nobody is touching:

  ```
  r_p,d = sum_i w_i,d-1 * r_i,d
  w_i,d = w_i,d-1 * (1 + r_i,d) / (1 + r_p,d)
  ```

* **Costs.** Turnover is notional traded over NAV, charged linearly at `cost_bps` on the
  way in and again on the way out.

The strategy sees none of that. It gets a price window and returns a score.

### 3. Reporting: written for the reader

`report.html` is a single file with the matplotlib figures inlined as base64. It emails
cleanly and opens on any machine. `metrics.json` and `returns.csv` sit beside it for
anyone who would rather use their own tools.

Past the standard block (CAGR, vol, Sharpe, Sortino, Calmar, drawdown depth and duration,
hit rate, skew, VaR and CVaR), the sections that move a decision:

| Section | The question it answers |
|---|---|
| Sharpe standard error and t-stat | Is this different from luck over this sample? |
| Cost sweep and breakeven bps | At what cost level does the edge die? |
| Long leg vs short leg | Is the alpha in the hard-to-borrow short book? |
| Beta and alpha vs SPY | Is this repackaged market exposure? |
| Turnover and cost drag | What does this cost to run? |
| Parameter grid, lookback x top fraction | Is the config a cherry-picked peak? |
| Yearly returns and rolling Sharpe | Does it work outside one lucky regime? |
| Assumptions and caveats | What does this backtest not capture? |

Breakeven comes from bisection on CAGR rather than from extrapolating the arithmetic mean,
because the arithmetic answer is systematically optimistic.

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
dollar neutrality are all inherited. `score` gets the measurement window with the skip
already applied, and higher means you want to be long.

Momentum and reversal differ by exactly one minus sign, and `tests/test_strategies.py`
asserts that their weights come out as exact negatives of each other on the same panel.

---

## No lookahead, enforced by a test

This is the failure that makes every other number in the report meaningless, so it is not
left to code review.

`tests/test_no_lookahead.py` runs a backtest, replaces **all** price data after some date
`T` with a different random walk, re-runs, and asserts the returns up to `T` are
bit-identical. A positive control alongside it asserts that the data after `T` does move,
so the test cannot pass on an engine that ignores prices. A third test hands the strategy a
panel truncated at the signal date and asserts it returns the same book.

What they guard: `CrossSectionalRankStrategy.measurement_window` slices
`prices.iloc[position - lookback : position - skip + 1]`, where `position` is the location
of `asof`. Prices are handed to the strategy whole, future included, on purpose. Trusting
the strategy and then testing the trust catches more than pre-slicing the panel would,
because pre-slicing hides a bug instead of surfacing it. The reasoning is in
[`docs/DESIGN.md`](docs/DESIGN.md).

---

## Configuration

One YAML per strategy, validated by pydantic with `extra="forbid"`, so a typo is an error
rather than a silently ignored key. Paths are relative to your working directory, not to
the config file: one rule, and it matches how the CLI, the Makefile and the container all
invoke things.

```yaml
name: momentum_6m_skip1m

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
make setup        # editable install and dev deps
make check        # lint, format check, typecheck and the suite, what CI runs
```

Individually: `make lint`, `make format`, `make typecheck`, `make test`, `make test-cov`,
`make docker`.

CI runs all of it on Python 3.11, 3.12 and 3.13, plus a Docker build. Every test is offline
and deterministic: the Yahoo client runs against recorded JSON fixtures and the panels
elsewhere are seeded random walks.

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

A focused slice, not a platform. The biggest gap by a distance is survivorship: the
universe is 40 names that are liquid **today**, so nothing delisted or acquired since 2010
is in it. After that, borrow cost is not charged, there is no capacity model so the cost
sweep understates size, execution is at the close with no slippage, the parameter grid is a
robustness check rather than a selection procedure, and corporate actions are whatever
Yahoo's adjustment says they are.

All of it is written up in [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md) and repeated in the
report footer, because a report that hides its assumptions is worse than no report. The
design decisions, and the alternatives I turned down, are in
[`docs/DESIGN.md`](docs/DESIGN.md).
