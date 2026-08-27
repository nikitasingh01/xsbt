# xsbt

A cross-sectional equity backtester. It fetches prices reproducibly, runs long/short
rank-based strategies over history, and writes a report that answers the question a PM
actually asks: **is this edge real, and can it be traded after costs?**

Momentum and reversal both ship with it, but the point of the repo is the machinery
underneath. Adding a third strategy is one file and one method.

```bash
make setup                 # editable install, pinned
make test                  # full suite, offline
make demo                  # fetch, run both strategies, build both reports
open runs/momentum/report.html
```

Python 3.11 or newer. `make test` needs no network and takes about a minute. `make demo` is
the only step that does: it pulls 41 symbols from Yahoo, and everything after the fetch runs
`--offline` against what it wrote.

`make setup` installs against [`constraints.txt`](constraints.txt), the exact versions the
checked-in reports came off. `make setup-latest` takes the ranges in `pyproject.toml`
instead, which is what the CI version matrix does.

Both runs are checked in under `runs/`, so you can open
[`runs/momentum/report.html`](runs/momentum/report.html) and read the output before
installing anything. The price cache is not, since it is 8 MB of vendor data that
`xsbt fetch` rebuilds.

---

## What you get

| Stage | Command | Output |
|---|---|---|
| Data | `xsbt fetch` | parquet snapshot and `manifest.json`, one SHA-256 per ticker |
| Backtest | `xsbt run` | `daily.parquet`, `weights.parquet`, `metrics.json`, `returns.csv` |
| Report | `xsbt report` | one self-contained `report.html` |
| Integrity | `xsbt verify` | re-hashes the cache and rebuilds adjusted close from its own dividends |

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
| Sharpe t-stat | 0.49 | -2.46 |
| Max drawdown | -34.9% | -64.6% |
| Turnover p.a. | 9.0x | 18.7x |
| Breakeven cost | 17bps | none, it loses at zero cost |
| P(true Sharpe > 0), search charged | 28% | 0% |

The t-stat carries a Newey-West error bar over one holding period rather than the iid one,
since a book held for a month leaves its daily returns correlated. Both bars are on the
report so the size of that adjustment is visible. The last row goes further and charges for
the 20 grid cells that ran, so it is asking whether the cell I picked beat the best cell the
search would throw up from noise alone.

Neither strategy is tradeable. Momentum cannot be told apart from zero over this sample, and
its 0.12 Sharpe sits under the 0.26 the same grid produces from noise, so it does not clear
its own search either. A 17bps breakeven leaves nothing over a realistic 10bps once size is
involved. Reversal is significantly negative, and its per-name table says why: the damage is
concentrated in shorting NVDA, TSLA, NFLX and AAPL.

Forty survivors give an 8-long, 8-short book, which is thin. Read these as a check that the
machinery works, not as an estimate of the effect. The report says so on its own front page.
A fresh fetch will land near these numbers rather than on them, because Yahoo restates
adjusted close whenever a dividend or split settles.

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

The seams are `Protocol`s: `PriceSource`, `Strategy`, `CostModel`. Any one can be swapped
without the others noticing. A different vendor is a new class with a `fetch` method; a
spread-aware cost model is a new class with a `charge` method.

**Data is a snapshot, not a cache.** Yahoo restates adjusted close on every dividend and
split, so re-fetching on each run is not reproducible and fails quietly rather than loudly.
One parquet per ticker holds the payload as fetched, `manifest.json` carries a SHA-256 per
file, and `snapshot_id` hashes the manifest. Same `config_fingerprint` plus same
`snapshot_id` means the same numbers, and `metrics.json` keeps the wall clock out so two
runs off one snapshot are byte-identical. The suite asserts that; `xsbt verify` re-hashes
the cache and exits non-zero on drift.

**The vendor's arithmetic is checked, not trusted.** A hash proves the bytes have not moved
since we fetched them. It says nothing about whether they were right, and adjusted close is
the one field Yahoo derived rather than observed, so a missed dividend would put a fake
return on an ex-date with nothing looking broken. The snapshot therefore also stores the
dividends and splits, and `fetch` and `verify` rebuild adjusted close from them and compare.
On the current snapshot all 41 names reconcile, worst gap 9.2e-07.

**The strategy is small because the engine is not.** The engine owns the rebalance calendar
(resolved against real sessions, so month end is the last trading day), the execution lag
(signal at the close of `t`, traded at `t + execution_lag_days`), the weight drift between
rebalances, and the cost charge on turnover. A strategy gets a price window and returns a
score. The reasoning behind each is in [`docs/DESIGN.md`](docs/DESIGN.md).

---

## The report

One self-contained HTML file with the figures inlined as base64, so it emails cleanly and
opens anywhere. `metrics.json` and `returns.csv` sit beside it for anyone who would rather
use their own tools.

Past the standard block (CAGR, vol, Sharpe, Sortino, Calmar, drawdown depth and duration,
hit rate, skew, VaR and CVaR), the sections that move a decision:

| Section | The question it answers |
|---|---|
| Sharpe standard error and t-stat | Is this different from luck over this sample? |
| Cost sweep and breakeven bps | At what cost level does the edge die? |
| Long leg vs short leg | Is the alpha in the hard-to-borrow short book? |
| Per-name contributions | Is the P&L forty names, or three? |
| Beta and alpha vs SPY | Is this repackaged market exposure? |
| Turnover and cost drag | What does this cost to run? |
| Parameter grid, lookback x top fraction | Is the config a cherry-picked peak? |
| Deflated Sharpe over that grid | Would the search alone have produced this? |
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

Import it in `strategies/__init__.py` so the registry sees it, then point a config at
`name: low_volatility`. Windowing, the eligibility filter, ranking, the top and bottom
split, equal weighting and dollar neutrality are all inherited. `score` gets the measurement
window with the skip already applied, and higher means you want to be long.

The base class is a convenience, not a requirement: `register` takes anything satisfying the
`Strategy` protocol, so something that weights its book its own way is named in a config the
same way and never touches the rank machinery. `tests/test_strategies.py` pins both halves
of that, and also that momentum and reversal come out as exact negatives of each other,
since they differ by a single minus sign.

---

## No lookahead, enforced by a test

This is the failure that makes every other number in the report meaningless, so it is not
left to code review. `tests/test_no_lookahead.py` runs a backtest, replaces **all** price
data after some date `T` with a different random walk, re-runs, and asserts the returns up
to `T` are bit-identical. A positive control alongside it asserts the data after `T` does
move, so the test cannot pass on an engine that ignores prices. A third test hands the
strategy a panel truncated at the signal date and asserts it returns the same book.

Prices are handed to the strategy whole, future included, on purpose. Trusting the strategy
and then testing the trust catches more than pre-slicing the panel would, because slicing
hides a bug instead of surfacing it. Full reasoning in [`docs/DESIGN.md`](docs/DESIGN.md).

---

## Configuration

One YAML per strategy, validated by pydantic with `extra="forbid"`, so a typo is an error
rather than a silently ignored key. Paths are relative to your working directory, not to the
config file: one rule, and it matches how the CLI, the Makefile and the container all invoke
things.

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

# Re-hash the cache against its manifest, then rebuild adjusted close from the
# dividends and splits stored alongside it. Non-zero exit if either disagrees.
xsbt verify --cache-dir data/cache
xsbt verify --cache-dir data/cache --no-adjustments   # hashes only, much faster
```

`--verbose` turns up the xsbt logger without making third-party libraries noisy.

---

## Development

```bash
make setup        # editable install against constraints.txt
make check        # lint, format check, typecheck and the suite, what CI runs
```

Individually: `make lint`, `make format`, `make typecheck`, `make test`, `make test-cov`,
`make docker`.

CI runs the check on Python 3.11, 3.12 and 3.13 unpinned, to prove the ranges in
`pyproject.toml` still resolve, then once more against `constraints.txt`, to prove the
published numbers still reproduce, then a Docker build. Every test is offline and
deterministic: the Yahoo client runs against recorded JSON fixtures and the panels elsewhere
are seeded random walks.

```
src/xsbt/
  cli.py            typer app: fetch | run | report | verify
  config.py         pydantic models, YAML loading, config fingerprint
  data/             base.py (protocols) yahoo.py cache.py repository.py
                    adjustment.py universe.py market.py
  strategies/       base.py (protocol, registry, rank base) momentum.py reversal.py
  engine/           calendar.py portfolio.py costs.py backtest.py
  analytics/        metrics.py attribution.py sensitivity.py
  report/           html.py plots.py template.html
tests/              fixtures/ holds the recorded Yahoo JSON
docs/               DESIGN.md, ASSUMPTIONS.md
```

---

## Scope

A focused slice, not a platform. The biggest gap by a distance is survivorship: the universe
is 40 names that are liquid **today**, so nothing delisted or acquired since 2010 is in it.
After that, borrow cost is not charged, there is no capacity model so the cost sweep
understates size, execution is at the close with no slippage, and the parameter grid is a
robustness check rather than a selection procedure.

All of it is in [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md) and repeated in the report
footer, because a report that hides its assumptions is worse than no report. The design
decisions, and the alternatives I turned down, are in [`docs/DESIGN.md`](docs/DESIGN.md).
