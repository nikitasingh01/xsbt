# Design notes

Why the code is shaped the way it is, and what I considered and did not do. The brief was
explicit that this is a development exercise rather than a research one, so most of these
decisions are about correctness and seams rather than about signal.

---

## 1. Three layers, joined by protocols

`data -> strategy -> engine -> analytics -> report`, with `Protocol` classes at the joins:

```python
class PriceSource(Protocol):
    name: str
    def fetch(self, ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame: ...

class Strategy(Protocol):
    @property
    def name(self) -> str: ...
    def target_weights(self, prices, asof, dollar_volume=None) -> pd.Series: ...

class CostModel(Protocol):
    name: str
    def charge(self, traded_notional: float) -> float: ...
```

Structural typing rather than inheritance, because a test double should not have to
import my base class to be a valid price source. `tests/helpers.py` has a `StubSource`
that is a dozen lines and inherits from nothing, which is the whole argument.

**Rejected:** a single `Backtester` class with `fetch`, `run` and `report` methods. It
reads fine at this size and becomes untestable at three times this size, because you
cannot exercise the engine without a network stub somewhere in the constructor.

---

## 2. The strategy gets the whole price panel, including the future

This looks wrong at first glance and is the decision I thought about hardest.

The alternative is to slice the panel to `prices.loc[:asof]` in the engine before handing
it over, so a strategy is physically incapable of seeing the future. That is genuinely
safer for one strategy.

I did not do it, for two reasons:

1. It makes lookahead **invisible instead of impossible**. The day someone writes a
   strategy that needs a longer window than the engine decided to hand over, they will
   ask for the full panel back, and the guarantee quietly evaporates. A guarantee that
   holds by accident of the current call site is not a guarantee.
2. Slicing on every rebalance date over a 16-year sample is a real copy cost, for a
   protection that a test provides for free.

So the contract is stated in the docstring, and `tests/test_no_lookahead.py` enforces it
by experiment rather than by construction: run, corrupt every price after `T` by a factor
of `1e6`, re-run, assert the pre-`T` returns are bit-identical. That test catches the
whole class of bug, including the ones you would not think to look for, such as an
analytics function that reindexes against the full sample.

If the codebase grew to a dozen strategies written by people who had not read this file,
I would add the slicing as belt and braces. At two strategies, the test is the honest
protection and the slice would be theatre.

---

## 3. Weights drift between rebalances

The naive engine holds constant weights between rebalance dates. That silently assumes a
daily rebalance back to target, which both costs nothing in the model and is not what the
strategy asked for.

`engine/portfolio.py` propagates the book properly:

```
r_p,d = sum_i w_i,d-1 * r_i,d
w_i,d = w_i,d-1 * (1 + r_i,d) / (1 + r_p,d)
```

so a name that doubles becomes a bigger share of the book, exactly as it would in a real
portfolio nobody is touching. Turnover is then the honest number: the distance from the
drifted book to the new target, not from the original target to the new one.

This also matters for the exposures chart. A dollar-neutral book does not stay
dollar-neutral, and the report shows the drift rather than a flat line at zero.

---

## 4. Cost is a drag on return, which makes repricing exact

`net = gross - (bps / 1e4) * turnover`.

Because the weight path drifts on **gross** return and the strategy's targets never
consult realised costs, the entire cost sweep can be computed from one backtest:

```python
def net_returns_at(result, cost_bps):
    return result.gross_returns - (cost_bps / 1e4) * result.turnover
```

That is exact, not an approximation, and it is why the report can show a five-level cost
ladder plus a bisected breakeven without re-running anything. It is worth stating out
loud in `analytics/sensitivity.py`, because the moment someone adds a cost-aware
rebalance rule (skip the trade if the expected cost exceeds the expected edge) this stops
being true and the sweep has to start re-running.

**Rejected:** charging cost against the capital base rather than the return. It compounds
differently and is arguably more correct for a funded book, but it makes the sweep
require a re-run for a difference that is second order at these turnover levels.

---

## 5. The cache is a snapshot, not a speedup

The thing I did not know before starting, and had to go and check, is that Yahoo
**rewrites history**. Adjusted close is recomputed backwards every time a dividend or
split lands, so the same query run a month apart gives you different prices for 2015.

That makes "reliably and reproducibly" in the brief a data-layer problem, not a caching
problem. So:

* one parquet per ticker holding the payload as fetched, raw OHLCV plus adjusted close,
* `manifest.json` with per-ticker fetch time, row count, first and last session, the
  requested window and a SHA-256 of the file,
* `snapshot_id` is a SHA-256 over the manifest, embedded in every run,
* `xsbt verify` re-hashes and exits non-zero on drift.

A run is therefore reproducible against **the bytes we actually saw**, which is the
strongest claim available without paying a vendor for point-in-time data. It is not the
same as saying the data is right, and `docs/ASSUMPTIONS.md` says so.

**Rejected:** SQLite or DuckDB. Parquet plus a JSON manifest is one file per ticker,
diffable, hashable, readable by anything, and has no migration story. A database earns
its keep at the point you need cross-sectional queries over a universe too big to hold in
memory, which is not this.

**Rejected:** storing only adjusted close. It halves the file size and throws away the
ability to ever reconstruct what the adjustment did.

---

## 6. Vectorised pandas, not an event loop

An event-driven simulator (order objects, a fill model, a broker) is the right shape for
intraday or for anything with partial fills and queue position. For a monthly-rebalanced
daily-close cross-sectional book it is a large amount of machinery whose only output is a
weight matrix multiplied by a return matrix.

The engine is roughly 50 lines of pandas and runs the 16-year backtest in well under a
second, which is what makes the parameter grid (25 re-runs) and the cost sweep affordable
inside a report build.

The trade is that adding realistic fills later means replacing `simulate()` rather than
extending it. I think that is the right bet: an event loop written speculatively for
requirements nobody has stated yet is usually the wrong event loop.

---

## 7. Configs are frozen, validated and hashed

pydantic v2 with `frozen=True` and `extra="forbid"`. The second one matters more than it
looks: without it, `lookback_dyas: 200` is silently ignored and you spend an afternoon
wondering why the parameter had no effect.

`fingerprint()` is a SHA-256 over the canonical JSON of the resolved config. Together with
`snapshot_id` it is the reproducibility claim: same fingerprint plus same snapshot means
same numbers, and if the numbers moved you can see which of the two moved first.

One gotcha worth recording, because it cost me time in `analytics/sensitivity.py`:
`model_copy(update=...)` does **not** re-run validators. Building the parameter grid by
copying a config with a new `lookback_days` would happily produce an invalid config where
`skip_days >= lookback_days`. The grid uses `model_validate` on a dumped dict instead.

---

## 8. A skipped rebalance is a real outcome

If fewer than `min_names` names are eligible on a rebalance date, the strategy returns an
empty series and the engine holds the previous book rather than trading into a two-name
portfolio. The count of skipped rebalances is recorded in `metadata.json` and shown in
the report's provenance block.

The alternative, quietly ranking whatever is available, produces a backtest whose early
years are three names levered to gross 1.0. That is not a small distortion and it is
invisible unless you go looking.

---

## 9. Session-counted annualisation

CAGR is computed over `len(returns) / 252` rather than over calendar days. Calendar dating
makes a backtest that happens to end on a Monday look different from one that ends on a
Friday, which is noise dressed up as a result.

Sharpe uses the Lo (2002) standard error:

```
SE(SR_annual) = sqrt(252 * (1 + SR_period^2 / 2) / T)
```

This assumes returns are iid. They are not, because monthly rebalancing leaves
autocorrelation in the daily series, so the reported t-statistic is optimistic. The
report says so in the caveats rather than quietly reporting a number with no scale on it.
A Newey-West correction would be the next improvement and is the first thing on the list
below.

There is also a `FLAT_VOL = 1e-15` floor in the metrics, because pandas returns something
like `2e-19` for the standard deviation of a constant series rather than a clean zero, and
dividing by that hands a reader a Sharpe ratio of 7e16. Small guard, absurd failure mode.

---

## 10. The report is one HTML file

matplotlib figures inlined as base64 data URIs, jinja2 for the page, no external assets
and no JavaScript. It survives being emailed, opens on a locked-down laptop, and archives
as a single artifact next to the run that produced it.

`metrics.json` is written beside it for anything programmatic, deliberately without a
timestamp so that two runs off one snapshot diff clean. The wall clock lives in
`metadata.json`.

**Rejected:** a notebook. Notebooks are excellent for the exploration and bad as a
deliverable: they carry an execution environment, they diff badly, and the output depends
on what order somebody ran the cells in.

**Rejected:** a golden-file test on the HTML. It fails on every wording change and passes
on every change that matters. `tests/test_report.py` asserts behaviour instead: the page
is self-contained, every section is present, the caveats are there, the verdict sentence
matches the t-statistic, and no reader ever sees a bare `nan`.

---

## 11. Tests assert arithmetic, not previous output

Where a number can be worked out by hand, the expected value is worked out in the
docstring and asserted exactly. `tests/test_portfolio.py` runs four tickers over ten days
with weights and P&L computed on paper. `tests/test_attribution.py` does the same for
beta, alpha and tracking error.

The failure mode I wanted to avoid is the test suite that pins whatever the code happened
to produce on the day it was written, which locks in bugs and calls it regression
coverage.

The Yahoo client is tested against recorded JSON fixtures, including a 404 and a 429, so
the suite is fully offline and CI never depends on a vendor being up.

---

## What I would do next

In rough order of what would change a decision:

1. **Newey-West standard errors** on the Sharpe, so the significance test stops assuming
   independence it does not have.
2. **A point-in-time universe.** Survivorship is the largest bias in these numbers by a
   distance, and everything else on this list is second order next to it. A historical
   index membership file would fix it, and until it exists no number here should be
   treated as an estimate of live performance.
3. **Borrow cost on the short leg**, even a crude flat annual rate, because the leg
   attribution is currently comparing a financed leg to an unfinanced one.
4. **Walk-forward parameter selection**, so the grid becomes a procedure rather than a
   diagnostic, with the reported result coming from out-of-sample periods only.
5. **A capacity model.** Linear cost in turnover understates size badly, and the
   breakeven number is the one a PM will quote back at you.
6. **Sector and size neutralisation** in the weighting step, which is a natural second
   method on the rank base class alongside `score`.
