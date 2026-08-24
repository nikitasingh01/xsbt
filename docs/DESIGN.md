# Design notes

Why the code is shaped the way it is, and what I considered and did not do. The brief was
explicit that this is a development exercise rather than a research one, so most of these
are about correctness and seams rather than about signal.

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

Structural typing rather than inheritance, because a test double should not have to import
my base class to be a valid price source. `tests/helpers.py` has a `StubSource` that is a
dozen lines and inherits from nothing, which is the whole argument.

**Rejected:** one `Backtester` class with `fetch`, `run` and `report` on it. Reads fine at
this size, untestable at three times it, because you cannot exercise the engine without a
network stub somewhere in the constructor.

---

## 2. The strategy gets the whole price panel, including the future

This looks wrong at first glance and is the decision I thought about hardest.

The alternative is to slice to `prices.loc[:asof]` in the engine, so a strategy is
physically incapable of seeing ahead. Safer for one strategy. I did not do it because it
makes lookahead **invisible rather than impossible**: the day someone needs a longer window
than the engine chose to hand over, they ask for the full panel back and the guarantee
quietly evaporates. A guarantee that holds by accident of the current call site is not one.

So `tests/test_no_lookahead.py` earns it by experiment instead: run, replace every price
after `T` with a different random walk off the same level, re-run, assert the pre-`T`
returns and weights are bit-identical. That catches the whole class, including the bugs you
would not think to look for, such as an analytics function reindexing against the full
sample.

Two details in that file took a second attempt:

* The first corruption scaled the future by `1e6`. That is worse, not blunter: multiplying
  every name by the same factor leaves the cross-sectional ranking exactly where it was, so
  it corrupts the data in the one way this strategy is blind to. A fresh walk reorders the
  names, which is what the signal actually reads.
* `test_the_lookahead_test_can_actually_fail` is a positive control asserting the post-`T`
  series *does* move. Without it the file would pass on an engine that ignored prices
  entirely, which is the failure a corruption test is most likely to hide.

A third test comes at it from the other side: hand the strategy a panel truncated at the
signal date and assert it returns the same book. The corruption test catches a leak once it
has changed an outcome; this one catches it where it would be introduced.

At a dozen strategies written by people who had not read this file, I would add the slicing
as well. At two, the test is the real protection and the slice would add nothing.

---

## 3. Weights drift between rebalances

The naive engine holds weights constant between rebalance dates, which silently assumes a
daily rebalance back to target that costs nothing and nobody asked for.
`engine/portfolio.py` propagates the book properly:

```
r_p,d  = sum_i w_i,d-1 * r_i,d
w_i,d  = w_i,d-1 * (1 + r_i,d) / (1 + r_p,d)
```

A name that doubles becomes a bigger share of the book, as it would in a portfolio nobody
is touching. Turnover then means the honest thing: the distance from the drifted book to
the new target, not from the old target to the new one. It also gives the exposure chart
something true to draw, since a dollar-neutral book does not stay dollar neutral.

---

## 4. Cost is a drag on return, which makes repricing exact

`net = gross - (bps / 1e4) * turnover`. Because the weight path drifts on **gross** return
and the targets never consult realised costs, the whole cost sweep comes out of one
backtest:

```python
def net_returns_at(result, cost_bps):
    return result.gross_returns - (cost_bps / 1e4) * result.turnover
```

Exact, not approximate, which is why the report shows a five-level ladder and a bisected
breakeven without re-running anything. It is called out in `analytics/sensitivity.py`
because the moment someone adds a cost-aware rebalance rule (skip the trade if cost exceeds
expected edge) it stops being true and the sweep has to start re-running.

**Rejected:** charging cost against the capital base. Arguably more correct for a funded
book, compounds differently, and makes the sweep need a re-run for a second-order
difference at these turnover levels.

---

## 5. The cache is a snapshot, not a speedup

The thing I did not know before starting is that Yahoo **rewrites history**. Adjusted close
is recomputed backwards on every dividend and split, so the same query a month apart gives
you different prices for 2015. That makes "reliably and reproducibly" in the brief a
data-layer problem rather than a caching one:

* one parquet per ticker holding the payload as fetched, raw OHLCV plus adjusted close,
* `manifest.json` with per-ticker fetch time, row count, first and last session, requested
  window and a SHA-256 of the file,
* `snapshot_id` is a SHA-256 over the manifest, embedded in every run,
* `xsbt verify` re-hashes and exits non-zero on drift.

A run is reproducible against **the bytes we actually saw**, which is the strongest claim
available without paying for point-in-time data. It is not the same as the data being
right, and `docs/ASSUMPTIONS.md` says so.

Two things about the client only showed up against the live endpoint, and both cost me
enough time to be worth writing down.

`requests.Session()` arrives with its own `User-Agent` already set, so
`headers.setdefault("User-Agent", browser_ua)` is a no-op, the request goes out as
`python-requests/2.32.5`, and Yahoo 429s every single call. The failure is well disguised:
it looks exactly like being rate limited for going too fast, so the instinct is to slow
down or add backoff, and neither does anything. It has to be plain assignment. The test for
it uses a real `requests.Session`, because the `FakeSession` double starts with empty
headers and hides the whole problem.

The second is that a rate limit is a fact about the session, not about the symbol that
tripped it. Retrying that one request with backoff and then going back to full speed means
the next symbol hits the same limit, forty times over. So a 429 doubles the inter-request
interval for the rest of the run, up to a ceiling, and `Retry-After` is honoured when the
server sends one.

**Rejected:** SQLite or DuckDB. Parquet plus a JSON manifest is diffable, hashable,
readable by anything, and has no migration story. A database earns its keep when you need
cross-sectional queries over a universe too big for memory, which is not this.

**Rejected:** storing only adjusted close. Halves the file size and throws away any chance
of reconstructing what the adjustment did.

---

## 6. Vectorised pandas, not an event loop

An event-driven simulator is the right shape for intraday, or for anything with partial
fills and queue position. For a monthly-rebalanced daily-close book it is a lot of
machinery whose only output is a weight matrix times a return matrix.

The engine is about 50 lines of pandas and runs the 16-year backtest in well under a
second, which is what makes the 25-cell parameter grid and the cost sweep affordable inside
a report build. The trade is that realistic fills later mean replacing `simulate()` rather
than extending it, and I think that is the right bet: an event loop written speculatively
for requirements nobody has stated is usually the wrong event loop.

---

## 7. Configs are frozen, validated and hashed

pydantic v2, `frozen=True` and `extra="forbid"`. The second matters more than it looks:
without it `lookback_dyas: 200` is silently ignored and you spend an afternoon wondering
why the parameter had no effect.

`fingerprint()` is a SHA-256 over the canonical JSON of the resolved config. With
`snapshot_id` it is the reproducibility claim: same fingerprint plus same snapshot means
the same numbers, and if they moved you can see which of the two moved first.

One gotcha, because it cost me time in `analytics/sensitivity.py`: `model_copy(update=...)`
does **not** re-run validators, so building the grid by copying a config with a new
`lookback_days` would happily produce one where `skip_days >= lookback_days`. The grid
rebuilds the strategy block with `model_validate` on a dumped dict instead.

---

## 8. A skipped rebalance is a real outcome

Below `min_names` eligible names the strategy returns an empty series and the engine holds
the previous book rather than trading into a two-name portfolio. The count is in
`metadata.json` and in the report's provenance block.

Quietly ranking whatever is available instead produces a backtest whose early years are
three names levered to gross 1.0, which is not a small distortion and is invisible unless
you go looking.

---

## 9. Session-counted annualisation

CAGR over `len(returns) / 252` rather than calendar days, because calendar dating makes a
backtest that ends on a Monday look different from one that ends on a Friday. Sharpe uses
the Lo (2002) standard error, `SE = sqrt(252 * (1 + SR_period^2 / 2) / T)`, which assumes
iid returns. Monthly rebalancing leaves autocorrelation in the daily series, so the
t-statistic is optimistic and the report says so rather than quietly printing a number with
no scale on it.

There is also a `FLAT_VOL = 1e-15` floor in the metrics, because pandas gives something
like `2e-19` for the standard deviation of a constant series rather than a clean zero, and
dividing by that hands a reader a Sharpe of 7e16. A small guard against a very silly
number.

---

## 10. The report is one HTML file

matplotlib inlined as base64, jinja2 for the page, no external assets and no JavaScript. It
survives being emailed, opens on a locked-down laptop, and archives next to the run that
produced it. `metrics.json` sits beside it for anything programmatic, deliberately without
a timestamp so two runs off one snapshot diff clean.

A number being arithmetically correct is not the same as it being worth printing. The first
real report said the long book was **562%** of the P&L and the short book **-462%**, which
is what you get dividing a net of +2.2% by legs of +12.2% and -10.0%. Exactly right, and it
tells a reader nothing except that the report might be broken. A long/short book nets a
small number out of two large offsetting ones almost by construction, so the split is now
dropped once it leaves the range a reader can interpret, and the page says why instead of
leaving a blank cell. The same pass caught "growth stops at roughly 0 bps" on reversal:
true, and useless, because that book loses money with costs switched off.

Neither was visible from the test suite, since both were working as designed. They are the
argument for the last line of my plan being "open the report and read it as a PM would".

**Rejected:** a notebook. Excellent for exploration, bad as a deliverable: it carries an
execution environment, diffs badly, and the output depends on cell order.

**Rejected:** a golden-file test on the HTML. Fails on every wording change and passes on
every change that matters. `tests/test_report.py` asserts behaviour instead: the page is
self-contained, every section is present, the verdict matches the t-statistic, and no
reader ever sees a bare `nan`.

---

## 11. Tests assert arithmetic, not previous output

Where a number can be worked out by hand, it is worked out in the docstring and asserted
exactly. `tests/test_portfolio.py` runs four tickers over ten days with weights and P&L
done on paper; `tests/test_attribution.py` does the same for beta, alpha and tracking
error. The failure mode I wanted to avoid is the suite that pins whatever the code happened
to produce on the day it was written, which locks in bugs and calls it regression coverage.

The Yahoo client runs against recorded JSON fixtures, including a 404 and a 429, so the
suite is fully offline and CI never depends on a vendor being up.

---

## What I would do next

In rough order of what would change a decision:

1. **Newey-West standard errors** on the Sharpe, so the significance test stops assuming
   independence it does not have.
2. **A point-in-time universe.** Survivorship is the largest bias here by a distance and
   everything below is second order next to it.
3. **Borrow cost on the short leg**, even a crude flat rate, because the leg attribution
   currently compares a financed leg to an unfinanced one.
4. **Walk-forward parameter selection**, so the grid becomes a procedure rather than a
   diagnostic and the reported result is out-of-sample.
5. **A capacity model.** Linear cost in turnover understates size badly, and breakeven is
   the number a PM will quote back at you.
6. **Sector and size neutralisation**, which is a natural second method on the rank base
   alongside `score`.
