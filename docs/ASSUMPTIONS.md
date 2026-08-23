# Assumptions and known limitations

Everything here is a place where the backtest is easier than reality. They are listed
roughly in order of how much they flatter the numbers. The short form of this list is in
the report footer, because a reader should not have to open the repo to find it.

---

## Data

### Survivorship bias (the big one)

`configs/universe_us_liquid.csv` is a fixed list of 40 US large caps that are liquid
**today**. Anything that was delisted, acquired or went to zero between 2010 and now is
not in it.

This is the largest single bias in these results, and it is not small. Every name in the
universe is one that survived the whole sample, which flatters the long book directly and
the short book indirectly, because the shorts are drawn from a pool that also survived.
The 2010 start date makes it worse rather than better: it is a longer window over which
to have quietly excluded the failures.

Fixing it properly needs a point-in-time index membership file, which is a licensing
problem rather than an engineering one. Until that exists, no number in the report should
be read as an estimate of live performance. It is a measure of whether the machinery
works and whether the effect is present in a clean sample.

### Yahoo restates history

Adjusted close is recomputed backwards every time a dividend or a split lands, so the
same query run a month apart returns different prices for 2015. The cache is a snapshot
with a `snapshot_id`, so a given run is reproducible against the bytes we actually saw,
but that is a weaker claim than the data being right.

Yahoo is also an unofficial API with no SLA, no documented schema and no correction
notices. Anything that mattered would be priced off a vendor with a contract behind it.

### Prices are adjusted close, and only adjusted close

Splits and dividends are whatever Yahoo's adjustment says they are. Spin-offs, rights
issues, and ticker changes are not handled at all. A ticker that was reused by a different
company inside the sample would produce a silently wrong series.

### No fundamental or reference data

No sector neutralisation, no market cap weighting, no earnings dates. The sector column
in the universe CSV is descriptive and is not used by anything.

### Gaps are left as gaps

`to_panel` uses the union of dates and does not forward-fill. A name that was not trading
is NaN, and the eligibility filter in `CrossSectionalRankStrategy.eligible` drops it for
that rebalance. Forward-filling would fabricate a flat day and a zero return, which is
worse than an absent one.

---

## Portfolio construction

### Equal weighting inside each leg

Long the top `top_fraction` and short the bottom, equal weighted, gross 1.0, dollar
neutral at the moment of trading. No volatility scaling, no risk model, no position
limits beyond the slice size.

This means a single high-beta name carries the same weight as a utility, and the book's
realised volatility is whatever the cross-section happens to give. Inverse-volatility
weighting would be a small change to `weights_from_scores` and is left out on purpose:
it is a research decision, and the brief was about the machinery.

### Dollar neutral, not beta neutral or sector neutral

The legs are equal in notional, which does not make them equal in market exposure. The
report fits the realised beta against SPY precisely so this shows up rather than hiding.
If the beta comes out materially away from zero, part of the Sharpe is a market call.

### Weights drift, and are not rebalanced back

Between monthly rebalances the book is left alone and weights drift with realised
returns. This is the honest model of what happens to a portfolio nobody touches, but it
means gross exposure wanders away from 1.0 within the month. The exposure chart in the
report shows the drift.

### A skipped rebalance holds the previous book

Below `min_names` eligible names, the strategy returns nothing and the engine holds. The
count of skipped rebalances is in `metadata.json` and in the report. In a sample this
size it should be zero or near it, and if it is not, that is worth knowing.

---

## Execution and costs

### Execution is at the close, with no slippage

A signal formed at the close of `t` is traded at the close of `t + execution_lag_days`
(default 1) at that close's price, and starts earning from the session after. There is no
model of the gap between deciding and filling, no participation limit, and no
partial fills.

The default lag of 1 is the conservative reading. Lag 0 is the market-on-close convention
and assumes you can compute the signal and get the order in before the same bell, which
is achievable for a monthly rebalance but is an assumption rather than a fact.

### Costs are linear in turnover, which understates size

`cost = (cost_bps / 1e4) * notional traded`, charged on the way in and again on the way
out. 10bps per unit traded on US large caps is deliberately unkind: the half-spread is
closer to 1 or 2bps. Erring high keeps a strategy from looking good only because the cost
line was optimistic.

What this does not capture is that cost is a function of **size**. There is no
square-root impact term, no capacity limit and no per-name spread. The breakeven number
in the report is the one a PM will quote back at you, and it is a linear-model breakeven.
At size, the true number is lower.

### Costs are a drag on return, not a charge against capital

`net = gross - rate * turnover`. This is what makes the cost sweep exact rather than
approximate, since the weight path drifts on gross return and the targets never consult
realised costs. Charging cost against the capital base compounds slightly differently and
is arguably more correct for a funded book. At these turnover levels the difference is
second order and the exactness is worth more.

### Shorting is assumed free

Borrow is assumed available and free. The short rebate is not credited and borrow cost is
not charged. Both assumptions are generous, and they are generous in a correlated way:
the names that rank worst on momentum are frequently the ones that are hardest and most
expensive to borrow, which is exactly where the short book claims its P&L.

The leg attribution section exists partly so this is visible. An edge that lives entirely
in the short book is a different proposition from one split evenly.

### No financing, margin or cash accounting

A long/short book at gross 1.0 posts margin and earns something on the short proceeds.
None of that is modelled. `risk_free_rate` in the config is a hurdle for Sharpe and
Sortino only, and is deliberately **not** credited to the P&L: pretending to know the
short rebate without borrow data would be worse than leaving it out.

### No taxes, no fees beyond the spread proxy

---

## Statistics

### The Sharpe standard error assumes iid returns

Following Lo (2002):

```
SE(SR_annual) = sqrt(252 * (1 + SR_period^2 / 2) / T)
```

Monthly rebalancing leaves autocorrelation in the daily series, so the true standard
error is larger and the reported t-statistic is optimistic. A Newey-West correction is
the right fix and is the first item on the list at the end of `DESIGN.md`.

### The t-statistic is not corrected for the parameters searched

The report shows a lookback by top-fraction grid, which is 25 configurations. The
t-statistic on the chosen cell takes the sample at face value and charges nothing for the
search. The report's verdict sentence says so explicitly.

This is why the grid is presented as a **robustness check** and not as a selection
procedure. The question it answers is "is the configured cell surrounded by other good
cells", not "which cell is best".

### The parameter grid is in-sample

Every cell is scored over the full sample. There is no walk-forward split, so the grid
cannot be used to choose a parameter without overfitting to the same data the result is
reported on.

### Annualisation is by session count

252 sessions per year throughout, and CAGR is computed over `len(returns) / 252` rather
than over calendar days. Calendar dating makes a backtest that happens to end on a Monday
look different from one that ends on a Friday.

### The market fit is a plain OLS

Beta and alpha come from an ordinary least squares fit of the net daily return on the
benchmark, both over cash. Constant beta across the whole sample, no rolling estimate, no
Newey-West on the standard errors of the fit, no multi-factor decomposition. The book
could have a beta of zero on average while being long the market in one regime and short
it in another, and this would not show it.

---

## Engineering

### Not tested against a second data source

There is no cross-check that Yahoo's adjusted close agrees with anyone else's. A
systematic adjustment error would flow through everything and nothing in the test suite
would catch it.

### The backtest is vectorised, not event-driven

There is no order object, no fill model and no broker. Adding realistic execution later
means replacing `simulate()` rather than extending it. That is a deliberate trade for a
monthly-rebalanced daily-close book, and it is the wrong trade for anything intraday.

### Single process, in memory

The whole panel is held in memory. At 40 names and 16 years that is trivial. At a few
thousand names it would need chunking or a columnar store, which is a rewrite of the data
layer rather than a tuning exercise.
