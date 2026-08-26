# Assumptions and known limitations

Where this backtest is easier than reality, roughly in order of how much each one flatters
the numbers. The short version is in the report footer, so a reader does not have to open
the repo to find it.

---

## Data

**Survivorship bias, and it is the big one.** `configs/universe_us_liquid.csv` is 40 US
large caps that are liquid *today*. Anything delisted, acquired or zeroed since 2010 is
absent. That flatters the long book directly and the short book indirectly, since the
shorts come from the same pool of survivors, and the 2010 start makes it worse rather than
better. Fixing it needs point-in-time index membership, which is a licensing problem and
not an engineering one. Until then, no number here estimates live performance. It measures
whether the machinery works.

**Yahoo restates history.** Adjusted close is recomputed backwards on every dividend and
split, so the same query a month later returns different 2015 prices. The cache pins a
`snapshot_id`, which makes a run reproducible against the bytes we actually saw. That is a
weaker claim than the data being right. Yahoo is also unofficial: no SLA, no documented
schema, no correction notices.

**Corporate actions are whatever the adjustment says.** Spin-offs, rights issues and
ticker reuse are not handled. A recycled ticker would give a silently wrong series.

**No fundamental or reference data.** The sector column is descriptive and unused.

**Gaps stay gaps.** `to_panel` takes the union of dates and does not forward-fill;
`CrossSectionalRankStrategy.eligible` drops NaN names for that rebalance. Filling would
fabricate a flat day and a zero return, which is worse than an absent one.

---

## Portfolio construction

**Equal weight inside each leg**, gross 1.0, dollar neutral at the moment of trading. No
vol scaling, no risk model, no position limits beyond the slice size, so a high-beta name
carries the same weight as a utility. Inverse-vol weighting is a small change to
`weights_from_scores` and is left out because it is a research decision, not a machinery
one.

**Dollar neutral is not beta neutral or sector neutral.** The report fits realised beta
against SPY precisely so this shows up instead of hiding.

**Weights drift and are not pulled back.** Gross exposure wanders away from 1.0 inside the
month. That is what happens to a book nobody touches, and the exposure chart shows it.

**A skipped rebalance holds the previous book.** Below `min_names` eligible names the
strategy returns nothing rather than trading into a two-name portfolio. The count is in
`metadata.json` and in the report.

---

## Execution and costs

**Close execution, no slippage.** A signal formed at the close of `t` is traded at the
close of `t + execution_lag_days` (default 1) and earns from the session after. No fill
model, no participation limit, no partial fills. Lag 1 is the conservative reading; lag 0
assumes the order is in before the same bell, which is achievable monthly but is an
assumption rather than a fact.

**Costs are linear in turnover, which understates size.** `cost = (cost_bps / 1e4) *
notional traded`, charged in and again out. 10bps on US large caps is deliberately unkind,
since half-spread is nearer 1 or 2. What is missing is that cost is a function of size: no
square-root impact, no capacity limit, no per-name spread. The breakeven number is the one
a PM will quote back at you, and at size the true number is lower.

**Cost is a drag on return, not a charge against capital.** `net = gross - rate *
turnover`, which is what makes the cost sweep exact rather than approximate. Charging
against the capital base compounds slightly differently and is second order here.

**Shorting is assumed free.** No borrow charged, no rebate credited. Generous in a
correlated way: the worst-ranked names are usually the hardest and dearest to borrow, and
that is exactly where the short book claims its P&L. The leg attribution exists partly so
this is visible.

**No financing, margin, cash accounting or tax.** `risk_free_rate` is a hurdle for Sharpe
and Sortino only, deliberately not credited to the P&L: pretending to know the short
rebate without borrow data is worse than leaving it out.

---

## Statistics

**The Sharpe error bar is adjusted for autocorrelation, not for the search.** The base
result is Lo (2002), `SE = sqrt(252 * (1 + SR_period^2 / 2) / T)`, which is derived for iid
returns. A book carried for a month is not iid, so the bar is rescaled by a Newey-West
factor over the run's own holding horizon, and both versions are on the report. On these two
runs the factor came out below 1, so the adjusted bar is the tighter of the two. That is the
estimator doing its job on a series that alternates rather than trends, and not a sign that
the correction was skipped.

**Nothing is charged for the parameter search.** The grid is 25 configurations and the
t-stat on the chosen cell takes the sample at face value. That is why the grid is a
robustness check and not a selection procedure: it answers "is this cell surrounded by
good cells", not "which cell is best". Every cell is scored in-sample. A borderline t-stat
should be read as generous for that reason.

**Annualisation is by session count**, 252 throughout, with CAGR over `len(returns) / 252`.
Calendar dating makes a run that ends on a Monday look different from one that ends on a
Friday.

**The market fit is a plain OLS** of net return on benchmark, both over cash, with a
constant beta across the whole sample. A book that is long the market in one regime and
short it in another comes back with a beta of zero and no hint that it did so.

---

## Engineering

**No second data source.** Nothing cross-checks Yahoo's adjustment, so a systematic error
there flows through everything and no test would catch it.

**Vectorised, not event-driven.** Realistic fills mean replacing `simulate()` rather than
extending it. Right trade for a monthly daily-close book, wrong for anything intraday.

**Single process, in memory.** Trivial at 40 names and 16 years. At a few thousand names
it is a data-layer rewrite, not a tuning exercise.
