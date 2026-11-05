# Phase 3 — Power-law measurements (pre-holdout: every window ends ≤ 2015-12)

Universe: real S&P 500 membership from 1996-06; before that, the top 500 CRSP stocks by market
cap (NYSE-only before 1962). 88 June formation dates, 1927–2014. Raw tables: `reports/p1_tails.csv`,
`p2_zipf.csv`, `p4_transition.csv`, `p5_*.csv`, `phase3_results.json`.

## P1 Tail shape — lognormal, not power law
- **Clauset–Shalizi–Newman test:** out of 320 (formation × horizon) cross-sections, the power law was **never** the significantly better fit. Lognormal won 256 (80%); 64 (20%) were indistinguishable.
- CSN's KS-minimizing x_min lands low (it keeps 70–85% of the sample as "tail"), so its alpha ≈ 1.9–2.0 is really describing the body of the distribution. On a fixed top-10% tail, the **Hill** estimates are more informative:

| horizon | Hill α (top 10%), range of decade means | reading |
|---|---|---|
| 1y | 5.7 – 9.3 | thin tail |
| 5y | 2.6 – 4.9 | |
| 10y | 2.0 – 3.3 | |
| 20y | 2.0 – 2.7 | fat tail |

  The tail gets fatter the longer you compound. That is what a lognormal does (its σ grows with √horizon); it is not a stable power law.
- **Tail over time:** no trend at 1, 5, or 10 years (t = 0.07, 0.42, −0.21). At 20 years the tail got slightly *thinner* (+0.07 α per decade, t = 2.6).
- **Wealth concentration (Bessembinder-style, dollar wealth above T-bills):** the top 4% of index members produced a median **31–86% of each era's net wealth creation**. In some windows they produced more than 100%, because the rest destroyed wealth in total. The share of members that beat T-bills over 5 years ranged from 11% (the 1929 cohorts) to 91% (by decade average).

## P2 Zipf deviation — no predictive power
- Top-500 Zipf alpha: median 0.94, 10th–90th percentile 0.69–1.19, 1926–2015 (0.62 in 1926, 1.18 in 1986, 0.74 in 1996, 0.89 in 2015). Close to Zipf's law, but it drifts by era.
- Actual top-10 share minus the Zipf-implied share (fit on ranks 11–500), regressed on the next 5 years of equal-weight minus cap-weight return: β = −0.44, HAC t = −1.93 (n = 85, overlapping windows). On the 17 non-overlapping windows: β = +0.05, t = 0.12. The 1-year version: t = −0.89.
- **Verdict:** there is no evidence that over-concentration predicts equal weight beating cap weight. If anything, the overlapping-window sign points the other way: concentrated markets kept favoring cap weight.

## P3 Gibrat — fails downward, not upward
- Next-3y log sales growth on log sales, FIAM 2000–2012 formations: slope **−0.096, t = −22.2** (n = 7,106, clustered by date). Big firms grow *slower*, which is ordinary mean reversion.
- Per-formation slopes are negative in every year. Rich-get-richer (growth rising with size) is not the average regime. P3's trading angle tests industries where the slope is positive.

## P4 Tail persistence — none
- P(top 5% in the next 5y window | top 5% in this one) = **4.4%**, against a 5.0% base rate (n = 1,783 tail-winner survivors, 29,891 firm pairs, 1927–2005). Tail winners do not repeat. The full decile transition matrix is in `p4_transition.csv`.

## P5 Breadth and late entry
- **The late-entry cost is large.** Future S&P additions (1996+) beat the market by a mean +40% in the year before inclusion (median +20%, n = 434). Over 3 years before: mean +200%, median +69%. Over 5 years: mean +402%, median +129%.
- **After inclusion:** mean −2.1% vs the market, median −6.8% in the first year (n = 497).
- **Tail capture:** among the 1,500 largest US stocks, 57–83% of the next decade's top-5% winners were *not yet in the S&P* at the start. They produced 23–63% of the tail's dollar wealth, and 42–71% of them later joined the index. This motivates the P5_early_entrants test in Phase 7.

## P8 Survival
- Median time in the S&P 500 (1996+ entrants, Kaplan–Meier, n = 1,237): **9.0 years**.
- Firms first reaching the top 500 in each 20-year era:

| entry era | median years listed | P(distress death within 20y) | P(any exit within 20y) |
|---|---|---|---|
| 1920s–30s | 38.4 | 4.5% | 27% |
| 1940s–50s | 37.2 | 1.4% | 26% |
| 1960s–70s | 22.4 | 1.5% | 47% |
| 1980s–90s | 13.6 | 7.4% | 62% |
| 2000s | 12.3 | 10.7% | 55% |

  Large firms die younger now, and more of them die in distress.

## P10 Breadth simulation (20-year windows, formations 1936–1995)
| stocks held | median 20y multiple | 5th-percentile multiple | share of draws beating cap weight |
|---|---|---|---|
| 1 | 5.5× | 0.70× | 35% |
| 5 | 7.0× | 2.8× | 40% |
| 20 | 7.6× | 3.8× | 43% |
| 100 | 7.8× | 4.5× | 44% |
| 500 | 7.3× | 4.7× | 43% |

The Kelly fraction for the index is 2.53 (half-Kelly 1.27, 630 months), so a fully invested unlevered index is inside half-Kelly. Tilts are capped at 1.3× each stock's cap weight, far below any per-stock Kelly bound.

## P11 Generative model — rejected
- Proportional growth (σ of relative log growth = 0.32/yr) with the empirical 9.7%/yr top-500 churn, simulated 1936→2015, produces a median top-10 share of **74%** (90% band 53–93%). The observed share was **18%** in 2015 (1936–2015 mean 26%).
- Adding the size mean-reversion that P3 found (κ = 0.0024/yr) barely helps: 64%.
- A pure rich-get-richer process concentrates the market far more than history shows. Something pulls leaders back: competition, antitrust, and the negative size–growth slope. **So the model is not used to forecast concentration for P2** (P2 also failed on its own).

## P12 Cross-era
P1, P2, and P4 were run on every US decade 1926–2015 (above). The results agree across eras: the lognormal fit wins, Zipf's law holds with alpha ≈ 1, and tail winners don't persist. **International stock-level data is not available free** (JST Macrohistory has index levels only), so the cross-market half of P12 is **not tested**.

## Bottom line for the power-law thesis
The *wealth concentration* part is real: a few stocks create most of the wealth, and the effect grows with horizon. The *power law* itself (a scale-free tail, persistence, rich-get-richer dynamics) is not supported. The shape is lognormal, winners don't repeat, growth slows with size, and a proportional-growth model can't match the concentration we actually observe. The one actionable piece is **breadth/late entry (P5)**, tested as a portfolio in Phase 7.
