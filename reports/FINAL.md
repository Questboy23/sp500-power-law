# S&P 500 Power-Law Research System — Final Report (2026-09-30)

## 1. Verdict
**The system did not beat the S&P 500, and the research says it shouldn't try.**
- **Gate:** none of 23 testable angles passed on 1996–2015 data. The gate was t ≥ 3, deflated Sharpe ≥ 0.95 at the full trial count, the right sign in ≥ 60% of 5-year windows, and FF5+momentum alpha t ≥ 2.
- **Final portfolio:** plain S&P 500 cap weight, plus one implementation rule (buy index additions 5 trading days late).
- **Holdout** (2016-01 → 2026-08, 128 months, opened once, net of costs):

| portfolio | annualized return |
|---|---|
| SYSTEM | 15.22% |
| BASE0 (reconstructed index) | 15.20% |
| S&P 500 total return | 15.16% |
| VOO | 15.13% |

- **Against VOO:** SYSTEM finished +0.08%/yr ahead, with 0.39% tracking error. That is reconstruction noise (total-share caps vs float weights), not skill.
- **The late-addition rule** contributed +0.016%/yr. Its underlying effect faded: −0.23% over days 0–4 (t −0.6, n 238), against −1.4% (t −5.0) before 2016.
- **Overfitting test:** the spec says an angle whose holdout IR is below half its pre-holdout IR is likely overfit. 8 of the 13 angles with a non-negative pre-holdout IR were flagged: P2, P8, A4, N4, N2, N3, P9_hype_composite, and A5. Of the four best pre-holdout angles, only A9 kept more than half its IR (0.37 → 0.23, holdout t 0.75), and no holdout result would clear t ≥ 3.

**Holdout procedure note.** The first holdout run crashed inside P6 on a 2024 formation that has no FIAM data. It crashed after the lock was set but before any result was computed or printed. The bug fix skips formations without characteristics and changes no parameters. The resume is recorded in `data/holdout/.consumed.json` (`resumed`) and in git (tag `phase10-freeze` plus the fix commit).

## 2. Power-law findings (details in `phase3_powerlaw.md`)
- **Tail shape is lognormal, not a power law.** Lognormal won 256 of 320 cross-sections and the power law won 0. The Hill α on the top 10% is about 6–9 at 1 year and about 2.0–2.7 at 20 years; the tail fattens with horizon, as a lognormal does. There is no trend in tail fatness across eras.
- **Wealth is concentrated.** The top 4% of members produced a median 31–86% of each era's wealth above T-bills.
- **Zipf's law holds but predicts nothing.** Top-500 alpha has median 0.94. Over-concentration versus Zipf predicted nothing: t = 0.12 on non-overlapping windows. As a bounded tilt (P2) it was the best pre-holdout angle (IR 0.50) and went to −0.34 in the holdout.
- **Gibrat's law fails downward.** Size slope −0.096 (t −22). Big firms grow slower.
- **Tail winners don't persist.** 4.4% of top-5% winners repeat, against a 5% base rate.
- **The index enters late.** Additions beat the market by a median +20% the year before joining and +69% over 3 years (mean +200%), then lag. 57–83% of each decade's tail winners were outside the S&P at its start. Holding early entrants still lost (−0.12%/yr pre-holdout).
- **Survival is getting worse.** The median S&P tenure is 9.0 years. The chance a top-500 firm dies in distress within 20 years rose from about 1.5% (1940s–70s entrants) to 7–11% (1980s+).
- **Generative model rejected.** Proportional growth plus the observed churn predicts a 64–74% top-10 share against the 18–26% observed.
- **Cross-market untested.** Cross-era agreement holds back to 1926 (US only); no free international stock-level data was available.

## 3. Angles (full table: `phase7_angles.csv`; holdout: `phase10_holdout.json`)
- **Trial count:** 59 distinct trials in `trials.csv`; 23 angles tested, 15 registered as untestable on free data.

| angle | pre IR (t) | holdout IR (t) |
|---|---|---|
| P2_zipf_switch | 0.50 (2.19) | −0.34 (−1.12) |
| P8_survival | 0.58 (1.69) | 0.06 (0.20) |
| A9_fluent_ticker | 0.37 (1.64) | 0.23 (0.75) |
| A4_insider_buys | 0.42 (1.22) | −0.61 (−2.01) |
| N4_low_beta | 0.31 (1.22) | −0.62 (−2.03) |
| P9_hype_composite | 0.09 | −0.72 (−2.35) |
| P6_tail_classifier | −0.31 | −0.76 (−2.48) |
| B8_scale_gets_cheaper | −0.11 | 0.68 (2.22) |
| A8_rd_underpriced | −0.31 | 0.58 (1.90) |
| LIVE_SHADOW (registered before the holdout) | — | −0.32 (−1.05) |
| all angles combined (exploratory) | 0.08 | −0.30 (−0.99) |

Signs flipped in both directions, which is what noise looks like.

**Ablation, pre-holdout, combined by group:**

| group | excess/yr | t |
|---|---|---|
| power-law | −0.00% | −0.02 |
| published | +0.01% | 0.25 |
| vibe | −0.08% | −0.92 |
| Claude-added | +0.15% | 1.17 |

None is significant, and none helps beyond BASE3.

## 4. Group verdicts
- **Power-law angles:** nothing. The measurements are interesting; the trades aren't.
- **Published signals:** inside large caps 1996–2015, none survived. Several flipped sign (MAX/lottery, R&D).
- **Vibe heuristics:** nothing, and 9 of 14 couldn't be tested on free data.
- **Implementation fixes:** the only positive results, both small.
  - I3, buy additions late: +0.016%/yr in the holdout.
  - Bounded tilts: kept TE under 1%.
  - Tax-loss harvesting: up to about 0.58%/yr in tax value, after tax only. Worth more than every angle combined.

## 5. Decay and weighting (`phase9_results.json`)
- **No decay was measurable.** IC is flat from a 0- to 40-year gap, and the half-life is unidentifiable (CI 0.1–693 years).
- **Equal weighting of training years won.** Mean IC: equal 0.040, regime-similar 0.032, recency 0.019.

## 6. Known weaknesses
- **Data grade:** free data only. CRSP comes from a Kaggle re-upload and has no delisting returns (distress exits set to −100%). FIAM characteristics exist only for 2000–2023. Membership for 1996–2000 may be missing a few names.
- **Only 3 independent pre-holdout windows**, and 128 holdout months.
- **LLM layer:** ChronoBERT embeddings (MIT licence; yearly vintages 1999–2024) were built for 19,042 10-Ks, each embedded with the vintage from the year before its filing. The look-ahead diagnostic showed little recoverable signal: past-return AUC was about 0.50 for every vintage. There was one weak contamination hint (the 2009 vintage beat the 2004 vintage on 2005–09 filings by +0.014, CI +0.005 to +0.024, over only five years). Name removal leaves about 3–4% of filings with a recognisable short name. P6 was tested without text features; adding them now would be a new trial, and the holdout is spent. See `phase5_llm.md`.
- **2025+ caps** hold share counts fixed at their 2024-12 level.

## 7. Live paper portfolio (Phase 11)
- **Setup:** three portfolios of $100,000 each, started at the 2026-09-30 close:
  - SYSTEM: the S&P 500 at SPY's actual weights, plus rule I3.
  - SHADOW: the LIVE_SHADOW tilt, which failed the gate.
  - VOO.
- **Marking:** every weekday at 17:30 by cron. The results page is reedsmoneymaker23.com/sp500. Live results are kept separate from the backtest.
- **Expectation, 1 year out:**
  - SYSTEM vs VOO: within about ±0.2%.
  - SHADOW vs VOO: within about ±0.9% (95% range), with an expected difference of zero.
