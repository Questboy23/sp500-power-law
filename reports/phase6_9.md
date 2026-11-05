# Phases 6–9: engine, angle tests, ablation, decay (pre-holdout: returns ≤ 2015-12)

## Phase 6: engine validation
- Zero tilt reproduces BASE0 to within 1e-12 per month, and its cost is exactly 0 (`tests/test_backtest.py`).
- Each stock's multiplier stays inside [0.70, 1.30]. Ex-ante TE comes from a trailing-36-month holdings simulation, scaled to ≤ 2% (tested).
- Costs are 10 bps per unit of active turnover, plus 5 bps on the smallest 100 names. BASE0 is charged nothing.
- Walk-forward windows: 2001–05, 2006–10, 2011–15. That is **3 independent 5-year windows**.

## Phase 7: every angle alone (`phase7_angles.csv`)
**0 of 23 testable angles passed.**
- Gate: t ≥ 3, DSR ≥ 0.95 at the full trial count, right sign in ≥ 60% of windows, FF5+Mom alpha t ≥ 2.
- The best results were P2_zipf_switch (t 2.19, DSR 0.56), P8_survival (t 1.69, FF6 alpha t 2.02), A9_fluent_ticker (t 1.64), and A4_insider_buys (t 1.22, alpha t 2.30).
- Several published effects came out with the wrong sign inside large caps: MAX / lottery (P9, t −0.70, FF6 alpha t −2.2), boring/low turnover (B11, t −1.0), and R&D increases (A8, t −1.2).
- P5_trim_5pct: a 5% single-name cap never bound in 1996–2015, so "never trim" **could not be tested pre-holdout**.
- P5_early_entrants: −0.12%/yr, t −1.13.

Baselines, 1996-07 → 2015-12 (234 months):

| | CAGR | vol | Sharpe | vs BASE0/yr | TE |
|---|---|---|---|---|---|
| BASE0 cap weight | 8.08% | 15.4% | 0.44 | — | — |
| BASE1 equal weight | 10.17% | 17.5% | 0.51 | +2.27% | 6.14% |
| BASE2 total market | 7.97% | 15.9% | 0.42 | −0.02% | 2.63% |
| BASE3 value/mom/prof tilt | 8.44% | 15.3% | 0.46 | +0.32% | 0.74% |
| BASE4 low-vol tilt | 8.00% | 14.7% | 0.44 | −0.18% | 1.23% |

## Phase 8: ablation (`phase8_results.json`, `phase8_ablation.csv`)
- **Gated combined model = BASE0**, because nothing passed.
- The exploratory equal-weight combination of all 18 trading angles made +0.04%/yr, IR 0.11, t 0.48.
- By group:

| group | excess/yr | IR | t | months |
|---|---|---|---|---|
| power-law (7 angles) | −0.00% | −0.00 | −0.02 | 234 |
| published | +0.03% | 0.16 | 0.7 | 234 |
| vibe | −0.05% | −0.12 | −0.5 | 186 |
| Claude-added | +0.15% | 0.26 | 1.2 | 234 |

- **Implementation layer:**
  - **I1:** unbounded tilts (100% of cap weight, no TE cap) had 3× the TE for the same IR (0.115 vs 0.109).
  - **I2:** shrinkage by walk-forward IC confidence (average confidence 0.29) cut TE to 0.12% and IR to 0.09.
  - **I3:** additions earn +7.4% excess in the 20 days before their first index day (t 13.3, n 439) and −1.4% in days 0–4 (t −5.0, n 484). **Adopted.**
  - **I5:** costs are about 0.5 bp/yr on the combined tilt.
  - **I6:** direct-indexing TLH was worth up to 0.58%/yr in tax value (after-tax upper bound only).
  - **I7:** half-Kelly never binds.

## Phase 9: decay and regime weighting (`phase9_results.json`)
- **Setup:** a returns-only cross-sectional ridge model (12-1 momentum, 1-month reversal, 5-year reversal, 36-month volatility, size) on the top 500 / S&P, 1929–2014. It is trained on 10-year windows and scored at gaps of 0, 5, 10, 20, and 40 years. For the same-era control, every gap is averaged over the same 36 target years (1979–2014).
- **Model rank IC by gap:** 0.006, 0.041, 0.017, 0.052, 0.023. There is no decay. The fitted half-life is unidentifiable (90% bootstrap CI 0.1–693 years).
- **By signal:** low volatility was the steadiest (IC 0.03–0.06 at every gap). Momentum was about zero or negative in large caps at every gap.
- **Weighting training years**, targets 1950–2014, n = 65:

| weighting | mean OOS IC | t |
|---|---|---|
| equal | **0.040** | 2.3 |
| regime-similar | 0.032 | 1.4 |
| recency | 0.019 | 1.1 |

  Regime minus recency: +0.013 (t 0.9). Regime minus equal: −0.007 (t −0.3). **Adopted: equal weighting.**
