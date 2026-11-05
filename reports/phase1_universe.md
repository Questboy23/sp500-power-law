# Phase 1 — Universe and labels

**Data grade: "free-data grade, CRSP-backed."** No WRDS access. The stock-level data is a CRSP
daily extract (1968-01-02 → 2024-12-31, 37,876 permnos, total returns incl. dividends, shares
outstanding) that other users re-posted on Kaggle (`meliodaz/crsp-with-mkt-ret-68-24`), plus a
CRSP monthly ticker file used to build the ticker↔PERMNO link. Personal-research use only;
raw rows are never republished (the site shows aggregates only).

| Piece | Source | Coverage |
|---|---|---|
| S&P 500 membership | fja05680/sp500 (Clenow/Norgate-derived, point-in-time tickers) | 1996-01-02 → 2026-08-18, 487–507 names/day |
| Returns + caps, 1996–2024 | CRSP daily (Kaggle) via ticker+date → PERMNO | 100.0% of member-months linked 1996–2023, 99.9% in 2024 |
| Returns + caps, 2025-01 → 2026-09 | Yahoo Adj Close; cap = close × shares implied by the 2024-12 CRSP cap | 487–494 of ~503 names (3% missing) |
| Total US market | CRSP daily, shrcd 10/11 | 3,565–7,462 stocks/month; tracks VTI at r=0.9987, +0.1%/yr |
| Fundamentals | SEC EDGAR XBRL companyfacts | 10-K facts 2009+, `filed` date kept for point-in-time use |
| Factors | Ken French library (FF5 + momentum) | through 2026-08 |

## Sanity check: reconstructed cap-weight vs real S&P 500 total return (^SP500TR)

- Mean absolute annual gap **0.33%/yr** (bar: ~0.5%). Worst year 2000 (+1.71%): CRSP total shares vs S&P float adjustment during the dot-com peak.
- Monthly correlation **0.99964**, tracking error **0.41%/yr**, CAGR 10.63% vs 10.47% (1996-03 → 2026-08, 366 months).
- Year-by-year table: `reports/phase1_tracking.csv`.

## Survivorship handling and known gaps

1. **Retroactive tickers.** The membership file sometimes uses a firm's *later* symbol (SPGI for 1996 McGraw-Hill, WBA for Walgreen, NEE for FPL). Resolved by PERMNO persistence: a PERMNO that ever carried the symbol and was alive on the date (8,967 member-months). 38 bankrupt "Q" tickers mapped by hand to the symbol CRSP carried while listed (ENRNQ→ENE, LEHMQ→LEH, AAMRQ→AMR, ...), listed in `src/universe.py:MANUAL`.
2. **No CRSP delisting returns (DLRET).** A permno that stops trading while distressed (last price < $5 and < 30% of its 12-month high) gets **−100%** in the following month, per spec, and is flagged (`labels.distress_flag`; 61 permnos). Other exits get 0% for the missing tail. Only 6 member-months were distress exits *while in the index* — most failures are removed from the S&P before they delist, and their collapse is in ordinary CRSP returns.
3. **2025+ caps hold share counts fixed** at their 2024-12 level (buyback drift over 21 months ignored).
4. **Membership-file doubt, 1996–2000:** 487–492 names vs 500; the maintainer flags possibly missing symbols in the first five years.
5. **Float:** CRSP shares are total shares, not S&P's float-adjusted count.

## Labels

June formation dates 1996–2024. `fwd_1y/5y/10y/20y` buy-and-hold total returns; `tail` = top 5% by 5-year return within each formation cross-section. Every row carries `label_end`; a 2011+ formation's 5-year label ends in the holdout and `holdout_guard.assert_pre_holdout` refuses it.

**Pre-holdout supply is small:** 5-year labels that end before 2016 exist for formations 1996–2010 — 15 overlapping cross-sections, **3 non-overlapping 5-year windows** (2001–05, 2006–10, 2011–15 test windows with 1996+ training). Every Phase 7 verdict carries that caveat.
