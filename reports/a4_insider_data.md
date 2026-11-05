# A4_insider_buys: data build notes

Code: `src/features/a4_insider.py` (`download | parse | build | all`). Output: `data/processed/a4_insider.parquet`, 10,021 rows = 20 June-30 formation dates (2007 to 2026) x S&P members on that date. Intermediates are in `data/raw/form4/`: 82 zips, `trans_ps.parquet`, `filers.parquet`, `schema_headers.json`, `build_stats.json`, `urls.txt`. **No returns or backtests were computed.**

## Source and download (verified)
- Index page: `https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets` (the old `/dera/data/form-345` URL redirects there). The code scrapes the zip links from this page on every run.
- Zip URLs follow `https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{YYYY}q{N}_form345.zip`. The newest quarter (2026q2) sits under a different path: `/files/datastandardsinnovation/data/insider-transactions-data-sets/2026q2_form345.zip`. Hard-coding the old path would miss it.
- 82 quarterly zips, 2006q1 to 2026q2, about 880 MB. User-Agent was `sp500-power-law research <SEC_CONTACT_EMAIL>`, with at least 0.2 s between requests (5 req/s max).
- Filing dates in the data run from 2006-01-03 to 2026-06-30, so the 2026-06-30 formation date has full coverage.

## Schema verification
Each zip holds `SUBMISSION, REPORTINGOWNER, NONDERIV_TRANS, NONDERIV_HOLDING, DERIV_TRANS, DERIV_HOLDING, FOOTNOTES, OWNER_SIGNATURE` (.tsv), plus a readme and a JSON metadata file. The header row of every quarter was saved to `schema_headers.json` and compared:
- **REPORTINGOWNER and NONDERIV_TRANS:** the same header in all 82 quarters.
- **SUBMISSION:** the same header in all quarters, except that one column (`AFF10B5ONE`, the 10b5-1 plan flag) was added from 2023q1. It is not used.
- **Fields used:** `SUBMISSION.{ACCESSION_NUMBER, FILING_DATE, DOCUMENT_TYPE, ISSUERCIK, ISSUERTRADINGSYMBOL}`, `REPORTINGOWNER.{ACCESSION_NUMBER, RPTOWNERCIK, RPTOWNER_RELATIONSHIP}` and `NONDERIV_TRANS.{ACCESSION_NUMBER, NONDERIV_TRANS_SK, TRANS_DATE, TRANS_CODE, TRANS_SHARES, TRANS_PRICEPERSHARE, TRANS_ACQUIRED_DISP_CD}`. Rows are joined on ACCESSION_NUMBER.
- **Dates:** the metadata JSON gives `DD-MON-YYYY` for dates. That format parsed for every FILING_DATE. 11 of 3.5M TRANS_DATE values did not parse and were dropped.
- **DOCUMENT_TYPE values:** 3, 3/A, 4, 4/A, 5, 5/A.

## Cleaning
| step | rows |
|---|---|
| NONDERIV rows with TRANS_CODE P or S | 3,502,773 |
| drop amendments (4/A, 5/A; keep original Form 4 and Form 5) | 3,431,997 |
| drop P rows not coded A (4,129) and S rows not coded D (4,982); drop trans_date > filing_date or before 1990 (701) | 3,422,174 |
| of which purchases (P+A) / sales (S+D) | 831,756 / 2,590,418 |

- **Duplicate rows:** duplicates on (ACCESSION_NUMBER, NONDERIV_TRANS_SK) were removed.
- **Dollar value = shares x price.** 2,227 rows were set to NaN because the price was more than 20x (or less than 1/20) of the issuer's median P/S price that month, or the row was above $10B. Counts are not affected by this. The largest values that remain were spot-checked and are real: RAI in 2015 (BAT), TSLA in 2021-22 (Musk), BAC in 2017 and OXY in 2022 (Berkshire).
- **Rows per filing before 2009:** 2006-2008 have about 3x more P/S rows per filing than later years, while the number of filings stays flat. This matches filers reporting every execution price on its own line before the SEC allowed weighted-average reporting (around Dec 2008). This is why a "trade" is counted as one event per (insider, trade date, direction), not one per row. Row counts are not comparable across that break.
- **Joint filings:** 8.7% of P/S rows come from filings with several reporting owners. Each filing is assigned to one primary owner: an Officer or Director first, otherwise the lowest owner CIK. This keeps a fund and its general partner from counting as two buyers.

## Routine vs opportunistic (Cohen, Malloy & Pomorski 2012)
An insider is the pair (reporting-owner CIK, issuer CIK). For a trade in calendar year Y:
- **classifiable:** the insider has at least one open-market trade (P or S, either direction) in **each** of Y-1, Y-2 and Y-3. This is CMP's history requirement.
- **routine:** classifiable, and there is some calendar month m with a trade in the same direction in (Y-1, m), (Y-2, m) and (Y-3, m). All of that insider's trades in that direction in year Y are then routine.
- **opportunistic:** classifiable and not routine.
- **unclassified:** everything else.

Classification is redone at each formation date F using only rows with FILING_DATE <= F.

What this means for the data:
- **Early years:** the data starts in 2006, so no trade before 2009 can be classified. The 2007 and 2008 formations are almost all unclassified (a4 is about 0). The 2009 formation is partial. **2010 is the first formation where the classification is meaningful.**
- **Opportunistic buys are rare among S&P 500 firms.** From 2010 on, the S&P-member totals are 920 opportunistic buy events, 1,583 routine and 10,464 unclassified. Per formation date, only about 17 to 47 firms have an opportunistic buyer, against 155 to 260 firms with any buyer. Most large-cap insider buyers do not trade in each of the 3 prior years, so CMP leaves them unclassified. Sales dominate, so `a4` is ≤ 0 for most firms (mean about -0.35, median -0.5).
- **Unclassified insiders, reported both ways as requested:** `a4` uses opportunistic insiders only. `a4_incl_uncl` treats unclassified insiders as opportunistic. The unclassified counts and values are also kept in their own columns.

## Output columns (`a4_insider.parquet`)
- **Keys:** `formation_date, permno, ticker, permno_src` (the `how` column from membership_linked: span / lifetime / carried / synthetic 900000+), and `member_key`.
- **Signals:**
  - `a4 = (n_buyers_opp - n_sellers_opp) / (n_buyers_opp + n_sellers_opp + 1)`
  - `a4_incl_uncl`: the same formula with unclassified insiders counted as opportunistic.
  - `a4_scaled = n_opp_buys / n_insiders_filing`, the angles.yaml wording. The denominator is the number of distinct primary owners filing any original Form 4 on the issuer in (F-12m, F].
- **Counts:** for cls in {opp, routine, uncl}:
  - `n_{cls}_buys`, `n_{cls}_sells`: events per (insider, trade date, direction).
  - `n_buyers_{cls}`, `n_sellers_{cls}`: distinct insiders.
  - `val_{cls}_buy`, `val_{cls}_sell`: dollars.
  - `net_opp_buy_value = val_opp_buy - val_opp_sell`.
- **Window:** trade date in (F-12 months, F] and FILING_DATE <= F.
- **Missing vs zero:** `has_form4_link` is False when no issuer CIK maps to the member. For those rows the a4 signals are NaN, not zero.
- **Timing check:** for 2025 and 2026 the file uses the updated `membership_linked.parquet` (rewritten by another process at 14:38 today, with `carried` and `synthetic` permnos). The fallback for permno = -1 in the code now only covers 11 membership rows, none of them members on a formation date.

## Permno mapping and match rate
Issuer CIK is mapped to permno in four steps:
1. Normalized `ISSUERTRADINGSYMBOL` plus the filing month-end is matched exactly to membership_linked (date, ticker). A second pass strips dots (BRK.B vs BRKB). A (CIK, permno) link is kept only if it holds at least 20% of that permno's matches. This gives 942 links.
2. For members that never matched in step 1, their ticker is matched to Form 4 tickers at any date (25 links). This fixes membership rows that carry a later ticker for the whole history: AABA for YHOO, TPR for COH, KDP for DPS, WYND for WYN, HSH for SLE, ATGE for DV, VIAV for JDSU.
3. `data/raw/edgar/company_tickers.json` current tickers for members still unmatched (5 links: FOXA, NWSA, BF.B, LEN and one 2026-08 addition).
4. A CIK is applied to all of its filings. A CIK linked to several permnos (25 of them) goes to the permno whose matched span is closest to the filing date. Ties go to every tied permno, which covers dual-class shares.

Results:
- **All issuers:** 855,043 of 3,422,174 cleaned P/S rows (25%) map to some S&P permno. The rest are non-S&P issuers, which is expected.
- **Members with a CIK link:** 98.6% in 2007, at least 99.0% in 2009-2022, and 100% in 2023-2026 (see the table).
- **Members still unlinked:**
  - Bankruptcy-suffixed tickers from 2007-2011 (CITGQ, CCTYQ, MTLQQ, RSHCQ, SUNEQ, ANRZQ).
  - FOXA before 2019 (the old News Corp / 21CF class-A permno).
  - VIAB (2007-2019).
  - FRC and SBNY. These are bank-level filers that report to the FDIC, not the SEC, so they have no EDGAR Form 4s.
- **Cross-check:** compared with the separate CUSIP-based `data/processed/permno_cik.parquet` (built by another process), 857 of the 875 overlapping links agree (97.9%). The 18 that differ are mostly holding-company reorganizations or parent/subsidiary CIKs. A few are low-share ticker collisions (for example, CIK 4611 on permno 27713 with 1 matching month). Those were not removed by hand.

## Coverage per formation date (S&P members)
| formation | members | linked | link% | any_form4_12m | any_PS_12m | opp_buys | routine_buys | uncl_buys | opp_sells | routine_sells | uncl_sells | firms_opp_buyer | firms_any_buyer | a4!=0 % | a4 mean | a4_incl mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2007-06-30 | 496 | 489 | 98.6 | 476 | 458 | 0 | 1 | 944 | 0 | 0 | 9564 | 0 | 231 | 0.0 | 0.0 | -0.606 |
| 2008-06-30 | 496 | 490 | 98.8 | 481 | 470 | 2 | 2 | 1515 | 40 | 23 | 7634 | 1 | 289 | 2.2 | -0.01 | -0.404 |
| 2009-06-30 | 499 | 494 | 99.0 | 488 | 466 | 73 | 29 | 1609 | 514 | 350 | 3931 | 33 | 304 | 34.7 | -0.149 | -0.278 |
| 2010-06-30 | 499 | 495 | 99.2 | 489 | 476 | 42 | 58 | 678 | 2052 | 1061 | 5838 | 20 | 220 | 52.1 | -0.319 | -0.577 |
| 2011-06-30 | 497 | 492 | 99.0 | 486 | 474 | 49 | 67 | 619 | 2078 | 1427 | 7014 | 19 | 208 | 53.1 | -0.32 | -0.612 |
| 2012-06-30 | 497 | 494 | 99.4 | 488 | 478 | 120 | 84 | 845 | 1767 | 1184 | 5632 | 27 | 250 | 51.5 | -0.311 | -0.519 |
| 2013-06-30 | 497 | 495 | 99.6 | 489 | 486 | 58 | 93 | 558 | 2412 | 1655 | 6913 | 19 | 205 | 61.2 | -0.392 | -0.654 |
| 2014-06-30 | 498 | 496 | 99.6 | 489 | 479 | 28 | 79 | 500 | 2145 | 1960 | 6454 | 18 | 169 | 63.1 | -0.411 | -0.68 |
| 2015-06-30 | 499 | 497 | 99.6 | 491 | 474 | 31 | 83 | 486 | 2251 | 2159 | 5783 | 17 | 177 | 67.9 | -0.444 | -0.653 |
| 2016-06-30 | 503 | 501 | 99.6 | 497 | 490 | 48 | 99 | 832 | 1794 | 2050 | 4537 | 30 | 258 | 65.4 | -0.399 | -0.515 |
| 2017-06-30 | 505 | 503 | 99.6 | 499 | 495 | 46 | 114 | 602 | 2229 | 2156 | 5677 | 28 | 209 | 66.5 | -0.427 | -0.639 |
| 2018-06-30 | 505 | 503 | 99.6 | 501 | 490 | 40 | 96 | 606 | 2115 | 2004 | 5440 | 22 | 205 | 65.0 | -0.406 | -0.607 |
| 2019-06-30 | 505 | 503 | 99.6 | 501 | 490 | 60 | 102 | 719 | 1827 | 2231 | 4464 | 40 | 214 | 64.6 | -0.374 | -0.563 |
| 2020-06-30 | 505 | 504 | 99.8 | 504 | 491 | 134 | 89 | 747 | 1683 | 2310 | 4704 | 47 | 244 | 62.4 | -0.353 | -0.511 |
| 2021-06-30 | 505 | 504 | 99.8 | 504 | 488 | 45 | 66 | 421 | 2028 | 2471 | 5615 | 22 | 180 | 60.4 | -0.377 | -0.654 |
| 2022-06-30 | 503 | 501 | 99.6 | 500 | 489 | 50 | 61 | 628 | 1700 | 2162 | 4379 | 31 | 204 | 60.4 | -0.364 | -0.588 |
| 2023-06-30 | 503 | 503 | 100.0 | 503 | 488 | 49 | 63 | 632 | 1339 | 2187 | 3629 | 25 | 157 | 54.9 | -0.312 | -0.564 |
| 2024-06-30 | 503 | 503 | 100.0 | 502 | 486 | 56 | 70 | 365 | 1414 | 2389 | 4621 | 24 | 158 | 55.7 | -0.316 | -0.603 |
| 2025-06-30 | 503 | 503 | 100.0 | 502 | 487 | 32 | 191 | 531 | 1439 | 1969 | 4746 | 22 | 165 | 54.5 | -0.317 | -0.594 |
| 2026-06-30 | 503 | 503 | 100.0 | 499 | 482 | 32 | 168 | 694 | 1392 | 2280 | 4664 | 27 | 198 | 53.7 | -0.305 | -0.534 |

Counts are (insider, trade date, direction) events. The numbers are from the final build in `data/raw/form4/build_stats.json`. The last three columns were computed from the output parquet.

## Unverified / caveats
- **Amendments:** original filings only. A 4/A that corrects a transaction is ignored, and one that adds a missed transaction is lost.
- **Late Form 4s:** handled correctly by the FILING_DATE <= F rule. Form 5 annual reports (0.8% of rows) show up only once they are filed.
- **Insider identity:** the key is owner CIK x issuer CIK. Someone who moves to a new firm starts with no history there. The joint-filing primary-owner rule is a heuristic.
- **Row cap:** `NONDERIV_TRANS` rows per accession max out at exactly 30 in the quarters checked (2007q1, 2012q1). This may be a cap in the DERA extract. It does not affect event counts much, because splits collapse to one event, but dollar values of very large multi-line filings could be understated. **Not verified.**
- **10b5-1 plans:** the `AFF10B5ONE` flag exists only from 2023q1 and is not used. Many "opportunistic" sales after 2023 may be plan sales.
- **Link timing:** the CIK-permno link uses matches from all years. This is identity information, not price information, but a CIK first seen with a matching ticker after F is still applied to its earlier filings.
- **Holdout:** formation dates of 2016 and later fall in the locked holdout period (config `HOLDOUT_START`). This file is feature data only. Any training or selection code must still call `holdout_guard.assert_pre_holdout`.
