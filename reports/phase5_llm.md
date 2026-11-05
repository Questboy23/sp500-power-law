# Phase 5: text and point-in-time LLM layer

**Status:** A5 and B4 are built for 1996-01 → 2026-09-29, and the permno↔CIK link is done.
ChronoBERT embeddings and the look-ahead diagnostic are in the sections below.
Code is in `src/text/` (`link.py`, `tenk.py`, `features.py`, `anonymize.py`, `embed.py`) and `src/llm_classify.py`.
No 2016+ return is read anywhere in this phase. The diagnostic filters CRSP to dates before 2016-01-01 and calls `assert_pre_holdout`.

## 1. permno ↔ CIK link (`data/processed/permno_cik.parquet`)

**Scope:** 1,204 permnos (1,169 CRSP permnos plus 35 synthetic 2025+ permnos) out of 1,272 ever-members.
**Columns:** permno, cik, method, confidence, name_sim, n10k, first_year, last_year.
A permno can have several CIKs over time, each covering its own years (for example Oracle Systems → Oracle Corp, and Disney 1996/2019).

### Methods

| Method | Rule |
|---|---|
| `cusip8` / `cusip6` | FIAM CUSIP → leoliu0/cik-cusip-mapping, which is built from 13D/13G filings. That map contains errors on the filer side (it links IBM to ETEC and GD to SunTrust), so a CUSIP hit also needs a name that agrees (similarity ≥ 0.75). |
| `ticker_now` | The current SEC ticker. Trusted outright for index members at 2024-12; otherwise it needs a name that agrees, because tickers get reused (ABX → Abacus). |
| `name` | FIAM `comp_name` fuzzy-matched against EDGAR current and former names. FIAM names are truncated to 16 characters and abbreviated, so there is an abbreviation-aware subsequence matcher. Needs similarity ≥ 0.92. |
| `manual` | About 160 hand-written historical names, mostly pre-2000 exits that have no FIAM name, still verified against EDGAR names. |

**Rules that apply to every link:**
- Each link needs at least one 10-K filed while the permno was trading (−30 days / +60 days).
- One forced CIK: U.S. Steel.
- **2025+ extension:** the latest CIK of every permno still trading at the CRSP end (2024-12) is open-ended (`last_year = 2100`).
- Synthetic permnos (≥ 900000) are linked by ticker (`ticker_synthetic`) and mapped only for 10-Ks filed from 2024 on.

### Match rate: share of S&P member-months whose permno has a CIK

| 1996 | 2000 | 2005 | 2010 | 2015 | 2016 | 2020 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|
| 93.1% | 96.1% | 98.6% | 99.0% | 99.2% | 99.0% | 99.3% | 99.3% | 99.6% | 99.7% |

### Still unlinked: 68 permnos (about 2,850 member-months)

| Group | Why it is unlinked | Examples |
|---|---|---|
| Foreign 20-F/40-F filers | No 10-K exists | Unilever, Royal Dutch, Barrick, Placer Dome |
| Filers with no EDGAR 10-Ks | They file with the FDIC or other regulators, not the SEC | First Republic Bank, Freddie Mac pre-2008 |
| Pre-2000 exits | No FIAM name, and my historical name guess failed to verify | AM, USH, NAE, SFS, ... |
| Membership-file retro-ticker permnos | Could not be identified | 26607 "COP" 1996-2001, 48020 "MSI", 26374 "CAR", 31042 "CNP" |

- **Confidence:** `confidence` in [0, 1] combines name similarity, CUSIP/ticker agreement and method count. `manual` links cap at 0.6.
- **Unverified:** I spot-checked about 150 rows by eye; the remaining rows were not checked.
- **Known residual errors:** a few `name`-only links pick a same-name subsidiary or predecessor for 1-3 early years, e.g. PPL Electric Utilities for PPL in 1995 and Consolidated Edison Co of NY for ConEd in 1994-97.

## 2. 10-K corpus (`data/raw/tenk/`, 2.5 GB)

- **Filings:** 23,833 10-K/10-K405 filings (1996-01 → 2026-09-29) for every linked CIK, plus 344 for the synthetic firms. 0 failures after one retry pass.
- **Storage:** one `{cik}/{accession}.json.gz` per filing, holding the cleaned full text of the main 10-K document (no exhibits), the Item 1 text, `n_words` and the source URL.
- **Source document:**
  - Modern filings: the primary document.
  - Blank `doc` or pre-2001 filings: the full-submission `.txt`, from which the `<TYPE>10-K` document is extracted.
- **Rate and resume:** declared User-Agent; 7.5 requests/second overall across 6 processes. Resume-safe.
- **Item 1 extraction:** the longest "Item 1. Business" → "Item 1A / 2 / 3" span, which skips the table of contents. Median 4,313 words.
  - 7.1% of filings give fewer than 200 words. Causes: business description incorporated by reference to the annual report exhibit (common in the 1990s), or odd headings.
  - For those filings, the embeddings fall back to the full text after the cover page.

## 3. Features (`data/processed/text_features.parquet`)

- **Size:** 22,180 rows, one per (permno, 10-K); dual-class permnos share one filing.
- **Columns:** permno, cik, accession, filed, fiscal_year, form, a5_similarity, prev_filed, b4_rival_mentions, b4_rival_mentions_v2, n_words, item1_words.
  - `fiscal_year` is approximate: the year of (filed − 100 days). Period-of-report was not extracted.
- **Rows filed after 2025-06-30:** 653, covering 611 permnos. 99.5% have A5 and 92.5% have B4.

### A5_lazy_filings
- **Definition:** TF-IDF cosine of this 10-K's full text against the same permno's previous 10-K, filed 180-730 days earlier (it crosses CIK changes).
- **Vocabulary:** 47,685 terms. The vocabulary and IDF are fit on 4,000 random pre-2016 filings (the latest is 2015-12-21), with sublinear TF and English stop words.
- **Level:** mean 0.85 in 1996-2004, rising to 0.92-0.94 after 2010, as 10-Ks became more boilerplate.

### B4_named_as_rival
- **Definition:** the number of OTHER firms' 10-Ks filed in the 12 months before this 10-K that mention the firm's distinctive name within 200 characters of "compet".
- **Corpus:** only the linked S&P ever-members' 10-Ks, not all of EDGAR.
- **Distinctive names:** EDGAR current + former names with Inc/Corp/Co/... stripped. Names shorter than 5 characters and single English words are dropped. English words = /usr/share/dict/words ∪ words written in lower case in ≥ 1% of pre-2016 10-Ks.
  - Dropped by that rule: Apple, Oracle, Target, Tiffany, AT&T, 3M.
  - EDGAR's inverted person-names ("PRICE T ROWE") are kept but rarely match prose.
- **Two versions:**
  - `b4_rival_mentions` (**v1**) is the rule set the pre-2016 values were first scored with. It is kept byte-identical.
  - `b4_rival_mentions_v2` fixes a v1 bug: v1 strips "international/industries/trust/bancorp" as trailing suffixes and lets generic single words through, so AIG became "american" (86 mentions a year).
  - The two correlate at 0.60. **v2 is the one I would use for any new work.**
- **Raw annual counts:** `data/processed/b4_annual.parquet` (permno, year, b4_mentions, b4_mentions_v2, corpus_10ks). The name lists are in `b4_names.parquet` and `b4_names_v2.parquet`.
- **Registered signal:** the 3-year change in B4. It is not computed here.
- **Top names in 2015 (v1):** Amazon 25, Google 23, Intel 22, Microsoft 21, J&J 19, Mastercard 14, GE, Pfizer and Merck 12 each.

### Pre-2016 values are frozen
The run-2 rebuild extended the corpus to 2026 and added the synthetic firms. After that rebuild:
- Pre-2016 `b4_rival_mentions` and `n_words` matched run 1 exactly.
- 2 of 14,032 A5 values differed. The previous filing had the same date in both runs, so a tie between same-day filings was probably ordered differently; I did not check this directly.
- I restored those 2 values from run 1. `features.build()` now pins pre-2016 A5/B4 to `text_features_run1_backup.parquet`.

### Coverage: share of June S&P members with a 10-K feature filed in the prior 12 months

| June | Members | CIK linked | Any 10-K | A5 | B4 v1 | B4 v2 |
|---|---|---|---|---|---|---|
| 1996 | 485 | 93.0% | 70.7% | 0.0% | 64.5% | 66.4% |
| 1997 | 487 | 94.3% | 90.3% | 69.4% | 83.2% | 84.8% |
| 1998 | 489 | 94.5% | 91.2% | 90.0% | 83.4% | 85.1% |
| 1999 | 489 | 95.1% | 91.8% | 90.4% | 84.5% | 85.9% |
| 2000 | 490 | 95.9% | 91.6% | 90.4% | 84.3% | 85.5% |
| 2001 | 493 | 97.2% | 93.9% | 91.5% | 87.0% | 88.0% |
| 2002 | 493 | 97.2% | 94.1% | 92.7% | 87.4% | 88.4% |
| 2003 | 492 | 98.0% | 93.7% | 92.9% | 87.2% | 88.0% |
| 2004 | 493 | 98.6% | 94.9% | 94.5% | 88.4% | 89.2% |
| 2005 | 494 | 98.6% | 93.5% | 93.1% | 87.2% | 88.1% |
| 2006 | 496 | 98.6% | 93.5% | 92.7% | 87.3% | 88.5% |
| 2007 | 496 | 98.6% | 94.0% | 92.5% | 87.3% | 88.7% |
| 2008 | 496 | 98.4% | 93.8% | 91.9% | 87.5% | 88.7% |
| 2009 | 499 | 98.6% | 94.8% | 94.2% | 88.4% | 89.8% |
| 2010 | 499 | 99.0% | 95.0% | 94.8% | 88.2% | 89.6% |
| 2011 | 497 | 99.2% | 94.2% | 93.8% | 87.5% | 88.7% |
| 2012 | 497 | 99.2% | 94.0% | 93.2% | 87.5% | 88.5% |
| 2013 | 497 | 99.2% | 94.0% | 92.6% | 87.3% | 88.3% |
| 2014 | 498 | 99.2% | 94.0% | 93.4% | 87.3% | 88.4% |
| 2015 | 499 | 99.4% | 93.8% | 93.6% | 87.2% | 88.2% |
| 2016 | 503 | 99.0% | 93.2% | 92.2% | 86.9% | 87.9% |
| 2017 | 505 | 99.2% | 93.9% | 92.9% | 87.5% | 88.3% |
| 2018 | 505 | 99.2% | 93.7% | 93.3% | 87.1% | 87.9% |
| 2019 | 505 | 99.4% | 93.3% | 92.7% | 86.7% | 87.5% |
| 2020 | 505 | 99.2% | 93.3% | 92.7% | 86.1% | 86.9% |
| 2021 | 505 | 99.2% | 94.1% | 93.3% | 87.1% | 87.9% |
| 2022 | 503 | 99.0% | 93.4% | 93.2% | 86.5% | 87.3% |
| 2023 | 503 | 99.2% | 93.4% | 93.0% | 86.7% | 87.5% |
| 2024 | 503 | 99.4% | 93.4% | 92.8% | 87.3% | 88.1% |
| 2025 | 503 | 99.6% | 94.0% | 93.4% | 88.3% | 89.1% |
| 2026 | 503 | 99.6% | 92.8% | 92.6% | 87.1% | 87.9% |

- **June 1996:** A5 is ~0 because the corpus starts in 1996, so there is no previous 10-K yet.
- **Gap between "CIK linked" and "Any 10-K" (~5 points):** mostly 10-Ks filed more than 12 months before June (filing lags, fiscal-year changes), plus firms whose 10-K was not mapped into the link's year window.

## 4. Point-in-time LLM: ChronoBERT

### Availability and licence (checked 2026-09-30 on the Hugging Face API)
- **Models:** `manelalab/chrono-bert-v1-{YYYY}1231`, one vintage per year for 1999-2024 (26 models, ModernBERT-base, 768-d, ~600 MB each).
- **Also available:** `chrono-gpt-v1-*` and `chrono-gpt-instruct-v1-*` for the same years.
- **Licence:** MIT on every model card.
- **Paper:** He, Lv, Manela, Wu (2025), "Chronologically Consistent Large Language Models".
- **Unverified:** that each vintage's training data really stops at its cutoff. That is the authors' claim, and §5 is the only check I ran.

### What was built
- **Files:**
  - `data/processed/text_embeddings.parquet`: 19,042 rows of permno, cik, accession, filed, vintage, text_src, emb_0..emb_63.
  - `text_embeddings_768.parquet`: the raw 768-d vectors.
  - Per-vintage parts: `text_emb_parts/`.
- **Vintage rule:** vintage = min(filing year − 1, 2024), so the cutoff is always before the filing date (checked on every row).
- **PCA:** 64 dimensions, fit on the 12,069 pre-2016 filings only; 86.8% of variance explained.
- **Text:** anonymised Item 1, first ~2,048 tokens as 4 chunks of 512. Mean-pooled over tokens, then averaged across chunks weighted by token count.
  - 5-10% of filings per year fall back to the full text after the cover page because Item 1 was shorter than 200 words (`text_src = full`).
- **Compute:** fp16 on Apple MPS at about 3 docs/s.
- **Model cache:** deleted after use; the weights re-download from HF.

### Anonymisation (`src/text/anonymize.py`)
What it removes:
- The firm's EDGAR current + former names (with suffix variants), FIAM names, the distinctive core and first distinctive word → `[COMPANY]`.
- Every ticker the firm ever had (CRSP, membership, FIAM, EDGAR) → `[TICKER]`.
- Honorific + name → `[PERSON]`.
- Any capitalised token whose lower-case form is not a corpus-common word (written lower-case in ≥ 1% of pre-2016 10-Ks) and not on a places/regulators allowlist → `[NAME]`. This catches most product, brand, subsidiary, competitor, executive and city names.

### Embedding coverage by filing year

| Years | 10-Ks | Embedded |
|---|---|---|
| 1996-1999 | 3,138 | 0 (no vintage precedes them) |
| 2000-2026 | 19,042 | 19,042 (100%) |

## 5. Look-ahead diagnostic (spec 9.6) and name leak

### Setup (`src/llm_classify.py`, outputs in `data/processed/llm_diag/`)
- **Sample:** 120 filings per year for 2000-2015 (1,920 filings), all filed before 2016-01-01.
- **Target:** is the firm's past-12-month return (the 12 month-ends before filing) above that year's median? CRSP is filtered to dates before 2016 and passes through `assert_pre_holdout`.
- **Model:** anonymised-Item-1 embedding → standardise → PCA-32 → L2 logistic regression, fit inside each fold.
- **Evaluation:** leave-one-filing-year-out, out-of-fold AUC. Chance = 0.5.
- **Embedding sets compared:**
  - The production point-in-time embeddings (v = Y−1).
  - The same filings embedded with two **fixed** vintages, 2004 and 2009.

### Out-of-fold AUC by filing year

| Filing year | PIT (v = Y−1) | Fixed v2004 | Fixed v2009 |
|---|---|---|---|
| 2000 | 0.458 | 0.459 | 0.472 |
| 2001 | 0.587 | 0.603 | 0.544 |
| 2002 | 0.441 | 0.448 | 0.461 |
| 2003 | 0.488 | 0.449 | 0.456 |
| 2004 | 0.503 | 0.485 | 0.470 |
| 2005 | 0.537 | 0.578 | 0.592 |
| 2006 | 0.507 | 0.488 | 0.502 |
| 2007 | 0.474 | 0.487 | 0.490 |
| 2008 | 0.475 | 0.525 | 0.531 |
| 2009 | 0.544 | 0.512 | 0.544 |
| 2010 | 0.483 | 0.534 | 0.517 |
| 2011 | 0.542 | 0.535 | 0.542 |
| 2012 | 0.511 | 0.473 | 0.497 |
| 2013 | 0.545 | 0.526 | 0.514 |
| 2014 | 0.468 | 0.399 | 0.417 |
| 2015 | 0.487 | 0.509 | 0.506 |

### Diff-in-diff by block (year-bootstrap CI)

| Block | AUC PIT | AUC v2004 | AUC v2009 | v2009 − v2004 [95% CI] |
|---|---|---|---|---|
| 2000-04 (both inside) | 0.496 | 0.489 | 0.481 | -0.008 [-0.033, +0.012] |
| 2005-09 (v2009 inside, v2004 outside) | 0.507 | 0.518 | 0.532 | +0.014 [+0.005, +0.024] |
| 2010-15 (both outside) | 0.506 | 0.496 | 0.499 | +0.003 [-0.008, +0.015] |

### Results

**1. The text barely encodes past returns at all.**
- Mean AUC is 0.50 for every embedding set. Year-level AUCs range from 0.40 to 0.60, which is the noise band for about 120 filings a year.
- So this probe has little power, and an absence of look-ahead here is weak evidence.

**2. Look-ahead diff-in-diff: a small effect in the look-ahead direction.**
- In 2005-09, where the 2009 model has seen the period and the 2004 model has not, the 2009 model's AUC is higher by **+0.014 [+0.005, +0.024]**.
- In the control blocks the gap is −0.008 (2000-04, both models inside) and +0.003 (2010-15, both outside).
- The CI is a bootstrap over only 5 years and there were two comparisons, so treat this as **suggestive, not established**.
- If it is real, it is about 1.5 AUC points: economically trivial next to how little the text predicts at all.

**3. The "inside vs outside" comparison for a single vintage is confounded by year effects.**
- It gives −0.017 for v2004 and +0.007 for v2009, which is inconsistent.

**4. Masked-LM re-identification: 0 of 113.**
- For 2011 filings, the first `[COMPANY]` was replaced with the mask token and the vintage-matched 2010 model was asked to fill it.
- The firm's name appeared in the top-10 fill-ins **0 times out of 113**.
- This test only checks single-token recall of the name. It does not rule out implicit identification through the embedding.

**5. Residual names (400 anonymised filings).**
- The first word of the firm's FIAM short name survives in 6.75% of filings, and a ticker survives in 0.5%.
- About half of the surviving words are harmless ordinary words ("home improvement", "analog", "toys", "state banking laws").
- The other half are real leaks of short brand forms that are English words or were not in EDGAR's legal name: "Sovereign", "Golden West", "Level 3", "Capital One Bank", "Discover Bank", "Frontier", "Alaska Airlines", "Regions", "Prudential Insurance", "Family Dollar", "www.micron.com".
- So an estimated **3-4% of filings still carry a recognisable short name.** A fix would be to also mask quoted defined terms (`("Sovereign")`) and domain names. Not done.

### Five anonymised examples (first ~60 words)

- **permno 50905, filed 2000-03-30** (`item1`)
  - *Raw:* ITEM 1. BUSINESS General John H. Harland Company (the "Company" or the "Registrant") was founded in 1923 as a general printer and lithographer. The Registrant is incorporated under the laws of Georgia and is headquartered in Atlanta. The Company works with banks, credit unions, brokerage houses and financial software companies, providing these institutions with products
  - *Anonymised:* ITEM 1. BUSINESS General [NAME] H. [COMPANY] (the "Company" or the "Registrant") was founded in 1923 as a general printer and lithographer. The Registrant is incorporated under the laws of Georgia and is headquartered in [NAME]. The Company works with banks, credit unions, brokerage houses and financial software companies, providing these institutions with products and

- **permno 70536, filed 2009-02-27** (`item1`)
  - *Raw:* Item 1. Business (a) General Development of Business Sigma-Aldrich Corporation (“the Company”) was incorporated under the laws of the State of Delaware in May 1975. Effective July 31, 1975 (“Reorganization”), the Company succeeded, as a reporting company, Sigma International, Ltd., the predecessor of Sigma Chemical Company (“Sigma”), and Aldrich Chemical Company, Inc. (“Aldrich”), both of
  - *Anonymised:* Item 1. Business (a) General Development of Business [COMPANY] (“the Company”) was incorporated under the laws of the State of Delaware in May 1975. Effective July 31, 1975 (“Reorganization”), the Company succeeded, as a reporting company, [COMPANY] International, [NAME]., the predecessor of [COMPANY] Chemical Company (“[COMPANY]”), and [NAME] Chemical Company, Inc. (“[NAME]”), both of which

- **permno 46922, filed 2010-04-28** (`item1`)
  - *Raw:* Item 1. Business Overview We are the third largest retail drugstore chain in the United States based on revenues and number of stores. We operate our drugstores in 31 states across the country and in the District of Columbia. As of February 27, 2010, we operated 4,780 stores. In our stores, we sell prescription drugs
  - *Anonymised:* Item 1. Business Overview We are the third largest retail drugstore chain in the United States based on revenues and number of stores. We operate our drugstores in 31 states across the country and in the District of [NAME]. As of February 27, 2010, we operated 4,780 stores. In our stores, we sell prescription drugs

- **permno 46703, filed 2001-04-02** (`full`)
  - *Raw:* (Item 1 extraction empty → full-text fallback)
  - *Anonymised:* ill affect, among other things, demand for products and services at our hotels and other properties, the level of room rates and occupancy that can be achieved by such properties and the availability and terms of financing; . our ability to maintain the properties in a first-class manner, including meeting capital expenditure requirements; . our

- **permno 11674, filed 2003-03-24** (`full`)
  - *Raw:* (Item 1 extraction empty → full-text fallback)
  - *Anonymised:* ays after the end of the Registrant’s fiscal year covered by this report on Form 10-K, is incorporated herein by reference to Part III (Items 10, 11, 12 and 13) of this Form 10-K. TABLE OF CONTENTS Definitions Forward-Looking Statements Part I Items 1 & 2. Business & Properties Item 3. Legal Proceedings Item 4.

Examples 4 and 5 are full-text fallbacks: they start mid-sentence 3,000 characters into the document, and example 5 is mostly a table of contents. That is weaker input than Item 1.

### Bottom line
- ChronoBERT is available, MIT-licensed, and was run point-in-time for every 2000-2026 filing.
- Anonymised Item-1 embeddings carry essentially no information about past returns.
- I find no material look-ahead: at most a +1.4-point AUC gap, borderline significant.
- Name removal is good but not complete (about 3-4% of filings leak a short name).
- If this goes into P6_tail_classifier, it should enter as a low-prior feature and be judged strictly out of sample.


## 6. Unverified or not done

- **Link:** about 1,050 rows were not individually checked. The pre-2000 `manual` names are my own recollection, and only verification against EDGAR name + filing-date overlap guards them.
- **Item 1 extraction:** no per-filing check. 7% are shorter than 200 words, and some Item 1 spans may run into Item 1A/2 when the headings are unusual.
- **Fiscal year:** approximated from the filing date.
- **B4 corpus:** S&P ever-members only, so mentions by non-S&P rivals are missed.
- **B4 names:** name matching is lexical. Mentions by ticker or brand ("Google" for Alphabet does match through the former name; "Chevy" for GM does not) are partly missed. Firms whose names are English words have no B4 (about 6% of permnos).
- **A5:** uses the full 10-K main document, including financial statements. Cohen-Malloy-Nguyen also report section-level variants, which were not built.
- **Anonymisation** is heuristic (see §5); product and brand names that are ordinary English words survive.
- **ChronoBERT:** there is no vintage before 1999-12-31, so 1996-1999 filings have no embedding. The HF model cards say MIT licence; the upstream training-data licences were not checked.
