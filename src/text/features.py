"""Text features from the downloaded 10-Ks.

A5_lazy_filings  TF-IDF cosine similarity of a 10-K's full text vs the same firm's previous 10-K
                 (Cohen, Malloy, Nguyen 2020). Vocabulary + IDF fit on pre-2016 filings only.
B4_named_as_rival  number of OTHER firms' 10-Ks filed in the 12 months before this firm's 10-K
                 that mention this firm's distinctive name within 200 characters of "compet".
                 Corpus = the 10-Ks of linked S&P 500 (ever-)members only (not all EDGAR filers).
                 Raw annual counts saved to data/processed/b4_annual.parquet; the registered
                 signal is the 3-year change, computed downstream.

Output: data/processed/text_features.parquet
  permno, cik, accession, filed, fiscal_year (approx: year of filed-100d), form,
  a5_similarity, prev_filed, b4_rival_mentions, n_words, item1_words
"""
from __future__ import annotations
import gzip, json, os, re
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from config import RAW, PROC, HOLDOUT_START

TENK = RAW / "tenk"


def load_doc(cik: int, acc: str) -> dict:
    with gzip.open(TENK / str(cik) / f"{acc}.json.gz", "rt") as f:
        return json.load(f)


def filing_index() -> pd.DataFrame:
    """Every downloaded filing, mapped to permno(s) through permno_cik (year window of each link)."""
    rows = []
    for d in TENK.iterdir():
        if d.is_dir():
            for f in d.iterdir():
                if f.name.endswith(".json.gz"):
                    rows.append((int(d.name), f.name[:-8]))
    idx = pd.DataFrame(rows, columns=["cik", "accession"])
    fl = pd.read_parquet(PROC / "edgar_filings.parquet", columns=["cik", "form", "filed", "accession"])
    idx = idx.merge(fl.drop_duplicates("accession"), on=["cik", "accession"], how="left")
    idx["filed"] = pd.to_datetime(idx.filed)
    idx["fiscal_year"] = (idx.filed - pd.Timedelta(days=100)).dt.year
    link = pd.read_parquet(PROC / "permno_cik.parquet")
    # CRSP ends 2024-12: a link still active then stays open-ended so 2025-26 10-Ks (live portfolio) map
    life = pd.read_parquet(PROC / "permno_life.parquet")["max"]
    alive = link.permno.map(life) >= pd.Timestamp("2024-12-01")
    last = link.groupby("permno").last_year.transform("max")
    link.loc[alive & (link.last_year == last) & (link.last_year >= 2024), "last_year"] = 2100
    m = idx.merge(link[["permno", "cik", "first_year", "last_year"]], on="cik", how="left")
    y = m.filed.dt.year
    m = m[m.permno.notna() & (y >= m.first_year) & (y <= m.last_year)].copy()
    m["permno"] = m.permno.astype(int)
    return m.drop(columns=["first_year", "last_year"]).sort_values(["permno", "filed"]).reset_index(drop=True)


# ------------------------------------------------------------------ A5
def _meta_words(cik, acc):
    d = load_doc(cik, acc)
    return d["n_words"], len(d["item1"].split())


def a5(idx: pd.DataFrame, n_fit: int = 4000, seed: int = 0) -> pd.DataFrame:
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = idx.drop_duplicates("accession")[["cik", "accession", "filed"]].reset_index(drop=True)
    pre = docs[docs.filed < HOLDOUT_START]
    fit = pre.sample(min(n_fit, len(pre)), random_state=seed)
    vec = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]{2,}\b", stop_words="english",
                          min_df=5, max_df=0.95, sublinear_tf=True, max_features=100_000, dtype=np.float32)
    vec.fit(load_doc(c, a)["full"] for c, a in zip(fit.cik, fit.accession))
    print("A5 vocab", len(vec.vocabulary_), "fit on", len(fit), "pre-2016 docs (max filed", fit.filed.max().date(), ")")

    def tr(chunk):
        return vec.transform([load_doc(c, a)["full"] for c, a in chunk])
    pairs = list(zip(docs.cik, docs.accession))
    chunks = [pairs[i:i + 200] for i in range(0, len(pairs), 200)]
    import scipy.sparse as sp
    X = sp.vstack(Parallel(n_jobs=6)(delayed(tr)(c) for c in chunks)).tocsr()   # rows L2-normalised
    row = {a: i for i, a in enumerate(docs.accession)}
    out = []
    # previous 10-K of the same firm = previous filing of the same permno (crosses CIK changes),
    # at least 180 days earlier and at most 2 years earlier
    for p, g in idx.sort_values("filed").groupby("permno"):
        prev_acc, prev_f = None, None
        for r in g.itertuples():
            sim = np.nan; pf = pd.NaT
            if prev_acc is not None and 180 <= (r.filed - prev_f).days <= 730:
                sim = float(X[row[r.accession]].multiply(X[row[prev_acc]]).sum()); pf = prev_f
            out.append((r.permno, r.accession, sim, pf))
            if prev_f is None or (r.filed - prev_f).days >= 180:
                prev_acc, prev_f = r.accession, r.filed
    return pd.DataFrame(out, columns=["permno", "accession", "a5_similarity", "prev_filed"])


# ------------------------------------------------------------------ B4
# v1 = the rule set the pre-2016 B4 values were first scored with (kept byte-identical).
# v2 = bug fix: v1 strips "international"/"industries"/... as trailing suffixes and lets generic
#      single words through, so AIG became "american" (86 hits/yr). v2 keeps those words in the name
#      and drops generic single-word names. Reported as b4_rival_mentions_v2 alongside v1.
SUFFIXES_V1 = {"inc", "incorporated", "corp", "corporation", "co", "company", "companies", "ltd", "limited", "plc",
               "llc", "lp", "holdings", "holding", "group", "the", "new", "de", "sa", "nv", "ag", "trust", "bancorp",
               "international", "intl", "enterprises", "industries", "and"}
SUFFIXES = {"inc", "incorporated", "corp", "corporation", "co", "company", "companies", "ltd", "limited", "plc",
            "llc", "lp", "holdings", "holding", "group", "the", "new", "de", "sa", "nv", "ag", "and"}
GENERIC = {"american", "national", "united", "first", "general", "international", "standard", "southern",
           "northern", "western", "eastern", "pacific", "atlantic", "central", "federal", "continental",
           "global", "universal", "royal", "consolidated", "allied", "associated", "public", "republic",
           "security", "capital", "financial", "energy", "technology", "technologies", "systems", "health"}


def _dict_words() -> set[str]:
    from src.text.anonymize import _words   # corpus-frequent lower-case words (pre-2016 10-Ks)
    try:
        dw = {w.strip() for w in open("/usr/share/dict/words") if w.strip().islower()}
    except OSError:
        dw = set()
    return dw | set(_words())


def distinctive(name: str, words: set[str], v2: bool = True) -> str | None:
    """'MICROSOFT CORP' -> 'microsoft'; None if the result is < 5 chars or a single common English word."""
    s = re.sub(r"/[A-Z]{2,3}/?", " ", name or "")
    s = re.sub(r"[^A-Za-z0-9&\-' ]", " ", s).lower()
    s = s.replace("&", " & ")
    toks = s.split()
    while toks and toks[0] == "the":
        toks.pop(0)
    while toks and toks[-1] in (SUFFIXES if v2 else SUFFIXES_V1) | {"com", "&"}:
        toks.pop()
    if not toks:
        return None
    d = " ".join(toks)
    if len(d.replace(" ", "")) < 5:
        return None
    if len(toks) == 1 and toks[0] in words:
        return None
    from src.text.anonymize import ALLOW
    if v2 and len(toks) == 1 and (toks[0] in GENERIC or toks[0] in ALLOW):
        return None
    if len(toks) == 1 and len(toks[0]) == 1:
        return None
    return d


def firm_names(link: pd.DataFrame, v2: bool = True) -> dict[int, set[str]]:
    """permno -> distinctive names from EDGAR current + former names of every linked CIK."""
    meta = pd.read_parquet(PROC / "edgar_meta.parquet").set_index("cik")
    words = _dict_words()
    out: dict[int, set[str]] = {}
    for p, ciks in link.groupby("permno").cik:
        names = set()
        for k in ciks:
            if k not in meta.index:
                continue
            raw = [meta.at[k, "name"]] + [x["name"] for x in json.loads(meta.at[k, "former_names"] or "[]")]
            for n in raw:
                d = distinctive(n, words, v2)
                # EDGAR inverts person-named firms ("PRICE T ROWE GROUP"): those rarely match prose; kept anyway
                if d:
                    names.add(d)
                    if "&" in d:
                        names.add(d.replace("&", "and"))
        out[int(p)] = names
    return out


def _compet_windows(text: str, w: int = 200) -> str:
    low = text.lower()
    spans = [(max(0, m.start() - w), m.end() + w) for m in re.finditer("compet", low)]
    if not spans:
        return ""
    merged = [list(spans[0])]
    for a, b in spans[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return " | ".join(low[a:b] for a, b in merged)


_AUTO = {}


def _automaton(ver: str):
    """Aho-Corasick automaton over all distinctive names, built once per worker process from
    data/processed/b4_names.parquet (written by b4() before the workers start)."""
    if ver not in _AUTO:
        import ahocorasick
        A = ahocorasick.Automaton()
        for p, n in pd.read_parquet(PROC / f"b4_names{ver}.parquet").itertuples(index=False):
            if n in A:
                A.get(n)[1].append(int(p))
            else:
                A.add_word(n, (n, [int(p)]))
        A.make_automaton()
        _AUTO[ver] = A
    return _AUTO[ver]


def _b4_doc(cik, acc, vers=("", "_v2")):
    win = re.sub(r"\s+", " ", _compet_windows(load_doc(cik, acc)["full"]))
    out = []
    for ver in vers:
        hits = set()
        for end, (name, pids) in _automaton(ver).iter(win):
            s = end - len(name) + 1
            if (s == 0 or not win[s - 1].isalnum()) and (end + 1 >= len(win) or not win[end + 1].isalnum()):
                hits.update(pids)
        out.append(hits)
    return acc, out


def b4(idx: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    link = pd.read_parquet(PROC / "permno_cik.parquet")
    vers = {"": firm_names(link, v2=False), "_v2": firm_names(link, v2=True)}
    for ver, names in vers.items():
        pd.DataFrame([(p, n) for p, ns in names.items() for n in sorted(ns)],
                     columns=["permno", "name"]).to_parquet(PROC / f"b4_names{ver}.parquet")
    docs = idx.drop_duplicates("accession")[["cik", "accession"]]
    res = Parallel(n_jobs=6, batch_size=32)(delayed(_b4_doc)(c, a) for c, a in zip(docs.cik, docs.accession))
    acc2permnos = idx.groupby("accession").permno.apply(set).to_dict()
    acc2filed = idx.drop_duplicates("accession").set_index("accession").filed
    n_docs = idx.drop_duplicates("accession").groupby(idx.drop_duplicates("accession").filed.dt.year).size().rename("corpus_10ks")
    pers, anns = [], []
    for vi, (ver, names) in enumerate(vers.items()):
        col = "b4_rival_mentions" + ver
        ment = []   # (target permno, mentioning accession, filed)
        for acc, hits in res:
            own = acc2permnos.get(acc, set())
            for p in hits[vi] - own:
                ment.append((p, acc, acc2filed[acc]))
        ment = pd.DataFrame(ment, columns=["permno", "accession", "filed"])
        # annual raw counts: distinct other-firm 10-Ks per calendar year that name the firm near "compet"
        ann = ment.groupby(["permno", ment.filed.dt.year.rename("year")]).accession.nunique().rename("b4_mentions" + ver)
        anns.append(ann)
        # per own filing: other firms' 10-Ks filed in the 12 months before this filing
        ment = ment.sort_values("filed")
        by_p = {p: g.filed.values for p, g in ment.groupby("permno")}
        rows = []
        for r in idx.itertuples():
            f = by_p.get(r.permno)
            n = 0
            if f is not None:
                lo, hi = np.datetime64(r.filed - pd.DateOffset(months=12)), np.datetime64(r.filed)
                n = int(np.searchsorted(f, hi, "left") - np.searchsorted(f, lo, "left"))
            rows.append((r.permno, r.accession, n if names.get(r.permno) else np.nan))
        pers.append(pd.DataFrame(rows, columns=["permno", "accession", col]).set_index(["permno", "accession"]))
    per = pd.concat(pers, axis=1).reset_index()
    ann = pd.concat(anns, axis=1).fillna(0).astype(int).reset_index()
    ann = ann.merge(n_docs, left_on="year", right_index=True, how="left")
    return per, ann


def build():
    idx = filing_index()
    print("filings mapped to permnos:", len(idx), "unique docs:", idx.accession.nunique())
    wc = Parallel(n_jobs=6, batch_size=64)(delayed(_meta_words)(c, a) for c, a in zip(idx.cik, idx.accession))
    idx["n_words"] = [w[0] for w in wc]; idx["item1_words"] = [w[1] for w in wc]
    a = a5(idx)
    per, ann = b4(idx)
    out = idx.merge(a, on=["permno", "accession"], how="left").merge(per, on=["permno", "accession"], how="left")
    cols = ["permno", "cik", "accession", "filed", "fiscal_year", "form", "a5_similarity", "prev_filed",
            "b4_rival_mentions", "b4_rival_mentions_v2", "n_words", "item1_words"]
    out = out[cols]
    # Pre-2016 values were first scored from run 1 (text_features_run1_backup.parquet); keep them fixed.
    # (Only 2 A5 rows ever differed, from ties between same-day filings in the previous-10-K choice.)
    bk = PROC / "text_features_run1_backup.parquet"
    if bk.exists():
        o = pd.read_parquet(bk); o = o[o.filed < HOLDOUT_START].set_index(["permno", "accession"])
        out = out.set_index(["permno", "accession"])
        keep = o.index.intersection(out.index)
        for c in ["a5_similarity", "prev_filed", "b4_rival_mentions"]:
            out.loc[keep, c] = o.loc[keep, c]
        out = out.reset_index()[cols]
    out.to_parquet(PROC / "text_features.parquet")
    ann.to_parquet(PROC / "b4_annual.parquet")
    print(out.groupby(out.filed.dt.year).agg(n=("accession", "size"), a5=("a5_similarity", "count"),
                                             a5_mean=("a5_similarity", "mean"), b4_mean=("b4_rival_mentions", "mean")).to_string())
    return out


if __name__ == "__main__":
    build()
