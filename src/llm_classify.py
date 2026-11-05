"""Look-ahead diagnostic for the point-in-time LLM layer (spec 9.6) + name-leak checks.

Everything here that touches returns uses filings filed BEFORE 2016-01-01 only, and the target is the
firm's PAST 12-month return (the 12 month-ends strictly before the filing date), so no 2016+ return
is ever read. holdout_guard.assert_pre_holdout is called on every frame a model learns from.

1. Past-return probe. Target = 1 if the firm's past-12m return is above the median of filings in the
   same calendar year. Features = mean-pooled ChronoBERT embedding of anonymised Item 1 (PCA-32 fit
   inside each training fold) -> L2 logistic regression, leave-one-filing-year-out. Out-of-fold AUC
   per year; chance = 0.5.
   Embeddings: (a) the production point-in-time vintage (Y-1) for every filing;
   (b) two FIXED vintages (2004, 2009) on the same sample. For a fixed vintage v, filings filed
   <= v are "inside" the model's training window, filings after v are "outside".
   Diff-in-diff: for years 2005-2009 the 2009 model is inside and the 2004 model outside; for
   2000-2004 both are inside; for 2010-2015 both outside. Look-ahead predicts
   AUC(v2009) - AUC(v2004) is larger in 2005-09 than in the other two blocks.
2. Name leak: (a) 5 anonymised excerpts; (b) residual-name scan (FIAM short names and tickers that
   survive anonymisation); (c) masked-LM re-identification: replace the first [COMPANY] with the
   mask token and check whether the vintage-matched ChronoBERT's top-10 fill-ins hit the firm's name.
"""
from __future__ import annotations
import json, re, sys
import numpy as np, pandas as pd
from config import PROC, REPORTS, HOLDOUT_START
from src.holdout_guard import assert_pre_holdout

DIAG_VINTAGES = (2004, 2009)
OUT = PROC / "llm_diag"


def past_returns(idx: pd.DataFrame) -> pd.Series:
    """Compounded return over the 12 month-ends strictly before each filing date."""
    crsp = pd.read_parquet(PROC / "crsp_monthly_universe.parquet", columns=["permno", "date", "ret"])
    crsp = crsp[crsp.date < HOLDOUT_START]                      # never read a 2016+ return
    assert_pre_holdout(crsp, "crsp for past returns")
    crsp["lr"] = np.log1p(crsp.ret.clip(lower=-0.99))
    crsp = crsp.sort_values(["permno", "date"])
    crsp["lr12"] = crsp.groupby("permno").lr.transform(lambda s: s.rolling(12, min_periods=11).sum())
    crsp["me"] = crsp.date + pd.offsets.MonthEnd(0)
    key = idx[["permno", "filed"]].copy()
    key["me"] = key.filed.dt.to_period("M").dt.to_timestamp() - pd.Timedelta(days=1)   # last month-end before filing
    m = key.merge(crsp[["permno", "me", "lr12"]], on=["permno", "me"], how="left")
    return pd.Series(np.expm1(m.lr12.values), index=idx.index)


def sample(n_per_year: int = 120, seed: int = 0) -> pd.DataFrame:
    from src.text.features import filing_index
    idx = filing_index()
    idx = idx[(idx.filed >= "2000-01-01") & (idx.filed < HOLDOUT_START)].drop_duplicates("accession")
    idx["past12"] = past_returns(idx)
    idx = idx[idx.past12.notna()]
    s = idx.groupby(idx.filed.dt.year, group_keys=False).apply(lambda g: g.sample(min(n_per_year, len(g)), random_state=seed))
    s = s.reset_index(drop=True)
    s["year"] = s.filed.dt.year
    s["y"] = (s.past12 > s.groupby("year").past12.transform("median")).astype(int)
    s = s.rename(columns={"filed": "date"}); assert_pre_holdout(s, "diag sample"); s = s.rename(columns={"date": "filed"})
    return s


def embed_fixed(s: pd.DataFrame):
    from src.text.embed import embed_frame, Embedder
    OUT.mkdir(parents=True, exist_ok=True)
    for v in DIAG_VINTAGES:
        p = OUT / f"fixed_v{v}.parquet"
        if p.exists():
            continue
        e = embed_frame(s, v, Embedder(v))
        e.to_parquet(p)


def _oof_auc(X: np.ndarray, y: np.ndarray, years: np.ndarray) -> pd.Series:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    out = {}
    for yr in np.unique(years):
        tr, te = years != yr, years == yr
        sc = StandardScaler().fit(X[tr]); pca = PCA(32, random_state=0).fit(sc.transform(X[tr]))
        f = lambda Z: pca.transform(sc.transform(Z))
        clf = LogisticRegression(C=0.1, max_iter=2000).fit(f(X[tr]), y[tr])
        out[yr] = roc_auc_score(y[te], clf.predict_proba(f(X[te]))[:, 1])
    return pd.Series(out)


def probe(s: pd.DataFrame) -> pd.DataFrame:
    ecols = [f"e{i}" for i in range(768)]
    res = {}
    main = pd.read_parquet(PROC / "text_embeddings_768.parquet")
    m = s.merge(main[["permno", "accession"] + ecols], on=["permno", "accession"], how="inner")
    res["pit_vY-1"] = (_oof_auc(m[ecols].values, m.y.values, m.year.values), len(m))
    for v in DIAG_VINTAGES:
        e = pd.read_parquet(OUT / f"fixed_v{v}.parquet")
        m = s.merge(e[["permno", "accession"] + ecols], on=["permno", "accession"], how="inner")
        res[f"fixed_v{v}"] = (_oof_auc(m[ecols].values, m.y.values, m.year.values), len(m))
    tab = pd.DataFrame({k: v[0] for k, v in res.items()})
    tab.index.name = "filing_year"
    return tab, {k: v[1] for k, v in res.items()}


def block_stats(tab: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    blocks = {"2000-04 (both inside)": range(2000, 2005), "2005-09 (v2009 inside, v2004 outside)": range(2005, 2010),
              "2010-15 (both outside)": range(2010, 2016)}
    rows = []
    for name, yrs in blocks.items():
        t = tab.loc[tab.index.isin(list(yrs))]
        d = (t["fixed_v2009"] - t["fixed_v2004"]).values
        bs = [rng.choice(d, len(d)).mean() for _ in range(n_boot)]
        rows.append(dict(block=name, auc_pit=t["pit_vY-1"].mean(), auc_v2004=t["fixed_v2004"].mean(),
                         auc_v2009=t["fixed_v2009"].mean(), diff_v2009_minus_v2004=d.mean(),
                         diff_ci_lo=np.percentile(bs, 2.5), diff_ci_hi=np.percentile(bs, 97.5)))
    for v in DIAG_VINTAGES:
        inside = tab.index <= v
        rows.append(dict(block=f"v{v}: inside (<= {v}) vs outside", auc_v2004=np.nan,
                         diff_v2009_minus_v2004=tab.loc[inside, f"fixed_v{v}"].mean() - tab.loc[~inside, f"fixed_v{v}"].mean()))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- name leak
def leak_examples(s: pd.DataFrame, k: int = 5, seed: int = 1) -> list[dict]:
    from src.text.embed import doc_text
    from src.text.features import load_doc
    ex = []
    for r in s.sample(k, random_state=seed).itertuples():
        raw = load_doc(r.cik, r.accession)["item1"]
        anon, src = doc_text(r.cik, r.accession, r.permno)
        ex.append(dict(permno=r.permno, cik=r.cik, filed=str(r.filed.date()), src=src,
                       raw=" ".join(raw.split()[:70]), anon=" ".join(anon.split()[:70])))
    return ex


def residual_scan(s: pd.DataFrame, n: int = 400, seed: int = 2) -> dict:
    """How often a FIAM short name token or a ticker survives anonymisation."""
    from src.text.embed import doc_text
    from config import RAW
    fiam = pd.read_parquet(RAW / "delisted" / "fiam_monthly_2000_2023.parquet", columns=["permno", "comp_name", "stock_ticker"])
    fn = fiam.groupby("permno").agg(nm=("comp_name", "last"), tk=("stock_ticker", "last"))
    hits_name = hits_tick = tot = 0
    sub = s[s.permno.isin(fn.index)].sample(min(n, len(s)), random_state=seed)
    for r in sub.itertuples():
        anon, _ = doc_text(r.cik, r.accession, r.permno)
        first = (fn.at[r.permno, "nm"] or "").split()[0] if fn.at[r.permno, "nm"] else ""
        tk = fn.at[r.permno, "tk"] or ""
        tot += 1
        if len(first) >= 4 and re.search(rf"(?i)\b{re.escape(first)}\b", anon):
            hits_name += 1
        if len(tk) >= 3 and re.search(rf"\b{re.escape(tk)}\b", anon):
            hits_tick += 1
    return dict(n=tot, first_name_word_survives=hits_name / max(tot, 1), ticker_survives=hits_tick / max(tot, 1))


def mlm_reid(s: pd.DataFrame, vintage: int = 2010, n: int = 150, seed: int = 3) -> dict:
    """Mask the first [COMPANY] slot; does the (vintage-matched) model's top-10 contain the firm's name?"""
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    from src.text.embed import doc_text, model_id
    from src.text.features import distinctive, _dict_words
    meta = pd.read_parquet(PROC / "edgar_meta.parquet").set_index("cik")
    words = _dict_words()
    tok = AutoTokenizer.from_pretrained(model_id(vintage))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    mdl = AutoModelForMaskedLM.from_pretrained(model_id(vintage)).to(dev).eval()
    pool = s[s.year == vintage + 1]
    hits = tried = 0
    for r in pool.sample(min(n, len(pool)), random_state=seed).itertuples():
        anon, _ = doc_text(r.cik, r.accession, r.permno)
        i = anon.find("[COMPANY]")
        if i < 0:
            continue
        txt = anon[max(0, i - 1200):i] + tok.mask_token + anon[i + 9:i + 1200]
        txt = txt.replace("[COMPANY]", "the company").replace("[NAME]", "X").replace("[TICKER]", "X").replace("[PERSON]", "X")
        enc = tok(txt, return_tensors="pt", truncation=True, max_length=512).to(dev)
        pos = (enc.input_ids[0] == tok.mask_token_id).nonzero()
        if len(pos) == 0:
            continue
        with torch.no_grad():
            logits = mdl(**enc).logits[0, pos[0, 0]]
        top = [tok.decode([t]).strip().lower() for t in logits.topk(10).indices.tolist()]
        name = distinctive(meta.name.get(r.cik, ""), set()) or ""
        first = name.split()[0] if name else ""
        tried += 1
        if first and any(len(t) >= 3 and (first.startswith(t) or t == first) for t in top):
            hits += 1
    return dict(vintage=vintage, filings=tried, top10_hits_firm_name=hits, rate=hits / max(tried, 1))


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    sp = OUT / "sample.parquet"
    s = pd.read_parquet(sp) if sp.exists() else sample()
    s.to_parquet(sp)
    embed_fixed(s)
    tab, ns = probe(s)
    bl = block_stats(tab)
    ex = leak_examples(s)
    rs = residual_scan(s)
    mr = mlm_reid(s)
    tab.to_csv(OUT / "auc_by_year.csv"); bl.to_csv(OUT / "auc_blocks.csv", index=False)
    json.dump(dict(n=ns, examples=ex, residual=rs, mlm=mr), open(OUT / "leak.json", "w"), indent=1, default=str)
    print(tab.round(3).to_string()); print(bl.round(3).to_string()); print(ns, rs, mr)


if __name__ == "__main__":
    run()
