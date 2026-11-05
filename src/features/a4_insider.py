"""A4_insider_buys: opportunistic open-market insider trading (Cohen, Malloy & Pomorski 2012).

Pipeline
  1. download : SEC DERA "Insider Transactions Data Sets" quarterly zips -> data/raw/form4/
  2. parse    : SUBMISSION + REPORTINGOWNER + NONDERIV_TRANS (codes P/S) -> data/raw/form4/trans_ps.parquet
                plus every Form 4 filer (for #insiders filing)            -> data/raw/form4/filers.parquet
  3. link     : issuer CIK -> CRSP permno via (ticker, filing month) against membership_linked,
                propagated per CIK, plus company_tickers.json for current names
  4. build    : per June-30 formation date x S&P member permno, point-in-time counts
                -> data/processed/a4_insider.parquet

Point-in-time rule: at formation F only rows with FILING_DATE <= F are visible, both for the
12-month window counts AND for the routine/opportunistic history used to classify them.

No returns are computed here.

Usage:  .venv/bin/python -m src.features.a4_insider [download|parse|build|all]
"""
from __future__ import annotations

import io
import os
import json
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "form4"
PROC = ROOT / "data" / "processed"
OUT = PROC / "a4_insider.parquet"
def _ua() -> str:
    """SEC EDGAR asks for a contact email in the User-Agent. Set SEC_CONTACT_EMAIL before running."""
    email = os.environ.get("SEC_CONTACT_EMAIL", "").strip()
    if not email:
        raise RuntimeError("SEC_CONTACT_EMAIL is not set. The SEC requires a contact email in the "
                           "User-Agent of every request. Set it first: export SEC_CONTACT_EMAIL=you@example.com")
    return f"sp500-power-law research {email}"


INDEX_URL = "https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets"
MIN_GAP = 0.2  # seconds between requests (5 req/s, under the SEC 10 req/s cap and the 8 req/s budget)

FORMATIONS = [pd.Timestamp(f"{y}-06-30") for y in range(2007, 2027)]

# ----------------------------------------------------------------------------- download

_last = [0.0]


def _get(sess: requests.Session, url: str, **kw) -> requests.Response:
    wait = MIN_GAP - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    for attempt in range(5):
        _last[0] = time.time()
        r = sess.get(url, timeout=300, **kw)
        if r.status_code == 200:
            return r
        time.sleep(2 ** attempt + 1)
    r.raise_for_status()
    return r


def zip_urls(sess: requests.Session) -> list[str]:
    html = _get(sess, INDEX_URL).text
    rel = sorted(set(re.findall(r'href="([^"]*insider-transactions-data-sets/\d{4}q\d_form345\.zip)"', html)))
    return ["https://www.sec.gov" + u if u.startswith("/") else u for u in rel]


def download() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    sess.headers.update({"User-Agent": _ua(), "Accept-Encoding": "gzip, deflate"})
    urls = zip_urls(sess)
    (RAW / "urls.txt").write_text("\n".join(urls) + "\n")
    print(f"{len(urls)} quarterly zips listed")
    for u in urls:
        dest = RAW / u.rsplit("/", 1)[1]
        if dest.exists():
            try:
                zipfile.ZipFile(dest).testzip()
                continue
            except Exception:
                dest.unlink()
        r = _get(sess, u)
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(r.content)
        zipfile.ZipFile(tmp).namelist()  # raises if truncated
        tmp.rename(dest)
        print(f"  {dest.name}  {len(r.content)/1e6:.1f} MB", flush=True)


# ----------------------------------------------------------------------------- parse

def _read(z: zipfile.ZipFile, name: str, cols: list[str]) -> pd.DataFrame:
    with z.open(name) as fh:
        return pd.read_csv(fh, sep="\t", usecols=cols, dtype=str, quoting=3, keep_default_na=False,
                           na_values=[""], on_bad_lines="warn", encoding="latin-1")


def _date(s: pd.Series) -> pd.Series:
    # DERA files use DD-MON-YYYY (e.g. 31-MAR-2015); fall back to ISO if a quarter differs
    d = pd.to_datetime(s, format="%d-%b-%Y", errors="coerce")
    miss = d.isna() & s.notna()
    if miss.any():
        d[miss] = pd.to_datetime(s[miss], errors="coerce")
    return d


def _primary_owner(ow: pd.DataFrame) -> pd.DataFrame:
    """One insider per accession. Joint filings (fund + GP, spouses, etc.) list several owners;
    prefer an Officer/Director over a pure 10% owner, then the lowest CIK."""
    rel = ow["RPTOWNER_RELATIONSHIP"].fillna("")
    ow = ow.assign(_rank=np.where(rel.str.contains("Officer|Director", case=False), 0, 1),
                   _cik=pd.to_numeric(ow["RPTOWNERCIK"], errors="coerce"))
    ow = ow.sort_values(["ACCESSION_NUMBER", "_rank", "_cik"])
    n = ow.groupby("ACCESSION_NUMBER").size().rename("n_owners")
    first = ow.drop_duplicates("ACCESSION_NUMBER").set_index("ACCESSION_NUMBER")
    return first[["_cik", "RPTOWNER_RELATIONSHIP"]].rename(
        columns={"_cik": "owner_cik", "RPTOWNER_RELATIONSHIP": "relationship"}).join(n)


def parse() -> None:
    zips = sorted(RAW.glob("*_form345.zip"))
    trans, filers, schema = [], [], {}
    for zp in zips:
        q = zp.name[:6]
        with zipfile.ZipFile(zp) as z:
            names = set(z.namelist())
            for f in ("SUBMISSION.tsv", "REPORTINGOWNER.tsv", "NONDERIV_TRANS.tsv"):
                with z.open(f) as fh:
                    schema.setdefault(f, {})[q] = fh.readline().decode("latin-1").rstrip("\r\n").split("\t")
            sub = _read(z, "SUBMISSION.tsv", ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE",
                                              "ISSUERCIK", "ISSUERTRADINGSYMBOL"])
            ow = _read(z, "REPORTINGOWNER.tsv", ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP"])
            nd = _read(z, "NONDERIV_TRANS.tsv", ["ACCESSION_NUMBER", "NONDERIV_TRANS_SK", "TRANS_DATE",
                                                 "TRANS_CODE", "TRANS_SHARES", "TRANS_PRICEPERSHARE",
                                                 "TRANS_ACQUIRED_DISP_CD"])
        sub["filing_date"] = _date(sub["FILING_DATE"])
        sub["issuer_cik"] = pd.to_numeric(sub["ISSUERCIK"], errors="coerce")
        sub["ticker"] = sub["ISSUERTRADINGSYMBOL"].fillna("").str.upper().str.strip()
        sub = sub.set_index("ACCESSION_NUMBER")[["filing_date", "DOCUMENT_TYPE", "issuer_cik", "ticker"]]
        prim = _primary_owner(ow)

        # every Form 4 filer (any transaction code), for the #insiders-filing scale
        f4 = sub[sub["DOCUMENT_TYPE"] == "4"].join(prim, how="inner")
        filers.append(f4.reset_index()[["ACCESSION_NUMBER", "filing_date", "issuer_cik", "ticker", "owner_cik"]])

        nd = nd[nd["TRANS_CODE"].isin(["P", "S"])].copy()
        nd["trans_date"] = _date(nd["TRANS_DATE"])
        nd["shares"] = pd.to_numeric(nd["TRANS_SHARES"], errors="coerce")
        nd["price"] = pd.to_numeric(nd["TRANS_PRICEPERSHARE"], errors="coerce")
        nd = nd.join(sub, on="ACCESSION_NUMBER", how="inner").join(prim, on="ACCESSION_NUMBER", how="inner")
        nd["quarter"] = q
        trans.append(nd[["ACCESSION_NUMBER", "NONDERIV_TRANS_SK", "quarter", "DOCUMENT_TYPE", "filing_date",
                         "trans_date", "issuer_cik", "ticker", "owner_cik", "relationship", "n_owners",
                         "TRANS_CODE", "TRANS_ACQUIRED_DISP_CD", "shares", "price"]])
        print(f"  {q}: sub={len(sub):,} P/S rows={len(nd):,}", flush=True)
    t = pd.concat(trans, ignore_index=True)
    t.columns = [c.lower() for c in t.columns]
    t = t.drop_duplicates(["accession_number", "nonderiv_trans_sk"])
    t.to_parquet(RAW / "trans_ps.parquet", index=False)
    fl = pd.concat(filers, ignore_index=True).drop_duplicates("ACCESSION_NUMBER")
    fl.columns = [c.lower() for c in fl.columns]
    fl.to_parquet(RAW / "filers.parquet", index=False)
    (RAW / "schema_headers.json").write_text(json.dumps(schema))
    print(f"trans_ps: {len(t):,} rows; filers: {len(fl):,} Form 4 filings")


def clean_trades(t: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Open-market purchases (P + A) and sales (S + D), original Form 4 / Form 5 only."""
    st = {"raw_P_S_rows": len(t)}
    t = t[t["document_type"].isin(["4", "5"])]  # amendments (4/A, 5/A) dropped to avoid double counting
    st["after_drop_amendments"] = len(t)
    buy = (t["trans_code"] == "P") & (t["trans_acquired_disp_cd"] == "A")
    sell = (t["trans_code"] == "S") & (t["trans_acquired_disp_cd"] == "D")
    st["P_not_A"] = int(((t["trans_code"] == "P") & ~buy).sum())
    st["S_not_D"] = int(((t["trans_code"] == "S") & ~sell).sum())
    t = t[buy | sell].copy()
    t["dir"] = np.where(t["trans_code"] == "P", "B", "S")
    t = t.dropna(subset=["trans_date", "filing_date", "issuer_cik", "owner_cik"])
    # trans dates after filing or implausibly old are data-entry errors
    bad = (t["trans_date"] > t["filing_date"]) | (t["trans_date"] < pd.Timestamp("1990-01-01"))
    st["bad_trans_date"] = int(bad.sum())
    t = t[~bad]
    # dollar value; typo-level price errors (>20x or <1/20 of the issuer's median P/S price in the
    # same month) and single rows above $10B are set to NaN (counts are unaffected)
    t["value"] = (t["shares"] * t["price"]).where((t["price"] > 0) & (t["shares"] > 0))
    ym = t["trans_date"].dt.to_period("M")
    med = t.groupby([t["issuer_cik"], ym])["price"].transform(lambda x: x[x > 0].median())
    r = t["price"] / med
    badv = ((r > 20) | (r < 0.05)) & t["value"].notna() | (t["value"] > 1e10)
    st["value_set_nan_outlier"] = int(badv.sum())
    t.loc[badv, "value"] = np.nan
    st["clean_rows"] = len(t)
    st["clean_buys"] = int((t["dir"] == "B").sum())
    st["clean_sells"] = int((t["dir"] == "S").sum())
    t["issuer_cik"] = t["issuer_cik"].astype("int64")
    t["owner_cik"] = t["owner_cik"].astype("int64")
    return t, st


# ----------------------------------------------------------------------------- link

def _norm_ticker(s: pd.Series) -> pd.Series:
    s = s.fillna("").str.upper().str.strip()
    s = s.str.replace(r"^(NYSE|NASDAQ|NASD|AMEX|OTC)[:\s]+", "", regex=True)
    s = s.str.replace(r"[\s\-/_]+", ".", regex=True).str.replace(r"\.+$", "", regex=True)
    return s


def build_link(fl: pd.DataFrame, memb: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Issuer CIK -> permno. Step 1: exact (normalized ticker, filing month-end) match against
    S&P membership. Step 2: majority permno per CIK over direct matches (a CIK that ever
    matched is mapped for all its filings; multi-permno CIKs keep a dated link). Step 3:
    company_tickers.json current ticker -> permno for CIKs still unmapped (current members)."""
    m = memb[["date", "ticker", "permno"]].copy()
    m["tk"] = _norm_ticker(m["ticker"])
    m = m.drop_duplicates(["date", "tk"])
    f = fl[["issuer_cik", "ticker", "filing_date"]].dropna().copy()
    f["tk"] = _norm_ticker(f["ticker"])
    f["date"] = f["filing_date"] + pd.offsets.MonthEnd(0)
    f = f.drop_duplicates(["issuer_cik", "tk", "date"])
    hit = f.merge(m[["date", "tk", "permno"]], on=["date", "tk"], how="inner")
    # also try with the dots removed (BRK.B vs BRKB)
    m2 = m.assign(tk=m["tk"].str.replace(".", "", regex=False)).drop_duplicates(["date", "tk"])
    f2 = f.assign(tk=f["tk"].str.replace(".", "", regex=False))
    hit2 = f2.merge(m2[["date", "tk", "permno"]], on=["date", "tk"], how="inner")
    hit = pd.concat([hit, hit2]).drop_duplicates(["issuer_cik", "date", "permno"])

    # per (cik, permno): number of matching months and span
    g = hit.groupby(["issuer_cik", "permno"]).agg(n=("date", "size"), first=("date", "min"),
                                                 last=("date", "max")).reset_index()
    # drop spurious links: a permno claimed by several CIKs keeps the CIKs with >=20% of its matches
    g["share_p"] = g["n"] / g.groupby("permno")["n"].transform("sum")
    g = g[g["share_p"] >= 0.2]
    g["src"] = "ticker_date"

    # step 2b: members never matched by (ticker, month) -- membership sometimes carries a later
    # ticker for the whole history (AABA for Yahoo, TPR for Coach, KDP for Dr Pepper Snapple).
    # Match their ticker to Form 4 tickers at ANY date; take the CIK with the most such filings.
    un = m[~m["permno"].isin(g["permno"]) & (m["date"] >= pd.Timestamp("2006-01-31"))]
    un = un.drop_duplicates(["tk", "permno"])
    any_ = f.groupby(["tk", "issuer_cik"]).agg(n=("date", "size"), first=("date", "min"),
                                               last=("date", "max")).reset_index()
    any_ = any_.merge(un[["tk", "permno"]], on="tk").sort_values("n", ascending=False)
    any_ = any_.drop_duplicates("permno")
    any_ = any_.assign(share_p=1.0, src="ticker_anytime", first=pd.NaT, last=pd.NaT)[g.columns]
    g = pd.concat([g, any_], ignore_index=True)

    # company_tickers.json for CIKs never matched: current ticker vs the latest membership month
    ct = pd.DataFrame(json.load(open(ROOT / "data" / "raw" / "edgar" / "company_tickers.json")).values())
    ct["tk"] = _norm_ticker(ct["ticker"])
    last_m = m[m["date"] == m["date"].max()]
    ctm = ct.merge(last_m[["tk", "permno"]], on="tk")
    ctm = ctm[~ctm["permno"].isin(g["permno"])]  # only members with no ticker-date link at all
    add = pd.DataFrame({"issuer_cik": ctm["cik_str"].astype("int64"), "permno": ctm["permno"],
                        "n": 0, "first": pd.NaT, "last": pd.NaT, "share_p": 1.0, "src": "company_tickers"})
    link = pd.concat([g, add], ignore_index=True)
    st = {"ticker_date_links": int((link["src"] == "ticker_date").sum()),
          "ticker_anytime_links": int((link["src"] == "ticker_anytime").sum()),
          "company_tickers_links": int((link["src"] == "company_tickers").sum()),
          "ciks_linked": int(link["issuer_cik"].nunique()),
          "permnos_linked": int(link["permno"].nunique()),
          "ciks_multi_permno": int((link.groupby("issuer_cik")["permno"].nunique() > 1).sum())}
    return link, st


def attach_permno(df: pd.DataFrame, link: pd.DataFrame, date_col: str = "filing_date") -> pd.DataFrame:
    """Map rows to permno by CIK. For CIKs linked to >1 permno, pick the permno(s) whose matched
    span is nearest the row's date (company_tickers links have no span and count as dist 0)."""
    single = link.groupby("issuer_cik")["permno"].nunique()
    one = link[link["issuer_cik"].isin(single[single == 1].index)][["issuer_cik", "permno"]]
    out1 = df.merge(one, on="issuer_cik", how="inner")
    multi = link[link["issuer_cik"].isin(single[single > 1].index)]
    rest = df[df["issuer_cik"].isin(multi["issuer_cik"])].reset_index(names="_rid")
    if len(rest):
        x = rest[["_rid", "issuer_cik", date_col]].merge(multi, on="issuer_cik")
        d = x[date_col]
        x["dist"] = np.where(x["first"].isna(), 0, np.where(d < x["first"], (x["first"] - d).dt.days,
                             np.where(d > x["last"], (d - x["last"]).dt.days, 0)))
        # nearest span wins; ties (dual-class shares under one CIK, e.g. FOX/FOXA) map to every tied permno
        x = x[x["dist"] == x.groupby("_rid")["dist"].transform("min")].drop_duplicates(["_rid", "permno"])
        out2 = rest.merge(x[["_rid", "permno"]], on="_rid").drop(columns="_rid")
        return pd.concat([out1, out2], ignore_index=True)
    return out1


# ----------------------------------------------------------------------------- classify

def classify(h: pd.DataFrame) -> pd.Series:
    """CMP (2012) classification of each trade in `h` (the point-in-time visible history).
    Keyed on insider = (owner_cik, issuer_cik). For a trade in calendar year Y:
      classifiable  : insider has >=1 open-market trade (P or S) in EACH of Y-1, Y-2, Y-3
      routine       : classifiable AND for the trade's direction there is some calendar month m
                      with a same-direction trade in (Y-1,m), (Y-2,m) and (Y-3,m)
      opportunistic : classifiable and not routine
      unclassified  : not classifiable
    Returns a Series of {'routine','opp','uncl'} aligned to h.index."""
    y = h["trans_date"].dt.year
    mth = h["trans_date"].dt.month
    key = h["owner_cik"].astype("int64") * 10_000_000_000 + h["issuer_cik"].astype("int64")

    yrs = pd.DataFrame({"k": key, "y": y}).drop_duplicates()
    s = set(zip(yrs["k"], yrs["y"]))
    cls_ok = np.array([(k, a - 1) in s and (k, a - 2) in s and (k, a - 3) in s
                       for k, a in zip(key.to_numpy(), y.to_numpy())], dtype=bool)

    ym = pd.DataFrame({"k": key, "d": h["dir"], "y": y, "m": mth}).drop_duplicates()
    sm = set(zip(ym["k"], ym["d"], ym["y"], ym["m"]))
    # routine insider-years: (k, d, Y) such that some m has (Y-1..Y-3, m)
    rt = set()
    for k, d, a, m in sm:
        if (k, d, a - 1, m) in sm and (k, d, a - 2, m) in sm:
            rt.add((k, d, a + 1))  # a, a-1, a-2 all present -> routine in year a+1
    routine = np.array([(k, d, a) in rt for k, d, a in zip(key.to_numpy(), h["dir"].to_numpy(), y.to_numpy())],
                       dtype=bool)
    lab = np.where(~cls_ok, "uncl", np.where(routine, "routine", "opp"))
    return pd.Series(lab, index=h.index)


# ----------------------------------------------------------------------------- build

CRSP_LAST = pd.Timestamp("2024-12-31")


def load_members() -> tuple[pd.DataFrame, dict]:
    """membership_linked with a usable key. CRSP ends 2024-12; if 2025+ rows carry permno -1
    those tickers get the permno the same ticker had at 2024-12-31 (if it was a member then).
    Any row still without a permno (pre-2025 link failures, 2025+ additions) gets a per-ticker
    pseudo key -(1e6 + i) so its filings can still be linked by ticker; the output reports
    permno = -1 for those rows and carries the ticker."""
    m = pd.read_parquet(PROC / "membership_linked.parquet")
    how = m["how"].replace("", "unlinked") if "how" in m else "crsp_link"
    m = m[["date", "ticker", "permno"]].copy()
    m["permno_src"] = how  # universe.py link method: span / lifetime / carried (2025+) / synthetic (900000+)
    last = m[(m["date"] == CRSP_LAST) & (m["permno"] >= 0)].drop_duplicates("ticker").set_index("ticker")["permno"]
    fix = (m["date"] > CRSP_LAST) & (m["permno"] < 0) & m["ticker"].isin(last.index)
    m.loc[fix, "permno"] = m.loc[fix, "ticker"].map(last).astype("int64")
    m.loc[fix, "permno_src"] = "ticker_at_2024-12"
    un = m["permno"] < 0
    codes = {t: -(1_000_000 + i) for i, t in enumerate(sorted(m.loc[un, "ticker"].unique()))}
    m.loc[un, "permno"] = m.loc[un, "ticker"].map(codes).astype("int64")
    m.loc[un, "permno_src"] = "pseudo_ticker"
    st = {"member_rows_2025plus_resolved_to_2024_permno": int(fix.sum()),
          "member_rows_pseudo_ticker_key": int(un.sum())}
    return m, st


def build() -> None:
    t_raw = pd.read_parquet(RAW / "trans_ps.parquet")
    fl = pd.read_parquet(RAW / "filers.parquet").dropna(subset=["issuer_cik", "owner_cik", "filing_date"])
    fl["issuer_cik"] = fl["issuer_cik"].astype("int64")
    t, st_clean = clean_trades(t_raw)
    memb, st_memb = load_members()
    link, st_link = build_link(fl, memb)
    st_link.update(st_memb)

    tp = attach_permno(t, link)
    flp = attach_permno(fl, link)
    st_link["P_S_rows_all_issuers"] = len(t)
    st_link["P_S_rows_mapped_to_sp_permno"] = len(tp)

    rows, cov = [], []
    for F in FORMATIONS:
        lo = F - pd.DateOffset(months=12)
        memF = memb[memb["date"] == F].drop_duplicates("permno")
        mem = memF["permno"]
        vis = t[t["filing_date"] <= F]                     # point-in-time visible history (all issuers)
        lab = classify(vis)
        vis = vis.assign(cls=lab)
        win = vis[(vis["trans_date"] > lo) & (vis["trans_date"] <= F)]
        win = attach_permno(win, link)
        win = win[win["permno"].isin(mem)]
        # one "trade" = one insider x trade date x direction (collapses split-price rows)
        win["event"] = list(zip(win["owner_cik"], win["trans_date"], win["dir"]))

        def agg(sub: pd.DataFrame, tag: str) -> pd.DataFrame:
            g = sub.groupby("permno")
            return pd.DataFrame({f"n_{tag}": g["event"].nunique(), f"n_{tag}ers": g["owner_cik"].nunique(),
                                 f"val_{tag}": g["value"].sum(min_count=1)})

        parts = []
        for cls in ("opp", "routine", "uncl"):
            for d, nm in (("B", "buy"), ("S", "sell")):
                a = agg(win[(win["cls"] == cls) & (win["dir"] == d)], f"{nm}")
                a.columns = [c.replace(f"_{nm}", f"_{cls}_{nm}") for c in a.columns]
                parts.append(a)
        fw = flp[(flp["filing_date"] > lo) & (flp["filing_date"] <= F) & flp["permno"].isin(mem)]
        parts.append(fw.groupby("permno")["owner_cik"].nunique().rename("n_insiders_filing").to_frame())
        df = pd.DataFrame(index=pd.Index(mem.to_numpy(), name="permno")).join(parts)
        cnt = [c for c in df.columns if c.startswith("n_")]
        df[cnt] = df[cnt].fillna(0).astype("int64")
        df["has_form4_link"] = df.index.isin(link["permno"])
        df = df.reset_index().merge(memF[["permno", "ticker", "permno_src"]], on="permno", how="left")
        df.insert(0, "formation_date", F)
        rows.append(df)
        cov.append({"formation": F.date(), "members": len(mem), "linked": int(df["has_form4_link"].sum()),
                    "any_form4_12m": int((df["n_insiders_filing"] > 0).sum()),
                    "any_PS_12m": int((df.filter(regex=r"^n_(opp|routine|uncl)_(buy|sell)$").sum(axis=1) > 0).sum()),
                    "opp_buys": int(df["n_opp_buy"].sum()), "routine_buys": int(df["n_routine_buy"].sum()),
                    "uncl_buys": int(df["n_uncl_buy"].sum()), "opp_sells": int(df["n_opp_sell"].sum()),
                    "routine_sells": int(df["n_routine_sell"].sum()), "uncl_sells": int(df["n_uncl_sell"].sum()),
                    "firms_opp_buyer": int((df["n_opp_buyers"] > 0).sum()),
                    "firms_any_buyer": int((df.filter(regex=r"^n_(opp|routine|uncl)_buyers$").sum(axis=1) > 0).sum())})
        print(cov[-1], flush=True)

    out = pd.concat(rows, ignore_index=True)
    out = out.rename(columns={"n_opp_buyers": "n_buyers_opp", "n_opp_sellers": "n_sellers_opp",
                              "n_routine_buyers": "n_buyers_routine", "n_routine_sellers": "n_sellers_routine",
                              "n_uncl_buyers": "n_buyers_uncl", "n_uncl_sellers": "n_sellers_uncl",
                              "n_opp_buy": "n_opp_buys", "n_routine_buy": "n_routine_buys",
                              "n_uncl_buy": "n_uncl_buys", "n_opp_sell": "n_opp_sells",
                              "n_routine_sell": "n_routine_sells", "n_uncl_sell": "n_uncl_sells"})
    out["net_opp_buy_value"] = out["val_opp_buy"].fillna(0) - out["val_opp_sell"].fillna(0)
    ob, os_ = out["n_buyers_opp"], out["n_sellers_opp"]
    out["a4"] = (ob - os_) / (ob + os_ + 1)
    ub, us = ob + out["n_buyers_uncl"], os_ + out["n_sellers_uncl"]
    out["a4_incl_uncl"] = (ub - us) / (ub + us + 1)            # unclassified insiders treated as opportunistic
    out["a4_scaled"] = out["n_opp_buys"] / out["n_insiders_filing"].where(out["n_insiders_filing"] > 0)
    # rows with no link to any Form 4 issuer are missing, not zero
    for c in ("a4", "a4_incl_uncl", "a4_scaled"):
        out.loc[~out["has_form4_link"], c] = np.nan
    lead = ["formation_date", "permno", "ticker", "permno_src", "a4", "a4_incl_uncl", "a4_scaled", "n_opp_buys", "n_routine_buys",
            "n_uncl_buys", "n_buyers_opp", "n_sellers_opp", "n_opp_sells", "net_opp_buy_value"]
    out = out[lead + [c for c in out.columns if c not in lead]]
    out["member_key"] = out["permno"]
    out.loc[out["permno"] < 0, "permno"] = -1
    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT} {out.shape}")
    stats = {"clean": st_clean, "link": st_link, "coverage": cov}
    (RAW / "build_stats.json").write_text(json.dumps(stats, default=str, indent=1))


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("download", "all"):
        download()
    if step in ("parse", "all"):
        parse()
    if step in ("build", "all"):
        build()
