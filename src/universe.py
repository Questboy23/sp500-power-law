"""Point-in-time S&P 500 universe (Rule 1: no survivorship bias).

Membership: fja05680/sp500 daily component lists (1996-01-02 onward), point-in-time tickers.
Returns/caps: CRSP daily (Kaggle re-upload, 1968-2024) joined on ticker+date via
ticker_permno_link.csv; Yahoo fills 2025+ only. CRSP total returns include dividends.

Delisting returns (DLRET) are NOT in this CRSP copy. For a permno whose data ends while it
is an index member, we classify the exit: a distress exit (last price < $5 and down > 70%
from its trailing 12m high) gets -100% in the exit month per the spec, and is flagged; every
other exit (merger, going private, move to another index) gets 0% for the missing tail.
reports/phase1_universe.md has the sensitivity at -30%."""
from __future__ import annotations
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow as pa
from config import RAW, PROC

DISTRESS_RET = -1.0


def membership_daily() -> pd.DataFrame:
    d = pd.read_csv(RAW / "sp500_components_updated.csv"); d["date"] = pd.to_datetime(d.date)
    return d


def membership_monthly() -> pd.DataFrame:
    """(month_end, ticker) for every ticker in the index on the last listed date of each month."""
    d = membership_daily().set_index("date").sort_index()
    idx = pd.date_range(d.index.min(), pd.Timestamp.today(), freq="ME")
    rows = []
    for m in idx:
        s = d.loc[:m]
        if not len(s):
            continue
        for t in s.iloc[-1].tickers.split(","):
            rows.append((m, t))
    return pd.DataFrame(rows, columns=["date", "ticker"])


# Bankrupt / renamed tickers in the membership file -> the ticker CRSP carried while listed.
MANUAL = {"AABA": "YHOO", "RSHCQ": "RSH", "EKDKQ": "EK", "MTLQQ": "GM", "FNMA": "FNM", "FMCC": "FRE",
          "CCTYQ": "CC", "LEHMQ": "LEH", "KDP": "DPS", "DCNAQ": "DCN", "WAMUQ": "WM", "DALRQ": "DAL",
          "WNDXQ": "WIN", "BTUUQ": "BTU", "ABKFQ": "ABK", "AAMRQ": "AMR", "INCLF": "N", "RDS.A": "RD",
          "NRTLQ": "NT", "DPHIQ": "DPH", "UAWGQ": "UAL", "WCOEQ": "WCOM", "ENRNQ": "ENE", "CPNLQ": "CPN",
          "BHMSQ": "BS", "ACKH": "ACK", "CITGQ": "CIT", "GAPTQ": "GAP", "SUNEQ": "WFR", "MZIAQ": "MZ",
          "OWENQ": "OCF", "SGID": "SGI", "FLTWQ": "FLE", "BMGCA": "BMG", "FLMIQ": "FLM", "ANRZQ": "ANR",
          "VSTNQ": "VC", "CCEP": "CCE"}


def _candidates(t: str) -> list[str]:
    c = [t, t.replace(".", ""), t.split(".")[0]]
    if len(t) >= 4 and t.endswith("Q"):
        c.append(t[:-1])
    if t in MANUAL:
        c.append(MANUAL[t])
    return list(dict.fromkeys(c))


def link_permnos(mm: pd.DataFrame) -> pd.DataFrame:
    """Per (ticker, month): prefer a link span naming this ticker on this date; otherwise any
    permno that ever carried a candidate ticker and was alive on this date (permnos persist
    through ticker changes, which is how the file's retroactive tickers resolve)."""
    link = pd.read_csv(RAW / "delisted" / "ticker_permno_link.csv", parse_dates=["first_month", "last_month"])
    link.loc[link.last_month >= "2024-06-01", "last_month"] = pd.Timestamp("2024-12-31")
    link["first_month"] = link.first_month - pd.offsets.MonthBegin(1)
    life = pd.read_parquet(PROC / "permno_life.parquet")
    by_t = {t: g for t, g in link.groupby("ticker")}
    out = []
    for t, g in mm.groupby("ticker"):
        dates = g.date.values
        perm = np.full(len(dates), -1); how = np.array([""] * len(dates), dtype=object)
        cands = [by_t[c] for c in _candidates(t) if c in by_t]
        rows = pd.concat(cands).sort_values("n_months", ascending=False) if cands else pd.DataFrame()
        for _, r in rows.iterrows():            # pass 1: exact span
            hit = (dates >= np.datetime64(r.first_month)) & (dates <= np.datetime64(r.last_month)) & (perm < 0)
            perm[hit] = r.permno; how[hit] = "span"
        for _, r in rows.iterrows():            # pass 2: permno lifetime
            if r.permno not in life.index:
                continue
            lo, hi = life.loc[r.permno, "min"], life.loc[r.permno, "max"] + pd.offsets.MonthEnd(0)
            hit = (dates >= np.datetime64(lo)) & (dates <= np.datetime64(hi)) & (perm < 0)
            perm[hit] = r.permno; how[hit] = "lifetime"
        out.append(pd.DataFrame({"date": dates, "ticker": t, "permno": perm, "how": how}))
    return pd.concat(out)


def crsp_monthly(permnos: set[int]) -> pd.DataFrame:
    tbl = pq.read_table(RAW / "delisted" / "crsp_daily_1968_2024.parquet",
                        columns=["permno", "date", "prc", "ret", "shrout"],
                        filters=[("permno", "in", list(permnos))])
    d = tbl.to_pandas(); d["date"] = pd.to_datetime(d.date)
    d = d[d.date >= "1994-01-01"]
    d["prc"] = d.prc.abs()
    d["m"] = d.date + pd.offsets.MonthEnd(0)
    g = d.sort_values("date").groupby(["permno", "m"])
    mon = pd.DataFrame({
        "ret": g.ret.apply(lambda r: np.prod(1 + r.dropna()) - 1 if r.notna().any() else np.nan),
        "prc": g.prc.last(), "shrout": g.shrout.last(), "last_day": g.date.last(), "n_days": g.ret.count()})
    mon = mon.reset_index().rename(columns={"m": "date"})
    mon["mcap"] = mon.prc * mon.shrout * 1e3
    return mon


def yahoo_monthly(tickers) -> pd.DataFrame:
    """2025+ only (CRSP copy ends 2024-12-31). Total return from Adj Close."""
    from filings import TAGS  # noqa: F401  (import check only)
    return pd.DataFrame()


def build():
    mm = membership_monthly()
    lk = link_permnos(mm)
    cov = (lk.permno >= 0).groupby(lk.date.dt.year).mean()
    perms = set(lk.permno[lk.permno >= 0].astype(int))
    mon = crsp_monthly(perms)
    mon.to_parquet(PROC / "crsp_monthly_universe.parquet")
    lk.to_parquet(PROC / "membership_linked.parquet")
    print("permno link rate by year:\n", cov.round(3).to_string())
    return lk, mon


if __name__ == "__main__":
    build()


def _yahoo_frame(t: str) -> pd.DataFrame | None:
    p = RAW / "yahoo" / f"{t.replace('.', '-')}.parquet"
    if not p.exists():
        return None
    x = pd.read_parquet(p); x.index = pd.to_datetime(x.index).tz_localize(None)
    return x


def yahoo_monthly_ret(t: str) -> pd.Series | None:
    x = _yahoo_frame(t)
    if x is None:
        return None
    return x["Adj Close"].resample("ME").last().pct_change()


def _latest_shares() -> dict:
    """Most recent dei/us-gaap shares outstanding per current ticker (EDGAR), for 2025+ caps."""
    import json
    fl = pd.read_parquet(PROC / "fundamentals_long.parquet")
    sh = fl[fl.field == "shares"].sort_values("filed").groupby("cik").last()
    ct = json.load(open(RAW / "edgar" / "company_tickers.json"))
    return {v["ticker"]: (int(v["cik_str"]), sh.val.get(int(v["cik_str"]))) for v in ct.values()}


def panel() -> pd.DataFrame:
    """Monthly panel: one row per (date, ticker) index member with that month's total return,
    the member's cap at the END OF THE PRIOR MONTH (weights never see the month they weight),
    and exit flags. Returns for month m belong to members as of the end of month m-1."""
    lk = pd.read_parquet(PROC / "membership_linked.parquet")
    mon = pd.read_parquet(PROC / "crsp_monthly_universe.parquet")
    mon = mon.sort_values(["permno", "date"])
    mon["mcap_lag"] = mon.groupby("permno").mcap.shift(1)
    mon["hi12"] = mon.groupby("permno").prc.transform(lambda p: p.rolling(12, min_periods=1).max())
    last = mon.groupby("permno").date.max()
    # membership at end of m-1 defines who is held during month m
    held = lk.copy(); held["date"] = held.date + pd.offsets.MonthEnd(1)
    p = held.merge(mon, on=["permno", "date"], how="left")
    # a held permno with no CRSP row this month exited during/before it
    prev = mon[["permno", "date", "prc", "hi12", "mcap"]].copy(); prev["date"] = prev.date + pd.offsets.MonthEnd(1)
    prev.columns = ["permno", "date", "prc_prev", "hi12_prev", "mcap_prev"]
    p = p.merge(prev, on=["permno", "date"], how="left")
    crsp_end = pd.Timestamp("2024-12-31")
    exited = p.ret.isna() & p.mcap_prev.notna() & (p.date <= crsp_end)
    distress = exited & (p.prc_prev < 5) & (p.prc_prev < 0.3 * p.hi12_prev)
    p["exit"] = np.where(distress, "distress", np.where(exited, "other", ""))
    p.loc[distress, "ret"] = DISTRESS_RET
    p.loc[exited & ~distress, "ret"] = 0.0
    p["mcap_lag"] = p.mcap_lag.fillna(p.mcap_prev)
    p = p.drop_duplicates(["date", "permno"])
    return p[["date", "ticker", "permno", "how", "ret", "mcap_lag", "prc", "mcap", "exit"]]


def extend_yahoo(p: pd.DataFrame) -> pd.DataFrame:
    """2025-01 onward from Yahoo: total return from Adj Close; cap = close x latest EDGAR shares
    rolled back by Yahoo's split-adjusted close. Shares are held fixed, so buyback drift inside
    these 21 months is ignored (flagged in the Phase 1 report)."""
    lk = pd.read_parquet(PROC / "membership_linked.parquet")
    held = lk[lk.date >= "2024-12-31"].copy(); held["date"] = held.date + pd.offsets.MonthEnd(1)
    shares = _latest_shares()
    last_crsp = p[p.date == "2024-12-31"].set_index("ticker").mcap
    rows = []
    for t, g in held.groupby("ticker"):
        x = _yahoo_frame(t)
        if x is None:
            continue
        r = x["Adj Close"].resample("ME").last().pct_change()
        c = x["Close"].resample("ME").last()
        sh = shares.get(t.replace(".", "-"), (None, None))[1]
        if t in last_crsp.index and pd.notna(last_crsp[t]) and "2024-12-31" in c.index:
            sh = last_crsp[t] / c.loc["2024-12-31"]
        if sh is None or not np.isfinite(sh):
            continue
        cap = c * sh
        pn = int(g.permno.iloc[-1]) if "permno" in g and pd.notna(g.permno.iloc[-1]) else -1
        for d in g.date:
            if d in r.index:
                rows.append((d, t, pn, "yahoo", r.loc[d], cap.shift(1).get(d, np.nan), c.get(d), cap.get(d), ""))
    y = pd.DataFrame(rows, columns=p.columns)
    return pd.concat([p[p.date <= "2024-12-31"], y[y.date > "2024-12-31"]], ignore_index=True)


def build_panel():
    p = extend_yahoo(panel())
    p.to_parquet(PROC / "panel_monthly.parquet")
    return p


def carry_forward_permnos():
    """CRSP ends 2024-12: for later months, reuse each ticker's last linked permno; tickers that
    joined after 2024 get synthetic ids 900000+ (no CRSP history; Yahoo-only)."""
    lk = pd.read_parquet(PROC / "membership_linked.parquet")
    last = lk[(lk.permno > 0)].sort_values("date").groupby("ticker").permno.last()
    late = lk.date > "2024-12-31"
    lk.loc[late, "permno"] = lk.loc[late, "ticker"].map(last).fillna(-1).astype(int)
    new = sorted(lk.loc[late & (lk.permno < 0), "ticker"].unique())
    syn = {t: 900000 + i for i, t in enumerate(new)}
    m = late & (lk.permno < 0)
    lk.loc[m, "permno"] = lk.loc[m, "ticker"].map(syn)
    lk.loc[late, "how"] = np.where(lk.loc[late, "permno"] >= 900000, "synthetic", "carried")
    lk.to_parquet(PROC / "membership_linked.parquet")
    return syn
