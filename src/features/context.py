"""Shared, truncatable data context for feature builders.

`Context(asof=None)` loads everything; `ctx.truncate(t)` returns a copy that has NO data dated
after t. A feature is look-ahead-free iff feature(ctx_full, t) == feature(ctx_full.truncate(t), t)
(tests/test_features.py checks this for every angle)."""
from __future__ import annotations
import copy, numpy as np, pandas as pd
from config import RAW, PROC

FIAM_COLS = ["sale_gr1", "sale_gr3", "rd_sale", "be_me", "gp_at", "op_at", "at_gr1", "capx_gr3", "rmax1_21d",
             "turnover_126d", "age", "qmj", "betadown_252d", "debt_me", "o_score", "z_score", "rvol_21d",
             "ivol_capm_252d", "ret_12_1", "ope_be", "sale_me", "market_equity", "at_turnover", "rd5_at",
             "ebit_sale", "cash_at", "netdebt_me", "sale_emp_gr1", "eqnpo_12m", "ocf_at", "ni_be", "at_be"]


def _sic_map() -> pd.DataFrame:
    """(permno, date) -> siccd by exact month-end price+shares match between the CRSP monthly
    ticker file and our CRSP-daily-derived monthly panel."""
    p = PROC / "sic_map.parquet"
    if p.exists():
        return pd.read_parquet(p)
    t = pd.read_parquet(RAW / "delisted" / "crsp_monthly_ticker_1945_2024.parquet", columns=["date", "siccd", "vol", "prc", "shrout"])
    t["date"] = pd.to_datetime(t.date) + pd.offsets.MonthEnd(0)
    t = t[t.date >= "1994-01-01"].copy()
    t["k"] = t.prc.abs().round(3).astype(str) + "_" + t.shrout.round(0).astype(str)
    u = pd.read_parquet(PROC / "crsp_monthly_universe.parquet", columns=["permno", "date", "prc", "shrout"])
    u["k"] = u.prc.abs().round(3).astype(str) + "_" + u.shrout.round(0).astype(str)
    j = u.merge(t[["date", "k", "siccd", "vol"]], on=["date", "k"], how="left").drop_duplicates(["permno", "date"])
    j["siccd"] = pd.to_numeric(j.siccd, errors="coerce")
    j = j.sort_values(["permno", "date"])
    j["siccd"] = j.groupby("permno").siccd.ffill()
    out = j[["permno", "date", "siccd", "vol"]]
    out.to_parquet(p)
    return out


def _daily_max() -> pd.DataFrame:
    """Max daily return within each calendar month, per permno (CRSP daily; Yahoo for 2025+)."""
    p = PROC / "daily_max.parquet"
    if p.exists():
        return pd.read_parquet(p)
    import pyarrow.parquet as pq
    perms = pd.read_parquet(PROC / "crsp_monthly_universe.parquet", columns=["permno"]).permno.unique().tolist()
    d = pq.read_table(RAW / "delisted" / "crsp_daily_1968_2024.parquet", columns=["permno", "date", "ret"],
                      filters=[("permno", "in", perms)]).to_pandas()
    d["date"] = pd.to_datetime(d.date); d = d[d.date >= "1995-01-01"]
    d["m"] = d.date + pd.offsets.MonthEnd(0)
    out = d.groupby(["permno", "m"]).ret.max().rename("maxret").reset_index().rename(columns={"m": "date"})
    out.to_parquet(p)
    return out


class Context:
    def __init__(self, load=True):
        if not load:
            return
        self.asof = None
        pan = pd.read_parquet(PROC / "panel_monthly.parquet")
        self.members = pan[["date", "ticker", "permno"]].copy()
        self.members["date"] = self.members.date - pd.offsets.MonthEnd(1)   # panel rows are held-in-month; membership is as of prior month end
        lk = pd.read_parquet(PROC / "membership_linked.parquet")
        self.lk = lk
        u = pd.read_parquet(PROC / "crsp_monthly_universe.parquet")
        self.R = u.pivot(index="date", columns="permno", values="ret")
        self.C = u.pivot(index="date", columns="permno", values="mcap")
        self._extend_yahoo()
        f = pd.read_parquet(RAW / "delisted" / "fiam_monthly_2000_2023.parquet", columns=["date", "permno"] + FIAM_COLS)
        f["date"] = pd.to_datetime(f.date.astype(str), format="%Y%m%d") + pd.offsets.MonthEnd(0)
        self.F = f.drop_duplicates(["date", "permno"]).set_index(["date", "permno"]).sort_index()
        self.sic = _sic_map()
        self.dmax = _daily_max()

    def _extend_yahoo(self):
        """2025+ monthly returns and caps for S&P members (permno from the 2024-12 link)."""
        from src.universe import _yahoo_frame
        from src.universe import _latest_shares
        shares = _latest_shares()
        lk = self.lk[(self.lk.date > "2024-06-30") & (self.lk.permno > 0)].drop_duplicates("permno", keep="last")
        R, C = {}, {}
        for _, r in lk.iterrows():
            x = _yahoo_frame(r.ticker)
            if x is None:
                continue
            ret = x["Adj Close"].resample("ME").last().pct_change()
            cl = x["Close"].resample("ME").last()
            c24 = self.C.at[pd.Timestamp("2024-12-31"), r.permno] if (r.permno in self.C.columns) else np.nan
            if pd.notna(c24) and pd.Timestamp("2024-12-31") in cl.index:
                sh = c24 / cl.loc["2024-12-31"]
            else:
                sh = shares.get(r.ticker.replace(".", "-"), (None, None))[1]
            if sh is None or not np.isfinite(sh):
                continue
            R[r.permno] = ret[ret.index > "2024-12-31"]
            C[r.permno] = (cl * sh)[cl.index > "2024-12-31"]
        if R:
            self.R = pd.concat([self.R, pd.DataFrame(R)]).sort_index()
            self.C = pd.concat([self.C, pd.DataFrame(C)]).sort_index()
            self.R = self.R[~self.R.index.duplicated(keep="first")]
            self.C = self.C[~self.C.index.duplicated(keep="first")]

    def truncate(self, t):
        t = pd.Timestamp(t)
        c = Context(load=False)
        c.asof = t
        c.members = self.members[self.members.date <= t]
        c.lk = self.lk[self.lk.date <= t]
        c.R = self.R.loc[:t]; c.C = self.C.loc[:t]
        c.F = self.F.loc[:t]
        c.sic = self.sic[self.sic.date <= t]
        c.dmax = self.dmax[self.dmax.date <= t]
        return c

    # ---- accessors used by features ----
    def sp(self, t):
        m = self.lk[(self.lk.date == t) & (self.lk.permno > 0)].permno.unique()
        return pd.Index(m)

    def fiam(self, t, cols, lag_months=0):
        d = pd.Timestamp(t) - pd.offsets.MonthEnd(lag_months) if lag_months else pd.Timestamp(t)
        try:
            return self.F.loc[d, cols]
        except KeyError:
            return pd.DataFrame(columns=cols)

    def sic2(self, t):
        s = self.sic[self.sic.date <= t].sort_values("date").groupby("permno").siccd.last()
        return (s // 100)

    def trailing_ret(self, t, months):
        blk = self.R.loc[(self.R.index > t - pd.offsets.MonthEnd(months)) & (self.R.index <= t)]
        ok = blk.notna().sum() >= months * 0.8
        return ((1 + blk.fillna(0)).prod() - 1)[ok]
