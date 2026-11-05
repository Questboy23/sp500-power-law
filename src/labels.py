"""Forward returns and tail labels at each June formation date (Section 2).

fwd_Ny = buy-and-hold total return over N years from the formation month end, compounding
through delisting. A permno that stops trading before the CRSP end while distressed (last
price < $5 and < 30% of its trailing 12m high) gets -100% (spec: missing bankruptcy
delisting return -> -100%, flagged). Other exits (mergers etc.) stop compounding at the last
return; the missing final delisting return is typically small for mergers.

`label_end` travels with every row so holdout_guard can refuse a label that uses 2016+ returns."""
from __future__ import annotations
import numpy as np, pandas as pd
from config import PROC, TAIL_Q

CRSP_END = pd.Timestamp("2024-12-31")


def monthly_returns_wide(src: str = "universe") -> pd.DataFrame:
    if src == "universe":
        m = pd.read_parquet(PROC / "crsp_monthly_universe.parquet")
    else:
        m = pd.read_parquet(PROC / "market_monthly.parquet")
    m = m.sort_values(["permno", "date"])
    m["hi12"] = m.groupby("permno").prc.transform(lambda p: p.rolling(12, min_periods=1).max())
    w = m.pivot(index="date", columns="permno", values="ret")
    last = m.groupby("permno").tail(1).set_index("permno")
    dist = last[(last.date < CRSP_END) & (last.prc < 5) & (last.prc < 0.3 * last.hi12)]
    for pn, r in dist.iterrows():
        nxt = r.date + pd.offsets.MonthEnd(1)
        if nxt not in w.index:
            w.loc[nxt] = np.nan
        w.loc[nxt, pn] = -1.0
    w = w.sort_index()
    w.attrs["distress_permnos"] = list(dist.index)
    return w


def extend_with_yahoo(w: pd.DataFrame) -> pd.DataFrame:
    """Append 2025+ months for permnos that were S&P members at 2024-12 (Yahoo Adj Close)."""
    from universe import yahoo_monthly_ret
    lk = pd.read_parquet(PROC / "membership_linked.parquet")
    lk = lk[(lk.date == "2024-12-31") & (lk.permno > 0)]
    add = {}
    for _, r in lk.iterrows():
        s = yahoo_monthly_ret(r.ticker)
        if s is not None:
            add[r.permno] = s[s.index > CRSP_END]
    ext = pd.DataFrame(add)
    out = pd.concat([w, ext]).sort_index() if len(ext) else w
    out.attrs = dict(w.attrs)
    return out


def fwd_return(w: pd.DataFrame, start: pd.Timestamp, months: int) -> pd.Series:
    blk = w.loc[(w.index > start) & (w.index <= start + pd.offsets.MonthEnd(months))]
    if len(blk) < months:   # window runs past the data
        return pd.Series(dtype=float)
    return (1 + blk.fillna(0)).prod() - 1


def formation_dates(first=1996, last=2026):
    return [pd.Timestamp(f"{y}-06-30") for y in range(first, last + 1)]


def build_labels(years=(1, 5, 10, 20)) -> pd.DataFrame:
    w = extend_with_yahoo(monthly_returns_wide("universe"))
    lk = pd.read_parquet(PROC / "membership_linked.parquet")
    rows = []
    for f in formation_dates():
        mem = lk[(lk.date == f) & (lk.permno > 0)]
        alive = [p for p in mem.permno.unique() if p in w.columns]
        rec = pd.DataFrame({"permno": alive}); rec["formation_date"] = f
        for y in years:
            r = fwd_return(w, f, 12 * y)
            rec[f"fwd_{y}y"] = rec.permno.map(r) if len(r) else np.nan
        rec["label_end"] = f + pd.offsets.MonthEnd(12 * 5)
        rows.append(rec)
    lab = pd.concat(rows, ignore_index=True)
    lab["tail"] = lab.groupby("formation_date").fwd_5y.transform(lambda s: (s >= s.quantile(TAIL_Q)).astype(float).where(s.notna()))
    lab["distress_flag"] = lab.permno.isin(w.attrs.get("distress_permnos", []))
    lab.to_parquet(PROC / "labels.parquet")
    return lab


if __name__ == "__main__":
    import sys; sys.path.insert(0, "src")
    l = build_labels()
    print(l.groupby("formation_date")[["fwd_1y", "fwd_5y", "tail"]].agg(["count", "mean"]).round(3).to_string())
