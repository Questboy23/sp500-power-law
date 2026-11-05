"""Follow-up question (2026-10-01): what would a top-100 portfolio look like, and can it be improved?
Monthly rebalance on prior month-end caps. 1926-2024 CRSP (market_long), 2025-26 from the S&P panel.
NOT out-of-sample: the 2016+ holdout was spent on 2026-09-30, so post-2015 numbers are in-sample."""
import json, numpy as np, pandas as pd
from config import PROC, REPORTS
from src.trials import log_trial

def panel():
    x = pd.read_parquet(PROC / "market_long.parquet")[["permno", "date", "ret", "mcap"]]
    p = pd.read_parquet(PROC / "panel_monthly.parquet")
    y = p[p.date > "2024-12-31"][["permno", "date", "ret", "mcap"]]
    x = pd.concat([x, y]).sort_values(["permno", "date"])
    x["w0"] = x.groupby("permno").mcap.shift(1)
    return x.dropna(subset=["ret", "w0"])

def run(x):
    out = {}
    for d, g in x.groupby("date"):
        g = g.sort_values("w0", ascending=False)
        r = {}
        for n in (10, 25, 50, 100, 500):
            t = g.head(n); r[f"top{n}_cw"] = np.average(t.ret, weights=t.w0)
        t = g.head(100)
        r["top100_ew"] = t.ret.mean()
        w = (t.w0 / t.w0.sum()).clip(upper=0.05); w = w / w.sum()
        for _ in range(5): w = w.clip(upper=0.05); w = w / w.sum()
        r["top100_cap5"] = float((w * t.ret).sum())
        r["top100_sqrt"] = np.average(t.ret, weights=np.sqrt(t.w0))
        r["ranks101_500_cw"] = np.average(g.iloc[100:500].ret, weights=g.iloc[100:500].w0)
        out[d] = r
    return pd.DataFrame(out).T.sort_index()

def stats(R, rf):
    rows = {}
    for c in R:
        r = R[c].dropna(); ex = r - rf.reindex(r.index).fillna(0)
        a = r - R["top500_cw"].reindex(r.index)
        rows[c] = {"cagr": float(np.prod(1 + r) ** (12 / len(r)) - 1), "vol": float(r.std() * np.sqrt(12)),
                   "sharpe": float(ex.mean() / ex.std() * np.sqrt(12)), "vs_top500_ann": float(a.mean() * 12),
                   "te": float(a.std() * np.sqrt(12)), "t": float(a.mean() / (a.std() / np.sqrt(len(a)))) if a.std() > 0 else 0.0,
                   "months": len(r)}
    return pd.DataFrame(rows).T

if __name__ == "__main__":
    from src.powerlaw.descriptive import rf_cum
    rf = rf_cum()
    R = run(panel()); R = R[R.index >= "1926-07-31"]
    R.to_parquet(PROC / "topn_monthly.parquet")
    res = {}
    for name, a, b in [("1926-2015 (pre-holdout era)", "1926-07-31", "2015-12-31"), ("1926-1969", "1926-07-31", "1969-12-31"),
                       ("1970-1995", "1970-01-31", "1995-12-31"), ("1996-2015", "1996-01-31", "2015-12-31"),
                       ("2016-2026 (IN-SAMPLE, holdout spent)", "2016-01-31", "2026-08-31")]:
        s = stats(R.loc[a:b], rf); res[name] = s.round(4).to_dict("index")
        print("\n==", name); print(s[["cagr", "vol", "sharpe", "vs_top500_ann", "te", "t", "months"]].round(3).to_string())
    for c in R.columns: log_trial("TOPN_followup", c, "followup")
    json.dump(res, open(REPORTS / "topn_followup.json", "w"), indent=1)
