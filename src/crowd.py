"""Follow-up (2026-10-03): is 'buy what everyone else is buying' optimal, and how does it differ from
the S&P 500? Quintile sorts inside the top 500 by cap, cap-weighted within quintile, monthly rebalance.
Pre-2016 is the clean era; 2016+ is in-sample (holdout spent 2026-09-30)."""
import json, numpy as np, pandas as pd
from config import PROC, REPORTS
from src.trials import log_trial

def load():
    x = pd.read_parquet(PROC / "market_long.parquet")[["permno", "date", "ret", "mcap", "shrout"]].sort_values(["permno", "date"])
    g = x.groupby("permno")
    x["w0"] = g.mcap.shift(1)
    lr = np.log1p(x.ret.fillna(0))
    x["mom12_1"] = np.expm1(g.ret.transform(lambda r: np.log1p(r.fillna(0)).rolling(11).sum().shift(2)))  # t-12..t-2
    x["rev1"] = g.ret.shift(1)                                                                              # last month
    v = pd.read_parquet(PROC / "sic_map.parquet")[["permno", "date", "vol"]]
    x = x.merge(v, on=["permno", "date"], how="left").sort_values(["permno", "date"])
    g = x.groupby("permno")
    x["turn"] = (x.vol / x.shrout)                       # same-month turnover; lagged below
    x["turn_l"] = g.turn.shift(1)
    x["vspike_l"] = g.turn.shift(1) / g.turn.transform(lambda s: s.shift(2).rolling(12, min_periods=9).mean())
    return x.dropna(subset=["ret", "w0"])

def sorts(x, var, q=5, start="1927-01-31"):
    out = {}
    for d, g in x[x.date >= start].groupby("date"):
        g = g.nlargest(500, "w0")
        mkt = np.average(g.ret, weights=g.w0)
        h = g.dropna(subset=[var])
        if len(h) < 250:
            continue
        b = pd.qcut(h[var].rank(method="first"), q, labels=False)
        r = {f"Q{k+1}": np.average(h.ret[b == k], weights=h.w0[b == k]) for k in range(q)}
        r["MKT"] = mkt; r["EW"] = g.ret.mean()
        out[d] = r
    return pd.DataFrame(out).T

def summarize(R, a, b):
    R = R.loc[a:b]
    if len(R) < 24:
        return None
    s = {}
    for c in R.columns:
        if c == "MKT":
            continue
        ex = R[c] - R.MKT
        s[c] = {"excess_vs_cap_weight_ann": float(ex.mean() * 12), "t": float(ex.mean() / (ex.std() / np.sqrt(len(ex)))), "months": len(ex)}
    hl = R.Q5 - R.Q1
    s["Q5_minus_Q1"] = {"excess_vs_cap_weight_ann": float(hl.mean() * 12), "t": float(hl.mean() / (hl.std() / np.sqrt(len(hl)))), "months": len(hl),
                        "worst_month": float(hl.min())}
    return s

if __name__ == "__main__":
    x = load()
    res = {}
    eras = [("1927-2015", "1927-01-31", "2015-12-31"), ("1927-1969", "1927-01-31", "1969-12-31"), ("1970-1995", "1970-01-31", "1995-12-31"),
            ("1996-2015", "1996-01-31", "2015-12-31"), ("2016-2024 IN-SAMPLE", "2016-01-31", "2024-12-31")]
    for var, label in [("mom12_1", "12-month winners (what has been bought up)"), ("rev1", "last month's winners (recent buying frenzy)"),
                       ("turn_l", "most-traded stocks (turnover)"), ("vspike_l", "trading-volume surge")]:
        R = sorts(x, var)
        res[var] = {"label": label, "eras": {n: summarize(R, a, b) for n, a, b in eras}}
        log_trial("CROWD_followup", var, "followup")
        print("\n###", label)
        for n, a, b in eras:
            s = summarize(R, a, b)
            if s:
                print(f"  {n:22s} top-fifth(Q5) {s['Q5']['excess_vs_cap_weight_ann']*100:+.2f}%/yr t={s['Q5']['t']:+.2f} | bottom(Q1) {s['Q1']['excess_vs_cap_weight_ann']*100:+.2f}% | Q5-Q1 {s['Q5_minus_Q1']['excess_vs_cap_weight_ann']*100:+.2f}% t={s['Q5_minus_Q1']['t']:+.2f} | EW {s['EW']['excess_vs_cap_weight_ann']*100:+.2f}% (n={s['Q5']['months']})")
    json.dump(res, open(REPORTS / "crowd_followup.json", "w"), indent=1, default=float)
