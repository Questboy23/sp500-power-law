"""Round 4 (angles.yaml: round4): stochastic-portfolio-theory diversity weighting, and an ML model on
characteristics inside the S&P 500."""
import json, numpy as np, pandas as pd
from config import PROC, REPORTS
from src.trials import log_trial, count_trials
from src.evaluate import deflated_sharpe

def tst(x):
    x = pd.Series(x).dropna(); return float(x.mean() / (x.std() / np.sqrt(len(x)))) if len(x) > 2 else np.nan

# ---------------- U1/U2: diversity weighting ----------------
def diversity():
    x = pd.read_parquet(PROC / "market_long.parquet")[["permno", "date", "ret", "mcap"]].sort_values(["permno", "date"])
    p = pd.read_parquet(PROC / "panel_monthly.parquet"); x = pd.concat([x, p[p.date > "2024-12-31"][["permno", "date", "ret", "mcap"]]])
    x = x.sort_values(["permno", "date"]); x["w0"] = x.groupby("permno").mcap.shift(1); x = x.dropna(subset=["ret", "w0"])
    out = {}; prev = {0.5: None, 0.8: None}; tos = {0.5: [], 0.8: []}
    for d, g in x.groupby("date"):
        g = g.nlargest(500, "w0").set_index("permno")
        cw = g.w0 / g.w0.sum(); r = g.ret
        row = {"CW": float((cw * r).sum())}
        for pw in (0.5, 0.8):
            w = g.w0 ** pw; w = w / w.sum()
            row[f"D{pw}"] = float((w * r).sum())
            # turnover vs drifted previous weights
            if prev[pw] is not None:
                idx = w.index.union(prev[pw].index)
                tos[pw].append(float((w.reindex(idx, fill_value=0) - prev[pw].reindex(idx, fill_value=0)).abs().sum() / 2))
            dr = w * (1 + r); prev[pw] = dr / dr.sum()
        out[d] = row
    R = pd.DataFrame(out).T
    res = {}
    for pw in (0.5, 0.8):
        to = float(np.mean(tos[pw]))
        act = R[f"D{pw}"] - R.CW - to * 2 * 10 / 1e4       # 10 bps on each side of traded weight
        r = {"monthly_one_way_turnover": to}
        for name, a, b in [("1927-2015", "1927", "2015"), ("1927-1969", "1927", "1969"), ("1970-1995", "1970", "1995"),
                           ("1996-2015", "1996", "2015"), ("2016-2026 (seen era)", "2016", "2026")]:
            s = act.loc[a:b]; r[name] = {"excess_ann_net": float(s.mean() * 12), "t": tst(s), "months": len(s)}
        r["dsr"] = deflated_sharpe(act.loc["1927":"2015"], count_trials())
        signs = {np.sign(r[e]["excess_ann_net"]) for e in ("1927-1969", "1970-1995", "1996-2015")}
        r["pass"] = bool(r["1927-2015"]["t"] >= 3 and r["dsr"] >= 0.95 and signs == {1.0})
        res[f"U{1 if pw == 0.5 else 2}_diversity_p{str(pw).replace('.', '')}"] = r
    return res, R

# ---------------- U5: ML inside the S&P ----------------
def ml(kind):
    from src.features.context import Context, FIAM_COLS
    from src.backtest import Engine
    ctx = Context(); E = Engine(end="2023-12-31")
    F = ctx.F.copy()
    F["market_equity"] = np.log(F.market_equity.where(F.market_equity > 0))
    months = [pd.Timestamp(m) for m in E.months if pd.Timestamp(m) >= pd.Timestamp("2000-02-29")]
    X, Y = {}, {}
    for m in months:
        f = m - pd.offsets.MonthEnd(1)                       # characteristics known at prior month end
        mem = E.B.columns[E.B.loc[m].notna()]
        try:
            xf = F.loc[f].reindex(mem)
        except KeyError:
            continue
        xf = xf.rank(pct=True) - 0.5                         # cross-sectional ranks
        y = E.R.loc[m].reindex(mem).rank(pct=True) - 0.5
        X[f] = xf; Y[f] = y
    forms = sorted(X)
    sig = {}
    model = None
    for f in forms:
        if f.month == 6 or model is None:
            tr = [g for g in forms if g + pd.offsets.MonthEnd(1) <= f]      # label month already over
            if len(tr) < 48:
                continue
            Xt = pd.concat([X[g] for g in tr]); yt = pd.concat([Y[g] for g in tr])
            ok = yt.notna(); Xt, yt = Xt[ok], yt[ok]
            if kind == "boost":
                from sklearn.ensemble import HistGradientBoostingRegressor
                model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=500, random_state=0)
                model.fit(Xt.values, yt.values)
            else:
                from sklearn.linear_model import Ridge
                model = Ridge(alpha=100.0).fit(Xt.fillna(0).values, yt.values)
        if model is None:
            continue
        xv = X[f].values if kind == "boost" else X[f].fillna(0).values
        p = pd.Series(model.predict(xv), index=X[f].index)
        sig[f] = (p.rank(pct=True) - 0.5) * 2
    from src.evaluate import rank_ic
    ics = [rank_ic(sig[f], Y[f]) for f in sig if f in Y]
    res_e = E.run(sig)
    act = (res_e["net"] - E.base0).dropna()
    act = act[act.index > min(sig)]
    out = {"first_signal": str(min(sig).date()), "monthly_rank_ic_mean": float(np.nanmean(ics)), "ic_t": tst(ics),
           "turnover_ann": float(res_e["turnover"].reindex(act.index).mean() * 12)}
    for name, a, b in [("2004-2015 (walk-forward)", "2004", "2015"), ("2016-2023 (model never saw; era seen by researcher)", "2016", "2023")]:
        s = act.loc[a:b]; out[name] = {"excess_ann_net": float(s.mean() * 12), "te": float(s.std() * np.sqrt(12)), "t": tst(s), "months": len(s)}
    out["dsr"] = deflated_sharpe(act.loc["2004":"2015"], count_trials())
    out["pass"] = bool(out["2004-2015 (walk-forward)"]["t"] >= 3 and out["dsr"] >= 0.95)
    if kind == "boost":
        try:
            from sklearn.inspection import permutation_importance
        except Exception:
            pass
    return out

if __name__ == "__main__":
    for k in ("U1_diversity_p05", "U2_diversity_p08", "U5_ml_boosting", "U5b_ml_ridge"):
        log_trial(k, "round4", "round4")
    res, R = diversity()
    res["U5_ml_boosting"] = ml("boost")
    res["U5b_ml_ridge"] = ml("ridge")
    res["_n_trials"] = count_trials()
    json.dump(res, open(REPORTS / "round4_results.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=lambda v: round(float(v), 4)))
