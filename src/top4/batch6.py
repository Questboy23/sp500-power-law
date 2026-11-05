import json, numpy as np, pandas as pd
from config import PROC, REPORTS
from src.top4.tournament import evaluate, summarize, ROUNDS, rk, FEATS
from src.trials import log_trial
from src.features.context import Context
P = pd.read_parquet(PROC / "top4_panel.parquet")
ctx = Context()
# regime features at each formation (past 12m, data <= t)
reg = {}
for t in sorted(P.formation.unique()):
    d = P[P.formation == t]
    top50 = d.nlargest(50, "cap"); ew = d.ret12.mean()
    t12 = t - pd.offsets.MonthEnd(12)
    c0 = ctx.C.loc[t12].reindex(d.permno) if t12 in ctx.C.index else None
    sh_now = d.cap.nlargest(10).sum() / d.cap.sum()
    sh_then = (c0.nlargest(10).sum() / c0.sum()) if c0 is not None else np.nan
    reg[t] = {"size_mom": float((top50.cap * top50.ret12).sum() / top50.cap.sum() - ew), "conc_chg": float(sh_now - sh_then),
              "mkt12": float(d.mkt12.iloc[0]), "disp": float(d.ret12.std())}
reg = pd.DataFrame(reg).T
w01 = lambda d: d.cap
y03 = lambda d: (rk(d.qmj) + rk(d.gp_at)).where(rk(d.vol36) >= 2 / 3)
sc = {h: pd.Series(np.nan, index=P.index) for h in ["Z01_size_momentum_switch", "Z02_concentration_trend_switch", "Z03_winner_size_echo"]}
for t in sorted(P.formation.unique()):
    d = P[P.formation == t]; r = reg.loc[t]
    sc["Z01_size_momentum_switch"].loc[d.index] = (rk(w01(d)) if r.size_mom > 0 else rk(y03(d))).values
    sc["Z02_concentration_trend_switch"].loc[d.index] = (rk(w01(d)) if r.conc_chg > 0 else rk(y03(d))).values
    prev = P[(P.formation == t - pd.DateOffset(years=1))]
    if len(prev):
        p = prev.cap.rank(pct=True)[prev.top4 == 1].median()
        sp = d.cap.rank(pct=True)
        sc["Z03_winner_size_echo"].loc[d.index] = d.vol36.where((sp - p).abs() <= 0.15).values
# Z04 ML with regime features
from sklearn.ensemble import HistGradientBoostingClassifier
X = P[FEATS].copy(); X["cap"] = np.log(X.cap); Xr = X.groupby(P.formation).rank(pct=True)
for c in reg.columns:
    Xr["R_" + c] = P.formation.map(reg[c]).values
s = pd.Series(np.nan, index=P.index)
for t in sorted(P.formation.unique()):
    tr = (P.formation <= t - pd.DateOffset(years=1)) & P.top4.notna()
    if P[tr].formation.nunique() < 4:
        continue
    cols = [c for c in Xr.columns if Xr.loc[tr, c].notna().sum() > 200]
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100, random_state=0).fit(Xr.loc[tr, cols].values, P.top4[tr].values)
    te = P.formation == t; s.loc[te] = m.predict_proba(Xr.loc[te, cols].values)[:, 1]
sc["Z04_ml_with_regime"] = s
res = {}; f = lambda x: f"{x*100:+5.1f}%"
print("regime calls (size_mom>0 → giants):", {int(t.year): ("giants" if v > 0 else "volatile-quality") for t, v in reg.size_mom.items()})
for h, x in sc.items():
    log_trial(h, "batch6", "top4")
    r = {k: summarize(evaluate(P, x, ROUNDS[k])) for k in (1, 2, 3)}
    res[h] = r
    print(f"{h:32s} " + " | ".join(f"R{k} {r[k]['hit_rate']*100:4.1f}% losers {r[k]['loser_rate']*100:4.1f}% picks {f(r[k]['picks_cagr'])} vs S&P {f(r[k]['sp_cagr'])} ({r[k]['years']}y)" for k in (1, 2, 3) if r[k]))
json.dump(res, open(REPORTS / "top4_batch6.json", "w"), indent=1, default=float)
