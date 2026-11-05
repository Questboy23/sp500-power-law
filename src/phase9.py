"""Phase 9: decay of predictive validity with training-to-prediction gap, and regime-similarity
vs recency vs equal weighting of training years. Pre-holdout only (targets <= 2014-06, so every
1-year label ends by 2015-06).

No angle passed Phase 7, so the decay curve is measured on (a) a returns-only cross-sectional
model available back to 1927 (momentum 12-1, 1-month reversal, 5y reversal [= P4 sign-flipped],
36m volatility, size) and (b) each of those signals alone. This answers the decay question with the
longest free history; it is not a claim that any of these beat the index."""
from __future__ import annotations
import json, numpy as np, pandas as pd
from scipy import stats
from config import REPORTS, PROC
from src.powerlaw.descriptive import Market
from src.evaluate import fit_decay
from src.trials import log_trial

FEATS = ["mom", "rev1", "rev60", "vol36", "size"]
LAGS = [0, 5, 10, 20, 40]


def build_panel(M: Market):
    rows = []
    for y in range(1929, 2015):
        t = pd.Timestamp(f"{y}-06-30")
        if t not in M.R.index:
            continue
        mem = M.members(t)
        R = M.R.loc[:t, [c for c in mem if c in M.R.columns]]
        if len(R) < 61:
            continue
        g = lambda k0, k1: (1 + R.iloc[len(R) - k0:len(R) - k1].fillna(0)).prod() - 1
        f = pd.DataFrame({"mom": g(12, 1), "rev1": -R.iloc[-1], "rev60": -g(60, 0),
                          "vol36": -R.iloc[-36:].std(), "size": -np.log(M.C.loc[t, R.columns])})
        f = f[R.iloc[-36:].notna().sum() >= 24]
        y1 = M.fwd(t, 12, f.index)
        if y1 is None:
            continue
        f = f.rank(pct=True) - 0.5
        f["y"] = y1.rank(pct=True) - 0.5
        f["year"] = y
        rows.append(f.dropna())
    return pd.concat(rows)


def regime_vectors(M: Market):
    x = pd.read_excel("data/raw/shiller_ie_data.xls", sheet_name="Data", skiprows=7)
    x = x[pd.to_numeric(x.Date, errors="coerce").notna()].copy()
    x["Date"] = x.Date.astype(float)
    x["year"] = x.Date.astype(int); x["mo"] = ((x.Date - x.year) * 100).round().astype(int)
    x = x[x.mo == 6].set_index("year")
    cpi = pd.to_numeric(x.CPI, errors="coerce")
    z = pd.read_csv(REPORTS / "p2_zipf.csv", parse_dates=["date"]); z["year"] = z.date.dt.year
    mkt = (M.R * M.C.shift(1)).sum(1) / M.C.shift(1).where(M.R.notna()).sum(1)
    vol = mkt.rolling(12).std() * np.sqrt(12)
    v = pd.DataFrame({"rate": pd.to_numeric(x["Rate GS10"], errors="coerce"), "infl": cpi.pct_change(),
                      "cape": pd.to_numeric(x.CAPE, errors="coerce")})
    v = v.join(z.set_index("year").top10_actual.rename("conc"))
    v["vol"] = [vol.get(pd.Timestamp(f"{y}-06-30"), np.nan) for y in v.index]
    return v.dropna()


def fit_predict(train, test, w=None):
    from sklearn.linear_model import Ridge
    m = Ridge(alpha=1.0).fit(train[FEATS], train.y, sample_weight=w)
    return stats.spearmanr(m.predict(test[FEATS]), test.y).statistic


def decay(P):
    years = sorted(P.year.unique())
    out = []
    for t in years:
        if t < 1950:
            continue
        test = P[P.year == t]
        for L in LAGS:
            end = t - 1 - L                       # last training formation (label ended by t)
            tr = P[(P.year <= end) & (P.year > end - 10)]
            if tr.year.nunique() < 8:
                continue
            out.append({"target": t, "lag": L, "ic": fit_predict(tr, test)})
            for f in FEATS:   # single signals: sign learned in W
                s = np.sign(stats.spearmanr(tr[f], tr.y).statistic)
                out.append({"target": t, "lag": L, "signal": f, "ic": s * stats.spearmanr(test[f], test.y).statistic})
    D = pd.DataFrame(out)
    D["signal"] = D.get("signal").fillna("MODEL") if "signal" in D else "MODEL"
    return D


def regime_test(P, V):
    Z = (V - V.mean()) / V.std()
    rows = []
    for t in sorted(P.year.unique()):
        if t < 1950 or t not in Z.index:
            continue
        test = P[P.year == t]; tr = P[P.year <= t - 1]
        yrs = tr.year
        eq = np.ones(len(tr))
        rec = 0.5 ** ((t - 1 - yrs.values) / 10.0)
        zz = Z.reindex(yrs.values)
        d = np.sqrt(((zz.values - Z.loc[t].values) ** 2).sum(1))
        reg = np.exp(-(d ** 2) / 2)
        reg = np.where(np.isfinite(reg), reg, 0)
        rows.append({"target": t, "equal": fit_predict(tr, test, eq), "recency": fit_predict(tr, test, rec),
                     "regime": fit_predict(tr, test, reg) if reg.sum() > 0 else np.nan})
    return pd.DataFrame(rows)


def main():
    M = Market()
    P = build_panel(M); P.to_parquet(PROC / "phase9_panel.parquet")
    D = decay(P); D.to_csv(REPORTS / "phase9_decay_raw.csv", index=False)
    log_trial("PHASE9_decay_model", "ridge5_W10", "9")
    res = {"n_targets": int(D.target.nunique()), "target_range": [int(D.target.min()), int(D.target.max())]}
    curves = {}
    for sig, g in D.groupby("signal"):
        # same-era control: only targets that have ALL lags (so each lag is averaged over the same years)
        full = g.groupby("target").lag.nunique(); same = full[full == len(LAGS)].index
        gg = g[g.target.isin(same)]
        m = gg.groupby("lag").ic.agg(["mean", "std", "count"])
        fit = fit_decay(gg.lag.values.astype(float), gg.ic.values)
        curves[sig] = {"ic_by_lag": m["mean"].round(4).to_dict(), "n_targets_same_era": int(len(same)),
                       "same_era_years": [int(same.min()), int(same.max())] if len(same) else None, **fit}
    res["curves"] = curves
    V = regime_vectors(M)
    RT = regime_test(P, V); RT.to_csv(REPORTS / "phase9_regime.csv", index=False)
    log_trial("PHASE9_weighting", "equal_recency_regime", "9")
    rt = {}
    for c in ("equal", "recency", "regime"):
        v = RT[c].dropna()
        rt[c] = {"mean_ic": float(v.mean()), "t": float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))), "n": len(v)}
    diff = (RT.regime - RT.recency).dropna()
    rt["regime_minus_recency"] = {"mean": float(diff.mean()), "t": float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))))}
    diff = (RT.regime - RT.equal).dropna()
    rt["regime_minus_equal"] = {"mean": float(diff.mean()), "t": float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))))}
    rt["winner"] = max(("equal", "recency", "regime"), key=lambda c: rt[c]["mean_ic"])
    res["weighting"] = rt
    json.dump(res, open(REPORTS / "phase9_results.json", "w"), indent=1, default=float)
    return res


if __name__ == "__main__":
    print(json.dumps(main(), indent=1, default=float))
