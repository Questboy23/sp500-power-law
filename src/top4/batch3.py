import json, numpy as np, pandas as pd
from config import PROC, REPORTS
from src.top4.tournament import score_all, FEATS, rk, ROUNDS
from src.trials import log_trial

def add_labels(P):
    P = P.copy(); P["dollar"] = P.cap * P.fwd_1y
    P["top4d"] = P.groupby("formation").dollar.transform(lambda s: (s >= s.quantile(0.96)).astype(float).where(s.notna()))
    tf = pd.read_parquet(PROC / "text_features.parquet"); tf["filed"] = pd.to_datetime(tf.filed)
    lv = []
    for t in sorted(P.formation.unique()):
        z = tf[(tf.filed <= t) & (tf.filed > t - pd.DateOffset(months=15))].sort_values("filed").groupby("permno").last().b4_rival_mentions_v2
        d = P[P.formation == t]; lv.append(pd.Series(d.permno.map(z).values, index=d.index))
    P["b4_level"] = pd.concat(lv)
    return P

def hyps():
    H = {}
    H["W01_size_only"] = lambda d: d.cap
    H["W02_size_momentum"] = lambda d: rk(d.cap) + rk(d.ret12_1)
    H["W03_size_quality"] = lambda d: rk(d.cap) + rk(d.qmj)
    H["W04_size_growth"] = lambda d: rk(d.cap) + rk(d.sale_gr1)
    H["W05_size_rd"] = lambda d: rk(d.cap) + rk(d.rd5_at)
    H["W06_size_near_high"] = lambda d: rk(d.cap) + rk(d.near_high12)
    H["W07_megacap_convexity"] = lambda d: (d.ret6 - d.ret6prev).where(d.cap.rank(ascending=False) <= 50)
    H["W08_dollar_momentum"] = lambda d: d.cap * d.ret12
    H["W09_large_rank_climber"] = lambda d: d.rank_climb.where(d.cap.rank(ascending=False) <= 200)
    H["W11_hub_firm"] = lambda d: rk(d.b4_level) + rk(d.cap)
    H["W12_size_low_vol"] = lambda d: rk(d.cap) - rk(d.vol36)
    return H

def ml_dollar(P):
    from sklearn.ensemble import HistGradientBoostingClassifier
    F = FEATS + ["qmj", "z_score", "o_score", "b4_level"]
    X = P[F].copy(); X["cap"] = np.log(X.cap); Xr = X.groupby(P.formation).rank(pct=True)
    s = pd.Series(np.nan, index=P.index)
    for t in sorted(P.formation.unique()):
        tr = (P.formation <= t - pd.DateOffset(years=1)) & P.top4d.notna()
        if P[tr].formation.nunique() < 4:
            continue
        cols = [c for c in Xr.columns if Xr.loc[tr, c].notna().sum() > 200]
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100, random_state=0)
        m.fit(Xr.loc[tr, cols].values, P.top4d[tr].values)
        te = P.formation == t
        s.loc[te] = m.predict_proba(Xr.loc[te, cols].values)[:, 1]
    return s

def evaluate(P, score, years):
    rows = []
    for y in years:
        t = pd.Timestamp(f"{y}-06-30")
        d = P[(P.formation == t) & P.fwd_1y.notna()]
        s = score.loc[d.index].dropna()
        if len(s) < 40:
            continue
        k = max(int(round(0.04 * len(d))), 5)
        picks = s.nlargest(k).index
        rows.append({"year": y, "k": k, "hits": float(d.top4d.loc[picks].sum()),
                     "picks_cw": float((d.cap.loc[picks] * d.fwd_1y.loc[picks]).sum() / d.cap.loc[picks].sum()),
                     "picks_ew": float(d.fwd_1y.loc[picks].mean()), "sp_cw": float((d.cap * d.fwd_1y).sum() / d.cap.sum()),
                     "sp_ew": float(d.fwd_1y.mean())})
    return pd.DataFrame(rows)

def summ(E):
    if not len(E):
        return None
    hr = E.hits / E.k; ex = E.picks_cw - E.sp_cw
    c = lambda r: float(np.prod(1 + r) ** (1 / len(r)) - 1)
    return {"years": len(E), "hits": int(E.hits.sum()), "picks": int(E.k.sum()), "hit_rate": float(E.hits.sum() / E.k.sum()),
            "lift": float(E.hits.sum() / E.k.sum() / 0.04), "t_hit": float((hr.mean() - 0.04) / (hr.std(ddof=1) / np.sqrt(len(hr)))) if hr.std() > 0 else np.nan,
            "picks_cw_cagr": c(E.picks_cw), "picks_ew_cagr": c(E.picks_ew), "sp_cagr": c(E.sp_cw), "sp_ew_cagr": c(E.sp_ew),
            "excess_mean": float(ex.mean()), "t_excess": float(ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex)))) if len(ex) > 2 else np.nan}

if __name__ == "__main__":
    P = add_labels(pd.read_parquet(PROC / "top4_panel.parquet"))
    sc = score_all(P, hyps(), ml=False); sc["W10_ml_dollar"] = ml_dollar(P)
    base_hr = summ(evaluate(P, sc["W01_size_only"], ROUNDS[1]))["hit_rate"]
    res = {}
    for h, s in sc.items():
        log_trial(h, "batch3", "top4")
        r1 = summ(evaluate(P, s, ROUNDS[1])); rec = {"round1": r1}
        p1 = bool(r1 and r1["lift"] > 1 and r1["t_hit"] >= 2 and r1["excess_mean"] > 0 and (h == "W01_size_only" or r1["hit_rate"] >= base_hr))
        rec["promoted_to_round2"] = p1
        r2 = summ(evaluate(P, s, ROUNDS[2])); r3 = summ(evaluate(P, s, ROUNDS[3]))
        rec["round2_shadow"] = r2; rec["round3_shadow"] = r3          # shown for every idea, labelled
        if p1:
            rec["promoted_to_round3"] = bool(r2 and r2["hit_rate"] > 0.04 and r2["excess_mean"] > 0)
        res[h] = rec
    json.dump(res, open(REPORTS / "top4_batch3.json", "w"), indent=1, default=float)
    pd.DataFrame(sc).assign(formation=P.formation, permno=P.permno).to_parquet(PROC / "top4_scores_batch3.parquet")
    f = lambda x: f"{x*100:+5.1f}%"
    print(f"{'hypothesis':26s} {'R1 $hits':>9s} {'hit%':>6s} {'t':>5s} {'picks':>7s} {'S&P':>7s} adv | R2 hit% picks S&P | R3 hit% picks S&P")
    for h, r in sorted(res.items(), key=lambda kv: -kv[1]["round1"]["hit_rate"]):
        a, b, c = r["round1"], r["round2_shadow"], r["round3_shadow"]
        adv = "YES" if r["promoted_to_round2"] else " no"
        adv2 = ("→R3" if r.get("promoted_to_round3") else "x") if r["promoted_to_round2"] else ""
        print(f"{h:26s} {a['hits']:>3d}/{a['picks']:<4d} {a['hit_rate']*100:5.1f}% {a['t_hit']:5.2f} {f(a['picks_cw_cagr'])} {f(a['sp_cagr'])} {adv} | {b['hit_rate']*100:4.1f}% {f(b['picks_cw_cagr'])} {f(b['sp_cagr'])} {adv2} | {c['hit_rate']*100:4.1f}% {f(c['picks_cw_cagr'])} {f(c['sp_cagr'])} ({c['years']}y)")
