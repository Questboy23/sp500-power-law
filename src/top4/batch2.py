import json, numpy as np, pandas as pd
from scipy import stats
from config import PROC, REPORTS
from src.top4.tournament import score_all, run, show, FEATS, rk

def hyps():
    H = {}
    H["V01_up_down_vol_ratio"] = lambda d: d.up_semi / d.down_semi
    H["V02_convex_payoff"] = lambda d: d.beta_up - d.beta_down
    H["V03_vol_minus_distress"] = lambda d: rk(d.vol36) - rk(d.o_score) + rk(d.z_score)
    H["V04_vol_x_quality"] = lambda d: rk(d.vol36) + rk(d.qmj)
    H["V05_two_stage_tail"] = lambda d: d.gp_at.where(rk(d.vol36) >= 0.7)
    H["V06_past_tail_asymmetry"] = lambda d: d.tail_asym + rk(d.vol36) * 0.01
    def net(d):
        hi, lo = d.attrs.get("prev_hi", 0.6), d.attrs.get("prev_lo", -0.4)
        mu = d.mkt12 + 0.2 * (d.ret12_1 - d.mkt12) + 0.05 * (rk(d.gp_at).fillna(0.5) - 0.5)
        sig = d.vol36; m = np.log1p(mu.clip(lower=-0.9))
        return pd.Series((1 - stats.norm.cdf((np.log1p(hi) - m) / sig)) - stats.norm.cdf((np.log1p(lo) - m) / sig), index=d.index)
    H["V07_net_tail_prob"] = net
    H["V08_small_profitable"] = lambda d: rk(-np.log(d.cap)) + rk(d.op_at)
    H["V09_fallen_but_solvent"] = lambda d: rk(d.dd36) + rk(d.z_score)
    H["V10_leader_gap_quality"] = lambda d: rk(d.leader_gap) + rk(d.qmj)
    H["V11_convexity_quality"] = lambda d: rk(d.ret6 - d.ret6prev) + rk(d.gp_at)
    H["V14_vol_scaled_momentum"] = lambda d: d.ret12_1 / d.vol36
    H["V15_small_value"] = lambda d: rk(-np.log(d.cap)) + rk(d.be_me)
    H["V16_bold_within_industry"] = lambda d: d.vol36 / d.vol36.groupby(d.sic2).transform("median")
    H["V17_vol_without_lottery"] = lambda d: rk(d.vol36) - rk(d.maxday)
    H["V18_innovator_on_sale"] = lambda d: rk(d.rd5_at) + rk(d.dd36)
    return H

def ml_two_sided(P):
    from sklearn.ensemble import HistGradientBoostingClassifier
    F = FEATS + ["up_semi", "down_semi", "beta_up", "beta_down", "tail_asym", "qmj", "z_score", "o_score"]
    X = P[F].copy(); X["cap"] = np.log(X.cap); Xr = X.groupby(P.formation).rank(pct=True)
    y = (P.top4.fillna(0) - P.bot4.fillna(0)).astype(int) + 1        # 0=bottom, 1=middle, 2=top
    s = pd.Series(np.nan, index=P.index)
    for t in sorted(P.formation.unique()):
        tr = (P.formation <= t - pd.DateOffset(years=1)) & P.top4.notna()
        if P[tr].formation.nunique() < 4:
            continue
        cols = [c for c in Xr.columns if Xr.loc[tr, c].notna().sum() > 200]
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100, random_state=0)
        m.fit(Xr.loc[tr, cols].values, y[tr].values)
        te = P.formation == t
        pr = m.predict_proba(Xr.loc[te, cols].values)
        s.loc[te] = pr[:, 2] - pr[:, 0]
    return s

if __name__ == "__main__":
    P = pd.read_parquet(PROC / "top4_panel.parquet")
    # last year's realized cutoffs for V07 (labels of formation t-1y ended at t)
    hi = P.groupby("formation").fwd_1y.quantile(0.96); lo = P.groupby("formation").fwd_1y.quantile(0.04)
    import src.top4.tournament as T
    orig = T.score_all
    scores = {}
    for t in sorted(P.formation.unique()):
        pass
    H = hyps()
    # inject cutoffs via attrs: wrap each function
    def with_cut(f):
        def g(d):
            prev = d.formation.iloc[0] - pd.DateOffset(years=1)
            d.attrs["prev_hi"] = float(hi.get(prev, 0.6)); d.attrs["prev_lo"] = float(lo.get(prev, -0.4))
            return f(d)
        return g
    H = {k: with_cut(v) for k, v in H.items()}
    scores = score_all(P, H, ml=False)
    scores["V13_ml_two_sided"] = ml_two_sided(P)
    b1 = pd.read_parquet(PROC / "top4_scores_batch1.parquet")
    surv = [h for h, r in json.load(open(REPORTS / "top4_batch1.json")).items() if r["promoted_to_round2"]]
    rr = pd.concat([b1[h].groupby(b1.formation).rank(pct=True) for h in surv], axis=1)
    scores["V12_ensemble_round1_survivors"] = pd.Series(rr.mean(1).values, index=P.index)
    res = run(P, scores, "batch2")
    json.dump(res, open(REPORTS / "top4_batch2.json", "w"), indent=1, default=float)
    pd.DataFrame(scores).assign(formation=P.formation, permno=P.permno).to_parquet(PROC / "top4_scores_batch2.parquet")
    show(res)
    print("\nloser rate (share of picks in the bottom 4%), round 1:")
    for h, r in res.items():
        if r["round1"]:
            print(f"  {h:32s} winners {r['round1']['hit_rate']*100:4.1f}%  losers {r['round1']['loser_rate']*100:4.1f}%")
