"""Top-4% tournament engine. Hypotheses are score functions on one formation's cross-section."""
from __future__ import annotations
import json, numpy as np, pandas as pd
from scipy import stats
from config import PROC, REPORTS
from src.trials import log_trial

ROUNDS = {1: range(2000, 2015), 2: range(2015, 2020), 3: range(2020, 2026)}
rk = lambda s: s.rank(pct=True)


def batch1():
    H = {}
    H["Q01_vol_math"] = lambda d: d.vol36
    def tailprob(d):
        mu = d.mkt12 + 0.2 * (d.ret12_1 - d.mkt12); sig = d.vol36
        c = d.attrs.get("prev_cutoff", 0.6)
        return 1 - stats.norm.cdf((np.log1p(c) - np.log1p(mu.clip(lower=-0.9))) / sig)
    H["Q02_lognormal_tailprob"] = lambda d: pd.Series(tailprob(d), index=d.index)
    H["Q03_near_52w_high"] = lambda d: d.near_high12
    H["Q04_fresh_3y_high"] = lambda d: d.at_high36 + rk(d.ret12) * 0.01
    H["Q05_rank_climber"] = lambda d: d.rank_climb
    H["Q06_small_in_large"] = lambda d: -np.log(d.cap)
    H["Q07_fallen_angel"] = lambda d: d.dd36
    H["Q08_convexity"] = lambda d: d.ret6 - d.ret6prev
    H["Q09_positive_skew"] = lambda d: d.skew36
    H["Q10_max_day"] = lambda d: d.maxday
    H["Q11_high_beta"] = lambda d: d.beta36
    H["Q12_crash_rebound_beta"] = lambda d: d.beta36 * max(0.0, -float(d.mkt12.iloc[0])) if float(d.mkt12.iloc[0]) < 0 else d.beta36 * np.nan
    H["Q13_revenue_accel"] = lambda d: d.sale_gr1 - ((1 + d.sale_gr3) ** (1 / 3) - 1)
    H["Q14_operating_leverage"] = lambda d: rk(d.gp_at / d.at_turnover) * rk(d.sale_gr1)
    H["Q15_rd_capital"] = lambda d: d.rd5_at
    H["Q16_reinvestment_runway"] = lambda d: rk(d.capx_gr3) * rk(d.op_at)
    H["Q17_asset_light_growth"] = lambda d: rk(d.sale_gr1) - rk(d.at_gr1)
    H["Q18_productivity_jump"] = lambda d: d.sale_emp_gr1
    H["Q19_earnings_surprise"] = lambda d: d.niq_su
    H["Q20_analyst_disagreement"] = lambda d: d.eps_stdevest / d.eps_meanest.abs().replace(0, np.nan)
    H["Q21_garp"] = lambda d: rk(d.sale_gr3) + rk(d.sale_me)
    H["Q22_rule_of_40"] = lambda d: d.sale_gr1 + d.ebit_sale
    H["Q23_young"] = lambda d: -d.age
    H["Q24_text_transformer"] = lambda d: -d.a5
    H["Q25_rival_buzz"] = lambda d: d.b4_chg
    H["Q26_embedding_uniqueness"] = lambda d: d.emb_unique
    H["Q27_embedding_pivot"] = lambda d: d.emb_pivot
    H["Q28_looks_like_recent_winners"] = lambda d: d.sim_winners
    H["Q29_insider_buying"] = lambda d: d.a4
    H["Q30_shareholder_yield"] = lambda d: d.eqnpo_12m
    H["Q31_leader_gap"] = lambda d: d.leader_gap
    H["Q34_cheap_high_vol"] = lambda d: rk(d.vol36) + rk(d.be_me)
    return H


FEATS = ["vol36", "ret12_1", "ret12", "ret6", "ret6prev", "near_high12", "at_high36", "dd36", "skew36", "beta36", "mkt12",
         "rank_climb", "maxday", "leader_gap", "a5", "b4_chg", "emb_unique", "emb_pivot", "sim_winners", "a4", "cap",
         "sale_gr1", "sale_gr3", "rd5_at", "rd_sale", "capx_gr3", "op_at", "at_gr1", "sale_emp_gr1", "niq_su", "eps_stdevest",
         "eps_meanest", "sale_me", "ebit_sale", "age", "eqnpo_12m", "be_me", "gp_at", "at_turnover"]


def score_all(P, H, ml=True, ensemble_of=None):
    """Returns {hypothesis: Series(score) aligned to P.index}, computed formation by formation."""
    out = {h: pd.Series(np.nan, index=P.index) for h in H}
    years = sorted(P.formation.unique())
    cutoff = P.groupby("formation").fwd_1y.quantile(0.96)
    for t in years:
        d = P[P.formation == t].copy()
        prev = t - pd.DateOffset(years=1)
        d.attrs["prev_cutoff"] = float(cutoff.get(prev, 0.6)) if prev in cutoff.index else 0.6
        for h, f in H.items():
            try:
                s = f(d)
                out[h].loc[d.index] = pd.to_numeric(pd.Series(s, index=d.index), errors="coerce").replace([np.inf, -np.inf], np.nan)
            except Exception as e:      # missing columns in some years -> NaN
                pass
    if ensemble_of:
        rk_all = pd.concat([out[h].groupby(P.formation).rank(pct=True) for h in ensemble_of], axis=1)
        out["Q33_ensemble_rank"] = rk_all.mean(1).where(rk_all.notna().sum(1) >= 5)
    if ml:
        from sklearn.ensemble import HistGradientBoostingClassifier
        s = pd.Series(np.nan, index=P.index)
        X = P[FEATS].copy(); X["cap"] = np.log(X.cap)
        Xr = X.groupby(P.formation).rank(pct=True)
        for t in years:
            tr = P.formation <= t - pd.DateOffset(years=1)
            tr = tr & P.top4.notna()
            if P[tr].formation.nunique() < 4:
                continue
            m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100, random_state=0)
            cols = [c for c in Xr.columns if Xr.loc[tr, c].notna().sum() > 200]
            m.fit(Xr.loc[tr, cols].values, P.top4[tr].values)
            te = P.formation == t
            s.loc[te] = m.predict_proba(Xr.loc[te, cols].values)[:, 1]
        out["Q32_ml_top4"] = s
    return out


def evaluate(P, score, years):
    rows = []
    for y in years:
        t = pd.Timestamp(f"{y}-06-30")
        d = P[(P.formation == t) & P.fwd_1y.notna()]
        s = score.loc[d.index].dropna()
        if len(s) < 100:
            continue
        k = max(int(round(0.04 * len(d))), 5)
        picks = s.nlargest(k).index
        hits = float(d.top4.loc[picks].sum()); losers = float(d.bot4.loc[picks].sum())
        cw = float((d.cap * d.fwd_1y).sum() / d.cap.sum())
        rows.append({"year": y, "k": k, "hits": hits, "losers": losers, "hit_rate": hits / k, "picks_ret": float(d.fwd_1y.loc[picks].mean()),
                     "sp_cw": cw, "sp_ew": float(d.fwd_1y.mean()), "coverage": len(s) / len(d)})
    return pd.DataFrame(rows)


def summarize(E):
    if not len(E):
        return None
    hr = E.hit_rate; ex = E.picks_ret - E.sp_cw
    t_hr = (hr.mean() - 0.04) / (hr.std(ddof=1) / np.sqrt(len(hr))) if len(hr) > 2 and hr.std() > 0 else np.nan
    t_ex = ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex))) if len(ex) > 2 else np.nan
    cum = lambda r: float(np.prod(1 + r) ** (1 / len(r)) - 1)
    return {"years": len(E), "picks": int(E.k.sum()), "hits": int(E.hits.sum()), "losers": int(E.losers.sum()),
            "loser_rate": float(E.losers.sum() / E.k.sum()), "hit_rate": float(E.hits.sum() / E.k.sum()),
            "lift": float(E.hits.sum() / E.k.sum() / 0.04), "t_hit": float(t_hr), "picks_cagr": cum(E.picks_ret), "sp_cagr": cum(E.sp_cw),
            "sp_ew_cagr": cum(E.sp_ew), "excess_mean": float(ex.mean()), "t_excess": float(t_ex), "coverage": float(E.coverage.mean())}


def run(P, scores, label):
    res = {}
    for h, s in scores.items():
        log_trial(h, label, "top4")
        r1 = summarize(evaluate(P, s, ROUNDS[1]))
        rec = {"round1": r1}
        p1 = bool(r1 and r1["lift"] > 1 and r1["t_hit"] >= 2 and r1["excess_mean"] > 0)
        rec["promoted_to_round2"] = p1
        if p1:
            r2 = summarize(evaluate(P, s, ROUNDS[2])); rec["round2"] = r2
            p2 = bool(r2 and r2["hit_rate"] > 0.04 and r2["excess_mean"] > 0)
            rec["promoted_to_round3"] = p2
            if p2:
                rec["round3"] = summarize(evaluate(P, s, ROUNDS[3]))
        res[h] = rec
    return res


def show(res):
    print(f"{'hypothesis':32s} {'R1 hits':>8s} {'hit%':>6s} {'lift':>5s} {'t':>5s} {'picks/yr':>9s} {'S&P/yr':>7s}  R1→ | R2 hit% picks vs S&P | R3 hit% picks vs S&P")
    for h, r in sorted(res.items(), key=lambda kv: -(kv[1]["round1"] or {"lift": 0})["lift"]):
        a = r["round1"]
        if not a:
            print(f"{h:32s} insufficient coverage"); continue
        line = f"{h:32s} {a['hits']:>3d}/{a['picks']:<4d} {a['hit_rate']*100:5.1f}% {a['lift']:5.2f} {a['t_hit']:5.2f} {a['picks_cagr']*100:+8.1f}% {a['sp_cagr']*100:+6.1f}%  {'YES' if r['promoted_to_round2'] else ' no'}"
        if r.get("round2"):
            b = r["round2"]; line += f" | {b['hit_rate']*100:4.1f}% {b['picks_cagr']*100:+5.1f}% vs {b['sp_cagr']*100:+5.1f}% {'→' if r['promoted_to_round3'] else 'x'}"
        if r.get("round3"):
            c = r["round3"]; line += f" | {c['hit_rate']*100:4.1f}% {c['picks_cagr']*100:+5.1f}% vs {c['sp_cagr']*100:+5.1f}% (cov {c['coverage']:.0%}, {c['years']}y)"
        print(line)


if __name__ == "__main__":
    P = pd.read_parquet(PROC / "top4_panel.parquet")
    H = batch1()
    scores = score_all(P, H, ml=True, ensemble_of=[h for h in H])
    res = run(P, scores, "batch1")
    json.dump(res, open(REPORTS / "top4_batch1.json", "w"), indent=1, default=float)
    pd.DataFrame(scores).assign(formation=P.formation, permno=P.permno).to_parquet(PROC / "top4_scores_batch1.parquet")
    show(res)
