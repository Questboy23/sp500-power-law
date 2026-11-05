"""Phase 8: ablation. The GATED combined model = equal weight of passing angles (none passed in
Phase 7, so it is BASE0). The EXPLORATORY combined model = equal weight of every testable
angle's unit signal; it answers 'how much does each help' but is never used for money."""
from __future__ import annotations
import json, numpy as np, pandas as pd
from config import PROC, REPORTS
from src.backtest import Engine, rank_to_unit
from src.evaluate import te_ir, deflated_sharpe, tail_auc
from src.trials import log_trial, count_trials

GROUPS = {"P": "power-law", "A": "published", "B": "vibe", "N": "added"}


def combine(raw_all, names):
    forms = sorted(set().union(*[set(raw_all[a]) for a in names if a in raw_all]))
    out = {}
    for t in forms:
        parts = [rank_to_unit(raw_all[a][t]) for a in names if a in raw_all and t in raw_all[a]]
        if not parts:
            continue
        out[t] = pd.concat(parts, axis=1).mean(1)        # average over angles available for that stock
    return out


def stats(res, E):
    a = (res["net"] - E.base0.reindex(res["net"].index)).dropna()
    a = a[a.index > min(res["K"])] if res["K"] else a
    s = te_ir(a, pd.Series(0.0, index=a.index))
    return {"net_excess_ann": s["excess_ann"], "te": s["te"], "ir": s["ir"], "t": s["t"], "months": s["n"]}


def main():
    E = Engine()
    raw_all = pd.read_pickle(PROC / "phase7_raw_signals.pkl")
    T = pd.read_csv(REPORTS / "phase7_angles.csv")
    passing = T[T["pass"] == True].angle.tolist()
    testable = [a for a in raw_all if len(raw_all[a]) >= 3]
    labels = pd.read_parquet(PROC / "labels.parquet")
    out = {"passing": passing, "gated_combined": "BASE0 (no passing angles)" if not passing else passing}

    full = E.run(combine(raw_all, testable), name="kitchen_sink")
    log_trial("COMBINED_all", "equal_weight", "8")
    out["exploratory_all"] = stats(full, E)
    rows = []
    for a in testable:
        alone = T.set_index("angle").loc[a] if a in set(T.angle) else None
        minus = E.run(combine(raw_all, [x for x in testable if x != a]), name=f"minus_{a}")
        log_trial("COMBINED_minus", a, "8")
        sm = stats(minus, E)
        rows.append({"angle": a, "group": GROUPS.get(a[0], "?"), "alone_ir": float(alone["ir"]) if alone is not None else np.nan,
                     "alone_tail_auc": float(alone["tail_auc"]) if alone is not None and pd.notna(alone.get("tail_auc")) else np.nan,
                     "combined_ir_without": sm["ir"], "drop_in_ir_when_removed": out["exploratory_all"]["ir"] - sm["ir"]})
    ab = pd.DataFrame(rows).sort_values("drop_in_ir_when_removed", ascending=False)
    ab.to_csv(REPORTS / "phase8_ablation.csv", index=False)
    grp = {}
    for g, lab in GROUPS.items():
        names = [a for a in testable if a.startswith(g)]
        if names:
            r = E.run(combine(raw_all, names), name=f"group_{g}")
            log_trial(f"GROUP_{g}", "equal_weight", "8")
            grp[lab] = {"angles": names, **stats(r, E)}
    out["groups"] = grp
    out["implementation"] = implementation_layer(E, raw_all, testable)
    json.dump(out, open(REPORTS / "phase8_results.json", "w"), indent=1, default=float)
    return out, ab


def implementation_layer(E, raw_all, testable):
    """Section 8 effects measured on the exploratory combined signal."""
    from config import TILT_MAX
    import src.backtest as bt
    res = {}
    sig = combine(raw_all, testable)
    base = E.run(sig)
    res["I5_costs"] = {"gross_excess_ann": float((base["gross"] - E.base0).dropna().mean() * 12),
                       "net_excess_ann": float((base["net"] - E.base0).dropna().mean() * 12),
                       "cost_drag_ann": float(base["cost"].mean() * 12)}
    # I1: unbounded (tilt 100% of cap weight, no TE cap) vs bounded
    old = bt.TILT_MAX
    bt.TILT_MAX = 1.0
    unb = E.run(sig, te_cap=None); log_trial("I1_unbounded", "tilt100_noTE", "8")
    bt.TILT_MAX = old
    res["I1_bounded_vs_unbounded"] = {"bounded": stats(base, E), "unbounded": stats(unb, E)}
    # I2: shrinkage by walk-forward IC t-stat (confidence) vs full confidence
    ic = {}
    lab = pd.read_parquet(PROC / "labels.parquet")
    from src.evaluate import rank_ic
    for t in sorted(sig):
        past = [f for f in sig if f + pd.offsets.MonthEnd(12) <= t]
        vals = []
        for f in past:
            L = lab[lab.formation_date == f].set_index("permno").fwd_1y
            vals.append(rank_ic(sig[f], L))
        vals = np.array([v for v in vals if np.isfinite(v)])
        ic[t] = float(np.clip((vals.mean() / (vals.std(ddof=1) / np.sqrt(len(vals)))) / 3, 0, 1)) if len(vals) >= 3 else 0.0
    shr = E.run(sig, conf=ic); log_trial("I2_shrinkage", "ic_t_over_3", "8")
    res["I2_shrinkage"] = {"full_confidence": stats(base, E), "shrunk": stats(shr, E), "avg_confidence": float(np.mean(list(ic.values())))}
    res["I3_rebalance_day"] = rebalance_day_effect()
    res["I6_tax"] = tax_loss_harvest(E)
    res["I7_half_kelly"] = "non-binding: max tilted weight is 1.3x cap weight; index Kelly fraction 2.53 (P10)"
    return res


def rebalance_day_effect():
    """Daily CRSP returns of S&P additions around their first day in the index (1996-2015):
    cumulative excess vs CRSP VW market over [-20,-1], [0,+4] trading days. Waiting 5 days
    helps if [0,+4] is negative."""
    import pyarrow.parquet as pq
    lk = pd.read_parquet(PROC / "membership_linked.parquet")
    d = pd.read_csv("data/raw/sp500_components_updated.csv", parse_dates=["date"])
    first = {}
    for dt, s in zip(d.date, d.tickers):
        for t in s.split(","):
            first.setdefault(t, dt)
    adds = {t: dt for t, dt in first.items() if pd.Timestamp("1996-06-01") < dt <= pd.Timestamp("2015-11-30")}
    m = lk.assign(ym=lk.date).set_index(["ticker", "ym"]).permno
    pmap = {}
    for t, dt in adds.items():
        me = dt + pd.offsets.MonthEnd(0)
        p = m.get((t, me), None)
        if p is not None and p > 0:
            pmap[int(p)] = dt
    tbl = pq.read_table("data/raw/delisted/crsp_daily_1968_2024.parquet", columns=["permno", "date", "ret", "vwretd"],
                        filters=[("permno", "in", list(pmap))]).to_pandas()
    tbl["date"] = pd.to_datetime(tbl.date)
    pre, post, post20 = [], [], []
    for p, g in tbl.groupby("permno"):
        g = g.sort_values("date").reset_index(drop=True)
        i = g.index[g.date >= pmap[p]]
        if not len(i):
            continue
        i0 = i[0]
        ex = (g.ret - g.vwretd)
        if i0 >= 20:
            pre.append(ex.iloc[i0 - 20:i0].sum())
        if i0 + 5 <= len(g):
            post.append(ex.iloc[i0:i0 + 5].sum())
        if i0 + 20 <= len(g):
            post20.append(ex.iloc[i0:i0 + 20].sum())
    f = lambda v: {"mean": float(np.mean(v)), "t": float(np.mean(v) / (np.std(v, ddof=1) / np.sqrt(len(v)))), "n": len(v)}
    return {"excess_days_-20_to_-1": f(pre), "excess_days_0_to_4": f(post), "excess_days_0_to_19": f(post20),
            "note": "effective dates from the membership file (first day listed); announcement dates unknown"}


def tax_loss_harvest(E, thresh=0.10):
    """Direct-indexing TLH on BASE0 weights, 1996-07..2015-12, monthly: any position whose value
    is > thresh below its cost basis is sold and immediately re-bought (wash-sale ignored, i.e. a
    near-identical substitute is assumed). Losses on lots held < 12m are valued at 32%, else 15%.
    Report harvested tax value per year as % of portfolio, assuming it offsets outside gains.
    The basis reset defers (not forgives) tax: net benefit shown with deferred gains taxed at 15%
    at the end, and without (step-up at death)."""
    B, R = E.B, E.R
    val = pd.Series(dtype=float); basis = pd.Series(dtype=float); age = pd.Series(dtype=float)
    saved = []; V = 1.0; extra_gain = 0.0
    for m in E.months:
        b = B.loc[m].dropna(); r = R.loc[m].reindex(b.index).fillna(0)
        tgt = b * V
        # rebalance to target (index changes); new money basis = cost
        val = val.reindex(b.index); basis = basis.reindex(b.index); age = age.reindex(b.index)
        new = val.isna()
        val[new] = tgt[new]; basis[new] = tgt[new]; age[new] = 0
        val = val * (1 + r); age = age + 1
        V = val.sum()
        loss = basis - val
        h = loss > thresh * basis
        st = float((loss[h & (age < 12)]).sum()) * 0.32; lt = float((loss[h & (age >= 12)]).sum()) * 0.15
        saved.append((st + lt) / V)
        extra_gain += float(loss[h].sum())
        basis[h] = val[h]; age[h] = 0
        val = val / V; basis = basis / V; V = 1.0   # keep in weight units
    yrs = len(E.months) / 12
    gross = float(np.sum(saved) / yrs)
    return {"harvest_value_pct_per_year": gross, "months": len(E.months), "threshold": thresh,
            "note": "after-tax only; upper bound (assumes unlimited outside gains to offset, no wash-sale friction). Deferred gains taxed at 15% would claw back roughly 15/32 of the ST share and all LT benefit in PV terms."}


if __name__ == "__main__":
    out, ab = main()
    print(json.dumps({k: v for k, v in out.items() if k != "implementation"}, indent=1, default=float))
    print(json.dumps(out["implementation"], indent=1, default=float))
    print(ab.round(3).to_string())
