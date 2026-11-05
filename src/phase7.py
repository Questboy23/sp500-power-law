"""Phase 7: test every angle individually on pre-holdout data (formations <= 2015-06, returns
<= 2015-12). Every run is logged to trials.csv. Also computes the baselines."""
from __future__ import annotations
import json, numpy as np, pandas as pd
from scipy import stats
from config import PROC, REPORTS, T_STAT_BAR, DSR_BAR, CONSISTENCY_BAR, HOLDOUT_START
from src.backtest import Engine, rank_to_unit, base1_equal_weight, base2_total_market, factor_signal, lowvol_signal
from src.evaluate import rank_ic, tail_auc, te_ir, deflated_sharpe, factor_alpha, sharpe
from src.factors import ff_monthly
from src.features.context import Context
from src.features import ANGLES, get
from src.trials import log_trial, count_trials
from src.holdout_guard import assert_pre_holdout

LAST_FORM = pd.Timestamp("2015-06-30")
WINDOWS = [("2001-01-01", "2005-12-31"), ("2006-01-01", "2010-12-31"), ("2011-01-01", "2015-12-31")]


def formations(first=1996, last=2015):
    return [pd.Timestamp(f"{y}-06-30") for y in range(first, last + 1)]


def signals_for(ctx, fn, forms):
    out = {}
    for t in forms:
        s = fn(ctx, t)
        if s is not None and len(s) >= 50:
            out[t] = s.astype(float)
    return out


def ic_auc(raw: dict, labels: pd.DataFrame):
    ics, aucs = [], []
    for t, s in raw.items():
        L = labels[labels.formation_date == t].set_index("permno")
        if t + pd.offsets.MonthEnd(12) < HOLDOUT_START:
            ics.append(rank_ic(s, L.fwd_1y))
        if t + pd.offsets.MonthEnd(60) < HOLDOUT_START:
            aucs.append(tail_auc(s, L["tail"]))
    ics = np.array([x for x in ics if np.isfinite(x)]); aucs = np.array([x for x in aucs if np.isfinite(x)])
    t_ic = ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics))) if len(ics) > 2 else np.nan
    return {"ic_mean": ics.mean() if len(ics) else np.nan, "ic_t": t_ic, "n_ic": len(ics),
            "tail_auc": aucs.mean() if len(aucs) else np.nan, "n_auc": len(aucs)}


def evaluate(res, E, ff, base3_active=None, n_trials=1):
    net = res["net"]; act = (net - E.base0.reindex(net.index)).dropna()
    first = min(res["K"]) if res["K"] else None
    if first is not None:
        act = act[act.index > first]
    st = te_ir(act + 0, pd.Series(0.0, index=act.index))
    fa = factor_alpha(act, ff.drop(columns=["RF"]).reindex(act.index))
    fa3 = {}
    if base3_active is not None:
        X = ff.drop(columns=["RF"]).reindex(act.index).assign(BASE3=base3_active.reindex(act.index))
        fa3 = factor_alpha(act, X)
    yearly = act.groupby(act.index.year).apply(lambda r: np.prod(1 + r) - 1)
    wins = []
    for a, b in WINDOWS:
        w = act[(act.index >= a) & (act.index <= b)]
        if len(w) >= 24:
            wins.append(np.prod(1 + w) - 1)
    streak = cur = 0
    for v in yearly.values:
        cur = cur + 1 if v < 0 else 0; streak = max(streak, cur)
    return {"months": len(act), "net_excess_ann": st["excess_ann"], "te": st["te"], "ir": st["ir"], "t": st["t"],
            "gross_excess_ann": float((res["gross"] - E.base0.reindex(res["gross"].index)).reindex(act.index).mean() * 12),
            "cost_ann": float(res["cost"].reindex(act.index).mean() * 12),
            "turnover_ann": float(res["turnover"].reindex(act.index).mean() * 12),
            "alpha_ff6_ann": fa.get("alpha_ann"), "alpha_ff6_t": fa.get("alpha_t"),
            "alpha_ff6_base3_ann": fa3.get("alpha_ann"), "alpha_ff6_base3_t": fa3.get("alpha_t"),
            "dsr": deflated_sharpe(act, n_trials), "n_windows": len(wins),
            "win_share_windows": float(np.mean([w > 0 for w in wins])) if wins else np.nan,
            "window_returns": [round(float(w), 4) for w in wins], "worst_window": float(min(wins)) if wins else np.nan,
            "win_share_years": float((yearly > 0).mean()), "longest_losing_years": int(streak),
            "mean_te_scale": float(res["scales"].mean()) if len(res["scales"]) else np.nan}


def gate(m):
    return bool(np.isfinite(m["t"]) and m["t"] >= T_STAT_BAR and np.isfinite(m["dsr"]) and m["dsr"] >= DSR_BAR
                and np.isfinite(m["win_share_windows"]) and m["win_share_windows"] >= CONSISTENCY_BAR
                and m["alpha_ff6_t"] is not None and np.isfinite(m["alpha_ff6_t"]) and m["alpha_ff6_t"] >= 2.0)


# ---------------- trained angles ----------------
P6_FEATS = ["sale_gr3", "gp_at", "rd_sale", "be_me", "ret_12_1", "at_gr1", "age", "qmj", "o_score", "debt_me",
            "rvol_21d", "market_equity", "op_at", "capx_gr3", "ebit_sale", "cash_at"]


def _xy(ctx, labels, t, feats, target):
    X = ctx.fiam(t, feats).reindex(ctx.sp(t)).apply(pd.to_numeric, errors="coerce")
    if "market_equity" in X:
        X["market_equity"] = np.log(X.market_equity.where(X.market_equity > 0))
    L = labels[labels.formation_date == t].set_index("permno")
    return X, L[target].reindex(X.index) if target in L else None


def trained_signals(ctx, labels, kind: str, forms):
    """Walk-forward: for test formation t, train on formations f with f + 60m <= t (label ended)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    feats = P6_FEATS if kind == "P6" else ["debt_me", "o_score", "z_score", "age", "rvol_21d", "netdebt_me"]
    target = "tail" if kind == "P6" else "death5"
    out = {}
    hype = get("P9_hype_composite")
    for t in forms:
        train_f = [f for f in formations(2000, 2026) if f + pd.offsets.MonthEnd(60) <= t]
        if len(train_f) < 3:
            continue
        tr = labels[labels.formation_date.isin(train_f)]
        assert_pre_holdout(tr[["formation_date", "label_end"]].rename(columns={"label_end": "date"}), f"{kind} training")
        Xs, ys = [], []
        for f in train_f:
            X, y = _xy(ctx, labels, f, feats, target)
            if y is None:
                continue
            ok = y.notna(); Xs.append(X[ok]); ys.append(y[ok])
        X = pd.concat(Xs); y = pd.concat(ys)
        if y.sum() < 10:
            continue
        mdl = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        mdl.fit(X.values, y.values)
        Xt, _ = _xy(ctx, labels, t, feats, target)
        if Xt.notna().sum().sum() == 0:      # no FIAM characteristics at t (FIAM ends 2023-12)
            continue
        p = pd.Series(mdl.predict_proba(Xt.values)[:, 1], index=Xt.index)
        if kind == "P6":
            h = hype.signal(ctx, t)   # already = -hype
            out[t] = (p.rank(pct=True) + h.reindex(p.index).rank(pct=True).fillna(0.5)) - 1.0   # rank(P) - rank(hype)
        else:
            cut = p.quantile(0.9)
            out[t] = (-(p >= cut).astype(float))   # underweight top hazard decile only
    return out


def death_labels(labels):
    """death5 = distress exit within 5y of formation (labels.distress_flag + the exit date)."""
    from src.powerlaw.descriptive import Market
    return labels


def main():
    ctx = Context(); E = Engine()
    ff = ff_monthly()
    labels = pd.read_parquet(PROC / "labels.parquet")
    labels["label_end"] = pd.to_datetime(labels.label_end)
    # death label: distressed permno whose final month falls within 5y after formation
    u = pd.read_parquet(PROC / "crsp_monthly_universe.parquet", columns=["permno", "date"])
    lastd = u.groupby("permno").date.max()
    dist = set(labels[labels.distress_flag].permno)
    labels["death5"] = [float(p in dist and f < lastd.get(p, pd.Timestamp("2100-01-01")) <= f + pd.offsets.MonthEnd(60))
                        for p, f in zip(labels.permno, labels.formation_date)]
    forms = formations(1996, 2015)
    results = {}

    # ---------- baselines ----------
    base = {"BASE0": E.base0, "BASE1": base1_equal_weight(E), "BASE2": base2_total_market(E)}
    b3 = E.run({t: rank_to_unit(s) for t, s in signals_for(ctx, factor_signal, forms).items()}, name="BASE3")
    b4 = E.run({t: rank_to_unit(s) for t, s in signals_for(ctx, lowvol_signal, forms).items()}, name="BASE4")
    base["BASE3"] = b3["net"]; base["BASE4"] = b4["net"]
    base3_active = b3["net"] - E.base0
    bl = {}
    for k, r in base.items():
        r = r.dropna(); r = r[r.index >= "1996-07-31"]
        a = (r - E.base0.reindex(r.index)).dropna()
        bl[k] = {"cagr": float(np.prod(1 + r) ** (12 / len(r)) - 1), "vol": float(r.std() * np.sqrt(12)),
                 "sharpe": sharpe(r - ff.RF.reindex(r.index)), "excess_vs_base0_ann": float(a.mean() * 12),
                 "te": float(a.std() * np.sqrt(12)), "months": len(r)}
        log_trial(k, "baseline", "7", "1996-07..2015-12", sharpe=bl[k]["sharpe"])

    # ---------- simple angles ----------
    raw_all = {}
    for a in ANGLES:
        raw = signals_for(ctx, get(a).signal, forms)
        raw_all[a] = raw
        if len(raw) < 3:
            results[a] = {"status": "insufficient data", "n_formations": len(raw)}
            log_trial(a, "v1", "7", notes="insufficient data")
            continue
        res = E.run({t: rank_to_unit(s) for t, s in raw.items()}, name=a)
        results[a] = {"n_formations": len(raw), "first_formation": str(min(raw).date()), **ic_auc(raw, labels), "_res": res}

    # ---------- trained ----------
    for kind, aid in (("P6", "P6_tail_classifier"), ("P8", "P8_survival")):
        raw = trained_signals(ctx, labels, kind, formations(2000, 2015))
        raw_all[aid] = raw
        if len(raw) < 3:
            results[aid] = {"status": "insufficient data", "n_formations": len(raw)}; log_trial(aid, "v1", "7", notes="insufficient"); continue
        res = E.run({t: s.clip(-1, 1) for t, s in raw.items()}, name=aid)
        results[aid] = {"n_formations": len(raw), "first_formation": str(min(raw).date()), **ic_auc(raw, labels), "_res": res}

    # ---------- portfolio-construction angles ----------
    results.update(portfolio_angles(ctx, E, forms))

    # log trials first so DSR uses the full count
    for a, r in results.items():
        if "_res" in r:
            log_trial(a, "v1", "7", "pre-holdout")
    N = count_trials()
    table = []
    for a, r in results.items():
        if "_res" not in r:
            table.append({"angle": a, **{k: v for k, v in r.items()}, "pass": False}); continue
        m = evaluate(r["_res"], E, ff, base3_active, n_trials=N)
        row = {"angle": a, **{k: v for k, v in r.items() if k != "_res"}, **m}
        row["pass"] = gate(m)
        table.append(row)
    T = pd.DataFrame(table)
    T = T.sort_values(["pass", "t"], ascending=[False, False])
    T.to_csv(REPORTS / "phase7_angles.csv", index=False)
    json.dump({"baselines": bl, "n_trials": N}, open(REPORTS / "phase7_baselines.json", "w"), indent=1, default=float)
    # keep net active series for ablation / decay
    act = {a: (r["_res"]["net"] - E.base0).dropna() for a, r in results.items() if "_res" in r}
    pd.DataFrame(act).to_parquet(PROC / "phase7_active.parquet")
    pd.to_pickle(raw_all, PROC / "phase7_raw_signals.pkl")
    return T, bl


def portfolio_angles(ctx, E, forms):
    out = {}
    # P2 Zipf switch: EW tilt when top-10 over-concentration exceeds its trailing 10y median
    z = pd.read_csv(REPORTS / "p2_zipf.csv", parse_dates=["date"]).set_index("date").deviation
    sig = {}
    for t in forms:
        hist = z[(z.index <= t) & (z.index > t - pd.DateOffset(years=10))]
        if t not in z.index or len(hist) < 5:
            continue
        m_next = t + pd.offsets.MonthEnd(1)
        b = E.B.loc[m_next].dropna()
        s = ((1 / len(b)) / b - 1).clip(-1, 1) if z[t] > hist.median() else b * 0
        sig[t] = s
    out["P2_zipf_switch"] = {"n_formations": len(sig), "n_switched_on": int(sum((s != 0).any() for s in sig.values())),
                             "_res": E.run(sig, name="P2")}
    # P5 no-trim: the trimmed portfolio (5% single-name cap). Reported as trim minus BASE0;
    # the "never trim" angle passes if trimming HURTS.
    sig = {}
    for t in forms:
        b = E.B.loc[t + pd.offsets.MonthEnd(1)].dropna()
        capped = b.clip(upper=0.05); capped = capped / capped.sum()
        for _ in range(5):
            capped = capped.clip(upper=0.05); capped = capped / capped.sum()
        sig[t] = ((capped / b - 1) / 0.30).clip(-1, 1)   # k = capped/b
    out["P5_trim_5pct"] = {"n_formations": len(sig), "_res": E.run(sig, te_cap=None, name="P5trim")}
    out["P5_early_entrants"] = early_entrants(ctx, E, forms)
    return out


def early_entrants(ctx, E, forms):
    """Members funded pro-rata; non-members ranked inside the top 500 US common stocks by cap get
    0.30 x their cap weight (relative to S&P total cap), held 12 months (CRSP returns)."""
    m = pd.read_parquet(PROC / "market_monthly.parquet", columns=["permno", "date", "ret", "mcap", "shrcd"])
    m = m[m.shrcd.isin([10, 11])]
    R = m.pivot_table(index="date", columns="permno", values="ret", aggfunc="first")
    C = m.pivot_table(index="date", columns="permno", values="mcap", aggfunc="first")
    rets = {}; n_add = {}; share = {}
    for i, t in enumerate(forms):
        if t not in C.index:
            continue
        mem = set(ctx.sp(t))
        top = C.loc[t].dropna().nlargest(500)
        cand = top[[p not in mem for p in top.index]]
        spcap = C.loc[t].reindex(list(mem)).sum()
        x = 0.30 * cand / spcap                       # outside weights
        n_add[t] = len(cand); share[t] = float(x.sum())
        end = forms[i + 1] if i + 1 < len(forms) else E.end
        for mo in E.months:
            if not (t < mo <= end):
                continue
            rin = E.base0.get(mo, np.nan)
            rout = R.loc[mo].reindex(x.index) if mo in R.index else pd.Series(np.nan, index=x.index)
            # a candidate that stops trading: its weight earns 0 that month (conservative; no delist ret)
            ro = float((x * rout.fillna(0)).sum() / (1 + x.sum()))
            rets[mo] = rin / (1 + x.sum()) + ro
    g = pd.Series(rets)
    # turnover ~ full replacement of outside sleeve yearly, both sides
    cost = pd.Series(0.0, index=g.index)
    for t in forms:
        mo = t + pd.offsets.MonthEnd(1)
        if mo in cost.index:
            cost[mo] = 2 * share.get(t, 0) * 15 / 1e4   # 10 + 5 bps: these are small relative to S&P members
    res = {"gross": g, "net": g - cost, "cost": cost, "turnover": cost / (15 / 1e4), "scales": pd.Series(dtype=float),
           "K": {t: None for t in forms if t in C.index}}
    return {"n_formations": len(n_add), "avg_names_added": float(np.mean(list(n_add.values()))),
            "avg_outside_weight": float(np.mean(list(share.values()))), "_res": res}


if __name__ == "__main__":
    T, bl = main()
    pd.set_option("display.width", 250)
    cols = ["angle", "pass", "n_formations", "ic_mean", "ic_t", "tail_auc", "net_excess_ann", "te", "ir", "t", "alpha_ff6_ann", "alpha_ff6_t", "dsr", "win_share_windows", "worst_window", "longest_losing_years"]
    print(T[[c for c in cols if c in T.columns]].round(3).to_string())
    print(json.dumps(bl, indent=1, default=float))
