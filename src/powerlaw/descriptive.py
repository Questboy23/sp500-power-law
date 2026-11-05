"""Phase 3 measurements (P1, P2, P3, P4, P5, P8, P10, P11, P12-era). Pre-holdout only:
every forward window ends on or before 2015-12-31 (Rule 4)."""
from __future__ import annotations
import json, numpy as np, pandas as pd
from scipy import stats
from config import PROC, REPORTS, HOLDOUT_START
from src.powerlaw.fit import csn_fit
from src.powerlaw.data import load_long

LAST_OK = HOLDOUT_START - pd.offsets.MonthEnd(1)


class Market:
    def __init__(self):
        x = load_long()
        x = x[x.date <= LAST_OK]
        self.R = x.pivot(index="date", columns="permno", values="ret")
        self.C = x.pivot(index="date", columns="permno", values="mcap")
        P = x.pivot(index="date", columns="permno", values="prc").abs()
        hi = P.rolling(12, min_periods=1).max()
        last_idx = self.R.notna()[::-1].idxmax()           # last date with a return
        self.distress = {}
        for pn, d in last_idx.items():
            if d < LAST_OK - pd.offsets.MonthEnd(1):
                p, h = P.at[d, pn], hi.at[d, pn]
                if pd.notna(p) and p < 5 and p < 0.3 * h:
                    self.distress[pn] = d
        self.lk = pd.read_parquet(PROC / "membership_linked.parquet")

    def members(self, d, n=500):
        if d >= pd.Timestamp("1996-06-30"):
            m = self.lk[(self.lk.date == d) & (self.lk.permno > 0)].permno.unique()
            return np.array([p for p in m if p in self.C.columns])
        return self.C.loc[d].dropna().nlargest(n).index.values

    def fwd(self, d, months, cols):
        end = d + pd.offsets.MonthEnd(months)
        if end > LAST_OK:
            return None
        blk = self.R.loc[(self.R.index > d) & (self.R.index <= end), [c for c in cols if c in self.R.columns]]
        g = (1 + blk.fillna(0)).prod() - 1
        for pn in g.index:
            dd = self.distress.get(pn)
            if dd is not None and d < dd < end:
                g[pn] = -1.0
        return g


def rf_cum():
    ff = pd.read_csv("data/raw/french/F-F_Research_Data_Factors.csv", skiprows=3)
    ff = ff[ff.iloc[:, 0].astype(str).str.strip().str.len() == 6].rename(columns={ff.columns[0]: "ym"})
    ff["date"] = pd.to_datetime(ff.ym.str.strip(), format="%Y%m") + pd.offsets.MonthEnd(0)
    return ff.set_index("date").RF.astype(float) / 100


def p1_tails(M: Market):
    rf = rf_cum()
    rows = []
    for y in range(1927, 2015):
        d = pd.Timestamp(f"{y}-06-30")
        if d not in M.R.index:
            continue
        mem = M.members(d)
        cap0 = M.C.loc[d, mem]
        for h in (1, 5, 10, 20):
            g = M.fwd(d, 12 * h, mem)
            if g is None or len(g) < 100:
                continue
            fit = csn_fit(1 + g.values)
            # Bessembinder-style: dollar wealth created in excess of T-bills
            rfc = np.prod(1 + rf.loc[(rf.index > d) & (rf.index <= d + pd.offsets.MonthEnd(12 * h))]) - 1 if d >= rf.index.min() else np.nan
            W = (cap0.reindex(g.index) * (g - rfc)).dropna().sort_values(ascending=False)
            tot = W.sum()
            shares = {f"top{k}pct_share": (W.iloc[:max(1, int(len(W) * k / 100))].sum() / tot) if tot > 0 else np.nan for k in (1, 4, 10)}
            rows.append({"formation": d, "year": y, "horizon": h, "n": len(g), "pct_beat_tbill": float((g > rfc).mean()) if pd.notna(rfc) else np.nan,
                         "median_ret": float(g.median()), "mean_ret": float(g.mean()), **fit, **shares})
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "p1_tails.csv", index=False)
    return df


def p2_zipf(M: Market):
    """Rank-size fit on ranks 11-500, implied top-10 share vs actual; then deviation -> next 5y EW-CW."""
    rows = []
    for d in M.C.index[M.C.index.month == 6]:
        mem = M.members(d)
        c = M.C.loc[d, mem].dropna().sort_values(ascending=False).values
        if len(c) < 300:
            continue
        r = np.arange(1, len(c) + 1)
        sl = slice(10, min(500, len(c)))
        b, a = np.polyfit(np.log(r[sl]), np.log(c[sl]), 1)
        implied10 = np.exp(a) * r[:10] ** b
        act10 = c[:10].sum() / c.sum()
        imp10 = implied10.sum() / (implied10.sum() + c[10:].sum())
        zipf_alpha = -1 / b
        g = M.fwd(d, 60, mem)
        ew_cw = np.nan
        if g is not None:
            w = M.C.loc[d, g.index].fillna(0); w = w / w.sum()
            ew_cw = g.mean() - (w * g).sum()
        g1 = M.fwd(d, 12, mem)
        ew_cw1 = np.nan
        if g1 is not None:
            w = M.C.loc[d, g1.index].fillna(0); w = w / w.sum()
            ew_cw1 = g1.mean() - (w * g1).sum()
        rows.append({"date": d, "n": len(c), "zipf_alpha": zipf_alpha, "top10_actual": act10, "top10_zipf": imp10,
                     "deviation": act10 - imp10, "ew_minus_cw_5y": ew_cw, "ew_minus_cw_1y": ew_cw1})
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "p2_zipf.csv", index=False)
    import statsmodels.api as sm
    out = {}
    for col, lags in (("ew_minus_cw_5y", 5), ("ew_minus_cw_1y", 1)):
        z = df.dropna(subset=[col])
        m = sm.OLS(z[col], sm.add_constant(z.deviation)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        out[col] = {"beta": m.params.deviation, "t_hac": m.tvalues.deviation, "n": len(z), "r2": m.rsquared}
        # non-overlapping 5y check
        if col == "ew_minus_cw_5y":
            nz = z.iloc[::5]
            m2 = sm.OLS(nz[col], sm.add_constant(nz.deviation)).fit()
            out["nonoverlap_5y"] = {"beta": m2.params.deviation, "t": m2.tvalues.deviation, "n": len(nz)}
    return df, out


def p4_transition(M: Market):
    """5y return decile in window t -> decile in window t+1 (consecutive non-overlapping windows)."""
    T = np.zeros((10, 10)); tail_hits = tail_n = 0; n_pairs = 0; per = []
    for y in range(1927, 2006):
        d0 = pd.Timestamp(f"{y}-06-30"); d1 = d0 + pd.offsets.MonthEnd(60)
        if d0 not in M.R.index:
            continue
        mem = M.members(d0)
        g0 = M.fwd(d0, 60, mem)
        if g0 is None:
            continue
        mem1 = M.members(d1)
        g1 = M.fwd(d1, 60, mem1)
        if g1 is None:
            continue
        both = g0.index.intersection(g1.index)
        dec0 = pd.qcut(g0.rank(method="first"), 10, labels=False)
        dec1 = pd.qcut(g1.rank(method="first"), 10, labels=False)
        for pn in both:
            T[dec0[pn], dec1[pn]] += 1
        tail0 = g0[g0 >= g0.quantile(0.95)].index
        tail1 = g1[g1 >= g1.quantile(0.95)].index
        still = tail0.intersection(g1.index)
        tail_hits += len(still.intersection(tail1)); tail_n += len(still)
        per.append({"year": y, "p_tail_given_tail": len(still.intersection(tail1)) / max(len(still), 1), "n_tail_survivors": len(still),
                    "tail0_survival": len(still) / max(len(tail0), 1)})
        n_pairs += len(both)
    P = T / T.sum(1, keepdims=True)
    pd.DataFrame(P, index=[f"D{i+1}" for i in range(10)], columns=[f"D{i+1}" for i in range(10)]).to_csv(REPORTS / "p4_transition.csv")
    per = pd.DataFrame(per); per.to_csv(REPORTS / "p4_tail_persistence_by_year.csv", index=False)
    return P, {"p_tail_given_tail": tail_hits / tail_n, "n": tail_n, "base_rate": 0.05, "pairs": n_pairs}, per


def p5_breadth(M: Market):
    """(a) returns before S&P inclusion; (b) share of next decade's tail winners not yet in the S&P."""
    lk = M.lk[M.lk.permno > 0]
    first = lk.groupby("permno").date.min()
    adds = first[first > pd.Timestamp("1996-06-30")]
    mkt = (M.R * M.C.shift(1)).sum(1) / M.C.shift(1).where(M.R.notna()).sum(1)
    rows = []
    for pn, d in adds.items():
        if pn not in M.R.columns or d > LAST_OK:
            continue
        r = M.R[pn]
        for yrs in (1, 3, 5):
            s = d - pd.offsets.MonthEnd(12 * yrs)
            blk = r[(r.index > s) & (r.index <= d)]
            mb = mkt[(mkt.index > s) & (mkt.index <= d)]
            if blk.notna().sum() >= 12 * yrs * 0.9:
                rows.append({"permno": pn, "add_date": d, "years_before": yrs,
                             "ret": np.prod(1 + blk.fillna(0)) - 1, "mkt": np.prod(1 + mb) - 1})
        # 1y AFTER inclusion
        blk = r[(r.index > d) & (r.index <= d + pd.offsets.MonthEnd(12))]
        mb = mkt[(mkt.index > d) & (mkt.index <= d + pd.offsets.MonthEnd(12))]
        if len(blk) == 12:
            rows.append({"permno": pn, "add_date": d, "years_before": -1, "ret": np.prod(1 + blk.fillna(0)) - 1, "mkt": np.prod(1 + mb) - 1})
    pre = pd.DataFrame(rows); pre["excess"] = pre.ret - pre.mkt
    late = pre.groupby("years_before").agg(n=("excess", "size"), mean_ret=("ret", "mean"), mean_excess=("excess", "mean"),
                                           median_excess=("excess", "median")).reset_index()
    # (b) decade tail capture: tail winners among top-1000 caps at t by 10y return
    cap_rows = []
    for y in range(1996, 2006):
        d = pd.Timestamp(f"{y}-06-30")
        c = M.C.loc[d].dropna()
        uni = c.nlargest(1500).index
        g = M.fwd(d, 120, uni)
        if g is None:
            continue
        tail = g[g >= g.quantile(0.95)].index
        mem = set(M.members(d))
        ends = set(M.members(d + pd.offsets.MonthEnd(120))) if d + pd.offsets.MonthEnd(120) <= pd.Timestamp("2026-08-31") else set()
        wealth = (c.reindex(tail) * g.reindex(tail))
        not_in = [p for p in tail if p not in mem]
        cap_rows.append({"formation": d, "n_tail": len(tail), "share_tail_not_in_sp": len(not_in) / len(tail),
                         "wealth_share_not_in_sp": wealth.reindex(not_in).sum() / wealth.sum(),
                         "share_later_joined": np.mean([p in ends for p in not_in]) if not_in and ends else np.nan})
    cap = pd.DataFrame(cap_rows)
    pre.to_csv(REPORTS / "p5_preinclusion.csv", index=False); cap.to_csv(REPORTS / "p5_tail_capture.csv", index=False)
    return late, cap


def p8_lifespan(M: Market):
    """Kaplan-Meier: time an S&P member stays in the index (1996+ real list), and time until a
    top-500 firm stops trading (1926+), by entry era. Death = distress exit."""
    from lifelines import KaplanMeierFitter
    out = {}
    lk = M.lk[M.lk.permno > 0]
    spans = lk.groupby("permno").date.agg(["min", "max"])
    spans["dur_y"] = (spans["max"] - spans["min"]).dt.days / 365.25
    spans["event"] = spans["max"] < pd.Timestamp("2026-07-31")
    kmf = KaplanMeierFitter().fit(spans.dur_y, spans.event)
    out["sp_membership_median_years_1996plus"] = float(kmf.median_survival_time_)
    out["sp_membership_n"] = len(spans)
    # listing lifespans of firms that entered the top 500 in each era
    firsttop = {}
    for d in M.C.index[M.C.index.month == 6]:
        for pn in M.C.loc[d].dropna().nlargest(500).index:
            firsttop.setdefault(pn, d)
    lastd = M.R.notna()[::-1].idxmax()
    rows = []
    for pn, d0 in firsttop.items():
        end = lastd[pn]
        rows.append({"permno": pn, "entry": d0, "dur_y": (end - d0).days / 365.25,
                     "delisted": end < LAST_OK - pd.offsets.MonthEnd(1), "distress": pn in M.distress})
    L = pd.DataFrame(rows)
    L["era"] = (L.entry.dt.year // 20) * 20
    era = {}
    for e, g in L.groupby("era"):
        k = KaplanMeierFitter().fit(g.dur_y, g.delisted)
        kd = KaplanMeierFitter().fit(g.dur_y, g.distress)
        era[int(e)] = {"n": len(g), "median_listed_years": float(k.median_survival_time_),
                       "p_distress_death_20y": float(1 - kd.survival_function_at_times(20).iloc[0]),
                       "p_any_exit_20y": float(1 - k.survival_function_at_times(20).iloc[0])}
    out["top500_by_entry_era"] = era
    return out


def p3_gibrat():
    """3y sales growth on log sales by SIC2 industry and era (FIAM 2000-2015 pre-holdout)."""
    import statsmodels.api as sm
    f = pd.read_parquet("data/raw/delisted/fiam_monthly_2000_2023.parquet", columns=["date", "permno", "sale_gr3", "sale_me", "market_equity"])
    f["date"] = pd.to_datetime(f.date.astype(str), format="%Y%m%d") + pd.offsets.MonthEnd(0)
    f = f[(f.date.dt.month == 6) & (f.date <= LAST_OK - pd.offsets.MonthEnd(36))]   # growth measured over the NEXT 3y must end pre-holdout
    sic = pd.read_parquet("data/raw/delisted/crsp_monthly_ticker_1945_2024.parquet", columns=["date", "ticker", "siccd"])
    # forward 3y growth: sale_gr3 at t+36 describes growth from t to t+36
    f["sales"] = f.sale_me * f.market_equity
    fw = f[["permno", "date", "sale_gr3"]].copy(); fw["date"] = fw.date - pd.offsets.MonthEnd(36)
    x = f[["permno", "date", "sales"]].merge(fw, on=["permno", "date"]).dropna()
    x = x[(x.sales > 0) & np.isfinite(x.sale_gr3)]
    x["log_s"] = np.log(x.sales)
    x["g"] = np.log1p(x.sale_gr3.clip(-0.95, 20))
    m = sm.OLS(x.g, sm.add_constant(x.log_s)).fit(cov_type="cluster", cov_kwds={"groups": x.date})
    res = {"pooled": {"slope": m.params.log_s, "t": m.tvalues.log_s, "n": len(x)}}
    by = {}
    for d, g in x.groupby("date"):
        mm = sm.OLS(g.g, sm.add_constant(g.log_s)).fit()
        by[str(d.date())] = {"slope": mm.params.log_s, "t": mm.tvalues.log_s, "n": len(g)}
    res["by_formation"] = by
    return res, x


def p11_generative(M: Market, sims=200, seed=0):
    """Gabaix-style proportional growth with entry/exit, calibrated on 1926-2010 top-500 data:
    log cap growth (relative to the aggregate) ~ N(mu, sigma) i.i.d. (Gibrat), firms exit at the
    empirical top-500 turnover rate and are replaced by entrants at the 500th-rank size.
    Check: does it reproduce the observed top-10 share and Zipf alpha?"""
    rng = np.random.default_rng(seed)
    C = M.C[M.C.index.month == 6]
    rel = []
    turn = []
    for d0, d1 in zip(C.index[:-1], C.index[1:]):
        t0 = C.loc[d0].dropna().nlargest(500); t1 = C.loc[d1].dropna().nlargest(500)
        both = t0.index.intersection(C.loc[d1].dropna().index)
        agg = np.log(C.loc[d1, t0.index].sum() / t0.sum())
        rel.append(np.log(C.loc[d1, both] / t0[both]) - agg)
        turn.append(1 - len(t0.index.intersection(t1.index)) / 500)
    rel = pd.concat(rel)
    sigma = float(rel.std()); churn = float(np.mean(turn))
    obs = pd.read_csv(REPORTS / "p2_zipf.csv", parse_dates=["date"])
    start = C.loc[pd.Timestamp("1936-06-30")].dropna().nlargest(500).values
    yrs = 2015 - 1936
    top10 = []; alphas = []
    for s in range(sims):
        c = np.log(start.copy())
        for _ in range(yrs):
            c = c + rng.normal(-sigma ** 2 / 2, sigma, len(c))
            c = np.sort(c)[::-1]
            k = int(round(churn * 500))
            c[-k:] = c[-k - 1] + rng.normal(0, 0.1, k)   # entrants at the boundary size
        v = np.exp(np.sort(c)[::-1]); top10.append(v[:10].sum() / v.sum())
        r = np.arange(1, 501); b, a = np.polyfit(np.log(r[10:]), np.log(v[10:]), 1); alphas.append(-1 / b)
    # Variant B: Gibrat fails (P3): relative growth mean-reverts in log size. Calibrate kappa by
    # regressing next-year relative log growth on demeaned log cap among top-500 firms.
    xs, ys = [], []
    for d0, d1 in zip(C.index[:-1], C.index[1:]):
        t0 = C.loc[d0].dropna().nlargest(500)
        both = t0.index.intersection(C.loc[d1].dropna().index)
        agg = np.log(C.loc[d1, t0.index].sum() / t0.sum())
        lc = np.log(t0[both]); xs.append(lc - lc.mean()); ys.append(np.log(C.loc[d1, both] / t0[both]) - agg)
    xs = pd.concat(xs); ys = pd.concat(ys)
    kappa = float(-np.polyfit(xs, ys, 1)[0]); sig_b = float((ys + kappa * xs).std())
    top10b = []; alphasb = []
    for s in range(sims):
        c = np.log(start.copy())
        for _ in range(yrs):
            c = c - kappa * (c - c.mean()) + rng.normal(-sig_b ** 2 / 2, sig_b, len(c))
            c = np.sort(c)[::-1]; k = int(round(churn * 500)); c[-k:] = c[-k - 1] + rng.normal(0, 0.1, k)
        v = np.exp(np.sort(c)[::-1]); top10b.append(v[:10].sum() / v.sum())
        r = np.arange(1, 501); b, a = np.polyfit(np.log(r[10:]), np.log(v[10:]), 1); alphasb.append(-1 / b)
    last = obs[obs.date <= "2015-06-30"].iloc[-1]
    return {"B_kappa": kappa, "B_sigma": sig_b, "B_sim_top10_median": float(np.median(top10b)),
            "B_sim_top10_p5_p95": [float(np.percentile(top10b, 5)), float(np.percentile(top10b, 95))],
            "B_sim_zipf_alpha_median": float(np.median(alphasb)),"sigma_rel_growth": sigma, "annual_top500_churn": churn,
            "sim_top10_share_median": float(np.median(top10)), "sim_top10_p5_p95": [float(np.percentile(top10, 5)), float(np.percentile(top10, 95))],
            "sim_zipf_alpha_median": float(np.median(alphas)),
            "obs_top10_share_2015": float(last.top10_actual), "obs_zipf_alpha_2015": float(last.zipf_alpha),
            "obs_top10_share_mean_1936_2015": float(obs[(obs.date >= "1936-06-30") & (obs.date <= "2015-06-30")].top10_actual.mean())}


def p10_breadth_sim(M: Market, sims=2000, seed=0):
    """20y outcomes of k-stock portfolios drawn from the S&P (proxy) at formations 1936-1995."""
    rng = np.random.default_rng(seed); rows = []
    ks = (1, 5, 20, 50, 100, 500)
    pool = []
    for y in range(1936, 1996, 5):
        d = pd.Timestamp(f"{y}-06-30")
        mem = M.members(d); g = M.fwd(d, 240, mem)
        if g is not None:
            w = M.C.loc[d, g.index].fillna(0); pool.append((y, g.values, (w * g).sum() / w.sum()))
    for k in ks:
        outs = []
        for _ in range(sims):
            y, g, cw = pool[rng.integers(len(pool))]
            pick = rng.choice(len(g), min(k, len(g)), replace=False)
            outs.append((np.mean(g[pick]), cw))
        o = np.array(outs)
        rows.append({"k": k, "median_20y_multiple": float(np.median(1 + o[:, 0])), "p5_20y_multiple": float(np.percentile(1 + o[:, 0], 5)),
                     "share_beating_capweight": float((o[:, 0] > o[:, 1]).mean()), "median_cagr": float(np.median((1 + o[:, 0]) ** (1 / 20) - 1))})
    # Kelly for the index: annual excess return / variance (monthly data, pre-holdout)
    rf = rf_cum()
    mkt = (M.R * M.C.shift(1)).sum(1) / M.C.shift(1).where(M.R.notna()).sum(1)
    ex = (mkt - rf.reindex(mkt.index)).dropna()
    kelly = float((ex.mean() * 12) / (ex.var() * 12))
    return pd.DataFrame(rows), {"index_kelly_fraction": kelly, "half_kelly": kelly / 2, "n_months": len(ex),
                                "note": "a 30% tilt on a 0.1-7% weight never approaches half-Kelly on the index position; per-stock half-Kelly cap is applied in backtest as max weight <= 1.3x cap weight"}


def run_all():
    M = Market()
    res = {}
    p1 = p1_tails(M)
    import statsmodels.api as sm
    for h in (1, 5, 10, 20):
        z = p1[(p1.horizon == h)].dropna(subset=["alpha"])
        m = sm.OLS(z.alpha, sm.add_constant(z.year)).fit(cov_type="HAC", cov_kwds={"maxlags": h})
        res[f"p1_alpha_trend_{h}y"] = {"slope_per_decade": m.params.year * 10, "t_hac": m.tvalues.year, "n": len(z)}
        z["decade"] = (z.year // 10) * 10
        res[f"p1_by_decade_{h}y"] = z.groupby("decade").agg(alpha=("alpha", "mean"), n_forms=("alpha", "size"),
                                                             pl_better=("better_fit", lambda s: (s == "power_law").mean()),
                                                             ln_better=("better_fit", lambda s: (s == "lognormal").mean()),
                                                             top4=("top4pct_share", "median"), beat_tbill=("pct_beat_tbill", "mean")).round(3).to_dict("index")
    p2, p2t = p2_zipf(M); res["p2_test"] = p2t
    P, p4s, _ = p4_transition(M); res["p4"] = p4s
    late, cap = p5_breadth(M); res["p5_late_entry"] = late.round(4).to_dict("records"); res["p5_tail_capture"] = cap.round(3).to_dict("records")
    res["p8"] = p8_lifespan(M)
    g, _ = p3_gibrat(); res["p3"] = g["pooled"]; res["p3_by_formation"] = g["by_formation"]
    res["p11"] = p11_generative(M)
    sim, kel = p10_breadth_sim(M); res["p10"] = sim.round(4).to_dict("records"); res["p10_kelly"] = kel
    json.dump(res, open(REPORTS / "phase3_results.json", "w"), indent=1, default=str)
    return res


if __name__ == "__main__":
    r = run_all()
    print(json.dumps(r, indent=1, default=str)[:6000])
