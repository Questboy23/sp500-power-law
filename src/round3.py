"""Round 3: structural edges (pre-registered in angles.yaml: round3)."""
import json, numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy import stats
from config import RAW, PROC, REPORTS
from src.trials import log_trial, count_trials
from src.evaluate import deflated_sharpe
from src.powerlaw.descriptive import rf_cum

R3 = RAW / "r3"


def daily_market():
    d = pq.read_table(RAW / "delisted" / "crsp_daily_1968_2024.parquet", columns=["date", "vwretd"]).to_pandas()
    d = d.drop_duplicates("date"); d["date"] = pd.to_datetime(d.date)
    s = d.set_index("date").vwretd.sort_index()
    y = pd.read_parquet(RAW / "yahoo" / "^SP500TR.parquet"); y.index = pd.to_datetime(y.index).tz_localize(None)
    ext = y["Adj Close"].pct_change(); ext = ext[ext.index > s.index.max()]
    s = pd.concat([s, ext]).dropna()
    rfm = rf_cum()
    m = s.index + pd.offsets.MonthEnd(0)
    n = pd.Series(1, index=s.index).groupby(m).transform("count")
    rfd = pd.Series(rfm.reindex(m).values, index=s.index).ffill() / n
    return s, rfd


def tstat(x):
    x = pd.Series(x).dropna(); return float(x.mean() / (x.std() / np.sqrt(len(x)))) if len(x) > 2 and x.std() > 0 else np.nan


def overlay(mask, r, rf, extra=0.5, cost_bp=1.0):
    """Extra exposure on masked days; returns daily extra-return series (0 on other days)."""
    ex = (r - rf).where(mask, 0.0) * extra
    entries = mask & ~mask.shift(1, fill_value=False)
    return ex - entries.astype(float) * extra * 2 * cost_bp / 1e4      # in + out


def summarize_overlay(o, a, b):
    x = o.loc[a:b]; mo = x.groupby(x.index + pd.offsets.MonthEnd(0)).sum()
    return {"extra_ann": float(mo.mean() * 12), "t_monthly": tstat(mo), "months": len(mo)}


def t1_t2(r, rf):
    fomc = pd.to_datetime(pd.read_csv(R3 / "fomc_dates.csv").announce)
    days = r.index
    pos = days.searchsorted(fomc)
    m1 = pd.Series(False, index=days)
    for p in pos:
        if 0 < p < len(days):
            m1.iloc[p] = True; m1.iloc[p - 1] = True
    m1[days < "1994-01-01"] = False
    month = days.to_period("M")
    rank = pd.Series(1, index=days).groupby(month).cumcount()
    rev = pd.Series(1, index=days)[::-1].groupby(month[::-1]).cumcount()[::-1]
    m2 = (rank <= 2) | (rev == 0)
    o1, o2 = overlay(m1, r, rf), overlay(m2, r, rf)
    ex = r - rf
    res = {"T1_pre_fomc": {"design_1994_2015": summarize_overlay(o1, "1994", "2015"),
                           "after_paper_2012_2026": summarize_overlay(o1, "2012", "2026"),
                           "mean_excess_on_fomc_days_bp": float(ex[m1].loc["1994":"2026"].mean() * 1e4),
                           "mean_excess_other_days_bp": float(ex[~m1].loc["1994":"2026"].mean() * 1e4),
                           "n_days": int(m1.sum())},
           "T2_turn_of_month": {"design_1968_2015": summarize_overlay(o2, "1968", "2015"),
                                "after_paper_1987_2026": summarize_overlay(o2, "1987", "2026"),
                                "since_2016": summarize_overlay(o2, "2016", "2026"),
                                "share_of_market_excess_from_TOM_days_1968_2015": float(ex[m2].loc["1968":"2015"].sum() / ex.loc["1968":"2015"].sum()),
                                "TOM_share_of_days": float(m2.loc["1968":"2015"].mean())}}
    return res, {"T1_pre_fomc": o1, "T2_turn_of_month": o2}


def monthly(r, rf):
    m = (1 + r).groupby(r.index + pd.offsets.MonthEnd(0)).prod() - 1
    f = rf.groupby(rf.index + pd.offsets.MonthEnd(0)).sum()
    rv = (r ** 2).groupby(r.index + pd.offsets.MonthEnd(0)).sum() * 12 / (r.groupby(r.index + pd.offsets.MonthEnd(0)).count() / 21)  # annualized
    return m, f, rv


def cboe(name):
    x = pd.read_csv(R3 / f"{name}_History.csv")
    x.columns = [c.strip().upper() for c in x.columns]
    dcol = x.columns[0]; vcol = name if name in x.columns else x.columns[-1] if name != "VIX" else "CLOSE"
    x[dcol] = pd.to_datetime(x[dcol], format="mixed")
    return x.set_index(dcol)[vcol].astype(float).sort_index()


def perf(r, m, f):
    r, m, f = r.dropna(), m.reindex(r.dropna().index), f.reindex(r.dropna().index).fillna(0)
    ex, mex = r - f, m - f
    b, a, *_ = stats.linregress(mex, ex)
    res = ex - b * mex
    cum, mc = (1 + r).cumprod(), (1 + m).cumprod()
    return {"cagr": float(cum.iloc[-1] ** (12 / len(r)) - 1), "mkt_cagr": float(mc.iloc[-1] ** (12 / len(r)) - 1),
            "vol": float(r.std() * np.sqrt(12)), "mkt_vol": float(m.std() * np.sqrt(12)),
            "sharpe": float(ex.mean() / ex.std() * np.sqrt(12)), "mkt_sharpe": float(mex.mean() / mex.std() * np.sqrt(12)),
            "beta": float(b), "alpha_ann": float(a * 12), "alpha_t": tstat(res),
            "max_dd": float((cum / cum.cummax() - 1).min()), "mkt_max_dd": float((mc / mc.cummax() - 1).min()), "months": len(r),
            "_resid": res}


def t3_t4_t5_t6(r, rf):
    m, f, rv = monthly(r, rf)
    out, series = {}, {}
    put = cboe("PUT").resample("ME").last().pct_change()
    bxm = cboe("BXM").resample("ME").last().pct_change()
    half = 0.5 * m + 0.5 * put
    for name, s, design, post in [("T3_put_write", put, ("1991-04-30", "2015-12-31"), ("2008-01-31", "2026-09-30")),
                                  ("T3b_half_put_half_sp", half, ("1991-04-30", "2015-12-31"), ("2008-01-31", "2026-09-30")),
                                  ("T4_covered_call", bxm, ("2002-05-31", "2015-12-31"), ("2005-01-31", "2026-09-30"))]:
        out[name] = {"design": perf(s.loc[design[0]:design[1]], m, f), "after_paper": perf(s.loc[post[0]:post[1]], m, f)}
        series[name] = s
    vix = cboe("VIX").resample("ME").last()
    vrp = (vix / 100) ** 2 - rv.reindex(vix.index)
    w = pd.Series(1.0, index=vrp.index)
    for d in vrp.index:
        h = vrp.loc[:d].iloc[-120:]
        if len(h) >= 60:
            lo, hi = h.quantile(1 / 3), h.quantile(2 / 3)
            w[d] = 1.3 if vrp[d] >= hi else (0.7 if vrp[d] <= lo else 1.0)
    w = w.shift(1).dropna()
    t5 = (w * m.reindex(w.index) + (1 - w) * f.reindex(w.index) - (w - 1).clip(lower=0) * 0.005 / 12).dropna()
    out["T5_vrp_timing"] = {"design": perf(t5.loc["1995-01-31":"2015-12-31"], m, f), "after_paper": perf(t5.loc["2008-01-31":"2026-09-30"], m, f)}
    series["T5_vrp_timing"] = t5
    inv = 1 / rv.shift(1)
    c = 1 / inv.expanding(min_periods=60).mean()
    wv = (c * inv).clip(upper=1.5)
    t6 = (wv * m + (1 - wv) * f - (wv - 1).clip(lower=0) * 0.005 / 12).dropna()
    out["T6_vol_managed_levered"] = {"design": perf(t6.loc["1973-01-31":"2015-12-31"], m, f), "since_2016": perf(t6.loc["2016-01-31":"2026-09-30"], m, f),
                                     "avg_weight": float(wv.loc["1973":"2015"].mean())}
    series["T6_vol_managed_levered"] = t6
    return out, series, m, f


def t7_seasonality():
    x = pd.read_parquet(PROC / "market_long.parquet")[["permno", "date", "ret", "mcap"]].sort_values(["permno", "date"])
    x["w0"] = x.groupby("permno").mcap.shift(1); x = x.dropna(subset=["ret", "w0"])
    R = x.pivot(index="date", columns="permno", values="ret"); W = x.pivot(index="date", columns="permno", values="w0")
    rows = {}; prevw = None; tos = []
    for i, d in enumerate(R.index):
        if d.year < 1947:
            continue
        lags = [d - pd.DateOffset(years=k) + pd.offsets.MonthEnd(0) for k in range(1, 21)]
        lags = [l for l in lags if l in R.index]
        hist = R.loc[lags]
        sea = hist.mean()[hist.notna().sum() >= 5]
        w = W.loc[d].dropna(); top = w.nlargest(500); w = top / top.sum()
        s = sea.reindex(w.index).dropna()
        k = pd.Series(1.0, index=w.index)
        if len(s) > 200:
            q = s.rank(pct=True); k[q[q >= 0.8].index] = 1.3; k[q[q <= 0.2].index] = 0.7
        wt = w * k; wt = wt / wt.sum()
        rr = R.loc[d].reindex(w.index).fillna(0)
        a = wt - w
        if prevw is not None:
            idx = a.index.union(prevw.index)
            tos.append(float((a.reindex(idx, fill_value=0) - prevw.reindex(idx, fill_value=0)).abs().sum()))
        prevw = a
        rows[d] = (float((wt * rr).sum()), float((w * rr).sum()))
    df = pd.DataFrame(rows, index=["strat", "bench"]).T
    to = np.mean(tos)
    act = df.strat - df.bench - to * 10 / 1e4
    res = {}
    for name, a, b in [("design_1947_2002", "1947", "2002"), ("after_paper_2003_2015", "2003", "2015"), ("untouched_for_this_signal_2016_2024", "2016", "2024"),
                       ("1947_2015", "1947", "2015")]:
        s = act.loc[a:b]; res[name] = {"excess_ann_net": float(s.mean() * 12), "t": tstat(s), "months": len(s)}
    res["monthly_active_turnover"] = float(to)
    return res, act


def t8_deletions():
    d = pd.read_csv(RAW / "sp500_components_updated.csv", parse_dates=["date"])
    lk = pd.read_parquet(PROC / "membership_linked.parquet")
    last = {}; prev = set()
    for dt, s in zip(d.date, d.tickers):
        cur = set(s.split(","))
        for t in prev - cur:
            last[t] = dt
        prev = cur
    x = pd.read_parquet(PROC / "market_long.parquet")[["permno", "date", "ret", "prc"]]
    R = x.pivot(index="date", columns="permno", values="ret"); P = x.pivot(index="date", columns="permno", values="prc").abs()
    mkt = pd.read_parquet(PROC / "topn_monthly.parquet").top500_cw
    rows = []
    t2p = lk.set_index(["ticker", "date"]).permno
    for t, dt in last.items():
        if not (pd.Timestamp("1996-06-01") <= dt <= pd.Timestamp("2023-12-31")):
            continue
        me = dt + pd.offsets.MonthEnd(0) - pd.offsets.MonthEnd(1)
        p = t2p.get((t, me))
        if p is None or p <= 0 or p not in R.columns:
            continue
        px = P.at[me, p] if me in P.index else np.nan
        if not np.isfinite(px) or px < 5:
            continue                                            # non-distress only (pre-registered)
        fut = R.loc[(R.index > dt + pd.offsets.MonthEnd(0)) & (R.index <= dt + pd.offsets.MonthEnd(12)), p]
        if fut.notna().sum() < 3:
            continue                                            # acquired / delisted at deletion: not investable
        mk = mkt.reindex(fut.index)
        rows.append({"ticker": t, "date": dt, "ret12": float(np.prod(1 + fut.fillna(0)) - 1), "mkt12": float(np.prod(1 + mk.fillna(0)) - 1),
                     "months_traded": int(fut.notna().sum())})
    E = pd.DataFrame(rows); E["excess"] = E.ret12 - E.mkt12
    res = {}
    for name, a, b in [("1996_2015", "1996", "2015"), ("2016_2023_untouched_for_this_signal", "2016", "2023")]:
        e = E[(E.date >= a) & (E.date <= b + "-12-31")]
        res[name] = {"n_events": len(e), "mean_excess_12m": float(e.excess.mean()), "median_excess_12m": float(e.excess.median()), "t": tstat(e.excess)}
    E.to_csv(REPORTS / "round3_deletions.csv", index=False)
    return res


def t9_earnings_month():
    from src.backtest import Engine
    pf = pd.read_parquet(PROC / "periodic_filings.parquet")
    link = pd.read_parquet(PROC / "permno_cik.parquet")
    pf["ym"] = pf.filed + pd.offsets.MonthEnd(0)
    E = Engine(end="2024-12-31")
    sig = {}
    lk = link[["permno", "cik", "first_year", "last_year"]]
    j = pf.merge(lk, on="cik")
    j = j[(j.filed.dt.year >= j.first_year - 1) & (j.filed.dt.year <= j.last_year + 1)]
    filed_months = j.groupby("permno").ym.apply(set)
    for t in E.months:
        t = pd.Timestamp(t)
        if t.year < 1997:
            continue
        form = t - pd.offsets.MonthEnd(1)                    # signal formed at prior month end for month t
        target = t - pd.offsets.MonthEnd(12)                 # same calendar month last year
        mem = E.B.columns[E.B.loc[t].notna()]
        s = pd.Series([1.0 if target in filed_months.get(p, ()) else 0.0 for p in mem], index=mem)
        sig[form] = s
    res_eng = E.run(sig)
    act = (res_eng["net"] - E.base0).dropna()
    out = {}
    for name, a, b in [("design_1997_2004", "1997", "2004"), ("after_paper_2005_2015", "2005", "2015"), ("untouched_2016_2024", "2016", "2024"), ("1997_2015", "1997", "2015")]:
        s = act.loc[a:b]; out[name] = {"excess_ann_net": float(s.mean() * 12), "t": tstat(s), "te": float(s.std() * np.sqrt(12)), "months": len(s)}
    out["avg_share_flagged"] = float(np.mean([s.mean() for s in sig.values()]))
    return out, act


if __name__ == "__main__":
    r, rf = daily_market()
    res = {}
    a, ov = t1_t2(r, rf); res.update(a)
    b, series, m, f = t3_t4_t5_t6(r, rf); res.update(b)
    res["T7_seasonality"], act7 = t7_seasonality()
    res["T8_deletion_rebound"] = t8_deletions()
    res["T9_earnings_month"], act9 = t9_earnings_month()
    for k in ["T1_pre_fomc", "T2_turn_of_month", "T3_put_write", "T3b_half_put_half_sp", "T4_covered_call", "T5_vrp_timing",
              "T6_vol_managed_levered", "T7_seasonality", "T8_deletion_rebound", "T9_earnings_month"]:
        log_trial(k, "round3", "round3")
    N = count_trials()
    # deflated Sharpe on each design-period edge series
    dsr = {}
    for k, o in ov.items():
        mo = o.groupby(o.index + pd.offsets.MonthEnd(0)).sum()
        dsr[k] = deflated_sharpe(mo.loc["1994" if k == "T1_pre_fomc" else "1968":"2015"], N)
    for k in ["T3_put_write", "T3b_half_put_half_sp", "T4_covered_call", "T5_vrp_timing", "T6_vol_managed_levered"]:
        dsr[k] = deflated_sharpe(res[k]["design"]["_resid"], N)
    dsr["T7_seasonality"] = deflated_sharpe(act7.loc["1947":"2002"], N)
    dsr["T9_earnings_month"] = deflated_sharpe(act9.loc["1997":"2004"], N)
    res["_dsr"] = dsr; res["_n_trials"] = N
    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if not k.startswith("_resid")}
        return o
    json.dump(strip(res), open(REPORTS / "round3_results.json", "w"), indent=1, default=float)
    print(json.dumps(strip(res), indent=1, default=lambda v: round(float(v), 4)))
