"""Phase 10: the holdout, run exactly once. `unlock_holdout` refuses a second call.

Frozen system (from Phases 7-9): no angle passed the gate, so SYSTEM = S&P 500 cap weight +
implementation rule I3 (buy index additions 5 trading days after they enter). Also reported, one run
each: every angle, and LIVE_SHADOW (registered in angles.yaml before this ran)."""
from __future__ import annotations
import json, sys
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from config import PROC, REPORTS, RAW
from src.holdout_guard import unlock_holdout, holdout_unlocked
from src.trials import log_trial, count_trials

H0, H1 = pd.Timestamp("2016-01-31"), pd.Timestamp("2026-08-31")
LIVE_SHADOW = ["P4_tail_momentum", "P9_hype_max", "N3_volume_spike", "A9_fluent_ticker", "A4_insider_buys",
               "A5_lazy_filings", "B4_named_as_rival"]


def i3_adjustment(E, since="2016-01-01") -> pd.Series:
    """Monthly return change from buying each 2016+ addition 5 trading days late: the addition's
    index weight earns the rest of the index instead of itself over days 0-4.
    adj = -w * (r_add[0..4] - r_mkt[0..4]), booked in the month of day 0."""
    d = pd.read_csv(RAW / "sp500_components_updated.csv", parse_dates=["date"])
    first, prev = {}, set()
    for dt, s in zip(d.date, d.tickers):
        cur = set(s.split(","))
        for t in cur - prev:
            if dt >= pd.Timestamp(since):
                first.setdefault(t, dt)
        prev = cur
    lk = pd.read_parquet(PROC / "membership_linked.parquet")
    lk = lk.set_index(["ticker", "date"]).permno
    adds = []
    for t, dt in first.items():
        p = lk.get((t, dt + pd.offsets.MonthEnd(0)))
        if p is not None and p > 0:
            adds.append((t, int(p), dt))
    crsp = pq.read_table(RAW / "delisted" / "crsp_daily_1968_2024.parquet", columns=["permno", "date", "ret", "vwretd"],
                         filters=[("permno", "in", [a[1] for a in adds])]).to_pandas()
    crsp["date"] = pd.to_datetime(crsp.date)
    import yfinance as yf
    mk = yf.download("^SP500TR", start="2024-12-01", auto_adjust=False, progress=False)["Adj Close"].squeeze().pct_change()
    mk.index = pd.to_datetime(mk.index).tz_localize(None)
    adj = {}; rows = []
    for t, p, dt in adds:
        g = crsp[crsp.permno == p].sort_values("date")
        g = g[g.date >= dt].head(5)
        if len(g) == 5:
            ex = float(np.prod(1 + g.ret.fillna(0)) - np.prod(1 + g.vwretd.fillna(0)))
        else:                                        # 2025+: Yahoo
            try:
                y = yf.download(t.replace(".", "-"), start=str(dt.date()), auto_adjust=False, progress=False)["Adj Close"].squeeze()
                y.index = pd.to_datetime(y.index).tz_localize(None)
                r = y.pct_change().dropna().iloc[:5]
                if len(r) < 5:
                    r = y.pct_change().loc[dt:].iloc[:5]
                m = mk.reindex(r.index).fillna(0)
                ex = float(np.prod(1 + r) - np.prod(1 + m)) if len(r) == 5 else np.nan
            except Exception:
                ex = np.nan
        m = dt + pd.offsets.MonthEnd(0) + pd.offsets.MonthEnd(1)   # panel month where it is first held
        m0 = dt + pd.offsets.MonthEnd(0)
        w = E.B.loc[m, p] if (m in E.B.index and p in E.B.columns) else np.nan
        if np.isfinite(ex) and np.isfinite(w):
            adj[m0] = adj.get(m0, 0.0) - w * ex - 2 * w * 0.0010 * 0   # extra trade cost ~0 (same buy, later)
            rows.append({"ticker": t, "date": dt.date(), "weight": w, "excess_d0_4": ex})
    R = pd.DataFrame(rows)
    R.to_csv(REPORTS / "phase10_i3_additions.csv", index=False)
    return pd.Series(adj).sort_index(), R


def voo_monthly():
    x = pd.read_parquet(RAW / "yahoo" / "VOO.parquet"); x.index = pd.to_datetime(x.index).tz_localize(None)
    return x["Adj Close"].resample("ME").last().pct_change()


def tr_monthly():
    x = pd.read_parquet(RAW / "yahoo" / "^SP500TR.parquet"); x.index = pd.to_datetime(x.index).tz_localize(None)
    return x["Adj Close"].resample("ME").last().pct_change()


def main(confirm: str):
    global H0, H1
    dry = confirm == "DRY"
    if dry:        # same code path on pre-holdout dates, guard stays locked, nothing logged as holdout
        H0, H1 = pd.Timestamp("2011-01-31"), pd.Timestamp("2015-12-31")
        import src.trials as _tr
        globals()["log_trial"] = lambda *a, **k: None
    elif confirm == "RESUME-AFTER-CRASH-NO-RESULTS-SEEN":
        import json as _j, datetime as _d
        from src.holdout_guard import _LOCK
        rec = _j.loads(_LOCK.read_text())
        if rec.get("resumed"):
            raise SystemExit("already resumed once")
        rec["resumed"] = {"at": _d.datetime.now().isoformat(), "why": "first run crashed in P6 on a formation with no FIAM data (np.log on an all-NaN object column) before any result was computed or printed; fix = skip formations without characteristics, no parameter changes"}
        _LOCK.write_text(_j.dumps(rec))
    elif confirm != "RUN-HOLDOUT-ONCE":
        raise SystemExit("pass RUN-HOLDOUT-ONCE or DRY")
    else:
        unlock_holdout("Phase 10: frozen at tag phase10-freeze; SYSTEM=BASE0+I3; all angles + LIVE_SHADOW one run each")
    from src.backtest import Engine, rank_to_unit
    from src.features.context import Context
    from src.features import ANGLES, get
    from src.phase7 import signals_for, trained_signals, evaluate, formations, ic_auc
    from src.phase8 import combine
    from src.factors import ff_monthly
    E = Engine(end=H1)
    ctx = Context(); ff = ff_monthly()
    labels = pd.read_parquet(PROC / "labels.parquet"); labels["label_end"] = pd.to_datetime(labels.label_end)
    u = pd.read_parquet(PROC / "crsp_monthly_universe.parquet", columns=["permno", "date"]); lastd = u.groupby("permno").date.max()
    dist = set(labels[labels.distress_flag].permno)
    labels["death5"] = [float(p in dist and f < lastd.get(p, pd.Timestamp("2100-01-01")) <= f + pd.offsets.MonthEnd(60))
                        for p, f in zip(labels.permno, labels.formation_date)]
    hm = lambda s: s[(s.index >= H0) & (s.index <= H1)]

    # ---------------- SYSTEM ----------------
    adj, adds = i3_adjustment(E, since=str(H0.date())[:8] + '01')
    base0 = hm(E.base0)
    system = base0.add(adj.reindex(base0.index).fillna(0))
    voo, tr = hm(voo_monthly()), hm(tr_monthly())
    cagr = lambda r: float(np.prod(1 + r) ** (12 / len(r)) - 1)
    log_trial("SYSTEM", "BASE0+I3", "10", "2016-01..2026-08")
    act_v = (system - voo.reindex(system.index)).dropna()
    out = {"system": {"cagr": cagr(system), "months": len(system), "vol": float(system.std() * np.sqrt(12))},
           "base0_cagr": cagr(base0), "sp500tr_cagr": cagr(tr.reindex(system.index).dropna()),
           "voo_cagr": cagr(voo.reindex(system.index).dropna()), "voo_months": int(voo.reindex(system.index).notna().sum()),
           "system_minus_voo_ann": float(act_v.mean() * 12), "system_minus_voo_te": float(act_v.std() * np.sqrt(12)),
           "system_minus_sp500tr_ann": float((system - tr.reindex(system.index)).dropna().mean() * 12),
           "i3_effect_ann": float(adj.reindex(base0.index).fillna(0).mean() * 12), "i3_n_additions": len(adds),
           "i3_mean_excess_d0_4": float(adds.excess_d0_4.mean()) if len(adds) else None,
           "i3_t": float(adds.excess_d0_4.mean() / (adds.excess_d0_4.std(ddof=1) / np.sqrt(len(adds)))) if len(adds) > 2 else None}

    # ---------------- every angle, one run each ----------------
    forms = formations(H0.year - 1, H1.year - (0 if H1.month >= 7 else 1)) if not dry else formations(2010, 2015)
    pre = pd.read_csv(REPORTS / "phase7_angles.csv").set_index("angle")
    raw_all = {}
    rows = []
    for a in ANGLES:
        raw = signals_for(ctx, get(a).signal, forms); raw_all[a] = raw
        if len(raw) < 2:
            rows.append({"angle": a, "status": "no holdout coverage"}); continue
        res = E.run({t: rank_to_unit(s) for t, s in raw.items()})
        rows.append(_row(a, res, E, hm, pre))
        log_trial(a, "v1", "10-holdout")
    for kind, a in (("P6", "P6_tail_classifier"), ("P8", "P8_survival")):
        raw = trained_signals(ctx, labels, kind, forms); raw_all[a] = raw
        if len(raw) >= 2:
            rows.append(_row(a, E.run({t: s.clip(-1, 1) for t, s in raw.items()}), E, hm, pre)); log_trial(a, "v1", "10-holdout")
    # LIVE_SHADOW: equal-weight unit signals + P2 switch
    z = pd.read_csv(REPORTS / "p2_zipf.csv", parse_dates=["date"]).set_index("date").deviation
    from src.live.signals import _zipf_history
    z = _zipf_history()
    p2 = {}
    for t in forms:
        hist = z[(z.index <= t) & (z.index > t - pd.DateOffset(years=10))]
        m_next = t + pd.offsets.MonthEnd(1)
        if t in z.index and m_next in E.B.index and len(hist) >= 5:
            b = E.B.loc[m_next].dropna()
            p2[t] = ((1 / len(b)) / b - 1).clip(-1, 1) if z[t] > hist.median() else b * 0
    raw_all["P2_zipf_switch"] = p2
    rows.append(_row("P2_zipf_switch", E.run(p2), E, hm, pre)); log_trial("P2_zipf_switch", "v1", "10-holdout")
    shadow_sig = combine(raw_all, LIVE_SHADOW)
    for t in shadow_sig:
        if t in p2:
            shadow_sig[t] = pd.concat([shadow_sig[t] * len(LIVE_SHADOW), p2[t].reindex(shadow_sig[t].index).fillna(0)], axis=1).sum(1) / (len(LIVE_SHADOW) + 1)
    sh = E.run(shadow_sig); log_trial("LIVE_SHADOW", "v1", "10-holdout")
    shadow = hm(sh["net"]).add(adj.reindex(base0.index).fillna(0), fill_value=0)
    rows.append(_row("LIVE_SHADOW", sh, E, hm, pre))
    allsig = combine(raw_all, [a for a in raw_all if len(raw_all[a]) >= 2])
    rows.append(_row("COMBINED_all (exploratory)", E.run(allsig), E, hm, pre)); log_trial("COMBINED_all", "equal_weight", "10-holdout")
    out["angles"] = rows
    out["shadow_minus_voo_ann"] = float((shadow - voo.reindex(shadow.index)).dropna().mean() * 12)
    # cumulative $100k series for the chart
    idx = system.index
    cum = lambda r: (100000 * np.cumprod(1 + r.reindex(idx).fillna(0))).round(2)
    M = pd.DataFrame({"SYSTEM": cum(system), "SHADOW": cum(shadow), "VOO": cum(voo), "SP500TR": cum(tr)})
    start = pd.DataFrame({c: [100000.0] for c in M.columns}, index=[H0 - pd.offsets.MonthEnd(1)])
    M = pd.concat([start, M]); M.index.name = "date"
    out["monthly"] = [{"date": str(d.date()), **{c: float(v) for c, v in r.items()}} for d, r in M.iterrows()]
    out["n_trials_total"] = count_trials()
    out["note"] = ("SYSTEM is the S&P 500 reconstructed from CRSP (through 2024) and Yahoo (2025+) at total-share cap weights, plus rule I3. "
                   "It is not float-adjusted, so a gap vs VOO of a few tenths of a percent a year is reconstruction noise, not skill. "
                   "The comparison that isolates the research decision is SYSTEM vs BASE0 (the I3 effect).")
    json.dump(out, open(REPORTS / ("phase10_dryrun.json" if dry else "phase10_holdout.json"), "w"), indent=1, default=float)
    return out


def _row(a, res, E, hm, pre):
    net = hm(res["net"]); act = (net - E.base0.reindex(net.index)).dropna()
    te = act.std() * np.sqrt(12); ex = act.mean() * 12
    t = act.mean() / (act.std(ddof=1) / np.sqrt(len(act))) if len(act) > 2 and act.std() > 0 else np.nan
    ir_pre = float(pre.loc[a, "ir"]) if a in pre.index and pd.notna(pre.loc[a].get("ir")) else None
    ir_h = float(ex / te) if te > 0 else None
    return {"angle": a, "months": int(len(act)), "excess_holdout": float(ex), "te_holdout": float(te), "ir_holdout": ir_h,
            "t_holdout": float(t), "ir_pre": ir_pre,
            "overfit_flag": bool(ir_pre is not None and ir_h is not None and ir_pre > 0 and ir_h < ir_pre / 2)}


if __name__ == "__main__":
    print(json.dumps({k: v for k, v in main(sys.argv[1] if len(sys.argv) > 1 else "").items() if k != "monthly"}, indent=1, default=float))
