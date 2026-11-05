"""Export reports/site_snapshot.json for reedsmoneymaker23.com/sp500. Aggregates only: no CRSP
rows leave this repo (the CRSP extract is personal-research use)."""
from __future__ import annotations
import json, datetime as dt, math
import numpy as np, pandas as pd
from config import REPORTS, ROOT
from src.trials import count_trials

LIVE = ROOT / "data" / "live"


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (pd.Timestamp, dt.date)):
        return str(o)[:10]
    return o


def live_block():
    if not (LIVE / "marks.csv").exists():
        return {"started": False}
    m = pd.read_csv(LIVE / "marks.csv")
    s = json.loads((LIVE / "state.json").read_text())
    last = m.iloc[-1]; first = m.iloc[0]
    ret = lambda c: float(last[c] / first[c] - 1)
    tom = None
    if "TOM" in m and m.TOM.notna().any():
        t0 = m[m.TOM.notna()].iloc[0]
        tom = {"start_date": str(t0.date), "trading_days": int(m.TOM.notna().sum() - 1), "value": float(last.TOM),
               "return": float(last.TOM / t0.TOM - 1), "voo_same_period": float(last.VOO / t0.VOO - 1),
               "vs_voo": float(last.TOM / t0.TOM - last.VOO / t0.VOO)}
    trades = pd.read_csv(LIVE / "trades.csv") if (LIVE / "trades.csv").exists() else pd.DataFrame()
    info = json.loads((LIVE / "shadow_signal_info.json").read_text()) if (LIVE / "shadow_signal_info.json").exists() else {}
    k = pd.Series(s["portfolios"]["SHADOW"].get("k", {}))
    return {"started": True, "start_date": s["start_date"], "last_mark": s["last_mark"],
            "trading_days": int(len(m) - 1), "weights_asof": s["weights_asof"],
            "marks": m.to_dict("records"),
            "returns": {c: ret(c) for c in ("SYSTEM", "SHADOW", "VOO", "SP500TR")},
            "vs_voo": {"SYSTEM": ret("SYSTEM") - ret("VOO"), "SHADOW": ret("SHADOW") - ret("VOO")},
            "values": {c: float(last[c]) for c in ("SYSTEM", "SHADOW", "VOO", "SP500TR")},
            "costs": {n: p["costs"] for n, p in s["portfolios"].items()},
            "pending_adds": {n: p.get("pending", {}) for n, p in s["portfolios"].items()},
            "n_holdings": {n: len(p.get("shares", {})) or 1 for n, p in s["portfolios"].items()},
            "trades": trades.tail(40).to_dict("records") if len(trades) else [],
            "shadow_info": info, "tom": tom,
            "system_vs_sp500tr": ret("SYSTEM") - ret("SP500TR"),
            "shadow_tilts": {"most_overweight": k.sort_values().tail(8).round(3).to_dict(),
                             "most_underweight": k.sort_values().head(8).round(3).to_dict()}}


def rounds_block():
    """Round 2 (13 'improve the S&P' ideas) and round 3 (9 structural ideas): one row each."""
    rows = []
    r2p = REPORTS / "round2_results.json"
    if r2p.exists():
        r2 = json.load(open(r2p))
        for k, v in r2.items():
            d = v["1927-2015"]
            rows.append({"round": 2, "id": k, "design": "1927-2015", "edge_ann": d["edge_ann"], "t": d["t"], "dsr": d.get("dsr"),
                         "later": v.get("2016-2026 (hindsight era, not a test)", {}).get("edge_ann"), "later_label": "2016-26 (hindsight)",
                         "pass": d.get("pass", False), "months": d["months"]})
    r3p = REPORTS / "round3_results.json"
    if r3p.exists():
        r3 = json.load(open(r3p)); dsr = r3.get("_dsr", {})
        def add(k, design, dkey, later, lkey, edge="extra_ann", t="t_monthly", lab=""):
            if k not in r3:
                return
            d, l = r3[k][dkey], r3[k].get(lkey, {})
            rows.append({"round": 3, "id": k, "design": design, "edge_ann": d.get(edge), "t": d.get(t), "dsr": dsr.get(k),
                         "later": l.get(edge), "later_t": l.get(t), "later_label": lab, "months": d.get("months"),
                         "pass": bool(d.get(t) is not None and d.get(t) >= 3 and (dsr.get(k) or 0) >= 0.95 and (l.get(edge) or 0) > 0 and (l.get(t) or 0) > 0)})
        add("T1_pre_fomc", "1994-2015", "design_1994_2015", None, "after_paper_2012_2026", lab="after paper, 2012-26")
        add("T2_turn_of_month", "1968-2015", "design_1968_2015", None, "after_paper_1987_2026", lab="after paper, 1987-26")
        for k, lab in [("T3_put_write", "after paper, 2008-26"), ("T3b_half_put_half_sp", "after paper, 2008-26"), ("T4_covered_call", "after paper, 2005-26"),
                       ("T5_vrp_timing", "after paper, 2008-26")]:
            add(k, "design", "design", None, "after_paper", edge="alpha_ann", t="alpha_t", lab=lab)
        add("T6_vol_managed_levered", "1973-2015", "design", None, "since_2016", edge="alpha_ann", t="alpha_t", lab="2016-26")
        add("T7_seasonality", "1947-2002", "design_1947_2002", None, "after_paper_2003_2015", edge="excess_ann_net", t="t", lab="after paper, 2003-15")
        add("T9_earnings_month", "1997-2004", "design_1997_2004", None, "after_paper_2005_2015", edge="excess_ann_net", t="t", lab="after paper, 2005-15")
        d8 = r3.get("T8_deletion_rebound", {})
        if d8:
            a, b = d8["1996_2015"], d8["2016_2023_untouched_for_this_signal"]
            rows.append({"round": 3, "id": "T8_deletion_rebound", "design": "1996-2015", "edge_ann": a["mean_excess_12m"], "t": a["t"], "dsr": None,
                         "later": b["mean_excess_12m"], "later_t": b["t"], "later_label": "2016-23", "months": a["n_events"], "pass": False,
                         "note": "per-event 12-month excess; n = events"})
    notes = {"T7_seasonality": "Technical pass only: after the paper it made +0.00%/yr (t 0.01). Not adopted.",
             "T2_turn_of_month": "Adopted for a live paper test. Weaker since 2000: +1.2%/yr, t 1.65.",
             "R9_trend_10m": "Cuts drawdowns (−45% vs −84%) but did not raise return; trailed after publication."}
    for r in rows:
        if r["id"] in notes:
            r["note"] = notes[r["id"]]
    return rows


def tournament_block():
    """Top-4% tournament: every hypothesis, round by round (round 2/3 shown only when reached, except
    batches whose rules judged later rounds directly)."""
    rows = []
    def pick(r):
        return None if not r else {"hit_rate": r.get("hit_rate"), "losers": r.get("loser_rate"), "picks": r.get("picks_cagr", r.get("picks_cw_cagr")),
                                   "sp": r.get("sp_cagr"), "years": r.get("years")}
    for b in range(1, 7):
        f = REPORTS / f"top4_batch{b}.json"
        if not f.exists():
            continue
        J = json.load(open(f))
        for h, r in J.items():
            if b == 3:
                rows.append({"batch": b, "id": h, "target": "dollars", "r1": pick(r["round1"]), "adv1": r["promoted_to_round2"], "r2": pick(r.get("round2_shadow")),
                             "r3": pick(r.get("round3_shadow")), "note": "dollar-wealth target; rounds 2-3 shown for context"})
            elif b == 5:
                rows.append({"batch": b, "id": h, "target": "returns", "r1": pick(r["round1_in_sample"]), "adv1": None, "r2": pick(r["round2"]),
                             "r3": pick(r["round3"]) if r.get("promoted_to_round3") else None, "note": "built from round-1 profile: judged on rounds 2-3"})
            elif b == 6:
                rows.append({"batch": b, "id": h, "target": "returns", "r1": pick(r["1"]), "adv1": None, "r2": pick(r["2"]), "r3": pick(r["3"]), "note": "regime switch"})
            else:
                rows.append({"batch": b, "id": h, "target": "returns", "r1": pick(r["round1"]), "adv1": r["promoted_to_round2"],
                             "r2": pick(r.get("round2")), "r3": pick(r.get("round3")), "note": ""})
    return rows


def main():
    T = pd.read_csv(REPORTS / "phase7_angles.csv")
    keep = ["angle", "pass", "n_formations", "first_formation", "ic_mean", "ic_t", "n_ic", "tail_auc", "n_auc",
            "net_excess_ann", "te", "ir", "t", "months", "alpha_ff6_ann", "alpha_ff6_t", "dsr", "n_windows", "status"]
    angles = T[[c for c in keep if c in T.columns]].to_dict("records")
    import yaml
    reg = yaml.safe_load(open(ROOT / "angles.yaml"))["angles"]
    untestable = [{"angle": a["id"], "reason": a.get("reason", "")} for a in reg if a.get("data_grade") == "X"]
    vibe = {a["id"]: a.get("vibe", "") for a in reg}
    for a in angles:
        a["vibe"] = vibe.get(a["angle"], "")
    p3 = json.load(open(REPORTS / "phase3_results.json"))
    p1 = pd.read_csv(REPORTS / "p1_tails.csv"); p1["decade"] = (p1.year // 10) * 10
    tails = p1.groupby(["horizon", "decade"]).agg(hill=("hill_alpha_top10", "mean"), n=("n", "mean"), forms=("year", "size"),
                                                   top4=("top4pct_share", "median"), lognormal=("better_fit", lambda s: (s == "lognormal").mean()),
                                                   powerlaw=("better_fit", lambda s: (s == "power_law").mean())).reset_index()
    z = pd.read_csv(REPORTS / "p2_zipf.csv")
    z = z[z.date <= "2015-06-30"][["date", "zipf_alpha", "top10_actual", "top10_zipf", "deviation"]]
    p8 = json.load(open(REPORTS / "phase8_results.json"))
    p9 = json.load(open(REPORTS / "phase9_results.json"))
    bl = json.load(open(REPORTS / "phase7_baselines.json"))
    hold = json.load(open(REPORTS / "phase10_holdout.json")) if (REPORTS / "phase10_holdout.json").exists() else None
    snap = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "data_grade": "free-data grade (CRSP extract via Kaggle re-upload, 1926-2024; Yahoo 2025+; FIAM/JKP characteristics 2000-2023; SEC EDGAR)",
        "verdict": {
            "headline": "Nothing beat the S&P 500 well enough to use, so the system holds the S&P 500.",
            "detail": ("None of the 23 testable angles passed the pre-registered gate (t ≥ 3, deflated Sharpe ≥ 0.95, right sign "
                       "in ≥ 60% of windows, factor alpha t ≥ 2) on 1996–2015 data. The locked 2016–2026 holdout, opened once, "
                       "backed the gate: the best-looking angles before 2016 faded or reversed afterward, and none would have passed. So the "
                       "portfolio is plain cap weight, plus one small rule: buy index additions 5 trading days late."),
        },
        "trials": count_trials(),
        "baselines": bl["baselines"],
        "angles": angles, "untestable": untestable,
        "powerlaw": {"tails_by_decade": tails.to_dict("records"), "zipf": z.to_dict("records"),
                     "p2_test": p3["p2_test"], "p3_gibrat": p3["p3"], "p4": p3["p4"],
                     "p5_late_entry": p3["p5_late_entry"], "p5_tail_capture": p3["p5_tail_capture"],
                     "p8": p3["p8"], "p10": p3["p10"], "p10_kelly": p3["p10_kelly"], "p11": p3["p11"]},
        "ablation": {"groups": p8["groups"], "exploratory_all": p8["exploratory_all"],
                     "table": pd.read_csv(REPORTS / "phase8_ablation.csv").to_dict("records")},
        "implementation": p8["implementation"],
        "decay": p9,
        "holdout": hold,
        "live": live_block(),
        "rounds": rounds_block(),
        "tournament": tournament_block(),
    }
    (REPORTS / "site_snapshot.json").write_text(json.dumps(_clean(snap), indent=1))
    print("wrote", REPORTS / "site_snapshot.json")


if __name__ == "__main__":
    main()
