import json, numpy as np, pandas as pd
from config import PROC, REPORTS
from src.top4.tournament import score_all, evaluate, summarize, ROUNDS, rk
from src.trials import log_trial
H = {}
hv = lambda d: rk(d.vol36) >= 2 / 3
H["Y01_profile_composite_volatile"] = lambda d: (rk(d.qmj) + rk(d.near_high12) + rk(d.gp_at) - rk(d.maxday) - rk(d.dd36)).where(hv(d))
H["Y02_strong_volatile"] = lambda d: d.near_high12.where(hv(d))
H["Y03_quality_volatile"] = lambda d: (rk(d.qmj) + rk(d.gp_at)).where(hv(d))
H["Y04_profile_all_members"] = lambda d: rk(d.qmj) + rk(d.near_high12) + rk(d.gp_at) - rk(d.maxday)
H["Y05_tailprob_x_profile"] = lambda d: rk(d.vol36) + rk(rk(d.qmj) + rk(d.near_high12) + rk(d.gp_at) - rk(d.maxday))
P = pd.read_parquet(PROC / "top4_panel.parquet")
sc = score_all(P, H, ml=False)
res = {}
f = lambda x: f"{x*100:+5.1f}%"
for h, s in sc.items():
    log_trial(h, "batch5", "top4")
    r = {k: summarize(evaluate(P, s, ROUNDS[k])) for k in (1, 2, 3)}
    adv = bool(r[2] and r[2]["hit_rate"] > 0.04 and r[2]["excess_mean"] > 0)
    res[h] = {"round1_in_sample": r[1], "round2": r[2], "round3": r[3], "promoted_to_round3": adv}
    a, b, c = r[1], r[2], r[3]
    print(f"{h:32s} R1(in-sample) {a['hit_rate']*100:4.1f}% picks {f(a['picks_cagr'])} | R2 {b['hit_rate']*100:4.1f}% losers {b['loser_rate']*100:4.1f}% picks {f(b['picks_cagr'])} vs S&P {f(b['sp_cagr'])} {'→' if adv else 'x'} | R3 {c['hit_rate']*100:4.1f}% picks {f(c['picks_cagr'])} vs {f(c['sp_cagr'])} ({c['years']}y)")
json.dump(res, open(REPORTS / "top4_batch5.json", "w"), indent=1, default=float)
