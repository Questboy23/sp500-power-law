"""Walk-forward factor selection inside the S&P 500 (angles.yaml: factor_selection)."""
import json, numpy as np, pandas as pd
import pyarrow.parquet as pq
from config import RAW, PROC, REPORTS
from src.backtest import Engine
from src.trials import log_trial, count_trials
from src.evaluate import deflated_sharpe

SKIP = {"date", "permno", "shrcd", "exchcd", "year", "month", "size_port", "rf", "stock_exret", "ret_eom", "stock_ticker", "cusip", "comp_name", "prc"}

def tst(x):
    x = pd.Series(x).dropna(); return float(x.mean() / (x.std() / np.sqrt(len(x)))) if len(x) > 2 and x.std() > 0 else np.nan

def main():
    E = Engine(end="2023-12-31")
    cols = [c for c in pq.read_schema(RAW / "delisted" / "fiam_monthly_2000_2023.parquet").names if c not in SKIP]
    F = pd.read_parquet(RAW / "delisted" / "fiam_monthly_2000_2023.parquet", columns=["date", "permno"] + cols)
    F["date"] = pd.to_datetime(F.date.astype(str), format="%Y%m%d") + pd.offsets.MonthEnd(0)
    F = F.drop_duplicates(["date", "permno"]).set_index(["date", "permno"]).apply(pd.to_numeric, errors="coerce")
    cols = [c for c in cols if F[c].notna().mean() > 0.5]
    months = [pd.Timestamp(m) for m in E.months if pd.Timestamp(m) >= pd.Timestamp("2000-02-29")]
    ranks, frets = {}, {}
    for m in months:
        f = m - pd.offsets.MonthEnd(1)
        if f not in F.index.get_level_values(0):
            continue
        mem = E.B.columns[E.B.loc[m].notna()]
        X = F.loc[f].reindex(mem)[cols]
        Rk = X.rank(pct=True)
        ranks[f] = Rk
        r = E.R.loc[m].reindex(mem); w = E.B.loc[m].reindex(mem)
        fr = {}
        for c in cols:
            q = Rk[c]
            hi, lo = q >= 2 / 3, q <= 1 / 3
            if hi.sum() > 30 and lo.sum() > 30:
                fr[c] = float((r[hi] * w[hi]).sum() / w[hi].sum() - (r[lo] * w[lo]).sum() / w[lo].sum())
        frets[m] = fr
    FR = pd.DataFrame(frets).T.sort_index()          # factor return realized in month m (signal at m-1)
    FR.to_parquet(PROC / "factor_returns_sp500.parquet")
    sig = {"F1_best5_trailing60": {}, "F2_best5_trailing12": {}, "F3_all_tstat_weighted": {}}
    chosen = {}
    for f, Rk in ranks.items():
        hist = FR[FR.index <= f]                      # realized through formation date
        if len(hist) < 60:
            continue
        for name, win in (("F1_best5_trailing60", 60), ("F2_best5_trailing12", 12)):
            mu = hist.iloc[-win:].mean().dropna()
            top = mu.abs().nlargest(5).index
            chosen.setdefault(name, []).append((str(f.date()), {c: round(float(np.sign(mu[c])), 0) for c in top}))
            comp = sum((Rk[c] - 0.5) * np.sign(mu[c]) for c in top) / len(top)
            sig[name][f] = (comp.rank(pct=True) - 0.5) * 2
        h = hist.iloc[-60:]
        t = (h.mean() / (h.std() / np.sqrt(len(h)))).clip(-3, 3).dropna()
        comp = sum((Rk[c].fillna(0.5) - 0.5) * t[c] for c in t.index) / t.abs().sum()
        sig["F3_all_tstat_weighted"][f] = (comp.rank(pct=True) - 0.5) * 2
    N0 = count_trials()
    out = {"n_factors": len(cols)}
    for name, s in sig.items():
        log_trial(name, "factor_selection", "factor")
        res = E.run(s)
        act = (res["net"] - E.base0).dropna(); act = act[act.index > min(s)]
        o = {}
        for lab, a, b in [("2005-2015", "2005", "2015"), ("2016-2023", "2016", "2023"), ("2005-2023", "2005", "2023")]:
            x = act.loc[a:b]; o[lab] = {"excess_ann_net": float(x.mean() * 12), "te": float(x.std() * np.sqrt(12)), "t": tst(x), "months": len(x)}
        o["dsr_2005_2015"] = deflated_sharpe(act.loc["2005":"2015"], count_trials())
        o["turnover_ann"] = float(res["turnover"].reindex(act.index).mean() * 12)
        o["pass"] = bool(o["2005-2015"]["t"] >= 3 and o["dsr_2005_2015"] >= 0.95 and o["2005-2015"]["excess_ann_net"] > 0 and o["2016-2023"]["excess_ann_net"] > 0)
        out[name] = o
    # which factors got picked most often (F1)
    from collections import Counter
    cnt = Counter(c for _, d in chosen["F1_best5_trailing60"] for c in d)
    out["F1_most_picked"] = cnt.most_common(12)
    cnt2 = Counter(c for _, d in chosen["F2_best5_trailing12"] for c in d)
    out["F2_most_picked"] = cnt2.most_common(12)
    # descriptive: best/worst factors in hindsight, 2000-2023 (NOT used by the strategies)
    full = FR.mean() * 12; tt = FR.mean() / (FR.std() / np.sqrt(len(FR)))
    out["hindsight_top_factors"] = pd.DataFrame({"ann": full, "t": tt}).sort_values("t", ascending=False).head(8).round(4).to_dict("index")
    out["hindsight_bottom_factors"] = pd.DataFrame({"ann": full, "t": tt}).sort_values("t").head(5).round(4).to_dict("index")
    # persistence check: does a factor's 2000-2011 return predict its 2012-2023 return?
    a, b = FR.loc[:"2011"].mean(), FR.loc["2012":].mean()
    out["persistence_corr_first_vs_second_half"] = float(a.corr(b))
    json.dump(out, open(REPORTS / "factor_selection.json", "w"), indent=1, default=float)
    return out

if __name__ == "__main__":
    print(json.dumps(main(), indent=1, default=lambda v: round(float(v), 4)))
