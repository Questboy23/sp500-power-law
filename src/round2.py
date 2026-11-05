"""Round 2: ways to improve the S&P 500 (pre-registered in angles.yaml: round2). See header there."""
import json, numpy as np, pandas as pd
from scipy import stats
from config import RAW, PROC, REPORTS
from src.trials import log_trial, count_trials
from src.evaluate import deflated_sharpe

F = RAW / "french" / "12_Industry_Portfolios.csv"
IND = ["NoDur", "Durbl", "Manuf", "Enrgy", "Chems", "BusEq", "Telcm", "Utils", "Shops", "Hlth", "Money", "Other"]

def section(title, annual=False):
    lines = open(F).read().splitlines()
    i = next(k for k, l in enumerate(lines) if l.strip() == title)
    rows = []
    for l in lines[i + 2:]:
        p = [s.strip() for s in l.split(",")]
        if len(p) < 13 or not p[0].isdigit():
            break
        rows.append(p)
    df = pd.DataFrame(rows, columns=["d"] + IND)
    fmt = "%Y" if annual else "%Y%m"
    df["d"] = pd.to_datetime(df.d, format=fmt) + (pd.offsets.YearEnd(0) if annual else pd.offsets.MonthEnd(0))
    df = df.set_index("d").astype(float).replace([-99.99, -999], np.nan)
    return df

def data():
    R = section("Average Value Weighted Returns -- Monthly") / 100
    N = section("Number of Firms in Portfolios"); S = section("Average Firm Size")
    cap = (N * S).shift(1)                                   # lag a month: no look-ahead
    W = cap.div(cap.sum(1), axis=0)
    BM = section("Value-Weighted Average of BE/ME", annual=True)
    from src.powerlaw.descriptive import rf_cum
    rf = rf_cum().reindex(R.index)
    mkt = (W * R).sum(1)
    return R, W, BM, rf, mkt

def tilt(R, W, kfun):
    out = {}
    for d in R.index:
        w = W.loc[d]
        if w.isna().all():
            continue
        k = kfun(d)
        x = w * k; x = x / x.sum()
        out[d] = float((x * R.loc[d]).sum())
    return pd.Series(out)

def const(k_map):
    k = pd.Series(1.0, index=IND)
    for i, v in k_map.items():
        k[i] = v
    return lambda d: k

def ranked(score, top=1.3, bot=0.7):
    def f(d):
        s = score(d)
        k = pd.Series(1.0, index=IND)
        if s is None or s.isna().sum() > 3:
            return k
        r = s.rank()
        k[r >= 10] = top; k[r <= 3] = bot
        return k
    return f

def capeseries():
    x = pd.read_excel(RAW / "shiller_ie_data.xls", sheet_name="Data", skiprows=7)
    x = x[pd.to_numeric(x.Date, errors="coerce").notna()].copy()
    d = x.Date.astype(float); y = d.astype(int); m = ((d - y) * 100).round().astype(int).clip(1, 12)
    s = pd.Series(pd.to_numeric(x.CAPE, errors="coerce").values, index=pd.to_datetime(dict(year=y, month=m, day=1)) + pd.offsets.MonthEnd(0))
    return s

def strategies():
    R, W, BM, rf, mkt = data()
    cum12 = (1 + R).rolling(11).apply(np.prod, raw=True).shift(2) - 1     # t-12..t-2
    cum60 = (1 + R).rolling(60).apply(np.prod, raw=True).shift(1) - 1
    def bm_at(d):
        yr = d.year if d.month >= 7 else d.year - 1                     # BE/ME of year y-1 known by June y
        k = pd.Timestamp(f"{yr - 1}-12-31")
        return BM.loc[k] if k in BM.index else None
    S = {}
    S["R1_tech_overweight"] = tilt(R, W, const({"BusEq": 1.3}))
    S["R1b_tech_double"] = tilt(R, W, const({"BusEq": 2.0}))
    S["R2_health_overweight"] = tilt(R, W, const({"Hlth": 1.3}))
    S["R3_defensive"] = tilt(R, W, const({"NoDur": 1.3, "Utils": 1.3, "Hlth": 1.3}))
    S["R5_sector_momentum"] = tilt(R, W, ranked(lambda d: cum12.loc[d] if d in cum12.index else None))
    S["R6_sector_reversal"] = tilt(R, W, ranked(lambda d: -cum60.loc[d] if d in cum60.index else None))
    S["R7_sector_value"] = tilt(R, W, ranked(bm_at))
    S["R8_equal_sector"] = R.mean(1)
    # timing on the market
    idx = (1 + mkt).cumprod()
    sma = idx.rolling(10).mean()
    on = (idx >= sma).shift(1).fillna(True)
    S["R9_trend_10m"] = mkt.where(on, rf)
    vol = mkt.rolling(6).std().shift(1) * np.sqrt(12)
    w = (0.15 / vol).clip(upper=1.0).fillna(1.0)
    S["R10_vol_target"] = w * mkt + (1 - w) * rf
    S["R11_sell_in_may"] = mkt.where(mkt.index.month.isin([11, 12, 1, 2, 3, 4]), rf)
    cape = capeseries().reindex(mkt.index).shift(1)
    med = capeseries().rolling(240, min_periods=120).median().reindex(mkt.index).shift(1)
    we = np.where(cape > med, 0.8, 1.0)
    S["R12_cape_timing"] = pd.Series(we, index=mkt.index) * mkt + (1 - pd.Series(we, index=mkt.index)) * rf
    tp = pd.read_parquet(PROC / "topn_monthly.parquet")
    # R14 needs top-500 EW: recompute cheaply from market_long
    x = pd.read_parquet(PROC / "market_long.parquet")[["permno", "date", "ret", "mcap"]].sort_values(["permno", "date"])
    x["w0"] = x.groupby("permno").mcap.shift(1); x = x.dropna(subset=["ret", "w0"])
    ew = x.groupby("date").apply(lambda g: g.nlargest(500, "w0").ret.mean(), include_groups=False)
    S["R14_half_equal_weight"] = 0.5 * tp.top500_cw + 0.5 * ew.reindex(tp.index)
    S["_bench_top500"] = tp.top500_cw
    return S, mkt, rf

def evaluate(S, mkt, rf):
    eras = {"1927-2015": ("1927-01-31", "2015-12-31"), "1927-1969": ("1927-01-31", "1969-12-31"),
            "1970-1995": ("1970-01-31", "1995-12-31"), "1996-2015": ("1996-01-31", "2015-12-31"),
            "2016-2026 (hindsight era, not a test)": ("2016-01-31", "2026-08-31")}
    timing = {"R9_trend_10m", "R10_vol_target", "R11_sell_in_may", "R12_cape_timing"}
    N = count_trials()
    out = {}
    for name, s in S.items():
        if name.startswith("_"):
            continue
        bench = S["_bench_top500"] if name == "R14_half_equal_weight" else mkt
        res = {}
        for era, (a, b) in eras.items():
            r = s.loc[a:b].dropna(); m = bench.reindex(r.index); f = rf.reindex(r.index).fillna(0)
            if len(r) < 24:
                continue
            act = r - m
            ex = r - f; mex = m - f
            if name in timing:   # CAPM alpha: does timing beat the right mix of market and T-bills?
                beta, alpha, _, _, se = stats.linregress(mex, ex)
                tstat = alpha / (np.std(ex - alpha - beta * mex, ddof=2) / np.sqrt(len(ex)))
                edge = alpha * 12
            else:
                tstat = act.mean() / (act.std() / np.sqrt(len(act))); edge = act.mean() * 12
            cum = (1 + r).cumprod(); mc = (1 + m).cumprod()
            res[era] = {"cagr": float(cum.iloc[-1] ** (12 / len(r)) - 1), "bench_cagr": float(mc.iloc[-1] ** (12 / len(r)) - 1),
                        "edge_ann": float(edge), "t": float(tstat), "te": float(act.std() * np.sqrt(12)),
                        "sharpe": float(ex.mean() / ex.std() * np.sqrt(12)), "bench_sharpe": float(mex.mean() / mex.std() * np.sqrt(12)),
                        "max_dd": float((cum / cum.cummax() - 1).min()), "bench_max_dd": float((mc / mc.cummax() - 1).min()),
                        "months": len(r)}
        d = res["1927-2015"]
        a = s.loc["1927-01-31":"2015-12-31"] - bench.loc["1927-01-31":"2015-12-31"]
        d["dsr"] = deflated_sharpe(a.dropna(), N)
        signs = [np.sign(res[e]["edge_ann"]) for e in ("1927-1969", "1970-1995", "1996-2015") if e in res]
        d["pass"] = bool(d["t"] >= 3 and d["dsr"] >= 0.95 and len(set(signs)) == 1 and signs[0] > 0)
        out[name] = res
    return out

if __name__ == "__main__":
    S, mkt, rf = strategies()
    for k in S:
        if not k.startswith("_"):
            log_trial(k, "round2", "round2", "1927-2015")
    out = evaluate(S, mkt, rf)
    json.dump(out, open(REPORTS / "round2_results.json", "w"), indent=1, default=float)
    pd.DataFrame({k: v for k, v in S.items()}).to_parquet(PROC / "round2_monthly.parquet")
    print(f"trials now {count_trials()}")
    print(f"{'hypothesis':24s} {'27-15 edge':>10s} {'t':>6s} {'DSR':>5s} {'27-69':>7s} {'70-95':>7s} {'96-15':>7s} {'16-26*':>7s}  Sharpe(27-15) vs mkt   maxDD vs mkt   pass")
    for k, r in out.items():
        e = lambda era: f"{r[era]['edge_ann']*100:+.2f}" if era in r else "  —"
        a = r["1927-2015"]
        print(f"{k:24s} {a['edge_ann']*100:+9.2f}% {a['t']:+6.2f} {a['dsr']:5.2f} {e('1927-1969'):>7s} {e('1970-1995'):>7s} {e('1996-2015'):>7s} {e('2016-2026 (hindsight era, not a test)'):>7s}  {a['sharpe']:.2f} vs {a['bench_sharpe']:.2f}     {a['max_dd']*100:.0f}% vs {a['bench_max_dd']*100:.0f}%   {a['pass']}")
