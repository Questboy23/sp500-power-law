"""LIVE_SHADOW multipliers from live data (angles.yaml: live_shadow). Same definitions as the
historical feature modules, recomputed from Yahoo daily data because CRSP/FIAM end before today."""
from __future__ import annotations
import numpy as np, pandas as pd
from config import PROC, REPORTS, TILT_MAX, TE_CAP
from src.backtest import rank_to_unit
from src.features.a9_fluent_ticker import pronounceable
from src.live.engine import yh


def _daily(tickers, years=6):
    import yfinance as yf
    df = yf.download([yh(t) for t in tickers], period=f"{years}y", auto_adjust=False, progress=False,
                     threads=True, group_by="ticker")
    lv0 = set(df.columns.get_level_values(0))
    adj = pd.DataFrame({t: df[yh(t)]["Adj Close"] for t in tickers if yh(t) in lv0})
    vol = pd.DataFrame({t: df[yh(t)]["Volume"] * df[yh(t)]["Close"] for t in tickers if yh(t) in lv0})
    adj.index = vol.index = pd.to_datetime(adj.index).tz_localize(None)
    return adj.dropna(how="all"), vol


def _zipf_dev(w: pd.Series) -> float:
    c = w.sort_values(ascending=False).values
    r = np.arange(1, len(c) + 1)
    b, a = np.polyfit(np.log(r[10:500]), np.log(c[10:500]), 1)
    imp = np.exp(a) * r[:10] ** b
    return float(c[:10].sum() / c.sum() - imp.sum() / (imp.sum() + c[10:].sum()))


def _zipf_history() -> pd.Series:
    """June deviations: 1926-2015 from Phase 3, 2016+ from the panel's caps (caps only)."""
    z = pd.read_csv(REPORTS / "p2_zipf.csv", parse_dates=["date"]).set_index("date").deviation
    p = pd.read_parquet(PROC / "panel_monthly.parquet")
    for d in sorted(p.date.unique()):
        d = pd.Timestamp(d)
        if d.month == 7 and d.year >= 2016:       # mcap_lag in July rows = June 30 caps
            c = p[p.date == d].mcap_lag.dropna()
            if len(c) > 400:
                z[d - pd.offsets.MonthEnd(1)] = _zipf_dev(c)
    return z.sort_index()


def compute(w: pd.Series, asof: pd.Timestamp) -> tuple[dict, dict]:
    tick = list(w.index)
    adj, dvol = _daily(tick)
    adj = adj.loc[:asof]; dvol = dvol.loc[:asof]
    ret = adj.pct_change(fill_method=None)
    sig = {}
    m = adj.resample("ME").last()
    sig["P4_tail_momentum"] = (m.iloc[-1] / m.iloc[-61] - 1) if len(m) > 61 else pd.Series(dtype=float)
    sig["P9_hype_max"] = -ret.iloc[-21:].max()
    sig["N3_volume_spike"] = -(dvol.iloc[-21:].sum() / (dvol.iloc[-252:].sum() / 12))
    sig["A9_fluent_ticker"] = pd.Series({t: float(pronounceable(t)) for t in tick})
    lk = pd.read_parquet(PROC / "membership_linked.parquet")
    t2p = lk[lk.date == lk.date.max()].drop_duplicates("ticker").set_index("ticker").permno
    p2t = {p: t for t, p in t2p.items()}
    a4 = pd.read_parquet(PROC / "a4_insider.parquet")
    a4["formation_date"] = pd.to_datetime(a4.formation_date)
    a4 = a4[a4.formation_date == a4.formation_date.max()].drop_duplicates("permno").set_index("permno").a4
    sig["A4_insider_buys"] = pd.Series({p2t[p]: v for p, v in a4.items() if p in p2t})
    tf = PROC / "text_features.parquet"
    if tf.exists():
        d = pd.read_parquet(tf); d["filed"] = pd.to_datetime(d.filed)
        recent = d[(d.filed <= asof) & (d.filed > asof - pd.DateOffset(months=15))]
        a5 = recent.dropna(subset=["a5_similarity"]).sort_values("filed").groupby("permno").a5_similarity.last()
        sig["A5_lazy_filings"] = pd.Series({p2t[p]: v for p, v in a5.items() if p in p2t})
        def b4_at(x):
            z = d[(d.filed <= x) & (d.filed > x - pd.DateOffset(months=15))].dropna(subset=["b4_rival_mentions"])
            return z.sort_values("filed").groupby("permno").b4_rival_mentions.last()
        b4 = (b4_at(asof) - b4_at(asof - pd.DateOffset(years=3))).dropna()
        sig["B4_named_as_rival"] = pd.Series({p2t[p]: v for p, v in b4.items() if p in p2t})
    z = _zipf_history()
    # same basis as the history (total-share caps from the panel), not SPY's float weights
    pan = pd.read_parquet(PROC / "panel_monthly.parquet")
    last = pan[pan.date == pan.date.max()]
    dev = _zipf_dev(last.mcap.dropna() if last.mcap.notna().sum() > 400 else last.mcap_lag.dropna())
    hist = z[z.index > asof - pd.DateOffset(years=10)]
    p2_on = bool(dev > hist.median())
    sig["P2_zipf_switch"] = ((1 / len(w)) / w - 1).clip(-1, 1) if p2_on else pd.Series(0.0, index=w.index)
    units = [rank_to_unit(s.reindex(tick).dropna()) if k != "P2_zipf_switch" else s for k, s in sig.items() if len(s)]
    S = pd.concat(units, axis=1).mean(1).reindex(tick).fillna(0).clip(-1, 1)
    k = 1 + TILT_MAX * S
    # ex-ante TE on trailing 36 months, scale down if above cap
    mr = m.pct_change(fill_method=None).iloc[-36:]
    ww = w * k; ww /= ww.sum()
    te = float((mr.fillna(0) * (ww - w).reindex(mr.columns).fillna(0)).sum(1).std() * np.sqrt(12))
    scale = min(1.0, TE_CAP / te) if te > 0 else 1.0
    k = 1 + TILT_MAX * scale * S
    info = {"asof": str(asof.date()), "zipf_deviation": dev, "zipf_trailing10y_median": float(hist.median()),
            "p2_switch_on": p2_on, "te_ex_ante_before_scale": te, "te_scale": scale,
            "coverage": {k2: int(v.reindex(tick).notna().sum()) for k2, v in sig.items()},
            "k_min": float(k.min()), "k_max": float(k.max())}
    return {t: float(v) for t, v in k.items()}, info
