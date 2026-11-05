"""Walk-forward engine (Section 8 + 9.1).

Portfolio in month m = S&P members held during m (panel rows), weights
    w_i ∝ b_i * k_i,   b_i = cap weight (prior month-end cap),  k_i = 1 + TILT_MAX * c * s_i
where s_i ∈ [-1, 1] is the angle's signal at the most recent June formation, c ∈ [0, 1] a
confidence scale (1 for single-angle tests, walk-forward shrinkage for the combined model),
scaled down further if the ex-ante tracking error (trailing 36m, holdings-based simulation)
exceeds TE_CAP. Members with no signal get k = 1. So s ≡ 0 reproduces BASE0 exactly.

Costs: COST_BPS per unit of ACTIVE turnover (|Δ(w-b)| after drift) + SMALL_EXTRA_BPS on the
smallest 100 names. BASE0 is charged nothing (index fund)."""
from __future__ import annotations
import numpy as np, pandas as pd
from config import PROC, TILT_MAX, TE_CAP, COST_BPS, SMALL_EXTRA_BPS, HOLDOUT_START
from src.holdout_guard import assert_pre_holdout


def rank_to_unit(sig: pd.Series) -> pd.Series:
    """Cross-sectional percentile -> [-1, 1]. Binary 0/1 signals map to {0, +1} (flag = overweight)."""
    s = sig.dropna()
    if len(s) == 0:
        return s
    u = set(np.unique(s.values))
    if u <= {0.0, 1.0}:
        return s.astype(float)
    r = s.rank(pct=True, method="average")
    return (2 * (r - r.min()) / (r.max() - r.min()) - 1) if r.max() > r.min() else s * 0


class Engine:
    def __init__(self, end=None):
        p = pd.read_parquet(PROC / "panel_monthly.parquet")
        p = p.dropna(subset=["ret", "mcap_lag"])
        p = p[p.mcap_lag > 0]
        self.end = pd.Timestamp(end) if end is not None else HOLDOUT_START - pd.offsets.MonthEnd(1)
        assert_pre_holdout(pd.Series(index=pd.DatetimeIndex([self.end]), dtype=float), "Engine end")
        p = p[p.date <= self.end]
        self.p = p
        self.months = sorted(p.date.unique())
        self.R = p.pivot_table(index="date", columns="permno", values="ret", aggfunc="first")
        self.B = p.pivot_table(index="date", columns="permno", values="mcap_lag", aggfunc="first")
        self.B = self.B.div(self.B.sum(1), axis=0)
        self.base0 = (self.B * self.R).sum(1)

    # ---------- ex-ante TE ----------
    def ex_ante_te(self, t, active: pd.Series) -> float:
        hist = self.R.loc[(self.R.index > t - pd.offsets.MonthEnd(36)) & (self.R.index <= t)]
        if len(hist) < 12:
            return np.nan
        a = active.reindex(hist.columns).fillna(0)
        ar = (hist.fillna(0) * a).sum(1)
        return float(ar.std() * np.sqrt(12))

    def multipliers(self, t, s: pd.Series, conf: float = 1.0, te_cap=TE_CAP) -> tuple[pd.Series, float, float]:
        """Return k (index permno) for formation date t, the scale applied, and ex-ante TE."""
        m_next = t + pd.offsets.MonthEnd(1)
        if m_next not in self.B.index:
            return pd.Series(dtype=float), 0.0, np.nan
        b = self.B.loc[m_next].dropna()
        s = s.reindex(b.index).fillna(0).clip(-1, 1)
        k = 1 + TILT_MAX * conf * s
        w = b * k; w = w / w.sum()
        te = self.ex_ante_te(t, w - b)
        scale = 1.0
        if te_cap is not None and np.isfinite(te) and te > te_cap:
            scale = te_cap / te
            k = 1 + TILT_MAX * conf * scale * s
        return k, scale, te

    def run(self, formation_signals: dict, conf: dict | float = 1.0, te_cap=TE_CAP, extra=None, name="") -> dict:
        """formation_signals: {formation_date: pd.Series(permno -> unit signal)}.
        extra: optional {formation_date: (pd.Series weights_outside_index, pd.DataFrame returns)} for P5."""
        forms = sorted(formation_signals)
        K = {}; scales = {}; tes = {}
        for t in forms:
            c = conf.get(t, 0.0) if isinstance(conf, dict) else conf
            K[t], scales[t], tes[t] = self.multipliers(t, formation_signals[t], c, te_cap)
        rets, costs, tos = {}, {}, {}
        prev_a = None; prev_w = None; prev_b = None
        for m in self.months:
            f = [t for t in forms if t < m]
            b = self.B.loc[m].dropna()
            if f:
                k = K[f[-1]].reindex(b.index).fillna(1.0)
            else:
                k = pd.Series(1.0, index=b.index)
            w = b * k; w = w / w.sum()
            r = self.R.loc[m].reindex(b.index).fillna(0)
            gross = float((w * r).sum())
            a = w - b
            if prev_w is not None:
                rp = self.R.loc[m].reindex(prev_w.index).fillna(0)   # not used for drift: drift uses last month's returns
            if prev_a is not None:
                idx = a.index.union(prev_a.index)
                to = float((a.reindex(idx).fillna(0) - prev_a.reindex(idx).fillna(0)).abs().sum())
            else:
                to = float(a.abs().sum())
            small = b.nsmallest(100).index
            to_small = float((a.reindex(small) - (prev_a.reindex(small).fillna(0) if prev_a is not None else 0)).abs().sum())
            cost = to * COST_BPS / 1e4 + to_small * SMALL_EXTRA_BPS / 1e4
            # drift the active weights through this month's returns for next month's turnover
            wd = w * (1 + r); wd = wd / wd.sum(); bd = b * (1 + r); bd = bd / bd.sum()
            prev_a = wd - bd; prev_w = w
            rets[m] = gross; costs[m] = cost; tos[m] = to
        g = pd.Series(rets); c = pd.Series(costs)
        return {"name": name, "gross": g, "net": g - c, "cost": c, "turnover": pd.Series(tos),
                "scales": pd.Series(scales), "te_ex_ante": pd.Series(tes), "K": K}


# ---------------- baselines ----------------
def base1_equal_weight(E: Engine) -> pd.Series:
    return E.R.where(E.B.notna()).mean(1)


def base2_total_market(E: Engine) -> pd.Series:
    m = pd.read_parquet(PROC / "market_monthly.parquet", columns=["permno", "date", "ret", "mcap", "shrcd"])
    m = m[m.shrcd.isin([10, 11])].sort_values(["permno", "date"])
    m["w"] = m.groupby("permno").mcap.shift(1)
    m = m.dropna(subset=["ret", "w"])
    s = m.groupby("date").apply(lambda g: np.average(g.ret, weights=g.w), include_groups=False)
    return s.reindex(E.months)


def factor_signal(ctx, t) -> pd.Series:
    """BASE3: mean rank of value (be_me), momentum (12-1), profitability (ope_be). Momentum from
    CRSP pre-2000 when FIAM is missing."""
    mem = ctx.sp(t)
    f = ctx.fiam(t, ["be_me", "ret_12_1", "ope_be"]).reindex(mem)
    mom = f.ret_12_1 if f.ret_12_1.notna().any() else None
    if mom is None or mom.isna().all():
        r = ctx.R.loc[(ctx.R.index > t - pd.offsets.MonthEnd(12)) & (ctx.R.index <= t - pd.offsets.MonthEnd(1))]
        mom = ((1 + r.fillna(0)).prod() - 1).reindex(mem)
    parts = [x.rank(pct=True) for x in (f.be_me, mom, f.ope_be) if x.notna().sum() > 50]
    return pd.concat(parts, axis=1).mean(1).dropna()


def lowvol_signal(ctx, t) -> pd.Series:
    r = ctx.R.loc[(ctx.R.index > t - pd.offsets.MonthEnd(36)) & (ctx.R.index <= t)]
    v = r.std()[r.notna().sum() >= 24]
    return (-v).reindex(ctx.sp(t)).dropna()
