"""Point-in-time feature panel for the top-4% tournament: one row per (June formation, S&P member)."""
from __future__ import annotations
import numpy as np, pandas as pd
from config import PROC, RAW

FORMS = [pd.Timestamp(f"{y}-06-30") for y in range(2000, 2026)]
FIAM_EXTRA = ["sale_gr1", "sale_gr3", "rd5_at", "rd_sale", "capx_gr3", "op_at", "at_gr1", "sale_emp_gr1", "niq_su",
              "eps_stdevest", "eps_meanest", "sale_me", "ebit_sale", "age", "eqnpo_12m", "be_me", "gp_at", "at_turnover",
              "market_equity", "prc_highprc_252d", "ret_12_1", "ivol_capm_252d", "rmax1_21d", "qmj", "z_score", "o_score"]


def build():
    from src.features.context import Context
    ctx = Context()
    lab = pd.read_parquet(PROC / "labels.parquet"); lab["formation_date"] = pd.to_datetime(lab.formation_date)
    f = pd.read_parquet(RAW / "delisted" / "fiam_monthly_2000_2023.parquet", columns=["date", "permno"] + FIAM_EXTRA)
    f["date"] = pd.to_datetime(f.date.astype(str), format="%Y%m%d") + pd.offsets.MonthEnd(0)
    f = f[f.date.dt.month == 6].drop_duplicates(["date", "permno"]).set_index(["date", "permno"])
    tf = pd.read_parquet(PROC / "text_features.parquet"); tf["filed"] = pd.to_datetime(tf.filed)
    emb = pd.read_parquet(PROC / "text_embeddings.parquet"); emb["filed"] = pd.to_datetime(emb.filed)
    ecols = [c for c in emb.columns if c.startswith("emb_")]
    a4 = pd.read_parquet(PROC / "a4_insider.parquet"); a4["formation_date"] = pd.to_datetime(a4.formation_date)
    a4 = a4.drop_duplicates(["formation_date", "permno"]).set_index(["formation_date", "permno"]).a4
    R, C = ctx.R, ctx.C
    base0 = (ctx.R * ctx.C.shift(1)).sum(1) / ctx.C.shift(1).where(ctx.R.notna()).sum(1)
    rows = []
    for t in FORMS:
        L = lab[(lab.formation_date == t)].set_index("permno")
        mem = [p for p in ctx.sp(t) if p in L.index]
        d = pd.DataFrame(index=pd.Index(mem, name="permno"))
        d["formation"] = t; d["fwd_1y"] = L.fwd_1y.reindex(mem)
        r = R.loc[:t, [p for p in mem if p in R.columns]]
        r36, r12 = r.iloc[-36:], r.iloc[-12:]
        idx = (1 + r.iloc[-37:].fillna(0)).cumprod()
        cap = C.loc[t].reindex(mem)
        d["cap"] = cap
        d["vol36"] = r36.std() * np.sqrt(12)
        d["ret12_1"] = (1 + r.iloc[-12:-1].fillna(0)).prod() - 1
        d["ret12"] = (1 + r12.fillna(0)).prod() - 1
        d["ret6"] = (1 + r.iloc[-6:].fillna(0)).prod() - 1
        d["ret6prev"] = (1 + r.iloc[-12:-6].fillna(0)).prod() - 1
        d["near_high12"] = idx.iloc[-1] / idx.iloc[-13:].max()
        d["at_high36"] = (idx.iloc[-1] >= idx.max() * 0.999).astype(float)
        d["dd36"] = 1 - idx.iloc[-1] / idx.max()
        d["skew36"] = r36.skew()
        m36 = base0.loc[r36.index]
        d["beta36"] = r36.apply(lambda s: np.cov(s.fillna(0), m36)[0, 1] / m36.var())
        up = m36 > 0
        d["up_semi"] = r36.apply(lambda s: np.sqrt((s[s > 0] ** 2).mean()) if (s > 0).sum() > 3 else np.nan)
        d["down_semi"] = r36.apply(lambda s: np.sqrt((s[s < 0] ** 2).mean()) if (s < 0).sum() > 3 else np.nan)
        def bt(s, msk):
            x, y = m36[msk], s[msk].fillna(0)
            return np.cov(y, x)[0, 1] / x.var() if msk.sum() > 5 else np.nan
        d["beta_up"] = r36.apply(lambda s: bt(s, up)); d["beta_down"] = r36.apply(lambda s: bt(s, ~up))
        d["tail_asym"] = (r36 > 0.10).sum() - (r36 < -0.10).sum()
        mk12 = float(np.prod(1 + base0.loc[r12.index]) - 1)
        d["mkt12"] = mk12
        crank_now = cap.rank(ascending=False)
        t36 = t - pd.offsets.MonthEnd(36)
        cap36 = C.loc[t36].reindex(mem) if t36 in C.index else pd.Series(np.nan, index=mem)
        d["rank_climb"] = np.log(cap36.rank(ascending=False)) - np.log(crank_now)
        dm = ctx.dmax[ctx.dmax.date == t].set_index("permno").maxret
        d["maxday"] = dm.reindex(mem)
        sic = ctx.sic2(t).reindex(mem); d["sic2"] = sic
        d["leader_gap"] = np.log(cap.groupby(sic).transform("max")) - np.log(cap)
        # FIAM
        if t in f.index.get_level_values(0):
            ff = f.loc[t].reindex(mem)
            for c in FIAM_EXTRA:
                d[c] = ff[c]
        # text
        rec = tf[(tf.filed <= t) & (tf.filed > t - pd.DateOffset(months=15))].sort_values("filed").groupby("permno").last()
        d["a5"] = rec.a5_similarity.reindex(mem)
        def b4_at(x):
            z = tf[(tf.filed <= x) & (tf.filed > x - pd.DateOffset(months=15))].sort_values("filed").groupby("permno").last()
            return z.b4_rival_mentions_v2
        d["b4_chg"] = (b4_at(t) - b4_at(t - pd.DateOffset(years=3))).reindex(mem)
        # embeddings
        er = emb[(emb.filed <= t) & (emb.filed > t - pd.DateOffset(months=15))].sort_values("filed")
        last = er.groupby("permno").last()
        prevmask = emb[(emb.filed <= t - pd.DateOffset(months=9)) & (emb.filed > t - pd.DateOffset(months=27))].sort_values("filed").groupby("permno").last()
        E = last[ecols].reindex(mem)
        En = E.div(np.linalg.norm(E.values, axis=1), axis=0)
        cent = En.groupby(sic).transform("mean")
        cn = cent.div(np.linalg.norm(cent.values, axis=1), axis=0)
        d["emb_unique"] = 1 - (En * cn).sum(1).where(E.notna().all(1))
        P = prevmask[ecols].reindex(mem); Pn = P.div(np.linalg.norm(P.values, axis=1), axis=0)
        d["emb_pivot"] = 1 - (En * Pn).sum(1).where(E.notna().all(1) & P.notna().all(1))
        d["_emb"] = list(En.values)
        d["a4"] = a4.loc[t].reindex(mem) if t in a4.index.get_level_values(0) else np.nan
        rows.append(d.reset_index())
    P = pd.concat(rows, ignore_index=True)
    # labels: top 4% within formation (by fwd_1y)
    P["top4"] = P.groupby("formation").fwd_1y.transform(lambda s: (s >= s.quantile(0.96)).astype(float).where(s.notna()))
    P["bot4"] = P.groupby("formation").fwd_1y.transform(lambda s: (s <= s.quantile(0.04)).astype(float).where(s.notna()))
    # Q28: similarity to previous 3 formations' winners (labels ended by t)
    sims = []
    for t in FORMS:
        cur = P[P.formation == t]
        prev = P[(P.formation < t) & (P.formation >= t - pd.DateOffset(years=3)) & (P.top4 == 1)]
        W = np.array([v for v in prev._emb if not np.isnan(v).any()]) if len(prev) else np.empty((0, 64))
        s = []
        for v in cur._emb:
            s.append(float(np.max(W @ v)) if len(W) and not np.isnan(v).any() else np.nan)
        sims.append(pd.Series(s, index=cur.index))
    P["sim_winners"] = pd.concat(sims)
    P = P.drop(columns="_emb")
    P.to_parquet(PROC / "top4_panel.parquet")
    return P


if __name__ == "__main__":
    P = build()
    print(P.shape); print(P.groupby(P.formation.dt.year).agg(n=("permno", "size"), top4=("top4", "sum"), fiam=("sale_gr1", lambda s: s.notna().mean()),
          emb=("emb_unique", lambda s: s.notna().mean()), sim=("sim_winners", lambda s: s.notna().mean())).round(2).T.to_string())
