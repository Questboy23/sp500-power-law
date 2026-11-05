import json, numpy as np, pandas as pd
from config import PROC, REPORTS, RAW
from src.top4.tournament import run, show, rk

def extras(P):
    from src.features.context import Context
    ctx = Context()
    emb = pd.read_parquet(PROC / "text_embeddings.parquet"); emb["filed"] = pd.to_datetime(emb.filed)
    ec = [c for c in emb.columns if c.startswith("emb_")]
    f = pd.read_parquet(RAW / "delisted" / "fiam_monthly_2000_2023.parquet", columns=["date", "permno", "ni_be", "gp_at", "at_turnover"])
    f["date"] = pd.to_datetime(f.date.astype(str), format="%Y%m%d") + pd.offsets.MonthEnd(0)
    f = f.drop_duplicates(["date", "permno"]).set_index(["date", "permno"])
    out = []
    for t in sorted(P.formation.unique()):
        d = P[P.formation == t].copy(); idx = d.index; pm = d.permno.values
        # industry returns
        ind12 = (d.cap * d.ret12).groupby(d.sic2).transform("sum") / d.cap.groupby(d.sic2).transform("sum")
        d["ind12"] = ind12; d["rel12"] = d.ret12 - ind12
        R = ctx.R.loc[:t, [p for p in pm if p in ctx.R.columns]]
        r36 = (1 + R.iloc[-36:].fillna(0)).prod() - 1
        d["ret36"] = d.permno.map(r36)
        d["ind36"] = (d.cap * d.ret36).groupby(d.sic2).transform("sum") / d.cap.groupby(d.sic2).transform("sum")
        d["vol6"] = d.permno.map(R.iloc[-6:].std() * np.sqrt(12))
        prev = P[P.formation == t - pd.DateOffset(years=1)]
        share = prev.groupby("sic2").top4.mean()
        d["ind_winner_share"] = d.sic2.map(share)
        # embeddings
        e = emb[(emb.filed <= t) & (emb.filed > t - pd.DateOffset(months=15))].sort_values("filed").groupby("permno").last()[ec]
        E = e.reindex(pm); ok = E.notna().all(1).values
        En = E.values / np.linalg.norm(E.values, axis=1, keepdims=True)
        S = En[ok] @ En[ok].T; np.fill_diagonal(S, -1)
        r12 = d.ret12.values[ok]
        nb = np.argsort(-S, axis=1)[:, :10]
        nbret = np.nanmean(r12[nb], axis=1)
        d["nb_ret12"] = np.nan; d.loc[idx[ok], "nb_ret12"] = nbret
        win = P[(P.formation < t) & (P.formation >= t - pd.DateOffset(years=5)) & (P.top4 == 1)]
        we = emb[(emb.filed <= t)].sort_values("filed")
        W = []
        for _, w in win.iterrows():
            z = we[(we.permno == w.permno) & (we.filed <= w.formation) & (we.filed > w.formation - pd.DateOffset(months=15))]
            if len(z):
                v = z[ec].iloc[-1].values; W.append(v / np.linalg.norm(v))
        d["sim5"] = np.nan
        if W:
            d.loc[idx[ok], "sim5"] = (En[ok] @ np.array(W).T).max(1)
        # FIAM lags
        if t in f.index.get_level_values(0) and (t - pd.offsets.MonthEnd(12)) in f.index.get_level_values(0):
            a = f.loc[t].reindex(pm); b = f.loc[t - pd.offsets.MonthEnd(12)].reindex(pm)
            d["first_profit"] = ((a.ni_be.values > 0) & (b.ni_be.values < 0)).astype(float)
            d["first_profit"] = d.first_profit.where(~np.isnan(a.ni_be.values) & ~np.isnan(b.ni_be.values))
            d["gm_chg"] = (a.gp_at / a.at_turnover).values - (b.gp_at / b.at_turnover).values
        out.append(d)
    return pd.concat(out)

def hyps():
    H = {}
    H["X01_catch_up"] = lambda d: rk(d.ind12) + rk(-d.rel12)
    H["X02_winner_industries"] = lambda d: d.ind_winner_share + rk(d.vol36) * 0.001
    H["X03_text_theme_laggard"] = lambda d: d.nb_ret12 - d.ret12
    H["X04_text_theme_momentum"] = lambda d: d.nb_ret12
    H["X05_first_profits"] = lambda d: d.first_profit + rk(d.sale_gr1).fillna(0) * 0.01
    H["X06_margin_expansion"] = lambda d: d.gm_chg
    H["X07_coiled_spring"] = lambda d: rk(d.vol36 / d.vol6) + rk(d.ret12_1)
    H["X08_sales_up_price_down"] = lambda d: rk(d.sale_gr1) - rk(d.ret12)
    H["X09_beaten_sector_best_firm"] = lambda d: rk(-d.ind36) + rk(d.qmj)
    H["X10_like_5y_winners"] = lambda d: d.sim5
    H["X11_insider_buying_in_volatile"] = lambda d: d.a4.where(rk(d.vol36) >= 0.5)
    return H

if __name__ == "__main__":
    from src.top4.tournament import score_all
    P = extras(pd.read_parquet(PROC / "top4_panel.parquet"))
    sc = score_all(P, hyps(), ml=False)
    res = run(P, sc, "batch4")
    json.dump(res, open(REPORTS / "top4_batch4.json", "w"), indent=1, default=float)
    pd.DataFrame(sc).assign(formation=P.formation.values, permno=P.permno.values).to_parquet(PROC / "top4_scores_batch4.parquet")
    show(res)
    print("\nwinners vs losers among picks, round 1:")
    for h, r in res.items():
        if r["round1"]:
            print(f"  {h:32s} winners {r['round1']['hit_rate']*100:4.1f}%  losers {r['round1']['loser_rate']*100:4.1f}%  coverage {r['round1']['coverage']:.0%}")
