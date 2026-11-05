import numpy as np, pandas as pd

def signal(ctx, t):
    """Slope of gross margin (gp_at / at_turnover) on log sales over 5 annual readings, if sales grew;
    0 when sales shrank. Positive slope = margins widen with scale."""
    rows = []
    for k in range(5):
        d = t - pd.offsets.MonthEnd(12 * k)
        f = ctx.fiam(d, ["gp_at", "at_turnover", "sale_me", "market_equity"])
        if not len(f):
            continue
        f = f.assign(gm=f.gp_at / f.at_turnover, ls=np.log((f.sale_me * f.market_equity).where(lambda s: s > 0)), k=k)
        rows.append(f[["gm", "ls", "k"]].dropna())
    if len(rows) < 4:
        return pd.Series(dtype=float)
    x = pd.concat(rows).reset_index()
    def one(g):
        if len(g) < 4 or g.ls.std() == 0:
            return np.nan
        grew = g.sort_values("k").ls.iloc[0] > g.sort_values("k").ls.iloc[-1]
        return np.polyfit(g.ls, g.gm, 1)[0] if grew else 0.0
    s = x.groupby("permno").apply(one, include_groups=False).clip(-2, 2)
    return s.reindex(ctx.sp(t)).dropna()
