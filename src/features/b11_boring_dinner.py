import pandas as pd

def signal(ctx, t):
    """PROXY: -turnover_126d rank within SIC2 (low trading attention)."""
    f = ctx.fiam(t, ["turnover_126d"])
    if not len(f):
        return pd.Series(dtype=float)
    f = f.reindex(ctx.sp(t)).dropna()
    sic = ctx.sic2(t).reindex(f.index)
    return (-f.turnover_126d).groupby(sic).rank(pct=True).dropna()
