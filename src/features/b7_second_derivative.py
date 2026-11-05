import pandas as pd

def signal(ctx, t):
    """+1 if sale_gr1 rose in each of the last 2 years (3 consecutive growth readings rising)."""
    g = [ctx.fiam(t - pd.offsets.MonthEnd(12 * k), ["sale_gr1"]).sale_gr1 for k in (0, 1, 2)]
    if not len(g[0]):
        return pd.Series(dtype=float)
    a, b, c = g[0], g[1].reindex(g[0].index), g[2].reindex(g[0].index)
    s = ((a > b) & (b > c)).astype(float).where(a.notna() & b.notna() & c.notna())
    return s.reindex(ctx.sp(t)).dropna()
