import pandas as pd

def signal(ctx, t):
    """+1 if R&D/sales rose by > 5 points of sales vs 12m earlier (and R&D > 0), else 0."""
    a = ctx.fiam(t, ["rd_sale"]).rd_sale
    b = ctx.fiam(t - pd.offsets.MonthEnd(12), ["rd_sale"]).rd_sale
    if not len(a):
        return pd.Series(dtype=float)
    d = (a - b.reindex(a.index))
    s = ((d > 0.05) & (a > 0)).astype(float).where(a.notna())
    return s.reindex(ctx.sp(t)).fillna(0.0)
