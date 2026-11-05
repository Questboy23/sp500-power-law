import pandas as pd
from src.features.a5_lazy_filings import _tf

def signal(ctx, t):
    """3y change in #other firms' 10-Ks mentioning this firm near 'compet' (counts as of t vs t-3y)."""
    d = _tf().dropna(subset=["b4_rival_mentions"])
    def at(x):
        z = d[(d.filed <= x) & (d.filed > x - pd.DateOffset(months=15))]
        return z.sort_values("filed").groupby("permno").b4_rival_mentions.last()
    now, then = at(t), at(t - pd.DateOffset(years=3))
    s = (now - then.reindex(now.index)).dropna()
    return s.reindex(ctx.sp(t)).dropna().astype(float)
