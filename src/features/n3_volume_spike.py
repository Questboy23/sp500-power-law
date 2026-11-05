import numpy as np, pandas as pd

def signal(ctx, t):
    """-(formation-month dollar volume / trailing 12m mean), CRSP monthly volume."""
    v = ctx.sic[(ctx.sic.date > t - pd.offsets.MonthEnd(12)) & (ctx.sic.date <= t)]
    if not len(v) or v.date.max() < t:
        return pd.Series(dtype=float)
    w = v.pivot(index="date", columns="permno", values="vol")
    spike = w.loc[t] / w.mean()
    return (-spike.replace([np.inf, -np.inf], np.nan)).reindex(ctx.sp(t)).dropna()
