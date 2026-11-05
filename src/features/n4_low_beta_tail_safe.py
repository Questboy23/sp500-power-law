import pandas as pd

def signal(ctx, t):
    f = ctx.fiam(t, ["betadown_252d"])
    if not len(f):
        return pd.Series(dtype=float)
    return (-1 * f["betadown_252d"]).reindex(ctx.sp(t)).dropna()
