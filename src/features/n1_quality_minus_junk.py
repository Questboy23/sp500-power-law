import pandas as pd

def signal(ctx, t):
    f = ctx.fiam(t, ["qmj"])
    if not len(f):
        return pd.Series(dtype=float)
    return (1 * f["qmj"]).reindex(ctx.sp(t)).dropna()
