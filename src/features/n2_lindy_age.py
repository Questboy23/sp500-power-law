import pandas as pd

def signal(ctx, t):
    f = ctx.fiam(t, ["age"])
    if not len(f):
        return pd.Series(dtype=float)
    return (1 * f["age"]).reindex(ctx.sp(t)).dropna()
