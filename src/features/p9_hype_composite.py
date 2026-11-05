import pandas as pd

def signal(ctx, t):
    """-(mean rank of MAX, turnover_126d, and within-SIC2 expensiveness (-be_me))."""
    f = ctx.fiam(t, ["rmax1_21d", "turnover_126d", "be_me"])
    if not len(f):
        return pd.Series(dtype=float)
    f = f.reindex(ctx.sp(t)).dropna(how="all")
    f["sic2"] = ctx.sic2(t).reindex(f.index)
    exp = (-f.be_me).groupby(f.sic2).rank(pct=True)
    comp = pd.concat([f.rmax1_21d.rank(pct=True), f.turnover_126d.rank(pct=True), exp], axis=1).mean(1)
    return (-comp).dropna()
