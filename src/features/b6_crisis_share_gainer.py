import pandas as pd

CRISES = [(2002, "2001"), (2010, "2008-09"), (2021, "2020")]   # June formation when the crisis-year 10-K is in FIAM

def signal(ctx, t):
    """Sales growth over the most recent crisis year minus its SIC2 median (share gain).
    For 2008-09 use growth over both years (sale_gr1 at June 2009 and June 2010)."""
    avail = [c for c in CRISES if pd.Timestamp(f"{c[0]}-06-30") <= t]
    if not avail:
        return pd.Series(dtype=float)
    y, _ = avail[-1]
    d = pd.Timestamp(f"{y}-06-30")
    g = ctx.fiam(d, ["sale_gr1"]).sale_gr1
    if y == 2010:
        g0 = ctx.fiam(pd.Timestamp("2009-06-30"), ["sale_gr1"]).sale_gr1
        g = (1 + g) * (1 + g0.reindex(g.index)) - 1
    sic = ctx.sic2(d).reindex(g.index)
    rel = g - g.groupby(sic).transform("median")
    return rel.reindex(ctx.sp(t)).dropna()
