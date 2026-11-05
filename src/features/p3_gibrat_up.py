import numpy as np, pandas as pd

def signal(ctx, t):
    """Industry slope of next-3y sales growth on log sales, estimated on the trailing 5
    formations whose 3y growth is fully observed by t (i.e. formations t-8y .. t-3y... using
    sale_gr3 reported at t-k for growth that ENDED at t-k)."""
    rows = []
    for k in range(0, 5):
        d_end = t - pd.offsets.MonthEnd(12 * k)             # growth observed at d_end covers d_end-36m .. d_end
        d_start = d_end - pd.offsets.MonthEnd(36)
        g = ctx.fiam(d_end, ["sale_gr3"]); s = ctx.fiam(d_start, ["sale_me", "market_equity"])
        if not len(g) or not len(s):
            continue
        x = g.join(s, how="inner"); x["sales"] = x.sale_me * x.market_equity
        x = x[(x.sales > 0) & np.isfinite(x.sale_gr3)]
        x["log_s"] = np.log(x.sales); x["g"] = np.log1p(x.sale_gr3.clip(-0.95, 20))
        x["sic2"] = ctx.sic2(d_start).reindex(x.index)
        rows.append(x.dropna())
    if not rows:
        return pd.Series(dtype=float)
    x = pd.concat(rows)
    slope = x.groupby("sic2").apply(lambda g: np.polyfit(g.log_s, g.g, 1)[0] if len(g) >= 15 else np.nan, include_groups=False)
    ind = ctx.sic2(t).reindex(ctx.sp(t))
    return ind.map(slope).dropna()
