import numpy as np, pandas as pd

def signal(ctx, t):
    """ROIIC proxy x reinvestment. ROIIC ~ change in operating profitability relative to asset
    growth over 3y: (op_at_t*AT_t - op_at_{t-3}*AT_{t-3}) / (AT_t - AT_{t-3}) with AT from
    compounding at_gr1. Reinvestment = rank(capx_gr3) + rank(rd_sale)."""
    now = ctx.fiam(t, ["op_at", "at_gr1", "capx_gr3", "rd_sale"])
    if not len(now):
        return pd.Series(dtype=float)
    g = [ctx.fiam(t - pd.offsets.MonthEnd(12 * k), ["at_gr1"]).at_gr1 for k in (0, 1, 2)]
    past = ctx.fiam(t - pd.offsets.MonthEnd(36), ["op_at"]).op_at
    growth = (1 + g[0]) * (1 + g[1].reindex(g[0].index)) * (1 + g[2].reindex(g[0].index))   # AT_t / AT_{t-3}
    at_ratio = growth.reindex(now.index)
    roiic = (now.op_at * at_ratio - past.reindex(now.index)) / (at_ratio - 1)
    roiic = roiic.where((at_ratio - 1).abs() > 0.05)
    reinv = now.capx_gr3.rank(pct=True) + now.rd_sale.fillna(0).rank(pct=True)
    s = roiic.clip(-1, 2).rank(pct=True) * reinv.rank(pct=True)
    return s.reindex(ctx.sp(t)).dropna()
