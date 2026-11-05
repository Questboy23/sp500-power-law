import pandas as pd

def signal(ctx, t):
    """-MAX: max daily return in the formation month (Bali, Cakici, Whitelaw 2011)."""
    m = ctx.dmax[ctx.dmax.date == t].set_index("permno").maxret
    return (-m).reindex(ctx.sp(t)).dropna()
