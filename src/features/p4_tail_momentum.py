import pandas as pd

def signal(ctx, t):
    """Trailing 5y total return, S&P members (deciles applied by the backtest's rank transform)."""
    r = ctx.trailing_ret(t, 60)
    return r.reindex(ctx.sp(t)).dropna()
