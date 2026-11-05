import pandas as pd
from config import PROC

_cache = {}

def signal(ctx, t):
    """(opportunistic buyers - opportunistic sellers)/(sum+1), 12m before t, filings <= t.
    Built by src/features/a4_insider.py from SEC Insider Transactions Data Sets (2006+)."""
    p = PROC / "a4_insider.parquet"
    if not p.exists():
        return pd.Series(dtype=float)
    if "d" not in _cache:
        _cache["d"] = pd.read_parquet(p)
    d = _cache["d"]
    d = d[pd.to_datetime(d.formation_date) == t]
    if not len(d):
        return pd.Series(dtype=float)
    return d.set_index("permno").a4.reindex(ctx.sp(t)).dropna()
