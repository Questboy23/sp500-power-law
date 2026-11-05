import pandas as pd
from config import PROC

_cache = {}

def _tf():
    if "d" not in _cache:
        p = PROC / "text_features.parquet"
        _cache["d"] = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["permno", "filed", "a5_similarity", "b4_rival_mentions"])
        _cache["d"]["filed"] = pd.to_datetime(_cache["d"]["filed"])
    return _cache["d"]

def signal(ctx, t):
    """Similarity of the latest 10-K filed on/before t (within 15 months) to the prior 10-K."""
    d = _tf()
    d = d[(d.filed <= t) & (d.filed > t - pd.DateOffset(months=15))].dropna(subset=["a5_similarity"])
    s = d.sort_values("filed").groupby("permno").a5_similarity.last()
    return s.reindex(ctx.sp(t)).dropna().astype(float)
