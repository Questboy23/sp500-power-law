"""Phase 6 validation: zero tilt reproduces BASE0 exactly (to float precision), with zero cost."""
import numpy as np, pandas as pd
from src.backtest import Engine


def test_zero_tilt_equals_base0():
    E = Engine()
    forms = {pd.Timestamp(f"{y}-06-30"): pd.Series(0.0, index=E.B.columns) for y in range(1996, 2016)}
    res = E.run(forms)
    diff = (res["gross"] - E.base0.reindex(res["gross"].index)).abs().max()
    assert diff < 1e-12, diff
    assert res["cost"].iloc[1:].abs().max() < 1e-12


def test_tilt_bounded():
    E = Engine()
    t = pd.Timestamp("2010-06-30")
    s = pd.Series(np.random.default_rng(0).uniform(-1, 1, E.B.shape[1]), index=E.B.columns)
    k, scale, te = E.multipliers(t, s, 1.0)
    assert k.max() <= 1.30 + 1e-12 and k.min() >= 0.70 - 1e-12
    b = E.B.loc[t + pd.offsets.MonthEnd(1)].dropna(); w = b * k.reindex(b.index); w /= w.sum()
    assert E.ex_ante_te(t, w - b) <= 0.02 + 1e-6
