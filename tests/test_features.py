"""Look-ahead tests for every angle (Phase 4.2).

1. Truncation invariance: signal(ctx, t) must equal signal(ctx.truncate(t), t). If a feature
   silently reads data dated after t, deleting that data changes its output and this fails.
2. Shift sensitivity: moving the formation date one year changes the signal (catches features
   that ignore t entirely, e.g. by reading the last row of a table)."""
import numpy as np, pandas as pd, pytest
from src.features.context import Context
from src.features import ANGLES, get

DATES = [pd.Timestamp(f"{y}-06-30") for y in (2004, 2009, 2013)]


@pytest.fixture(scope="module")
def ctx():
    return Context()


@pytest.mark.parametrize("angle", ANGLES)
def test_truncation_invariance(ctx, angle):
    m = get(angle)
    for t in DATES:
        full = m.signal(ctx, t)
        cut = m.signal(ctx.truncate(t), t)
        a, b = full.sort_index(), cut.sort_index()
        assert a.index.equals(b.index), f"{angle} {t.date()}: index differs after truncation"
        assert np.allclose(a.values, b.values, equal_nan=True), f"{angle} {t.date()}: values differ after truncation (look-ahead)"


@pytest.mark.parametrize("angle", ANGLES)
def test_shift_changes_signal(ctx, angle):
    m = get(angle)
    s0 = m.signal(ctx, DATES[1]); s1 = m.signal(ctx, DATES[1] + pd.offsets.MonthEnd(12))
    if len(s0) == 0 or len(s1) == 0:
        pytest.skip(f"{angle}: no coverage at test dates")
    common = s0.index.intersection(s1.index)
    if angle == "A9_fluent_ticker":   # a ticker property: constant by design unless the ticker changes
        return
    if angle == "B6_crisis_share_gainer":  # constant between crises by design (2010 -> 2011 same crisis)
        return
    assert not np.allclose(s0.reindex(common).values, s1.reindex(common).values, equal_nan=True), f"{angle}: signal identical one year later"


def test_holdout_guard():
    from src.holdout_guard import assert_pre_holdout, HoldoutViolation, holdout_unlocked
    if holdout_unlocked():
        pytest.skip("holdout already consumed")
    with pytest.raises(HoldoutViolation):
        assert_pre_holdout(pd.DataFrame({"date": [pd.Timestamp("2016-02-29")]}), "t")
    with pytest.raises(HoldoutViolation):
        assert_pre_holdout(pd.DataFrame({"formation_date": [pd.Timestamp("2011-06-30")], "label_end": [pd.Timestamp("2016-06-30")]}), "labels")
    assert_pre_holdout(pd.DataFrame({"date": [pd.Timestamp("2015-12-31")]}), "ok")
