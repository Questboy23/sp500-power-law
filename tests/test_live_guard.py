import pandas as pd
from src.live.engine import _final_days


def test_today_bar_dropped_before_close_and_kept_after():
    days = pd.DatetimeIndex(["2026-11-16", "2026-11-17"])
    am = pd.Timestamp("2026-11-17 11:19", tz="America/New_York")
    pm = pd.Timestamp("2026-11-17 17:30", tz="America/New_York")
    assert list(_final_days(days, am)) == [pd.Timestamp("2026-11-16")]
    assert list(_final_days(days, pm)) == list(days)
