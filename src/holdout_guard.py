"""Rule 4: the 2016+ holdout is locked until Phase 10.

Every training / tuning / feature-selection entry point calls `assert_pre_holdout`
on the frame it is about to learn from. The only way through is `unlock_holdout`,
which writes data/holdout/.consumed.json and refuses to run twice.
"""
from __future__ import annotations
import json, datetime as _dt
import pandas as pd
from config import HOLDOUT_START, ROOT

_LOCK = ROOT / "data" / "holdout" / ".consumed.json"


class HoldoutViolation(RuntimeError):
    pass


def _max_date(obj) -> pd.Timestamp | None:
    if obj is None:
        return None
    if isinstance(obj, (pd.Timestamp, _dt.date, str)):
        return pd.Timestamp(obj)
    if isinstance(obj, pd.Index):
        return pd.Timestamp(obj.max()) if len(obj) else None
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        cands = []
        if isinstance(obj.index, pd.DatetimeIndex) and len(obj.index):
            cands.append(obj.index.max())
        if isinstance(obj.index, pd.MultiIndex):
            for lvl in obj.index.levels:
                if isinstance(lvl, pd.DatetimeIndex) and len(lvl):
                    cands.append(obj.index.get_level_values(lvl.name).max())
        if isinstance(obj, pd.DataFrame):
            for c in ("date", "formation_date", "end_date", "label_end"):
                if c in obj.columns and len(obj):
                    cands.append(pd.to_datetime(obj[c]).max())
        return max(cands) if cands else None
    return None


def assert_pre_holdout(obj, what: str = "data", allow_label_end: bool = False) -> None:
    """Raise if anything in `obj` falls on or after HOLDOUT_START.

    Forward-return labels are the subtle case: a 2012 formation date with a 5y
    label *ends* in 2017, so its label uses holdout returns. Frames that carry
    a `label_end` column are checked on that column, not just `date`."""
    if holdout_unlocked():
        return
    m = _max_date(obj)
    if m is not None and m >= HOLDOUT_START:
        raise HoldoutViolation(f"{what}: max date {m.date()} >= holdout start {HOLDOUT_START.date()}")


def holdout_unlocked() -> bool:
    return _LOCK.exists()


def unlock_holdout(note: str) -> None:
    if _LOCK.exists():
        raise HoldoutViolation(f"holdout already consumed: {_LOCK.read_text()}")
    _LOCK.parent.mkdir(parents=True, exist_ok=True)
    _LOCK.write_text(json.dumps({"consumed_at": _dt.datetime.now().isoformat(), "note": note}))
