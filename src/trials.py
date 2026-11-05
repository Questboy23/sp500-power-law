"""Rule 6: append-only trial ledger. Every backtest run gets a row; rows are never deleted."""
from __future__ import annotations
import csv, datetime as _dt, json
from config import ROOT

LEDGER = ROOT / "trials.csv"
FIELDS = ["ts", "trial_id", "angle", "variant", "phase", "window", "sharpe", "ir", "t_stat", "notes"]


def log_trial(angle: str, variant: str, phase: str, window: str = "", sharpe=None, ir=None, t_stat=None, notes: str | dict = "") -> int:
    new = not LEDGER.exists()
    n = count_trials()
    with LEDGER.open("a", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        if new:
            w.writeheader()
        w.writerow({"ts": _dt.datetime.now().isoformat(timespec="seconds"), "trial_id": n + 1, "angle": angle,
                    "variant": variant, "phase": phase, "window": window,
                    "sharpe": sharpe, "ir": ir, "t_stat": t_stat,
                    "notes": json.dumps(notes) if isinstance(notes, dict) else notes})
    return n + 1


def count_trials(distinct: bool = True) -> int:
    if not LEDGER.exists():
        return 0
    with LEDGER.open() as fh:
        rows = list(csv.DictReader(fh))
    if distinct:
        return len({(r["angle"], r["variant"]) for r in rows})
    return len(rows)
