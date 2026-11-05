"""Global constants. Every date boundary in the project comes from here."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

HOLDOUT_START = pd.Timestamp("2016-01-01")   # locked; see src/holdout_guard.py
DATA_START = pd.Timestamp("1996-01-01")      # membership file starts 1996-01-02
FORWARD_YEARS = 5                            # tail label horizon
TAIL_Q = 0.95                                # top 5% = tail winner

# Section 8
TILT_MAX = 0.30          # each stock's tilt at most +/-30% of its cap weight
TE_CAP = 0.02            # 2% annualized tracking error
COST_BPS = 10            # per unit turnover
SMALL_EXTRA_BPS = 5      # extra for the smallest 100 members
REBAL_LAG_DAYS = 5       # trade 5 trading days after the index rebalance

# Section 9
T_STAT_BAR = 3.0
DSR_BAR = 0.95
CONSISTENCY_BAR = 0.60

# House benchmark for the live paper portfolio
LIVE_BENCHMARK = "VOO"   # Vanguard S&P 500 ETF
