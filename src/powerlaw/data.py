"""Long total-market monthly panel 1926-2024 (CRSP) and the top-500 'S&P proxy' used for
eras before the real membership file starts (1996)."""
import numpy as np, pandas as pd
from config import RAW, PROC


def build_long():
    a = pd.read_parquet(RAW / "delisted" / "crsp_monthly_permno_1926_2000.parquet", columns=["permno", "date", "prc", "ret", "shrout"])
    a["date"] = pd.to_datetime(a.date) + pd.offsets.MonthEnd(0)
    a = a[a.date < "1994-01-01"]
    a["mcap"] = a.prc.abs() * a.shrout * 1e3
    m = pd.read_parquet(PROC / "market_monthly.parquet")
    m = m[m.shrcd.isin([10, 11])][["permno", "date", "prc", "ret", "shrout", "mcap"]]
    x = pd.concat([a[["permno", "date", "prc", "ret", "shrout", "mcap"]], m], ignore_index=True)
    x = x.sort_values(["permno", "date"]).drop_duplicates(["permno", "date"])
    x.to_parquet(PROC / "market_long.parquet")
    return x


def load_long():
    return pd.read_parquet(PROC / "market_long.parquet")


def wide(x=None):
    x = load_long() if x is None else x
    return x.pivot(index="date", columns="permno", values="ret"), x.pivot(index="date", columns="permno", values="mcap")


def sp_members(date, lk=None, caps=None, n=500):
    """S&P 500 members on `date` (real list from 1996-06, else top-n CRSP caps)."""
    if date >= pd.Timestamp("1996-01-31"):
        lk = pd.read_parquet(PROC / "membership_linked.parquet") if lk is None else lk
        return lk[(lk.date == date) & (lk.permno > 0)].permno.unique()
    c = caps.loc[date].dropna()
    return c.nlargest(n).index.values


if __name__ == "__main__":
    x = build_long(); print(x.shape, x.date.min(), x.date.max())
