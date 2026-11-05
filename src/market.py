"""Total US market monthly panel from CRSP daily (common stocks, shrcd 10/11), 1994-2024.
Used for BASE2 and for P5 (tail winners that were not yet S&P members)."""
import numpy as np, pandas as pd, pyarrow.parquet as pq
from config import RAW, PROC


def build():
    f = pq.ParquetFile(RAW / "delisted" / "crsp_daily_1968_2024.parquet")
    parts = []
    for b in f.iter_batches(batch_size=5_000_000, columns=["permno", "date", "shrcd", "exchcd", "prc", "ret", "shrout"]):
        d = b.to_pandas()
        d["date"] = pd.to_datetime(d.date)
        d = d[d.date >= "1994-01-01"]
        if not len(d):
            continue
        d["date"] = pd.to_datetime(d.date); d["prc"] = d.prc.abs()
        d["lr"] = np.log1p(d.ret)
        d["m"] = d.date + pd.offsets.MonthEnd(0)
        g = d.sort_values("date").groupby(["permno", "m"])
        parts.append(pd.DataFrame({"lr": g.lr.sum(min_count=1), "prc": g.prc.last(), "shrout": g.shrout.last(),
                                   "shrcd": g.shrcd.last(), "exchcd": g.exchcd.last(), "hi": g.prc.max(),
                                   "last_day": g.date.max()}).reset_index())
    m = pd.concat(parts)
    # a permno-month can straddle batches: re-aggregate
    m = m.sort_values("last_day").groupby(["permno", "m"]).agg(lr=("lr", lambda x: x.sum(min_count=1)), prc=("prc", "last"),
                                                                shrout=("shrout", "last"), shrcd=("shrcd", "last"),
                                                                exchcd=("exchcd", "last"), hi=("hi", "max")).reset_index()
    m["ret"] = np.expm1(m.lr); m["mcap"] = m.prc * m.shrout * 1e3
    m = m.rename(columns={"m": "date"}).drop(columns="lr")
    m.to_parquet(PROC / "market_monthly.parquet")
    print(m.shape, m.permno.nunique())


if __name__ == "__main__":
    build()
