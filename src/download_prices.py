"""Download daily prices for every ticker that was ever an S&P 500 member (per the
point-in-time membership file), plus benchmarks. Yahoo only keeps currently-listed
history, so delisted names mostly come back empty; that gap is measured, not hidden."""
import pandas as pd, yfinance as yf, time, sys
from config import RAW

BENCH = ["^GSPC", "^SP500TR", "SPY", "VOO", "RSP", "VTI", "^RUA"]

def yahoo(t):  # BRK.B -> BRK-B
    return t.replace(".", "-")

def main():
    d = pd.read_csv(RAW / "sp500_components_updated.csv")
    tick = sorted(set(t for s in d.tickers for t in s.split(",")))
    out = RAW / "yahoo"; out.mkdir(exist_ok=True)
    todo = [t for t in tick + BENCH if not (out / f"{yahoo(t)}.parquet").exists() and not (out / f"{yahoo(t)}.empty").exists()]
    print(len(todo), "to download", flush=True)
    for i in range(0, len(todo), 50):
        chunk = todo[i:i+50]
        df = yf.download([yahoo(t) for t in chunk], start="1994-01-01", auto_adjust=False, actions=True,
                         group_by="ticker", progress=False, threads=True)
        for t in chunk:
            y = yahoo(t)
            try:
                x = df[y].dropna(how="all", subset=["Close"])
            except KeyError:
                x = pd.DataFrame()
            if len(x):
                x.to_parquet(out / f"{y}.parquet")
            else:
                (out / f"{y}.empty").touch()
        print(i + len(chunk), flush=True)
        time.sleep(1)

if __name__ == "__main__":
    main()
