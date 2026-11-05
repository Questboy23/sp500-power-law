"""Live paper portfolios (Phase 11). PAPER ONLY: nothing in this repo can place an order.

Three portfolios, $100,000 each, bought at the close of one start day:
  SYSTEM - what the research recommends: the S&P 500 at its actual float-adjusted weights (SPY's
           daily holdings file) plus implementation rule I3: an index addition is bought 5 trading
           days after it first appears in the holdings (pre-holdout event study: additions lose
           1.4% vs the market over days 0-4, t = -5.0, n = 484). Deletions are sold the day they
           disappear. Quarterly true-up to index weights 5 trading days after each quarterly
           rebalance (3rd Friday of Mar/Jun/Sep/Dec).
  SHADOW - the same, times bounded tilt multipliers k = 1 + 0.30*s from LIVE_SHADOW (every angle
           computable from live data, equal-weighted). It FAILED the validation gate; it runs in the
           open so the gate's decision can be checked.
  VOO    - Vanguard S&P 500 ETF, bought and held. The benchmark.

Bookkeeping is in real shares on Yahoo's split-adjusted Close, processed day by day: a split
multiplies shares, and a dividend is reinvested in the paying stock at that day's close (DRIP).
The same rule applies to all three portfolios. Costs: 10 bps per dollar traded after the initial
purchase. The initial purchase is free for all three, including VOO."""
from __future__ import annotations
import io, json, datetime as dt
import numpy as np, pandas as pd, requests
from config import ROOT

LIVE = ROOT / "data" / "live"
STATE, MARKS, TRADES = LIVE / "state.json", LIVE / "marks.csv", LIVE / "trades.csv"
START_CASH = 100_000.0
COST = 0.0010
SPY_URL = "https://www.ssga.com/us/en/intermediary/etfs/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx"
REF = "^SP500TR"


def yh(t: str) -> str:
    return t.replace(".", "-")


# ---------------------------------------------------------------- data
def index_weights(save: bool = True) -> tuple[pd.Series, str]:
    """SPY holdings, 3 tries; if the download fails, fall back to the newest saved weights so the
    day can still be marked (index changes are then picked up on the next successful run)."""
    import time
    r = None
    for i in range(3):
        try:
            r = requests.get(SPY_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
            r.raise_for_status()
            break
        except Exception as e:                                   # noqa: BLE001
            r = None; print(f"SPY holdings fetch failed ({type(e).__name__}); retry {i + 1}/3"); time.sleep(20 * (i + 1))
    if r is None:
        saved = sorted(LIVE.glob("spy_weights_*.csv"))
        if not saved:
            raise SystemExit("no SPY holdings and no saved weights")
        print(f"FALLBACK: using {saved[-1].name}")
        w = pd.read_csv(saved[-1], index_col=0).weight
        return w / w.sum(), saved[-1].stem.replace("spy_weights_", "") + " (saved)"
    hdr = pd.read_excel(io.BytesIO(r.content), header=None, nrows=3)
    asof = str(hdr.iloc[2, 1]).replace("As of ", "").strip()
    x = pd.read_excel(io.BytesIO(r.content), skiprows=4).dropna(subset=["Ticker", "Weight"])
    x = x[x.Ticker.astype(str).str.fullmatch(r"[A-Z]{1,5}(\.[A-Z])?")]
    w = x.groupby("Ticker").Weight.sum()
    w = w / w.sum()
    if save:
        LIVE.mkdir(parents=True, exist_ok=True)
        w.rename("weight").to_csv(LIVE / f"spy_weights_{pd.Timestamp(asof).date()}.csv")
    return w, asof


def market_data(tickers, start):
    """Close (split-adjusted), Dividends, Stock Splits per ticker; index = trading days."""
    import yfinance as yf
    (LIVE / "yf_cache").mkdir(exist_ok=True)
    yf.set_tz_cache_location(str(LIVE / "yf_cache"))
    tk = sorted(set(yh(t) for t in tickers))
    df = yf.download(tk, start=str(start), auto_adjust=False, actions=True, progress=False,
                     threads=8, group_by="ticker")
    lv0 = set(df.columns.get_level_values(0))
    get = lambda f: pd.DataFrame({t: df[t][f] for t in tk if t in lv0 and f in df[t]})
    close, div, spl = get("Close"), get("Dividends"), get("Stock Splits")
    for x in (close, div, spl):
        x.index = pd.to_datetime(x.index).tz_localize(None)
    close = close.dropna(how="all")
    return close, div.reindex(close.index).fillna(0), spl.reindex(close.index).fillna(0)


# ---------------------------------------------------------------- state
def load() -> dict:
    return json.loads(STATE.read_text())


def save(s: dict) -> None:
    STATE.write_text(json.dumps(s, indent=1, default=str))


def log_trades(rows) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(TRADES, mode="a", header=not TRADES.exists(), index=False)


# NYSE full-day holidays (published in advance; verify each January). Used ONLY to know, before a day
# starts, whether it is a turn-of-month day, so LIVE_TOM never needs tomorrow's data.
NYSE_HOLIDAYS = pd.to_datetime([
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31", "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
    "2028-01-17", "2028-02-21", "2028-04-14", "2028-05-29", "2028-06-19", "2028-07-04", "2028-09-04", "2028-11-23", "2028-12-25"])
TOM_EXTRA, TOM_COST = 0.5, 0.0002


def is_tom(d: pd.Timestamp) -> bool:
    cal = pd.bdate_range(d - pd.Timedelta(days=40), d + pd.Timedelta(days=40))
    cal = cal[~cal.isin(NYSE_HOLIDAYS)]
    mon = cal[(cal.year == d.year) & (cal.month == d.month)]
    if d not in mon:
        return False
    return bool(d == mon[-1] or d in mon[:3])


def rf_daily() -> float:
    from src.factors import ff_monthly
    return float(ff_monthly().RF.iloc[-1]) / 21


def port_value(p: dict, px: pd.Series) -> float:
    if "value" in p:                                   # LIVE_TOM is tracked as a value, not shares
        return float(p["value"])
    v = p["cash"]
    for t, sh in p["shares"].items():
        c = px.get(yh(t), np.nan)
        if np.isfinite(c):
            v += sh * c
    return float(v)


# ---------------------------------------------------------------- lifecycle
def init(start_date: str, shadow_k: dict, note: str = "") -> dict:
    if STATE.exists():
        raise SystemExit("live state exists; refusing to re-initialize (that would reset the record)")
    w, asof = index_weights()
    d = pd.Timestamp(start_date)
    close, _, _ = market_data(list(w.index) + ["VOO", REF], (d - pd.Timedelta(days=10)).date())
    if d not in close.index:
        raise SystemExit(f"no close for {d.date()} yet")
    px = close.loc[d]
    have = [t for t in w.index if np.isfinite(px.get(yh(t), np.nan))]
    missing = sorted(set(w.index) - set(have))
    w = w[have] / w[have].sum()
    k = pd.Series(shadow_k, dtype=float).reindex(w.index).fillna(1.0)
    ws = (w * k) / (w * k).sum()
    mk = lambda wt: {t: START_CASH * wt[t] / px[yh(t)] for t in wt.index}
    s = {"start_date": str(d.date()), "note": note, "weights_asof": asof, "last_mark": None,
         "universe": sorted(w.index), "missing_at_start": missing, "ref_start": float(px[REF]),
         "portfolios": {
             "SYSTEM": {"shares": mk(w), "cash": 0.0, "pending": {}, "costs": 0.0},
             "SHADOW": {"shares": mk(ws), "cash": 0.0, "pending": {}, "costs": 0.0,
                        "k": {t: float(v) for t, v in k.items()}},
             "VOO": {"shares": {"VOO": START_CASH / px["VOO"]}, "cash": 0.0, "pending": {}, "costs": 0.0}}}
    save(s)
    log_trades([{"date": d.date(), "portfolio": "ALL", "action": "INIT", "ticker": "",
                 "dollars": START_CASH, "note": note}])
    _mark(s, close, d)
    s["last_mark"] = str(d.date())
    save(s)
    return s


def _third_friday(y: int, m: int) -> pd.Timestamp:
    d = pd.Timestamp(y, m, 15)
    return d + pd.Timedelta(days=(4 - d.weekday()) % 7)


def _is_trueup_day(d: pd.Timestamp, tdays: pd.DatetimeIndex) -> bool:
    for m in (3, 6, 9, 12):
        after = tdays[tdays > _third_friday(d.year, m)]
        if len(after) >= 5 and after[4] == d:
            return True
    return False


def daily() -> list:
    """Run after the close. For every trading day after the last mark: splits, dividends (DRIP),
    index changes (today's holdings file is applied on the latest day only), due additions,
    quarterly true-up, then mark."""
    s = load()
    last = pd.Timestamp(s["last_mark"])
    w, asof = index_weights()
    held = set(s["universe"]) | set(w.index)
    for p in s["portfolios"].values():
        held |= set(p.get("shares", {})) | set(p.get("pending", {}))
    close, div, spl = market_data(sorted(held) + ["VOO", REF], (last - pd.Timedelta(days=15)).date())
    days = close.index[close.index > last]
    # only mark a day once its closes exist (Yahoo posts index levels before stock closes)
    complete = [d for d in days if pd.notna(close.at[d, "VOO"]) and close.loc[d].notna().mean() >= 0.9]
    days = pd.DatetimeIndex(complete)
    if not len(days):
        print("no complete new trading day yet"); save(s); return []
    trades = []
    for d in days:
        px = close.loc[:d].ffill().iloc[-1]
        tom = s["portfolios"].get("TOM")
        if tom is not None and "VOO" in close and d > pd.Timestamp(tom["start"]):
            prev = close["VOO"].loc[:d].iloc[:-1].dropna()
            if len(prev):
                rv = (close.at[d, "VOO"] + (div.at[d, "VOO"] if "VOO" in div else 0)) / prev.iloc[-1] - 1
                on, was = is_tom(d), tom.get("on", False)
                f = rf_daily()
                step = rv + (TOM_EXTRA * (rv - f) if on else 0.0) - (TOM_EXTRA * TOM_COST if on != was else 0.0)
                tom["value"] *= 1 + step
                tom["costs"] += tom["value"] * TOM_EXTRA * TOM_COST if on != was else 0.0
                tom["on"] = on
        for name, p in s["portfolios"].items():
            for t in list(p.get("shares", {})):
                y = yh(t)
                r = spl.at[d, y] if y in spl else 0
                if r and r > 0:
                    p["shares"][t] *= r
                dv = div.at[d, y] if y in div else 0
                if dv and dv > 0 and np.isfinite(px.get(y, np.nan)):
                    p["shares"][t] += p["shares"][t] * dv / px[y]
        if d == days[-1]:
            trades += _index_changes(s, w, px, d, close.index)
        for name in ("SYSTEM", "SHADOW"):
            trades += _execute_due(s, name, w, px, d, close.index)
            if _is_trueup_day(d, close.index):
                trades += _trueup(s, name, w, px, d)
        _mark(s, close, d)
        s["last_mark"] = str(d.date())
    s["weights_asof"] = asof
    save(s)
    log_trades(trades)
    return trades


def _index_changes(s, w, px, d, tdays) -> list:
    cur, prev = set(w.index), set(s["universe"])
    added, removed = sorted(cur - prev), sorted(prev - cur)
    out = []
    for name in ("SYSTEM", "SHADOW"):
        p = s["portfolios"][name]
        for t in removed:
            if t in p["shares"]:
                c = px.get(yh(t), np.nan)
                val = p["shares"][t] * c if np.isfinite(c) else 0.0
                p["cash"] += val * (1 - COST)
                p["costs"] += val * COST
                del p["shares"][t]
                out.append({"date": d.date(), "portfolio": name, "action": "SELL index deletion",
                            "ticker": t, "dollars": round(val, 2), "note": ""})
            p["pending"].pop(t, None)
        for t in added:
            p["pending"][t] = str(d.date())
            out.append({"date": d.date(), "portfolio": name, "action": "SCHEDULE buy +5 trading days",
                        "ticker": t, "dollars": 0, "note": "rule I3"})
    s["universe"] = sorted(cur)
    return out


def _execute_due(s, name, w, px, d, tdays) -> list:
    p = s["portfolios"][name]
    out = []
    for t, seen in list(p["pending"].items()):
        n = int(((tdays > pd.Timestamp(seen)) & (tdays <= d)).sum())
        c = px.get(yh(t), np.nan)
        if n < 5 or not np.isfinite(c) or t not in w.index:
            continue
        V = port_value(p, px)
        k = p.get("k", {}).get(t, 1.0)
        dollars = V * w[t] * k
        if dollars > p["cash"]:                      # fund the rest pro-rata from holdings
            need = dollars - p["cash"]
            inv = V - p["cash"]
            for u in p["shares"]:
                p["shares"][u] *= 1 - need / inv
            p["cash"] += need
        p["cash"] -= dollars
        p["shares"][t] = p["shares"].get(t, 0.0) + dollars * (1 - COST) / c
        p["costs"] += dollars * COST
        del p["pending"][t]
        out.append({"date": d.date(), "portfolio": name, "action": "BUY index addition (+5d)",
                    "ticker": t, "dollars": round(dollars, 2), "note": f"first seen {seen}"})
    if p["cash"] > 1 and not p["pending"]:           # deletion proceeds with nothing to fund
        c0 = p["cash"]
        inv = port_value(p, px) - c0
        for u in p["shares"]:
            p["shares"][u] *= 1 + c0 * (1 - COST) / inv
        p["costs"] += c0 * COST
        p["cash"] = 0.0
        out.append({"date": d.date(), "portfolio": name, "action": "REINVEST cash pro-rata",
                    "ticker": "", "dollars": round(c0, 2), "note": ""})
    return out


def _trueup(s, name, w, px, d) -> list:
    p = s["portfolios"][name]
    V = port_value(p, px)
    k = p.get("k", {})
    names = [t for t in w.index if t in p["shares"] and np.isfinite(px.get(yh(t), np.nan))]
    tgt = pd.Series({t: w[t] * k.get(t, 1.0) for t in names})
    tgt = tgt / tgt.sum()
    investable = V - p["cash"]
    traded = sum(abs(investable * tgt[t] - p["shares"][t] * px[yh(t)]) for t in names)
    cost = traded * COST
    for t in names:
        p["shares"][t] = (investable - cost) * tgt[t] / px[yh(t)]
    p["costs"] += cost
    return [{"date": d.date(), "portfolio": name, "action": "QUARTERLY TRUE-UP", "ticker": "",
             "dollars": round(traded, 2), "note": f"cost {cost:.2f}"}]


def _mark(s, close, d) -> None:
    px = close.loc[:d].ffill().iloc[-1]
    rec = {"date": str(d.date())}
    for name, p in s["portfolios"].items():
        if name == "TOM" and pd.Timestamp(p["start"]) > d:
            rec[name] = np.nan; continue
        rec[name] = round(port_value(p, px), 2)
    rec["SP500TR"] = round(START_CASH * px[REF] / s["ref_start"], 2) if REF in px else np.nan
    m = pd.DataFrame([rec])
    if MARKS.exists():
        old = pd.read_csv(MARKS, dtype={"date": str})
        m = pd.concat([old[old.date != rec["date"]], m]).sort_values("date")
    m.to_csv(MARKS, index=False)
