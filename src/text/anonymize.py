"""Best-effort anonymisation of 10-K Item 1 text before it goes into a point-in-time LLM.

Removes (1) the firm's own names: EDGAR current + former conformed names of every linked CIK,
FIAM names, their distinctive cores and first distinctive word; (2) its tickers (CRSP history,
membership file, FIAM, EDGAR current); (3) person names after honorifics; (4) capitalised tokens
that are not English dictionary words and not on a small allowlist (catches most product,
brand, subsidiary, competitor and executive names; also removes some place names).
It will miss names that are ordinary English words ("Apple", "Target", "Oracle") when they are
not the firm's own name, and inverted EDGAR names ("PRICE T ROWE").
"""
from __future__ import annotations
import json, re
from functools import lru_cache
import pandas as pd
from config import RAW, PROC

ALLOW = set("""
january february march april may june july august september october november december
monday tuesday wednesday thursday friday saturday sunday
america american americas u.s us usa united states north south east west central europe european asia asian
pacific atlantic africa african canada canadian mexico mexican china chinese japan japanese india indian
germany german france french italy italian spain spanish brazil latin kingdom england british uk
alabama alaska arizona arkansas california colorado connecticut delaware florida georgia hawaii idaho
illinois indiana iowa kansas kentucky louisiana maine maryland massachusetts michigan minnesota
mississippi missouri montana nebraska nevada hampshire jersey mexico york carolina dakota ohio oklahoma
oregon pennsylvania rhode tennessee texas utah vermont virginia washington wisconsin wyoming
sec fda epa ftc fcc doe dod irs gaap ifrs nyse nasdaq ipo ceo cfo coo cto r&d eu nafta opec
item form annual report company corporation registrant internet web
""".split())
STOP_TICKERS = {"A", "I", "IT", "ALL", "ON", "AN", "OR", "BE", "SO", "GO", "ARE", "FOR", "ONE", "NOW", "CAN",
                "KEY", "LOW", "BIG", "CAT", "CAR", "DAY", "SEE", "WELL", "HAS", "CASH", "PAY", "SPOT", "TWO",
                "NEW", "HE", "IN", "AT", "BY", "TO", "US", "UK", "EU"}


@lru_cache(1)
def _words() -> frozenset:
    """'Common word' = written in lower case in >= 1% of a sample of 500 downloaded 10-Ks (pre-2016).
    Corpus-based rather than a dictionary: /usr/share/dict/words contains 'cisco', 'apple', ... ."""
    cache = PROC / "common_words.json"
    if cache.exists():
        return frozenset(json.loads(cache.read_text()))
    import gzip, random
    from collections import Counter
    files = [f for d in (RAW / "tenk").iterdir() if d.is_dir() for f in d.iterdir() if f.name.endswith(".json.gz")]
    random.Random(0).shuffle(files)
    df, n = Counter(), 0
    for f in files:
        with gzip.open(f, "rt") as fh:
            j = json.load(fh)
        if j["filed"] >= "2016-01-01":
            continue
        df.update(set(re.findall(r"(?<![A-Za-z])[a-z][a-z'\-]+", j["full"])))
        n += 1
        if n >= 500:
            break
    words = sorted(w for w, c in df.items() if c >= max(5, 0.01 * n))
    cache.write_text(json.dumps(words))
    return frozenset(words)


_SUF = r"(?:,?\s+(?:inc\.?|incorporated|corp\.?|corporation|co\.?|company|companies|ltd\.?|limited|plc|llc|l\.p\.|lp|n\.v\.|s\.a\.|ag|holdings?|group))*"


def _name_variants(raw: str) -> set[str]:
    s = re.sub(r"/[A-Z]{2,3}/?", " ", raw or "")
    s = re.sub(r"\s+", " ", s).strip(" ,.")
    out = {s}
    core = re.sub(r"(?i)\b(inc|incorporated|corp|corporation|co|company|companies|ltd|limited|plc|llc|lp|holdings?|group|the|new)\b\.?", " ", s)
    core = re.sub(r"[,\.]", " ", core); core = re.sub(r"\s+", " ", core).strip()
    if len(core) >= 3:
        out.add(core)
        out.add(core.replace("&", "and"))
        first = core.split()[0]
        if len(first) >= 4 and first.lower() not in _words() and first.lower() not in ALLOW:
            out.add(first)                      # 'Cisco' from 'CISCO SYSTEMS INC'; skips 'General'
            out.add(first + "'s")
    return {v for v in out if len(v) >= 3}


@lru_cache(1)
def _tables():
    link = pd.read_parquet(PROC / "permno_cik.parquet")
    meta = pd.read_parquet(PROC / "edgar_meta.parquet").set_index("cik")
    fiam = pd.read_parquet(RAW / "delisted" / "fiam_monthly_2000_2023.parquet", columns=["permno", "comp_name", "stock_ticker"])
    tl = pd.read_csv(RAW / "delisted" / "ticker_permno_link.csv")
    mem = pd.read_parquet(PROC / "membership_linked.parquet")
    from src.text.link import MANUAL
    names, ticks = {}, {}
    for p, ciks in link.groupby("permno").cik:
        ns, ts = set(), set()
        for k in ciks:
            if k in meta.index:
                ns.add(meta.at[k, "name"])
                ns |= {x["name"] for x in json.loads(meta.at[k, "former_names"] or "[]")}
                ts |= {t for t in (meta.at[k, "tickers"] or "").split(",") if t}
        names[int(p)] = ns; ticks[int(p)] = ts
    for p, g in fiam.groupby("permno"):
        if p in names:
            names[p] |= set(g.comp_name.dropna()); ticks[p] |= set(g.stock_ticker.dropna())
    for p, g in tl.groupby("permno"):
        if p in ticks:
            ticks[p] |= set(g.ticker.dropna())
    for p, g in mem[mem.permno > 0].groupby("permno"):
        if p in ticks:
            ticks[p] |= {t.split(".")[0] for t in g.ticker}
    for p, n in MANUAL.items():
        if p in names:
            names[p].add(n)
    return names, ticks


@lru_cache(4096)
def _firm_regex(permno: int):
    names, ticks = _tables()
    vs = set()
    for n in names.get(permno, ()):
        vs |= _name_variants(n)
    vs = sorted(vs, key=len, reverse=True)
    name_rx = re.compile(r"(?i)(?<![A-Za-z0-9])(?:the\s+)?(?:" + "|".join(re.escape(v).replace(r"\ ", r"\s+") for v in vs) + r")" + _SUF + r"(?![A-Za-z0-9])") if vs else None
    tk = sorted({t for t in ticks.get(permno, ()) if len(t) >= 2 and t.upper() not in STOP_TICKERS}, key=len, reverse=True)
    tick_rx = re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(re.escape(t) for t in tk) + r")(?![A-Za-z0-9])") if tk else None
    return name_rx, tick_rx


_PERSON = re.compile(r"\b(?:Mr|Ms|Mrs|Dr|Messrs)\.?\s+(?:[A-Z][a-z]+\.?\s+){0,3}[A-Z][a-zA-Z\-']+")
_CAP = re.compile(r"(?<![\w\[])([A-Z][A-Za-z0-9&'\-]*[A-Za-z0-9])(?![\w\]])")


def anonymize(text: str, permno: int) -> str:
    name_rx, tick_rx = _firm_regex(int(permno))
    if name_rx is not None:
        text = name_rx.sub("[COMPANY]", text)
    if tick_rx is not None:
        text = tick_rx.sub("[TICKER]", text)
    text = _PERSON.sub("[PERSON]", text)
    words = _words()

    def cap(m):
        tok = m.group(1)
        low = re.sub(r"'s$", "", tok.lower()).strip("'")
        start = m.start()
        prev = text[max(0, start - 3):start]
        sentence_initial = start == 0 or re.search(r"[\.\!\?\n:]\s*$", prev) is not None
        if low in ALLOW or low.rstrip("s") in words or low in words:
            return tok
        if tok.isupper() and len(tok) <= 2:
            return tok
        if sentence_initial and low in words:
            return tok
        return "[NAME]"
    text = _CAP.sub(cap, text)
    text = re.sub(r"(\[NAME\][\s,]*){2,}", "[NAME] ", text)
    return text
