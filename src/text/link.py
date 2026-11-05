"""permno <-> CIK link for every permno that was ever an S&P 500 member (1996+).

Methods, strongest first (a permno can get several CIKs, e.g. holding-company reorganisations;
each (permno, cik) row must be backed by >= 1 10-K filed while the permno traded):
  cusip8      FIAM 8-digit CUSIP -> leoliu0/cik-cusip-mapping (built from 13D/13G filings)
  cusip6      same, issuer-level 6-digit match
  ticker_now  permno alive in 2024-12 -> current SEC company_tickers.json ticker
  name        FIAM comp_name (16-char truncated) fuzzy vs EDGAR current + former names
  manual      hand-written historical company name (pre-2000 exits have no FIAM name) fuzzy vs EDGAR
confidence in [0, 1]: agreement across methods + name similarity + filing-date overlap.
"""
from __future__ import annotations
import json, re
import numpy as np, pandas as pd
from rapidfuzz import fuzz, process
from config import RAW, PROC

SUFFIX = re.compile(r"\b(INCORPORATED|INC|CORPORATION|CORP|COMPANY|CO|LIMITED|LTD|PLC|LLC|LP|L P|NV|N V|SA|AG|"
                    r"THE|CP|COS|US|HOLDINGS?|HLDGS?|HLDG|GROUP|GRP|GP|TRUST|NEW|DE|DEL|MD|NY|NJ|OH|PA|VA|NC|CA|NV)\b")

# Pre-2000 (and a few later non-FIAM) S&P members with no FIAM name. Written from the CRSP
# historical ticker + domain knowledge; every entry is still verified against EDGAR names AND
# 10-K filing dates, so a wrong guess fails to link rather than linking wrongly (mostly).
MANUAL = {
    10364: "ASARCO", 11260: "CHRYSLER CORP", 12095: "GENERAL SIGNAL CORP", 12458: "RYERSON TULL",
    12546: "INCO LTD", 14883: "UNION CARBIDE CORP", 15966: "MOBIL CORP", 16870: "PACIFIC ENTERPRISES",
    17566: "ARMCO INC", 18112: "FOSTER WHEELER", 18147: "AEROQUIP VICKERS", 18374: "HONEYWELL INC",
    18551: "OUTBOARD MARINE CORP", 18948: "ROCKWELL INTERNATIONAL CORP", 19051: "TRANSAMERICA CORP",
    19254: "DRESSER INDUSTRIES", 19377: "BENEFICIAL CORP", 19553: "AMOCO CORP", 19916: "SEAGRAM CO LTD",
    21514: "SONAT INC", 21821: "CONSOLIDATED NATURAL GAS CO", 22082: "PANENERGY CORP", 22680: "MILACRON INC",
    22753: "CHARMING SHOPPES", 22891: "MERCANTILE STORES CO", 24264: "ALCAN ALUMINIUM LTD",
    25005: "GRACE W R & CO", 25022: "COMCAST CORP", 25056: "ENSERCH CORP", 25371: "FEDERAL PAPER BOARD CO",
    26542: "TENNECO INC", 26550: "FIRST INTERSTATE BANCORP", 26729: "MCDONNELL DOUGLAS CORP",
    26884: "RUBBERMAID INC", 27051: "AMP INC", 27115: "GREAT WESTERN FINANCIAL CORP",
    27263: "CORESTATES FINANCIAL CORP", 27596: "SALOMON INC", 27764: "STRIDE RITE CORP",
    28118: "NACCO INDUSTRIES", 30067: "DSC COMMUNICATIONS CORP", 30330: "CAPITAL CITIES ABC INC",
    31536: "ECHLIN INC", 32205: "GIANT FOOD INC", 32563: "HARNISCHFEGER INDUSTRIES", 33312: "SUNAMERICA INC",
    33785: "LORAL CORP", 33814: "LOUISIANA LAND & EXPLORATION CO", 34040: "VARITY CORP", 35211: "PENNZOIL",
    36978: "STONE CONTAINER CORP", 40484: "HBO & CO", 41179: "NALCO CHEMICAL CO", 41718: "CHASE MANHATTAN CORP",
    42833: "PACIFICORP", 43916: "DIGITAL EQUIPMENT CORP", 44062: "SPRINGS INDUSTRIES", 44652: "AMERICAN STORES CO",
    44792: "INTERGRAPH CORP", 45858: "PORTLAND GENERAL CORP", 46295: "FLEETWOOD ENTERPRISES", 47079: "CITICORP",
    47255: "L BRANDS", 47490: "RUSSELL CORP", 47992: "BEVERLY ENTERPRISES", 48354: "BANKERS TRUST NEW YORK CORP",
    50024: "WELLS FARGO & CO", 50404: "MCI COMMUNICATIONS CORP", 50905: "HARLAND JOHN H CO", 50956: "USF&G CORP",
    51772: "BANKBOSTON CORP", 52250: "GLOBAL CROSSING LTD", 52821: "MERCANTILE BANCORPORATION",
    53081: "PROVIDIAN CORP", 53401: "COMMUNITY PSYCHIATRIC CENTERS", 53786: "BROWNING FERRIS INDUSTRIES",
    53938: "REPUBLIC NEW YORK CORP", 54690: "STARWOOD HOTELS & RESORTS WORLDWIDE", 56266: "AHMANSON H F & CO",
    56450: "FIRST CHICAGO NBD CORP", 57381: "WMX TECHNOLOGIES", 57592: "DATA GENERAL CORP",
    58480: "BALLY ENTERTAINMENT CORP", 58640: "NORTEL NETWORKS CORP", 58827: "BANKAMERICA CORP",
    59432: "CALIBER SYSTEM INC", 59926: "MANOR CARE INC", 60046: "AMDAHL CORP", 61284: "BARNETT BANKS INC",
    62164: "CRAY RESEARCH INC", 62252: "GENERAL RE CORP", 62324: "MOORE CORP LTD", 63096: "PIONEER HI BRED INTERNATIONAL",
    63845: "ALEXANDER & ALEXANDER SERVICES", 64020: "LUBYS", 64346: "RAYCHEM CORP", 64451: "ARCHSTONE SMITH TRUST",
    64531: "PROVIDENT COMPANIES INC", 64629: "MCDERMOTT INTERNATIONAL", 65066: "SAFETY KLEEN CORP",
    65568: "ECHO BAY MINES LTD", 65859: "AMERITECH CORP", 66018: "NYNEX CORP", 66026: "PACIFIC TELESIS GROUP",
    67571: "GREEN TREE FINANCIAL CORP", 68049: "RYANS FAMILY STEAK HOUSES", 70376: "SHONEYS INC",
    71765: "CONRAIL INC", 71984: "FRUIT OF THE LOOM", 74617: "TANDEM COMPUTERS", 75030: "TELE COMMUNICATIONS INC",
    75092: "SAFECO CORP", 75181: "PLACER DOME INC", 75423: "ORYX ENERGY CO", 75647: "MORTON INTERNATIONAL",
    76070: "GIDDINGS & LEWIS", 76512: "LAIDLAW INC", 76754: "BAY NETWORKS INC", 78022: "CIT GROUP",
    78788: "HFS INC", 78946: "DEAN WITTER DISCOVER & CO", 78968: "U S BANCORP", 79119: "U S HEALTHCARE INC",
    79250: "UNITED STATES SURGICAL CORP", 80410: "WESTERN ATLAS INC", 80506: "ASCEND COMMUNICATIONS",
    80678: "CASE CORP", 80694: "SANTA FE PACIFIC GOLD CORP", 82776: "ITT CORP", 83988: "GRACE W R & CO",
    85869: "BATTLE MOUNTAIN GOLD CO", 86326: "GLOBAL CROSSING LTD", 87127: "CYPRUS AMAX MINERALS CO",
    89392: "KING WORLD PRODUCTIONS", 93236: "PHARMACIA & UPJOHN INC",
    # REITs / foreign issuers missing from FIAM, dead before 2024
    28388: "VORNADO REALTY TRUST", 75341: "DUKE REALTY CORP", 75592: "PLUM CREEK TIMBER CO",
    78927: "DEVELOPERS DIVERSIFIED REALTY CORP", 79129: "GGP INC", 79237: "TRANSOCEAN", 80399: "MACERICH CO",
    80406: "PROLOGIS", 80711: "APARTMENT INVESTMENT & MANAGEMENT CO", 85238: "EQUITY OFFICE PROPERTIES TRUST",
    85265: "SL GREEN REALTY CORP", 86314: "XL GROUP", 92156: "COVIDIEN", 93159: "ENSCO", 12076: "SIGNET JEWELERS",
    12542: "NIELSEN HOLDINGS", 13142: "CAPRI HOLDINGS", 14011: "MALLINCKRODT", 14704: "IHS MARKIT",
    15401: "BAXALTA INC", 15404: "COLUMBIA PIPELINE GROUP", 16538: "TECHNIPFMC", 92389: "EQUITY RESIDENTIAL",
    90251: "FREESCALE SEMICONDUCTOR", 91388: "HILTON HOTELS CORP", 61209: "TRANE",
}


# Hand-verified CIKs where automated matching fails (CUSIP map gap + name mismatch).
FORCE = {76644: 1163302}   # USX-U.S. Steel Group -> United States Steel Corp (2001+)


def norm(s: str) -> str:
    s = (s or "").upper().replace("&", " AND ")
    s = re.sub(r"/[A-Z]{2,3}/?", " ", s)          # EDGAR state tags like /DE/
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    s = SUFFIX.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def _subseq(x: str, y: str) -> bool:
    it = iter(y)
    return all(ch in it for ch in x)


def abbr_sim(a: str, b: str) -> float:
    """FIAM abbreviates ("NATL SEMICON", "CTRYWIDE FINL", "1ST DATA"): every token of `a` must be a
    same-initial subsequence of a later token of `b`, in order. 1.0 for compact-equal strings."""
    a = a.replace("1ST", "FIRST"); b = b.replace("1ST", "FIRST")
    if not a or not b:
        return 0.0
    if a.replace(" ", "") == b.replace(" ", ""):
        return 1.0
    ta, tb = a.split(), b.split()
    if len(ta[0]) < 2:
        return 0.0
    j = 0
    for t in ta:
        while j < len(tb) and not (tb[j][0] == t[0] and _subseq(t, tb[j])):
            j += 1
        if j == len(tb):
            return 0.0
        j += 1
    extra = len(tb) - len(ta)
    if len(ta) == 1:            # one-word names ("APPLIED", "FOX") are only safe as exact matches
        return 0.93 if extra <= 0 else 0.7
    return 0.93 if extra <= 0 else 0.9 if extra == 1 else 0.8


def _edgar_names() -> pd.DataFrame:
    meta = pd.read_parquet(PROC / "edgar_meta.parquet")
    rows = [(r.cik, r.name) for r in meta.itertuples()]
    for r in meta[meta.former_names != "[]"].itertuples():
        for fn in json.loads(r.former_names):
            rows.append((r.cik, fn["name"]))
    n = pd.DataFrame(rows, columns=["cik", "name"]).drop_duplicates()
    n["n"] = n.name.map(norm)
    return n


def build() -> pd.DataFrame:
    mem = pd.read_parquet(PROC / "membership_linked.parquet")
    mem = mem[mem.permno > 0]
    permnos = sorted(mem.permno.unique())
    life = pd.read_parquet(PROC / "permno_life.parquet")          # CRSP daily first/last trade
    life = life.reindex(permnos)
    mlife = mem.groupby("permno").date.agg(["min", "max"])
    life["min"] = life["min"].fillna(mlife["min"]); life["max"] = life["max"].fillna(mlife["max"])
    alive_now = set(mem[mem.date >= "2024-12-01"].permno) | set(life.index[life["max"] >= "2024-12-01"])

    fil = pd.read_parquet(PROC / "edgar_filings.parquet", columns=["cik", "form", "filed"])
    fil = fil[fil.form.isin(["10-K", "10-K405", "10-KSB"])]
    fil["filed"] = pd.to_datetime(fil.filed)
    kdates = fil.groupby("cik").filed.apply(lambda s: np.sort(s.values)).to_dict()

    def overlap(p, cik):
        d = kdates.get(cik)
        if d is None:
            return 0
        a, b = life.at[p, "min"], life.at[p, "max"]
        return int(((d >= np.datetime64(a - pd.Timedelta(days=30))) &
                    (d <= np.datetime64(b + pd.Timedelta(days=60)))).sum())

    cands = []   # (permno, cik, method, score)
    # --- CUSIP
    fiam = pd.read_parquet(RAW / "delisted" / "fiam_monthly_2000_2023.parquet",
                           columns=["permno", "cusip", "comp_name", "stock_ticker", "date"])
    fiam = fiam[fiam.permno.isin(permnos)]
    cc = pd.read_csv(RAW / "linking" / "cik-cusip-maps.csv").dropna()
    cc["cik"] = cc.cik.astype(int)
    c8 = cc.groupby("cusip8").cik.apply(set).to_dict(); c6 = cc.groupby("cusip6").cik.apply(set).to_dict()
    for p, cus in fiam[["permno", "cusip"]].drop_duplicates().itertuples(index=False):
        for cik in c8.get(cus, ()):
            cands.append((p, cik, "cusip8", 1.0))
        for cik in c6.get(cus[:6], ()):
            cands.append((p, cik, "cusip6", 0.9))
    # --- current ticker
    ct = json.loads((RAW / "edgar" / "company_tickers.json").read_text())
    t2c = {}
    meta = pd.read_parquet(PROC / "edgar_meta.parquet", columns=["cik", "tickers"])
    for cik, ts in meta[meta.tickers != ""].itertuples(index=False):   # submissions' current tickers
        for t in ts.split(","):
            t2c.setdefault(t.upper().replace("-", "."), int(cik))
    t2c.update({v["ticker"].upper().replace("-", "."): int(v["cik_str"]) for v in ct.values()})
    lastt = mem.sort_values("date").groupby("permno").ticker.last()
    for p in alive_now:
        t = lastt[p].replace("-", ".")
        if t in t2c:
            cands.append((p, t2c[t], "ticker_now", 0.9))
    # --- names
    en = _edgar_names()
    en = en[en.cik.isin(kdates.keys())]
    choices = en.n.tolist()
    trunc = [x[:14] for x in choices]
    by2 = {}
    for i, x in enumerate(choices):
        by2.setdefault(x[:2], []).append(i)
    fname = fiam.groupby("permno").comp_name.apply(lambda s: list(pd.unique(s)))
    for p in permnos:
        names = [(nm, "name") for nm in fname.get(p, [])]
        if p in MANUAL:
            names.append((MANUAL[p], "manual"))
        for nm, meth in names:
            q = norm(nm)
            if len(q) < 3:
                continue
            hits = process.extract(q, choices, scorer=fuzz.ratio, limit=8, score_cutoff=86)
            if meth == "name" and len(nm) >= 15:      # FIAM names are truncated to 16 chars
                hits += process.extract(q[:14], trunc, scorer=fuzz.ratio, limit=8, score_cutoff=92)
            for i in by2.get(q[:2], ()):
                sc = abbr_sim(q, choices[i])
                if sc >= 0.9:
                    hits.append((None, sc * 100, i))
            for _, sc, i in hits:
                cands.append((p, int(en.cik.iloc[i]), meth, sc / 100))
    for p, cik in FORCE.items():
        if p in set(permnos):
            cands.append((p, cik, "cusip8+manual_cik", 1.0))
    c = pd.DataFrame(cands, columns=["permno", "cik", "method", "score"])
    c["n10k"] = [overlap(p, k) for p, k in zip(c.permno, c.cik)]
    c = c[c.n10k > 0]
    # name similarity between the permno's known names and the CIK's EDGAR names (current + former)
    cnames = en.groupby("cik").n.apply(list).to_dict()
    pnames = {p: [norm(x) for x in fname.get(p, [])] + ([norm(MANUAL[p])] if p in MANUAL else []) for p in permnos}

    def nsim(p, cik):
        best = 0
        for a in pnames[p]:
            for b in cnames.get(cik, []):
                best = max(best, fuzz.token_sort_ratio(a, b), fuzz.ratio(a[:14], b[:14]), 100 * abbr_sim(a, b))
        return best / 100 if pnames[p] else np.nan

    g = c.groupby(["permno", "cik"]).agg(methods=("method", lambda s: "+".join(sorted(set(s)))),
                                         n10k=("n10k", "max")).reset_index()
    g["nsim"] = [nsim(p, k) for p, k in zip(g.permno, g.cik)]
    cus = g.methods.str.contains("cusip8"); tk = g.methods.str.contains("ticker_now")
    nm = g.methods.str.contains("name|manual")
    # leoliu0's map is built from 13D/13G filings and carries filer-side errors (IBM -> ETEC,
    # GD -> SunTrust), so a CUSIP hit needs a name that agrees; a current-ticker hit is trusted
    # outright; a name-only hit needs a near-exact name.
    in_now = g.permno.isin(set(mem[mem.date >= "2024-12-01"].permno))
    ok = (tk & (in_now | (g.nsim >= 0.75))) | (cus & (g.nsim >= 0.75)) | (nm & (g.nsim >= 0.92)) | (g.methods.str.contains("cusip6") & (g.nsim >= 0.92))
    g = g[ok].copy()
    g["confidence"] = (0.45 * g.nsim.fillna(0.8) + 0.25 * (cus | tk) + 0.15 * tk + 0.15 * (cus & nm)).clip(0, 1).round(3)
    g.loc[g.methods == "manual", "confidence"] = (0.6 * g.nsim).round(3)
    # per permno: greedy year assignment. Highest-priority CIK takes every year it files a 10-K in
    # the permno's life; a lower-priority CIK is kept only for years nobody else covers
    # (reorganisations: Oracle Systems -> Oracle Corp; Disney 1996/2019).
    g["prio"] = g.confidence + 0.001 * g.n10k
    keep = []
    for p, grp in g.groupby("permno"):
        a, b = life.at[p, "min"] - pd.Timedelta(days=30), life.at[p, "max"] + pd.Timedelta(days=60)
        covered = set()
        for r in grp.sort_values("prio", ascending=False).itertuples():
            d = pd.DatetimeIndex(kdates[r.cik]); d = d[(d >= a) & (d <= b)]
            yrs = set(d.year) - covered
            weak = not re.search("cusip8|ticker_now", r.methods)
            # weak (name-only) CIKs fill uncovered years only with a near-exact name and >= 3 10-Ks;
            # as the sole link they need >= 2 10-Ks in the permno's life
            if not yrs or (covered and weak and not (r.nsim >= 0.97 and len(d) >= 3)) or (weak and len(d) < 2):
                continue
            covered |= yrs
            keep.append((p, r.cik, r.methods, r.confidence, r.nsim, len(d), min(yrs), max(yrs)))
    out = pd.DataFrame(keep, columns=["permno", "cik", "method", "confidence", "name_sim", "n10k", "first_year", "last_year"])
    # a cik linked to several permnos with overlapping lives is fine (dual-class shares).
    out = out.sort_values(["permno", "first_year"]).reset_index(drop=True)
    out.to_parquet(PROC / "permno_cik.parquet")
    report(out)
    return out


def report(out: pd.DataFrame, mem: pd.DataFrame | None = None) -> pd.DataFrame:
    if mem is None:
        mem = pd.read_parquet(PROC / "membership_linked.parquet")
    m = mem.copy()
    linked = set(out.permno)
    # 2025+ rows carry permno -1: map through the ticker's last pre-2025 permno
    t2p = m[m.permno > 0].sort_values("date").groupby("ticker").permno.last()
    m.loc[m.permno < 0, "permno"] = m.loc[m.permno < 0, "ticker"].map(t2p).fillna(-1).astype(int)
    m["has_cik"] = m.permno.isin(linked)
    r = m.groupby(m.date.dt.year).has_cik.agg(["mean", "size"]).rename(columns={"mean": "share_member_months_with_cik", "size": "member_months"})
    print(r.to_string())
    print("permnos linked:", len(linked), "of", mem[mem.permno > 0].permno.nunique())
    print(out.method.value_counts().head(15))
    return r


if __name__ == "__main__" and "extend" not in __import__("sys").argv:
    build()


def extend_live(link: pd.DataFrame | None = None) -> pd.DataFrame:
    """Post-CRSP (2025+) extension, applied on top of build() without touching earlier rows:
    (1) the latest CIK of every permno still trading at the CRSP end (2024-12) stays linked
        open-ended (last_year = 2100);
    (2) synthetic permnos (>= 900000, tickers new to the index since 2025, see membership_linked
        how='synthetic') are linked by ticker via company_tickers.json / EDGAR submissions tickers,
        mapped only for 10-Ks filed 2024+ (enough for A5's previous-10-K; keeps pre-2016 corpora fixed)."""
    link = pd.read_parquet(PROC / "permno_cik.parquet") if link is None else link
    link = link[link.permno < 900000].copy()
    life = pd.read_parquet(PROC / "permno_life.parquet")["max"]
    alive = link.permno.map(life) >= pd.Timestamp("2024-12-01")
    last = link.groupby("permno").last_year.transform("max")
    link.loc[alive & (link.last_year == last) & (link.last_year >= 2024), "last_year"] = 2100
    mem = pd.read_parquet(PROC / "membership_linked.parquet")
    syn = mem[mem.permno >= 900000].sort_values("date").groupby("permno").ticker.last()
    ct = json.loads((RAW / "edgar" / "company_tickers.json").read_text())
    t2c = {}
    meta = pd.read_parquet(PROC / "edgar_meta.parquet", columns=["cik", "tickers"])
    for cik, ts in meta[meta.tickers != ""].itertuples(index=False):
        for t in ts.split(","):
            t2c.setdefault(t.upper().replace("-", "."), int(cik))
    t2c.update({v["ticker"].upper().replace("-", "."): int(v["cik_str"]) for v in ct.values()})
    rows, miss = [], []
    for p, t in syn.items():
        k = t2c.get(t.replace("-", ".")) or {"SATS": 1415404}.get(t)   # EchoStar renamed SATS -> ECHO
        if k is None:
            miss.append(t); continue
        rows.append((p, k, "ticker_synthetic", 0.9, np.nan, 0, 2024, 2100))
    add = pd.DataFrame(rows, columns=link.columns)
    out = pd.concat([link, add], ignore_index=True)
    out.to_parquet(PROC / "permno_cik.parquet")
    print("open-ended live links:", int((out.last_year == 2100).sum()), "synthetic linked:", len(add), "unmatched:", miss)
    return out


if __name__ == "__main__" and "extend" in __import__("sys").argv:
    extend_live()
