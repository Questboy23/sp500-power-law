"""EDGAR fundamentals (XBRL companyfacts, 2009+) and filing index (submissions).

Point-in-time rule: every fact carries its SEC `filed` date; features may only use
facts with filed <= formation date."""
from __future__ import annotations
import json, zipfile
import pandas as pd
from config import RAW, PROC

TAGS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet"],
    "op_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss"],
    "gross_profit": ["GrossProfit"],
    "cost_of_rev": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "rd": ["ResearchAndDevelopmentExpense", "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"],
    "sga": ["SellingGeneralAndAdministrativeExpense"],
    "advertising": ["AdvertisingExpense", "MarketingAndAdvertisingExpense"],
    "assets": ["Assets"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "liabilities": ["Liabilities"],
    "lt_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "interest_exp": ["InterestExpense", "InterestExpenseNonoperating"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "dna": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
    "shares": ["EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding"],
    "cur_assets": ["AssetsCurrent"], "cur_liab": ["LiabilitiesCurrent"],
}
FLOW = {"revenue", "op_income", "net_income", "gross_profit", "cost_of_rev", "rd", "sga", "advertising",
        "interest_exp", "capex", "dna", "cfo"}


def extract_companyfacts():
    tag2f = {t: f for f, ts in TAGS.items() for t in ts}
    pri = {t: i for f, ts in TAGS.items() for i, t in enumerate(ts)}
    rows = []
    with zipfile.ZipFile(RAW / "edgar" / "companyfacts.zip") as z:
        for n in z.namelist():
            j = json.loads(z.read(n))
            cik = int(j.get("cik") or 0)
            for tax in ("us-gaap", "dei"):
                for tag, v in j.get("facts", {}).get(tax, {}).items():
                    if tag not in tag2f:
                        continue
                    f = tag2f[tag]
                    for unit, facts in v.get("units", {}).items():
                        for x in facts:
                            form = x.get("form", "")
                            if not form.startswith("10-K") and f != "shares":
                                continue
                            if f in FLOW:   # keep ~annual durations only
                                if "start" not in x:
                                    continue
                                dur = (pd.Timestamp(x["end"]) - pd.Timestamp(x["start"])).days
                                if not 350 <= dur <= 380:
                                    continue
                            rows.append((cik, f, pri[tag], x["end"], x.get("filed"), x["val"], form))
    df = pd.DataFrame(rows, columns=["cik", "field", "pri", "end", "filed", "val", "form"])
    df["end"] = pd.to_datetime(df.end); df["filed"] = pd.to_datetime(df.filed)
    # earliest filing of each (cik, field, period) with the preferred tag = the first public value
    df = df.sort_values(["cik", "field", "end", "pri", "filed"]).drop_duplicates(["cik", "field", "end", "pri"])
    df = df.sort_values(["cik", "field", "end", "filed", "pri"]).drop_duplicates(["cik", "field", "end"])
    df.drop(columns=["pri"]).to_parquet(PROC / "fundamentals_long.parquet")
    print(df.shape, df.cik.nunique())


def extract_submissions():
    """cik -> current tickers, former names, and every 10-K filing date (for text + timing)."""
    meta, filings = [], []
    with zipfile.ZipFile(RAW / "edgar" / "submissions.zip") as z:
        for n in z.namelist():
            try:
                j = json.loads(z.read(n))
            except ValueError:
                continue
            if "cik" in j:
                cik = int(j["cik"])
                meta.append((cik, j.get("name"), ",".join(j.get("tickers", []) or []), j.get("sic"),
                             json.dumps(j.get("formerNames", []))))
                rec = j.get("filings", {}).get("recent", {})
            else:   # overflow file CIK##########-submissions-###.json
                cik = int(n[3:13]); rec = j
            forms = rec.get("form", [])
            for i, f in enumerate(forms):
                if f in ("10-K", "10-K405", "10-KSB", "10-K/A", "4", "DEF 14A"):
                    filings.append((cik, f, rec["filingDate"][i], rec["accessionNumber"][i],
                                    rec.get("primaryDocument", [""] * len(forms))[i]))
    pd.DataFrame(meta, columns=["cik", "name", "tickers", "sic", "former_names"]).to_parquet(PROC / "edgar_meta.parquet")
    pd.DataFrame(filings, columns=["cik", "form", "filed", "accession", "doc"]).to_parquet(PROC / "edgar_filings.parquet")
    print(len(meta), len(filings))


if __name__ == "__main__":
    import sys
    if "subs" not in sys.argv: extract_companyfacts()
    extract_submissions()
