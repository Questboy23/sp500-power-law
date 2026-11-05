"""10-Q/10-K filing dates per CIK from EDGAR submissions bulk (for T9 expected-announcement months)."""
import json, zipfile, pandas as pd
from config import RAW, PROC
rows = []
with zipfile.ZipFile(RAW / "edgar" / "submissions.zip") as z:
    for n in z.namelist():
        try:
            j = json.loads(z.read(n))
        except ValueError:
            continue
        if "cik" in j:
            cik = int(j["cik"]); rec = j.get("filings", {}).get("recent", {})
        else:
            cik = int(n[3:13]); rec = j
        for f, d in zip(rec.get("form", []), rec.get("filingDate", [])):
            if f in ("10-Q", "10-K", "10-K405", "10-Q405"):
                rows.append((cik, f, d))
df = pd.DataFrame(rows, columns=["cik", "form", "filed"]); df["filed"] = pd.to_datetime(df.filed)
df.to_parquet(PROC / "periodic_filings.parquet"); print(len(df), df.filed.min(), df.filed.max())
