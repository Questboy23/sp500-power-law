"""Download 10-K main documents from EDGAR and store Item 1 + cleaned full text.

One gzip JSON per filing: data/raw/tenk/{cik}/{accession}.json.gz with keys
cik, accession, form, filed, source_url, n_words, item1, full.
Resume-safe (existing files are skipped; failures are logged to data/raw/tenk/_failed.tsv and
retried on the next run). SEC fair-access: declared User-Agent, <= 8 requests/second overall
(6 worker processes x 1.25 req/s each = 7.5 req/s).
"""
from __future__ import annotations
import gzip, html, json, os, re, sys, time
from multiprocessing import Pool
import pandas as pd, requests
from config import RAW, PROC

def _ua() -> str:
    """SEC EDGAR asks for a contact email in the User-Agent. Set SEC_CONTACT_EMAIL before running."""
    email = os.environ.get("SEC_CONTACT_EMAIL", "").strip()
    if not email:
        raise RuntimeError("SEC_CONTACT_EMAIL is not set. The SEC requires a contact email in the "
                           "User-Agent of every request. Set it first: export SEC_CONTACT_EMAIL=you@example.com")
    return f"sp500-power-law research {email}"


OUT = RAW / "tenk"
N_PROC = 6
PER_PROC_INTERVAL = N_PROC / 7.5          # seconds between requests in one process

_sess = None
_last = 0.0


def _get(url: str) -> requests.Response:
    global _sess, _last
    if _sess is None:
        _sess = requests.Session(); _sess.headers.update({"User-Agent": _ua(), "Accept-Encoding": "gzip, deflate"})
    for attempt in range(5):
        wait = PER_PROC_INTERVAL - (time.monotonic() - _last)
        if wait > 0:
            time.sleep(wait)
        _last = time.monotonic()
        try:
            r = _sess.get(url, timeout=60)
        except requests.RequestException:
            time.sleep(2 ** attempt); continue
        if r.status_code in (429, 503):
            time.sleep(10 * (attempt + 1)); continue
        return r
    raise RuntimeError(f"gave up {url}")


# ---------------------------------------------------------------- text cleaning
_BLOCK = re.compile(r"(?i)<\s*(br|/p|/div|/tr|/li|/h\d|/table|p|div|tr|li|h\d)\b[^>]*>")
_TAG = re.compile(r"(?s)<[^>]+>")


def html_to_text(s: str) -> str:
    s = re.sub(r"(?is)<(script|style|ix:header|head)\b.*?</\1\s*>", " ", s)
    s = re.sub(r"(?is)<div[^>]*display:\s*none[^>]*>.*?</div>", " ", s)
    s = _BLOCK.sub("\n", s)
    s = _TAG.sub(" ", s)
    s = html.unescape(s).replace("\xa0", " ")
    return s


def clean(s: str) -> str:
    s = re.sub(r"(?i)</?(PAGE|S|C|FN|F\d+|CAPTION|TABLE)>", " ", s)    # SGML page/table markers
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n\n", s)
    s = re.sub(r"(?m)^\s*-?\s*\d{1,3}\s*-?\s*$", "", s)                    # bare page numbers
    return s.strip()


def sgml_main_doc(txt: str, forms=("10-K", "10-K405")) -> str | None:
    """Pull the main 10-K <DOCUMENT> out of a full-submission .txt."""
    for m in re.finditer(r"(?is)<DOCUMENT>(.*?)</DOCUMENT>", txt):
        d = m.group(1)
        t = re.search(r"(?i)<TYPE>\s*([^\s<]+)", d)
        if t and t.group(1).upper() in forms:
            body = re.search(r"(?is)<TEXT>(.*)</TEXT>", d)
            return body.group(1) if body else d
    return None


# ---------------------------------------------------------------- Item 1
_START = re.compile(r"(?im)^\W{0,10}item\s*1\s*[\.\:\-–—]?\s*(?:\(?\s*[a-z]?\s*\)?)?\s*[\.\:\-–—]?\s*(description\s+of\s+)?business\b")
_START2 = re.compile(r"(?i)\bitem\s*1\s*[\.\:\-–—]?\s*(description\s+of\s+)?business\b")
_END = re.compile(r"(?i)\bitem\s*(1\s*a\b|1a\b|2\s*[\.\:\-–—]?\s*(description\s+of\s+)?propert|3\s*[\.\:\-–—]?\s*legal)")


def item1(text: str) -> str:
    """Longest 'Item 1. Business' -> 'Item 1A'/'Item 2'/'Item 3' span (skips the table of contents)."""
    best = ""
    for rx in (_START, _START2):
        for m in rx.finditer(text):
            e = _END.search(text, m.end())
            seg = text[m.start(): e.start() if e else min(len(text), m.start() + 400_000)]
            if len(seg) > len(best):
                best = seg
        if len(best) > 2000:
            break
    return best.strip()


# ---------------------------------------------------------------- worker
def _path(cik: int, acc: str):
    return OUT / str(cik) / f"{acc}.json.gz"


def fetch_one(row) -> tuple[str, str]:
    cik, acc, form, filed, doc = row
    p = _path(cik, acc)
    if p.exists():
        return acc, "skip"
    nd = acc.replace("-", "")
    try:
        body = None
        if doc and not doc.lower().endswith((".txt", ".pdf", ".paper")) and doc.lower() != "paper":
            url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{nd}/{doc}"
            r = _get(url)
            if r.status_code == 200:
                body = html_to_text(r.content.decode("utf-8", "replace"))
        if body is None or len(body) < 5000:
            url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}.txt"
            r = _get(url)
            if r.status_code != 200:
                return acc, f"http{r.status_code}"
            raw = r.content.decode("latin-1", "replace")
            d = sgml_main_doc(raw, (form, "10-K", "10-K405"))
            if d is None:
                return acc, "nodoc"
            body = html_to_text(d) if re.search(r"(?i)<(html|p|div|font|table)\b", d[:20000]) else d
        full = clean(body)
        it1 = item1(full)
        rec = dict(cik=int(cik), accession=acc, form=form, filed=filed, source_url=url,
                   n_words=len(full.split()), item1=it1, full=full)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        with gzip.open(tmp, "wt", compresslevel=6) as f:
            json.dump(rec, f)
        os.replace(tmp, p)
        return acc, "ok"
    except Exception as e:  # noqa: BLE001
        return acc, f"err:{type(e).__name__}:{str(e)[:80]}"


def filing_list() -> pd.DataFrame:
    link = pd.read_parquet(PROC / "permno_cik.parquet")
    f = pd.read_parquet(PROC / "edgar_filings.parquet")
    f = f[f.form.isin(["10-K", "10-K405"]) & (f.filed >= "1996-01-01") & f.cik.isin(set(link.cik))]
    return f.drop_duplicates("accession").sort_values("filed", ascending=False).reset_index(drop=True)


def run(limit: int | None = None):
    OUT.mkdir(parents=True, exist_ok=True)
    fl = filing_list()
    todo = [r for r in fl[["cik", "accession", "form", "filed", "doc"]].itertuples(index=False, name=None)
            if not _path(r[0], r[1]).exists()]
    if limit:
        todo = todo[:limit]
    print(f"{len(fl)} filings, {len(todo)} to fetch", flush=True)
    stats, t0, failed = {}, time.time(), []
    with Pool(N_PROC) as pool:
        for i, (acc, st) in enumerate(pool.imap_unordered(fetch_one, todo, chunksize=4), 1):
            k = st.split(":")[0]; stats[k] = stats.get(k, 0) + 1
            if k not in ("ok", "skip"):
                failed.append((acc, st))
            if i % 200 == 0:
                print(f"{i}/{len(todo)} {stats} {i / (time.time() - t0):.2f}/s", flush=True)
    pd.DataFrame(failed, columns=["accession", "status"]).to_csv(OUT / "_failed.tsv", sep="\t", index=False)
    print("done", stats, flush=True)


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else None)
