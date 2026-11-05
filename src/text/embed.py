"""Point-in-time ChronoBERT embeddings of anonymised 10-K Item 1 text (spec rule 3).

Model: manelalab/chrono-bert-v1-{YYYY}1231 (He, Lv, Manela, Wu 2025; MIT licence), one vintage per
year 1999..2024, each trained only on text dated up to 31 Dec YYYY. A filing filed in year Y gets
vintage min(Y-1, 2024) -- the cutoff is always strictly before the filing date. Filings filed
before 2000 get no embedding (no vintage precedes them).

Text: anonymised Item 1 (src/text/anonymize.py); if Item 1 extraction gave < 200 words, the full
text after the first 3,000 characters (skips the cover page). First 2,048 tokens, 4 chunks of 512,
mean-pooled over tokens (attention mask) then averaged over chunks weighted by token count.
fp16 on MPS.

Outputs
  data/processed/text_emb_parts/v{YYYY}.parquet  resume-safe per-vintage parts (768-d)
  data/processed/text_embeddings_768.parquet      all parts
  data/processed/text_embeddings.parquet          permno, cik, accession, filed, vintage, emb_0..emb_63
                                                  (PCA fit on pre-2016 filings only)
"""
from __future__ import annotations
import os, shutil, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from config import ROOT, PROC, HOLDOUT_START

os.environ.setdefault("HF_HOME", str(ROOT / "data" / "raw" / "hf_cache"))
PARTS = PROC / "text_emb_parts"
MAX_TOK, CHUNK = 2048, 512


def model_id(v: int) -> str:
    return f"manelalab/chrono-bert-v1-{v}1231"


def vintage_for(filed: pd.Timestamp) -> int | None:
    v = min(filed.year - 1, 2024)
    return v if v >= 1999 else None


def doc_text(cik: int, acc: str, permno: int) -> tuple[str, str]:
    from src.text.features import load_doc
    from src.text.anonymize import anonymize
    d = load_doc(cik, acc)
    src = "item1"
    t = d["item1"]
    if len(t.split()) < 200:
        t, src = d["full"][3000:], "full"
    t = " ".join(t.split()[:2600])            # ~2,048 tokens after BPE; bounds regex time
    return anonymize(t, permno), src


class Embedder:
    def __init__(self, vintage: int, device: str | None = None, dtype=None):
        import torch
        from transformers import AutoTokenizer, AutoModel
        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_id(vintage))
        self.model = AutoModel.from_pretrained(model_id(vintage)).to(self.device).eval()
        if self.device != "cpu":
            self.model = self.model.half()

    def embed(self, texts: list[str], batch: int = 16) -> np.ndarray:
        torch = self.torch
        chunks, owner = [], []
        for i, t in enumerate(texts):
            ids = self.tok(t, add_special_tokens=False, truncation=True, max_length=MAX_TOK)["input_ids"]
            if not ids:
                ids = [self.tok.unk_token_id]
            for s in range(0, len(ids), CHUNK - 2):
                chunks.append([self.tok.cls_token_id] + ids[s:s + CHUNK - 2] + [self.tok.sep_token_id]); owner.append(i)
        order = np.argsort([len(c) for c in chunks])            # length-bucketed batches
        vecs = np.zeros((len(chunks), self.model.config.hidden_size), np.float32)
        ntok = np.array([len(c) for c in chunks], np.float32)
        with torch.no_grad():
            for b in range(0, len(order), batch):
                ix = order[b:b + batch]
                L = max(len(chunks[j]) for j in ix)
                inp = torch.full((len(ix), L), self.tok.pad_token_id, dtype=torch.long)
                att = torch.zeros((len(ix), L), dtype=torch.long)
                for r, j in enumerate(ix):
                    inp[r, :len(chunks[j])] = torch.tensor(chunks[j]); att[r, :len(chunks[j])] = 1
                inp, att = inp.to(self.device), att.to(self.device)
                h = self.model(input_ids=inp, attention_mask=att).last_hidden_state.float()
                m = att.unsqueeze(-1).float()
                vecs[ix] = ((h * m).sum(1) / m.sum(1)).cpu().numpy()
        out = np.zeros((len(texts), vecs.shape[1]), np.float32); w = np.zeros(len(texts), np.float32)
        for k, i in enumerate(owner):
            out[i] += vecs[k] * ntok[k]; w[i] += ntok[k]
        return out / np.maximum(w, 1)[:, None]


def _prep(rows):
    return [doc_text(c, a, p) for c, a, p in rows]


def embed_frame(fr: pd.DataFrame, vintage: int, emb: "Embedder | None" = None, log_every: int = 256) -> pd.DataFrame:
    from joblib import Parallel, delayed
    emb = emb or Embedder(vintage)
    rows = list(zip(fr.cik, fr.accession, fr.permno))
    out, t0 = [], time.time()
    step = 256
    for s in range(0, len(rows), step):
        part = rows[s:s + step]
        sub = [part[i:i + 32] for i in range(0, len(part), 32)]
        prepped = [x for chunk in Parallel(n_jobs=4)(delayed(_prep)(c) for c in sub) for x in chunk]
        E = emb.embed([t for t, _ in prepped])
        if not np.isfinite(E).all():
            raise FloatingPointError("non-finite embedding (fp16 overflow?)")
        out.append(pd.DataFrame(E, columns=[f"e{i}" for i in range(E.shape[1])]).assign(
            accession=[r[1] for r in part], permno=[r[2] for r in part], text_src=[x[1] for x in prepped]))
        if (s // step) % max(1, log_every // step) == 0:
            print(f"  v{vintage} {s + len(part)}/{len(rows)} {(s + len(part)) / (time.time() - t0):.1f} docs/s", flush=True)
    return pd.concat(out, ignore_index=True)


def _drop_model_cache(v: int):
    """Remove the vintage's snapshot AND the blobs it points to (newer hub layouts keep weights in a
    shared hub/blobs store, so deleting the model dir alone frees nothing)."""
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir(Path(os.environ["HF_HOME"]) / "hub")
        revs = [r.commit_hash for repo in info.repos if repo.repo_id == f"manelalab/chrono-bert-v1-{v}1231" for r in repo.revisions]
        if revs:
            info.delete_revisions(*revs).execute()
    except Exception:  # noqa: BLE001
        pass
    shutil.rmtree(Path(os.environ["HF_HOME"]) / "hub" / f"models--manelalab--chrono-bert-v1-{v}1231", ignore_errors=True)


def run(vintages: list[int] | None = None, keep_models: bool = False):
    from src.text.features import filing_index
    PARTS.mkdir(parents=True, exist_ok=True)
    idx = filing_index()
    idx["vintage"] = idx.filed.map(vintage_for)
    idx = idx[idx.vintage.notna()].copy(); idx["vintage"] = idx.vintage.astype(int)
    idx = idx.drop_duplicates(["permno", "accession"])
    for v in sorted(vintages or idx.vintage.unique()):
        p = PARTS / f"v{v}.parquet"
        fr = idx[idx.vintage == v]
        if p.exists() and len(pd.read_parquet(p, columns=["accession"])) >= len(fr):
            continue
        print(f"vintage {v}: {len(fr)} filings", flush=True)
        e = embed_frame(fr, v)
        e.merge(fr[["permno", "accession", "cik", "filed", "vintage"]], on=["permno", "accession"]).to_parquet(p)
        if not keep_models:
            _drop_model_cache(v)
    combine()


def combine(n_pca: int = 64):
    from sklearn.decomposition import PCA
    parts = [pd.read_parquet(p) for p in sorted(PARTS.glob("v*.parquet"))]
    df = pd.concat(parts, ignore_index=True)
    ecols = [c for c in df.columns if c.startswith("e") and c[1:].isdigit()]
    df.to_parquet(PROC / "text_embeddings_768.parquet")
    pre = df.filed < HOLDOUT_START
    pca = PCA(n_pca, random_state=0).fit(df.loc[pre, ecols].values)
    Z = pca.transform(df[ecols].values).astype(np.float32)
    out = df[["permno", "cik", "accession", "filed", "vintage", "text_src"]].reset_index(drop=True)
    out = pd.concat([out, pd.DataFrame(Z, columns=[f"emb_{i}" for i in range(n_pca)])], axis=1)
    out.to_parquet(PROC / "text_embeddings.parquet")
    print("embeddings:", len(out), "PCA var explained:", round(float(pca.explained_variance_ratio_.sum()), 3),
          "fit on", int(pre.sum()), "pre-2016 filings")


if __name__ == "__main__":
    run([int(x) for x in sys.argv[1:]] or None)
