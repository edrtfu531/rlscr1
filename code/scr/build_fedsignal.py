
# build_fedsignal.py

import argparse, re, sys, math
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, pipeline

# Helpers
def load_statements_from_dir(directory: Path) -> pd.DataFrame:
    """
    Reads all .txt files from a directory, parsing dates from filenames.
    """
    dates, texts = [], []
    print(f"Loading statements from: {directory}/*.txt")
    files = sorted(list(directory.glob("*.txt")))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {directory}")

    for file_path in files:
        date_str = file_path.stem
        try:
            dates.append(pd.to_datetime(date_str, format="%Y%m%d"))
            texts.append(file_path.read_text(encoding="utf-8"))
        except (ValueError, IOError) as e:
            print(f"Could not process file {file_path.name}: {e}")
            continue
    
    df = pd.DataFrame({"date": dates, "text": texts})
    print(f"Successfully loaded {len(df)} statements.")
    return df

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()

HIKE_PAT = re.compile(r"\b(raise|increas|higher|tighten(ed|ing)?|hike|lift)\b", flags=re.I)
HOLD_PAT = re.compile(r"\b(maintain|keep|unchanged|maintained)\b", flags=re.I)
CUT_PAT  = re.compile(r"\b(lower|decreas|reduce|cut|eas(e|ed|ing))\b", flags=re.I)

def stance_probs(text: str) -> Tuple[float,float,float]:
    if not isinstance(text, str): return 0.0, 1.0, 0.0
    h = len(HIKE_PAT.findall(text))
    o = len(HOLD_PAT.findall(text))
    c = len(CUT_PAT.findall(text))
    
    total = h + o + c
    if total == 0:
        return 0.0, 1.0, 0.0 # Neutral

    # Normalize counts to frequencies
    # Multiply by temperature to keep distribution sharp
    x = np.array([h/total, o/total, c/total], dtype=float) * 5.0
    p = softmax(x)
    return float(p[0]), float(p[1]), float(p[2])


def chunk_text(s: str, tok, max_length: int) -> List[dict]:
    if not s: return []
    parts = [p.strip() for p in re.split(r'(?<=[\.\?!])\s+', s) if p.strip()]
    chunks = []
    cur = ""
    for p in parts:
        tmp = (cur + " " + p).strip() if cur else p
        enc = tok(tmp, return_tensors="pt", truncation=True, max_length=max_length)
        if enc["input_ids"].shape[1] < max_length:
            cur = tmp
        else:
            if cur: chunks.append(tok(cur, return_tensors="pt", truncation=True, max_length=max_length))
            cur = p
    if cur: chunks.append(tok(cur, return_tensors="pt", truncation=True, max_length=max_length))
    return chunks

@torch.no_grad()
def embed_texts(texts: List[str], model_name: str = "bert-base-uncased", max_length: int = 256, device: Optional[str] = None) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Embedding texts using {model_name} on {device}...")
    tok = AutoTokenizer.from_pretrained(model_name)
    enc = AutoModel.from_pretrained(model_name).to(device).eval()
    vecs = []
    for s in texts:
        if not isinstance(s, str) or len(s.strip()) == 0:
            vecs.append(np.zeros(enc.config.hidden_size, dtype=np.float32))
            continue
        batches = chunk_text(s, tok, max_length=max_length)
        if not batches:
            vecs.append(np.zeros(enc.config.hidden_size, dtype=np.float32))
            continue
        chunk_vecs = []
        for b in batches:
            b = {k: v.to(device) for k, v in b.items()}
            out = enc(**b)
            last, mask = out.last_hidden_state, b["attention_mask"].unsqueeze(-1)
            pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            chunk_vecs.append(pooled.squeeze(0).detach().cpu().numpy())
        vecs.append(np.mean(chunk_vecs, axis=0).astype(np.float32))
    return np.vstack(vecs)

@torch.no_grad()
def finbert_sentiment(texts: List[str], max_length: int = 256) -> np.ndarray:
    print("Calculating FinBERT sentiment...")
    mdl = "ProsusAI/finbert"
    tok = AutoTokenizer.from_pretrained(mdl)
    clf = pipeline("sentiment-analysis", model=mdl, tokenizer=tok, top_k=None, truncation=True)
    label_to_idx = {"positive": 0, "negative": 1, "neutral": 2}
    out = np.zeros((len(texts), 3), dtype=np.float32)
    for i, s in enumerate(texts):
        if not isinstance(s, str) or not s.strip(): continue
        chunks = chunk_text(s, tok, max_length=max_length) or [tok(s, return_tensors="pt", truncation=True, max_length=max_length)]
        scores = []
        for b in chunks:
            txt = tok.decode(b["input_ids"][0], skip_special_tokens=True)
            res = clf(txt)
            dist = np.zeros(3, dtype=np.float32)
            for d in res[0]:
                j = label_to_idx[d["label"].lower()]
                dist[j] = float(d["score"])
            scores.append(dist)
        m = np.mean(scores, axis=0)
        out[i] = m / (m.sum() if m.sum() > 0 else 1.0)
    return out

def pca_reduce(X: np.ndarray, k: int, train_mask: np.ndarray) -> Tuple[np.ndarray, dict]:
    print(f"Performing PCA to reduce embedding dimension to {k}...")
    X = X.astype(np.float64)
    mu = X[train_mask].mean(axis=0, keepdims=True)
    Xc_train = X[train_mask] - mu
    _, _, Vt = np.linalg.svd(Xc_train, full_matrices=False)
    k = min(k, Vt.shape[0])
    Z_full = (X - mu) @ Vt[:k].T
    return Z_full.astype(np.float32), {"mu": mu, "Vt_k": Vt[:k]}


# Main build
def build_daily_fedsignal(fomc_dir: Path, out_path: Path, train_end: str, emb_model: str = "bert-base-uncased",
                          max_length: int = 256, returns_csv: Path = None,
                          decay_rate = 0.95):
    # Load from directory
    fomc = load_statements_from_dir(fomc_dir)
    fomc["date"] = pd.to_datetime(fomc["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    fomc = fomc.dropna(subset=["date"]).sort_values("date")

    # Filter statements to include the last one before analysis period
    start_date_filter = pd.to_datetime("2008-08-01")
    print(f"Filtering statements to start from {start_date_filter.date()}...")
    fomc = fomc[fomc["date"] >= start_date_filter].reset_index(drop=True)
    if fomc.empty:
        raise ValueError(f"No FOMC statements found on or after {start_date_filter.date()}.")

    # Stance probs
    print("Calculating stance probabilities...")
    stance = np.array([stance_probs(t) for t in fomc["text"].tolist()], dtype=np.float32)
    p_hike, p_hold, p_cut = stance[:,0], stance[:,1], stance[:,2]
    stance_score = p_hike - p_cut

    # FinBERT sentiment and Embeddings
    sent = finbert_sentiment(fomc["text"].tolist(), max_length=max_length)
    emb = embed_texts(fomc["text"].tolist(), model_name=emb_model, max_length=max_length)
    if emb.shape[1] > 64:
        te = pd.Timestamp(train_end)
        train_mask = (fomc["date"] <= te).to_numpy()
        emb, _ = pca_reduce(emb, k=64, train_mask=train_mask)

    # Build per-statement frame
    cols = {
        "p_hike": p_hike, "p_hold": p_hold, "p_cut":  p_cut, "stance_score": stance_score,
        "finbert_sent_pos": sent[:,0], "finbert_sent_neg": sent[:,1], "finbert_sent_neu": sent[:,2],
    }
    for i in range(emb.shape[1]): cols[f"emb_{i+1}"] = emb[:, i]
    stmt = pd.DataFrame({"date": fomc["date"]})
    for k, v in cols.items(): stmt[k] = v.astype(np.float32)

    if not (returns_csv and returns_csv.exists()):
        raise FileNotFoundError(f"The required returns.csv file was not found at '{returns_csv}'.")

    daily = pd.read_csv(returns_csv, parse_dates=["date"])[["date"]].drop_duplicates().sort_values("date")
    
    stmt  = stmt.sort_values("date")
    stmt["statement_date"] = stmt["date"]

    out = pd.merge_asof(daily, stmt, on="date", direction="backward")
    out["days_since"] = (out["date"] - out["statement_date"]).dt.days
    
    out["days_since"] = out["days_since"].fillna(999)
    decay_vec = np.power(decay_rate, out["days_since"].values)

    # Enforce next-day availability with a 1-row shift
    feat_cols = [c for c in out.columns if c not in ["date", "statement_date", "days_since"]]
    # Vectorized decay application
    for col in feat_cols:
        out[col] = out[col] * decay_vec
        out[col] = out[col].fillna(0.0)

    out[feat_cols] = out[feat_cols].shift(1)
    out = out[["date"] + feat_cols]

    output_start_date = pd.to_datetime("2009-01-01")
    out = out[out["date"] >= output_start_date].reset_index(drop=True)

    # Save
    out.to_csv(out_path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fomc_dir", type=Path, default=Path("../data/fomc_statements"))
    ap.add_argument("--out", type=Path, default=Path("../data/fomc_statements/fedsignal.csv"))
    ap.add_argument("--returns_csv", type=Path, default=Path("../data/market/returns.csv"), required=True)
    ap.add_argument("--emb_model", type=str, default="bert-base-uncased")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--train_end", type=str, default="2017-12-31")
    ap.add_argument("--decay_rate", type=float, default=0.95, 
                    help="Lambda for exponential decay. 0.95 means signal retains 95% strength per day.")

    args = ap.parse_args()
    build_daily_fedsignal(
        args.fomc_dir,
        args.out,
        emb_model=args.emb_model,
        max_length=args.max_length,
        returns_csv=args.returns_csv,
        train_end=args.train_end,
        decay_rate=args.decay_rate
    )

if __name__ == "__main__":
    main()

