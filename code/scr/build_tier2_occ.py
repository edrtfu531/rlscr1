# build_tier2_occ.py

from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional, Union
import pandas as pd
import argparse, yaml
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

def _load_csv_with_date(fp: Union[str, Path],
                        date_candidates: Optional[Iterable[str]] = None) -> pd.DataFrame:
    if date_candidates is None:
        date_candidates = (
            "date","Date","DATE","observation_date","timestamp","Timestamp","time","Time","TIME",
            "dt","DT","month","Month","MONTH","DATE ","index","level_0","Unnamed: 0","Unnamed: 0_level_0"
        )

    df = pd.read_csv(fp)

    date_col = next((c for c in df.columns if c.lower() == "date"), None)
    if date_col is None:
        date_col = next((c for c in date_candidates if c in df.columns), None)

    if date_col is None:
        best_col, best_hits = None, -1
        for c in df.columns:
            try:
                parsed = pd.to_datetime(df[c], errors="coerce")
                hits = int(parsed.notna().sum())
                if hits > max(5, int(0.3 * len(parsed))) and hits > best_hits:
                    best_col, best_hits = c, hits
            except Exception:
                continue
        date_col = best_col

    if date_col is None:
        raise ValueError(f"{fp} does not have a recognizable date column.")

    if date_col != "date":
        df = df.rename(columns={date_col: "date"})

    # Normalize dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().all():
        try:
            df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
        except Exception:
            pass

    # Make tz-naive safely
    try:
        if hasattr(df["date"].dt, "tz_convert"):
            df["date"] = df["date"].dt.tz_convert(None)
    except Exception:
        pass

    try:
        if hasattr(df["date"].dt, "tz_localize"):
            df["date"] = df["date"].dt.tz_localize(None)
    except Exception:
        # Already tz-naive
        pass

    df["date"] = df["date"].dt.normalize()
    df = df.dropna(subset=["date"])

    # Deduplicate by date, sort, reset index
    df = df[~df["date"].duplicated(keep="last")].sort_values("date").reset_index(drop=True)
    return df


def _load_returns_matrix(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp, parse_dates=["date"])
    try:
        df["date"] = df["date"].dt.tz_convert(None)
    except TypeError:
        pass
    try:
        df["date"] = df["date"].dt.tz_localize(None)
    except TypeError:
        pass

    df["date"] = df["date"].dt.normalize()
    piv = df.pivot_table(index="date", columns="ticker", values="ret", aggfunc="first").sort_index()
    piv.columns = [f"ret_{c}" for c in piv.columns]
    return piv.reset_index()

# DP-means (DPMM)
def calibrate_lambda_sq(U_train: np.ndarray, q_nn: float = 0.90) -> float:
    if U_train.shape[0] < 5:
        return 1.0
    from sklearn.neighbors import NearestNeighbors
    d, _ = NearestNeighbors(n_neighbors=2).fit(U_train).kneighbors(U_train)
    lam = float(np.quantile(d[:, 1], q_nn))
    return max(1e-6, lam**2)


def dp_means_online(U: np.ndarray, dates: np.ndarray, is_shock: np.ndarray, lam_sq: float):
    # Online DP-means across time. Only shock days create/receive assignments.
    centers = []          # list of centroid vectors
    counts = []           # counts for incremental means
    ids = np.zeros(U.shape[0], dtype=np.int32)

    for t in range(U.shape[0]):
        if not is_shock[t]:
            continue
        u = U[t]
        if len(centers) == 0:
            centers.append(u.copy()); counts.append(1); ids[t] = 1
            continue
        
        # compute squared dists to existing centers
        d2 = np.array([np.sum((u - c)**2) for c in centers], dtype=np.float64)
        j = int(np.argmin(d2))
        if d2[j] > lam_sq:
            # new cluster
            centers.append(u.copy()); counts.append(1); ids[t] = len(centers)
        else:
            # assign + update center (incremental mean)
            ids[t] = j+1
            counts[j] += 1
            eta = 1.0 / counts[j]
            centers[j] = (1-eta)*centers[j] + eta*u
    return ids

# Smoothing & semantics
def smooth_activation(mask: np.ndarray, win: int = 3) -> np.ndarray:
    if win <= 1:
        return mask.astype(np.int32)
    out = np.zeros_like(mask, dtype=np.int32)
    for i in range(len(mask)):
        lo = max(0, i - (win - 1))
        # only past & today
        if mask[lo:i+1].max() > 0:
            out[i] = 1
    return out


def summarize_channel(
    df_day: pd.DataFrame,
    active_idx: np.ndarray,
    train_idx,
    macro_prefix="shock_", ret_prefix="ret_"
):
    # Return name, description, signature macro features, top movers (tickers).
    if active_idx.sum() == 0:
        return {
            "name": "Unnamed Channel",
            "description": "No train-period occurrences.",
            "keywords": [],
            "macro_signature": [],
            "top_movers_pos": [],
            "top_movers_neg": [],
        }

    # Macro signature: mean(active) - median(overall) per macro col
    mac_cols = [c for c in df_day.columns if c.startswith(macro_prefix)]
    base = df_day.loc[train_idx, mac_cols].median(axis=0, skipna=True)
    mu   = df_day.loc[active_idx, mac_cols].mean(axis=0, skipna=True)
    delta = (mu - base).sort_values(key=lambda s: s.abs(), ascending=False)

    # choose top 5 by |delta|
    top_sig = delta.head(5)

    # Returns movers: mean on active days
    ret_cols = [c for c in df_day.columns if c.startswith(ret_prefix)]
    movers = df_day.loc[active_idx, ret_cols].mean(axis=0, skipna=True).sort_values()
    top_neg = [c.replace(ret_prefix,"") for c in movers.head(5).index.tolist()]
    top_pos = [c.replace(ret_prefix,"") for c in movers.tail(5).index.tolist()]

    # crude keyword/name from macro signature
    def pretty_feat(c):
        key = c.replace(macro_prefix,"")
        return key

    arrows = []
    for k, v in top_sig.items():
        arrows.append(f"{pretty_feat(k)}{'↑' if v>0 else '↓'}")

    # Name template heuristics
    name = " / ".join(arrows[:3]) if arrows else "Macro Shock"
    desc = f"Macro signature: {', '.join(arrows)}. Top movers: +{', '.join(top_pos[-3:])}; -{', '.join(top_neg[:3])}."

    return {
        "name": name,
        "description": desc,
        "keywords": [k.replace(macro_prefix,"") for k in top_sig.index.tolist()],
        "macro_signature": [(k.replace(macro_prefix,""), float(v)) for k, v in top_sig.items()],
        "top_movers_pos": top_pos[-5:],
        "top_movers_neg": top_neg[:5],
    }

# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier1", type=Path, required=True)
    ap.add_argument("--returns", type=Path, required=True)
    ap.add_argument("--shock", type=Path, required=True)
    ap.add_argument("--train_end", type=str, required=True)
    ap.add_argument("--out_channels", type=Path, default=Path("data/tier2_channels.csv"))
    ap.add_argument("--out_ledger_y", type=Path, default=Path("data/tier2_shock_ledger.yaml"))
    ap.add_argument("--out_ledger_p", type=Path, default=Path("data/tier2_shock_ledger.csv"))
    ap.add_argument("--out_macro_sig", type=Path, default=Path("data/macro_signature.csv"))
    ap.add_argument("--smooth_win", type=int, default=3)
    ap.add_argument("--q_nn", type=float, default=0.90)
    args = ap.parse_args()

    # Load sources
    tier1 = _load_csv_with_date(args.tier1)  # date, Z, is_shock, U_1..U_k
    shock = _load_csv_with_date(args.shock)  # macro features

    rename = {c: (c if c.startswith("shock_") else f"shock_{c}")
              for c in shock.columns if c != "date"}
    shock = shock.rename(columns=rename)
    ret   = _load_returns_matrix(args.returns)

    # returns + macro
    df_day = ret.merge(shock, on="date", how="left").sort_values("date").reset_index(drop=True)
    
    mac_cols = [c for c in df_day.columns if c.startswith("shock_")]
    ret_cols = [c for c in df_day.columns if c.startswith("ret_")]
    df_day[mac_cols] = df_day[mac_cols].ffill()
    df_day[ret_cols] = df_day[ret_cols].fillna(0.0)

    df = df_day.merge(tier1, on="date", how="left")
    U_cols = [c for c in df.columns if c.startswith("U_")]
    if not U_cols: raise ValueError("Tier1 must contain U_1..U_k columns.")
    if "is_shock" not in df.columns: raise ValueError("Tier1 must contain is_shock column.")

    dates = df["date"].to_numpy()
    U = df[U_cols].to_numpy(dtype=np.float32)
    is_shock = df["is_shock"].fillna(0).to_numpy(dtype=np.int32)

    # Train mask (for lambda calibration & naming)
    train_end = pd.Timestamp(args.train_end)
    mask_train = (df["date"] <= train_end).to_numpy()
    mask_train_shock = (mask_train & (is_shock==1))

    # Whiten U on train shock days
    if mask_train_shock.sum() >= 5:
        U_train_shock = U[mask_train_shock]
        mu = U_train_shock.mean(axis=0, keepdims=True)
        sd = U_train_shock.std(axis=0, keepdims=True) + 1e-9
        U_whiten = (U - mu) / sd

        # Calibrate lambda on whitened train shocks
        lam_sq = calibrate_lambda_sq(U_whiten[mask_train_shock], q_nn=args.q_nn)

        raw_ids = dp_means_online(U_whiten, dates, is_shock, lam_sq=lam_sq)
    else:
        lam_sq = 1.0
        raw_ids = dp_means_online(U, dates, is_shock, lam_sq=lam_sq)

    # Build per-channel activations (binary), then smooth
    channel_ids = sorted(set(raw_ids) - {0})
    chan_mat = {}
    for cid in channel_ids:
        mask = (raw_ids == cid).astype(np.int32)
        chan_mat[f"channel_{cid}"] = smooth_activation(mask, win=args.smooth_win)

    out_channels = df[["date"]].copy()
    for k, v in chan_mat.items():
        out_channels[k] = v
        
    # save channels
    args.out_channels.parent.mkdir(parents=True, exist_ok=True)
    out_channels.to_csv(args.out_channels, index=False)

    # Build ShockLedger
    rows = []
    for cid in channel_ids:
        act = (raw_ids == cid).astype(bool)
        # training occurrences for naming/semantics
        train_idx = (df_day["date"] <= train_end).to_numpy()
        active_train = (raw_ids == cid) & train_idx
        meta = summarize_channel(df_day, active_train, train_idx)

        idx = np.where(act)[0]
        first_seen = dates[idx[0]] if len(idx)>0 else None
        last_seen  = dates[idx[-1]] if len(idx)>0 else None
        centroid = np.nanmean(U[idx], axis=0).tolist() if len(idx)>0 else []

        rows.append({
            "channel_id": int(cid),
            "name": meta["name"],
            "description": meta["description"],
            "keywords": meta["keywords"],
            "macro_signature": meta["macro_signature"],
            "top_movers_pos": meta["top_movers_pos"],
            "top_movers_neg": meta["top_movers_neg"],
            "first_seen": pd.Timestamp(first_seen) if first_seen is not None else pd.NaT,
            "last_seen": pd.Timestamp(last_seen) if last_seen is not None else pd.NaT,
            "n_days": int(len(idx)),
            "n_days_train": int(active_train.sum()),
            "u_centroid": centroid,
            "version": "1.0",
        })

    ledger_df = pd.DataFrame(rows).sort_values("channel_id").reset_index(drop=True)
    
    # Save
    args.out_ledger_p.parent.mkdir(parents=True, exist_ok=True)
    ms_rows = []
    for _, r in ledger_df.iterrows():
        for item in (r["macro_signature"] or []):
            feat, val = item  # ("usd_lret_z", 0.73)
            ms_rows.append({"channel_id": r["channel_id"], "feature": feat, "delta": float(val)})
    macro_sig = pd.DataFrame(ms_rows)
    macro_sig.to_csv(args.out_macro_sig, index=False)

    flat_ledger = ledger_df.drop(columns=["macro_signature", "keywords", "top_movers_pos", "top_movers_neg", "u_centroid"])
    flat_ledger.to_csv(args.out_ledger_p, index=False)

    # Save YAML
    def pyify(x):
        if isinstance(x, pd.Timestamp) or np.issubdtype(type(x), np.datetime64):
            return pd.Timestamp(x).date().isoformat()  # "YYYY-MM-DD"
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        if isinstance(x, np.ndarray):
            return [pyify(v) for v in x.tolist()]
        if isinstance(x, dict):
            return {k: pyify(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [pyify(v) for v in x]
        return x

    rows_yaml = [pyify(r) for r in rows]

    args.out_ledger_y.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_ledger_y, "w", encoding="utf-8") as f:
        yaml.safe_dump(rows_yaml, f, sort_keys=False, allow_unicode=True)

if __name__ == "__main__":
    main()

