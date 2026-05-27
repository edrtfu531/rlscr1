# make_mapping_from_ledger.py

from __future__ import annotations
import argparse, ast, json, math, os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import yaml

# State dims
DEFAULT_DIMS = [
    "hy_oas_dzbps",
    "dgs10_dzbps",
    "ust_2y_10y_dzbps",
    "usd_lret_z",
    "wti_lret_z",
    "vix_dz",
]


ALIAS = {
    "curve2s10s_dzbps": "ust_2y_10y_dzbps",
    "usd_dzret":        "usd_lret_z",
    "usd_lret_z":       "usd_lret_z",
    "oil_wti_lret_z":   "wti_lret_z",
    "oil_lret_z":       "wti_lret_z",
    "vix_dz":           "vix_dz",
    "hy_oas_dzbps":     "hy_oas_dzbps",
    "dgs10_dzbps":      "dgs10_dzbps",
}

def alias_feat(name: str) -> str:
    return ALIAS.get(name, name)

def _load_ledger(path: Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in [".yml", ".yaml"]:
        data = yaml.safe_load(open(p, "r", encoding="utf-8")) or []
        df = pd.json_normalize(data)
    else:
        df = pd.read_csv(p)

    if "channel_id" not in df and "id" in df:
        df = df.rename(columns={"id": "channel_id"})
    if "channel_name" not in df:
        for k in ["name", "label", "channel"]:
            if k in df:
                df = df.rename(columns={k: "channel_name"})
                break
            
    if "channel_name" not in df.columns:
        df["channel_name"] = df.get("channel_id").apply(lambda x: f"channel_{int(x)}" if pd.notna(x) else None)

    # Parse first_seen
    if "first_seen" in df.columns:
        df["first_seen"] = pd.to_datetime(df["first_seen"], errors="coerce")
    else:
        df["first_seen"] = pd.NaT

    sig_col = None
    for c in ["macro_signature", "macro.sig", "signature.macro", "macroSignature"]:
        if c in df.columns:
            sig_col = c; break

    def parse_sig(x):
        if isinstance(x, (list, dict)): return x
        if pd.isna(x): return []
        if isinstance(x, str):
            x = x.strip()
            try:
                return json.loads(x)
            except Exception:
                try:
                    return ast.literal_eval(x)
                except Exception:
                    return []
        return []


    if sig_col:
        df["macro_signature"] = df[sig_col].apply(parse_sig)
    else:
        print("[WARN] Ledger has no 'macro_signature' column. All channel mappings will be empty.")
        df["macro_signature"] = [[] for _ in range(len(df))]
    
    # Keep only necessary columns
    keep = ["channel_id", "channel_name", "first_seen", "macro_signature"]
    return df[[c for c in keep if c in df.columns]].copy()

def _panel_cov(panel_path: Path, dims: List[str], fit_end: str | None) -> np.ndarray:
    if panel_path is None:
        raise ValueError("CRITICAL: Scaling is set to 'maha', but --panel path is None.")
    
    if not Path(panel_path).exists():
        raise FileNotFoundError(f"CRITICAL: Panel file not found at: {panel_path}")
    
    df = pd.read_csv(panel_path)

    # Date filtering
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date")
        if fit_end:
            df = df.loc[df["date"] <= pd.to_datetime(fit_end)]
            
    # Alias columns
    colmap = {c: alias_feat(c) for c in df.columns}
    df = df.rename(columns=colmap)

    missing = [d for d in dims if d not in df.columns]
    if missing:
        raise ValueError(
            f"CRITICAL: Cannot compute Mahalanobis covariance. "
            f"Panel is missing columns: {missing}\n"
            f"Available: {list(df.columns)}"
        )

    X = df[dims].dropna()
    
    if len(X) < len(dims) + 10:
        raise ValueError(f"Not enough history to compute covariance. Rows={len(X)}, Dims={len(dims)}")
        
    return np.cov(X.values.T, bias=False)

def _sig_to_weights(sig: Any, dims: List[str]) -> Dict[str, float]:
    # Convert macro_signature → {dim: weight} keeping only dims.
    out = {}
    if isinstance(sig, dict):
        items = list(sig.items())
    elif isinstance(sig, list):
        items = []
        for it in sig:
            if isinstance(it, dict):
                name = it.get("feature") or it.get("name") or it.get("dim") or next(iter(it.keys()))
                val  = it.get("value")   or it.get("weight") or it.get(name)
                items.append((name, val))
            elif isinstance(it, (list, tuple)) and len(it) >= 2:
                items.append((it[0], it[1]))
    else:
        items = []

    for k, v in items:
        try:
            k2 = alias_feat(str(k))
            if k2 in dims:
                out[k2] = float(v)
        except Exception:
            continue
    return out


def _scale_weights_maha(w: np.ndarray, Sigma: np.ndarray, s0: float) -> np.ndarray:
    if not w.size:
        return w
    try:
        inv = np.linalg.pinv(Sigma)
        s = math.sqrt(float(w.T @ inv @ w))
        if s > 0 and s0 >= 0:
            return w * (s0 / s)
        return w
    except Exception:
        return w
    
def build_mapping(
    ledger_path: Path,
    out_path: Path,
    dims: List[str],
    train_end: str | None,
    scaling: str,
    panel_path: Path | None,
    fit_end: str | None,
    s0: float,
    reg_threshold: float = 0.0,
) -> Dict[str, Any]:
    # Load ledger
    df = _load_ledger(ledger_path)

    # Filter by first_seen <= train_end to ensure leak-safety
    if train_end is not None:
        cutoff = pd.to_datetime(train_end)
        
        # Check for NaT (missing dates)
        missing_dates = df["first_seen"].isna()
        if missing_dates.any():
            print(f"[WARN] {missing_dates.sum()} channels have unknown 'first_seen' dates. Excluding them to prevent leakage.")
            
        # Only include if date is known AND <= cutoff
        mask = (~df["first_seen"].isna()) & (df["first_seen"] <= cutoff)
        df = df.loc[mask].copy()

    # Prepare covariance if needed
    Sigma = None
    if scaling.lower() == "maha":
        Sigma = _panel_cov(panel_path, dims, fit_end)

    channels_map: Dict[str, Dict[str, float]] = {}
    kept = 0
    for _, row in df.iterrows():
        ch_name = row.get("channel_name") or (f"channel_{int(row['channel_id'])}" if pd.notna(row.get("channel_id")) else None)
        if not ch_name:
            continue
        sig = row.get("macro_signature", [])
        w_dict = _sig_to_weights(sig, dims)
        w_vec = np.array([w_dict.get(d, 0.0) for d in dims], dtype=float)

        # Drop channels that have no overlap with dims
        if not np.any(np.abs(w_vec) > 0):
            channels_map[ch_name] = {}
            continue

        if scaling.lower() == "maha":
            w_vec = _scale_weights_maha(w_vec, Sigma if Sigma is not None else np.eye(len(dims)), s0)
        elif scaling.lower() == "none":
            pass
        else:
            raise ValueError(f"Unknown scaling '{scaling}'. Use: none|maha.")

        if reg_threshold > 0.0:
            w_vec[np.abs(w_vec) < reg_threshold] = 0.0

        ch_weights = {d: float(v) for d, v in zip(dims, w_vec) if abs(v) > 0}
        channels_map[ch_name] = ch_weights
        kept += 1

    spec = {
        "dims": dims,
        "channels": channels_map,
        "meta": {
            "source_ledger": str(ledger_path),
            "train_end": str(train_end) if train_end else None,
            "fit_end": str(fit_end) if fit_end else None,
            "scaling": scaling,
            "s0": float(s0) if scaling.lower() == "maha" else None,
            "num_channels": int(kept),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(spec, f, sort_keys=False)
    return spec

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ledger", required=True, type=Path, help="Tier-2 ledger (YAML or csv)")
    p.add_argument("--out",    required=True, type=Path, help="Path to write mapping YAML")
    p.add_argument("--dims",   nargs="+", default=DEFAULT_DIMS, help="State dims in desired order")
    p.add_argument("--train_end", type=str, default=None, help="Include channels with first_seen <= this date (YYYY-MM-DD)")
    p.add_argument("--scaling", choices=["none","maha"], default="maha",
                     help="Weight scaling: none | per-channel max-abs=1 | Mahalanobis to s0")
    p.add_argument("--panel", type=Path, default=None, help="Panel csv for Sigma_x (needed for maha)")
    p.add_argument("--fit_end", type=str, default=None, help="Cap Sigma_x estimation to this date (YYYY-MM-DD)")
    p.add_argument("--s0", type=float, default=1.0, help="Target Mahalanobis size for scaling='maha'")
    p.add_argument(
        "--regularize_weights",
        type=float,
        default=0.0,
        help=(
            "Hard L1-style threshold τ. After scaling, set weights with |w_i| < τ to 0. "
            "τ=0 disables regularization (default)."
        ),
    )
    args = p.parse_args()

    spec = build_mapping(
        ledger_path=args.ledger,
        out_path=args.out,
        dims=[alias_feat(d) for d in args.dims],
        train_end=args.train_end,
        scaling=args.scaling,
        panel_path=args.panel,
        fit_end=args.fit_end,
        s0=args.s0,
        reg_threshold=args.regularize_weights,
    )

if __name__ == "__main__":
    main()


