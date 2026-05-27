
# run_ccm.py
from __future__ import annotations
import argparse, os, yaml, numpy as np
import pandas as pd

from pathlib import Path
from typing import Dict, Optional
from io_utils import load_dataset_npz_or_csvs
from ccm_trainer import train
from evaluation import evaluate
from graph_utils import build_laplacian_series

# Helpers to read inputs
def _read_wide_mu(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if {"date","ticker"}.issubset(df.columns):
        cand = [c for c in df.columns if c not in ("date","ticker")]
        if not cand:
            raise ValueError("pred_cs must have a value column besides date,ticker")
        val = cand[0]
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        wide = df.pivot(index="date", columns="ticker", values=val).sort_index()
    else:
        wide = df.copy()
        # If a date column exists, make it the index
        if "date" in wide.columns:
            wide["date"] = pd.to_datetime(wide["date"]).dt.tz_localize(None).dt.normalize()
            wide = wide.set_index("date")
        else:
            wide.index = pd.to_datetime(wide.index).tz_localize(None).normalize()
        wide = wide.sort_index()
    return wide

def _load_risk_scale(cfg: dict, dates_train=None, dates_val=None, dates_test=None):
    """
    Load scenario_context.csv and produce risk_scale arrays aligned to train/test dates.
    """
    sc_path = (
        (cfg.get("data") or {}).get("scenario_context")
        or str(Path("scenario_context.csv"))
    )
    risk_train = None
    risk_val = None
    risk_test = None
    
    try:
        sc = pd.read_csv(sc_path)
        # find date column
        dcol = None
        for c in sc.columns:
            if "date" in c.lower():
                dcol = c; break
        if dcol is None:
            raise ValueError("scenario_context.csv missing a date column")
        sc[dcol] = pd.to_datetime(sc[dcol]).dt.tz_localize(None).dt.normalize()
        sc = sc.set_index(dcol).sort_index()

        # pick risk column
        risk_col = None
        for c in [ "risk_scale",]:
            if c in sc.columns:
                risk_col = c; break
        if risk_col is None:
            sc["risk_scale"] = 1.0
            risk_col = "risk_scale"

        ser = sc[risk_col].astype(float).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(lower=0.0)

        def _align(idx):
            if idx is None:
                return None
            s = ser.reindex(idx).ffill().fillna(1.0)
            return s.to_numpy(dtype=float)

        risk_train = _align(dates_train)
        risk_val   = _align(dates_val)   if dates_val   is not None else None
        risk_test  = _align(dates_test)

    except Exception as e:
        # Neutral defaults
        if dates_train is not None:
            risk_train = np.ones(len(dates_train), dtype=float)
        if dates_val is not None:
            risk_val = np.ones(len(dates_val), dtype=float)
        if dates_test is not None:
            risk_test = np.ones(len(dates_test), dtype=float)

    return risk_train, risk_val, risk_test

def _prepare_pack(pack):
    """
    - Build per-day Laplacians from edges for train/test
    - Convert pandas DataFrames to NumPy arrays for the trainer
    - Provide dates_train/test used for alignment
    """
    tickers = pack["tickers"]

    # Build Laplacian series aligned to mu indices
    mu_tr = pack["mu_train"]
    mu_te = pack["mu_test"]
    mu_val = pack.get("mu_val")
    Ltr_map = build_laplacian_series(pack["edges_train"], tickers, mu_tr.index)
    Lte_map = build_laplacian_series(pack["edges_test"],  tickers, mu_te.index)
    Lval_map = build_laplacian_series(pack["edges_val"], tickers, mu_val.index)
    
    pack = dict(pack)
    pack["L_train"] = np.stack([Ltr_map[d] for d in mu_tr.index], axis=0)
    pack["L_test"]  = np.stack([Lte_map[d] for d in mu_te.index], axis=0)
    pack["L_val"] = np.stack([Lval_map[d] for d in mu_val.index], axis=0)

    pack["dates_train"] = mu_tr.index
    pack["dates_test"] = mu_te.index
    pack["dates_val"] = mu_val.index

    
    # Convert to NumPy
    for k in [
        "mu_train","r_sim_train","r_real_train",
        "mu_val","r_sim_val","r_real_val",
        "mu_test","r_sim_test","r_real_test",
        "is_dead_train","is_dead_val","is_dead_test",
    ]:
        if k in pack and hasattr(pack[k], "to_numpy"):
            pack[k] = pack[k].to_numpy()

    for k in ["betas_train","betas_val","betas_test"]:
        if k in pack and hasattr(pack[k], "to_numpy"):
            pack[k] = pack[k].to_numpy()
    return pack


# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}

    # ML path
    pack = load_dataset_npz_or_csvs(cfg)
    pack = _prepare_pack(pack)

    # scenario_context.csv integration
    risk_tr, risk_val, risk_te = _load_risk_scale(
        cfg,
        dates_train=pack.get("dates_train"),
        dates_val=pack.get("dates_val"),
        dates_test=pack.get("dates_test"),
    )
    if risk_tr is not None:
        pack["risk_scale_train"] = risk_tr
    if risk_val is not None:
        pack["risk_scale_val"] = risk_val
    if risk_te is not None:
        pack["risk_scale_test"] = risk_te

    os.makedirs(cfg["output"]["dir"], exist_ok=True)
    state = train(pack, cfg)

    evaluate(pack, state, cfg)

if __name__ == "__main__":
    main()

