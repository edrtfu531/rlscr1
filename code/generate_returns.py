# generate_returns.py
from __future__ import annotations

import os, json, argparse
from pathlib import Path
import yaml
from typing import List
import numpy as np
import pandas as pd


def load_universe_from_json(json_path: str, run_id: int):
    # Load the ticker universe
    json_path = Path(json_path)
    if not json_path.is_file():
        raise FileNotFoundError(f"Universe JSON not found: {json_path}")

    with open(json_path, "r") as f:
        universes = json.load(f)

    matches = [u for u in universes if int(u["run_id"]) == int(run_id)]
    if not matches:
        raise ValueError(f"run_id {run_id} not found in {json_path}")

    u = matches[0]
    tickers = [str(t).upper() for t in u["tickers"]]

    meta = {
        "run_id": int(u["run_id"]),
        "subset_type": u.get("subset_type", "unknown"),
        "n_tickers": len(tickers),
        "json_path": str(json_path),
    }
    return tickers, meta


def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Empty config: {path}")
    return cfg


def load_panel(market_dir: str, tickers: List[str], start=None, end=None) -> pd.DataFrame:
    frames = {}
    all_dates = set()
    # First pass to collect dates
    for t in tickers:
        fp = os.path.join(market_dir, f"{t}.csv")
        if os.path.exists(fp):
            d = pd.read_csv(fp, usecols=["date"])["date"]
            all_dates.update(pd.to_datetime(d))
    master_idx = pd.DatetimeIndex(sorted(list(all_dates)))
    if start is not None:
        master_idx = master_idx[master_idx >= start]
    if end is not None:
        master_idx = master_idx[master_idx <= end]

    # Process each ticker
    for t in tickers:
        fp = os.path.join(market_dir, f"{t}.csv")

        if not os.path.exists(fp):
            df = pd.DataFrame(index=master_idx, columns=["open","high","low","close","volume"])
            df[:] = 0.0 
            df["is_dead"] = 1.0
        else:
            df = pd.read_csv(fp, parse_dates=["date"]).sort_values("date")

            # Clean numeric columns
            for c in ["open","high","low","close","volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            df = df.set_index("date").sort_index()

            df = df.reindex(master_idx)

            df["is_dead"] = df["close"].isna()
            df["is_dead"] = df["is_dead"].astype("float32") # 1.0 = Dead, 0.0 = Alive

            for col in ["open", "high", "low", "close"]:
                df[col] = df[col].ffill()
                df[col] = df[col].fillna(0.0)

            # Volume: Fill with 0
            df["volume"] = df["volume"].fillna(0.0)

        frames[t] = df[["open","high","low","close","volume", "is_dead"]]

    # Build panel
    panel = pd.DataFrame(index=master_idx)

    for t, df in frames.items():
        panel[f"{t}_open"]   = df["open"].astype("float32")
        panel[f"{t}_high"]   = df["high"].astype("float32")
        panel[f"{t}_low"]    = df["low"].astype("float32")
        panel[f"{t}_close"]  = df["close"].astype("float32")
        panel[f"{t}_volume"] = df["volume"].astype("float32")
        panel[f"{t}_is_dead"] = df["is_dead"].astype("float32")

    # Calculate Returns
    for t in tickers:
        curr = panel[f"{t}_close"]
        prev = curr.shift(1)

        r = (curr - prev) / prev
        r = r.replace([np.inf, -np.inf], np.nan)
        r = r.fillna(0.0)

        panel[f"{t}_return"] = r.astype("float32")

    panel = panel.iloc[1:]  
    return panel.reset_index().rename(columns={"index": "date"})



def panel_returns_to_long(panel: pd.DataFrame, out_path: str = "returns.csv") -> pd.DataFrame:
    if "date" not in panel.columns:
        raise ValueError("panel must contain a 'date' column")

    ret_cols = [c for c in panel.columns if c.endswith("_return")]
    if not ret_cols:
        raise ValueError("No '*_return' columns found in panel")

    long_frames = []
    for c in ret_cols:
        # Extract ticker
        ticker = c.rsplit("_", 1)[0]
        dead_col = f"{ticker}_is_dead"

        cols = ["date", c]
        has_dead = dead_col in panel.columns
        if has_dead:
            cols.append(dead_col)

        tmp = panel[cols].copy()

        rename_map = {c: "ret"}
        if has_dead:
            rename_map[dead_col] = "is_dead"

        tmp = tmp.rename(columns=rename_map)

        if "is_dead" not in tmp.columns:
            tmp["is_dead"] = 0.0

        tmp["ticker"] = ticker
        long_frames.append(tmp)

    # Concatenate
    long_ret = pd.concat(long_frames, ignore_index=True)
    long_ret["date"] = pd.to_datetime(long_ret["date"]).dt.strftime("%Y-%m-%d")
    long_ret = long_ret[["date", "ticker", "ret", "is_dead"]]
    long_ret = long_ret.sort_values(["date", "ticker"]).reset_index(drop=True)
    long_ret.to_csv(out_path, index=False)
    return long_ret


def split_train_test(panel, train_end, test_end):
    tr = panel[panel["date"] <= pd.Timestamp(train_end)].copy()
    te = panel[(panel["date"] > pd.Timestamp(train_end)) & (panel["date"] <= pd.Timestamp(test_end))].copy()
    return tr.reset_index(drop=True), te.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", default=None)
    parser.add_argument(
        "--universe_json",
        type=str,
        default=None,
        help="Path to experiment_universes_*.json (tickers per run).",
    )
    parser.add_argument(
        "--run_id",
        type=int,
        default=None,
        help="Which run_id inside universe_json to use for this experiment.",
    )

    args = parser.parse_args()
    cfg = load_cfg(args.config)
    tickers, universe_meta = load_universe_from_json(
        args.universe_json, args.run_id
    )

    market_dir = cfg["data"]["market_dir"]
    base_out_dir = Path(cfg["output"]["dir"])

    run_id = universe_meta.get("run_id")
    subset_type = universe_meta.get("subset_type", "manual")

    if run_id is not None:
        universe_tag = f"run_{int(run_id):02d}_{subset_type}"
        out_dir = base_out_dir / universe_tag
    else:
        universe_tag = "default_universe"
        out_dir = base_out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    start = cfg["data"]["start_date"]
    test_end = cfg["data"]["test_end"]

    panel = load_panel(market_dir, tickers, start=start, end=test_end)
    returns_path = out_dir / f"returns_{universe_tag}.csv"
    panel_returns_to_long(panel, out_path=returns_path)


if __name__ == "__main__":
    main()
