# io_utils.py

from __future__ import annotations
import os
from typing import Optional, Dict, Tuple, List
import pandas as pd

def _read_any(path: str, name: str) -> pd.DataFrame:
    if path is None or str(path).strip() == "":
        raise FileNotFoundError(f"[{name}] missing path in config.")
    p = os.path.abspath(str(path))
    if not os.path.exists(p):
        raise FileNotFoundError(f"[{name}] not found: {p}")
    if p.lower().endswith(".csv"):
        df = pd.read_csv(p)
    elif p.lower().endswith((".parquet", ".pq")):
        df = pd.read_parquet(p)
    else:
        raise ValueError(f"[{name}] unsupported file type: {p}")
    if df is None or len(df) == 0:
        raise ValueError(f"[{name}] empty dataframe: {p}")
    return df

def _normalize_dates(df: pd.DataFrame, explicit: Optional[str] = None) -> pd.DataFrame:
    df = df.copy()
    date_col = None
    if explicit and explicit in df.columns:
        date_col = explicit
    else:
        for cand in ("date", "Date", "DATE", "Unnamed: 0", "Unnamed:0"):
            if cand in df.columns:
                date_col = cand
                break
        if date_col is None and getattr(df.index, "name", None) in ("date","Date","DATE"):
            df = df.reset_index()
            date_col = "date"

    if date_col is None:
        raise ValueError(f"Could not find a date column in {df.columns.tolist()}")

    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    if df["date"].isna().all():
        raise ValueError("All values in 'date' failed to parse")
    return df

def _wide_or_pivot(df_or_path, name: str, value_col: Optional[str] = None) -> pd.DataFrame:
    if isinstance(df_or_path, (str, os.PathLike)):
        df = _read_any(df_or_path, name)
    else:
        df = df_or_path
        if df is None or len(df) == 0:
            raise ValueError(f"[{name}] got None/empty dataframe.")

    df = _normalize_dates(df)

    # pivot
    if "ticker" in df.columns:
        if value_col is None:
            for cand in ("r_tilde_with_alpha","ret","alpha","value","pred","r","return","r_tilde"):
                if cand in df.columns:
                    value_col = cand
                    break
        if value_col is None or value_col not in df.columns:
            raise ValueError(f"[{name}] long-format requires a value column; got {df.columns.tolist()}")
        piv = df.pivot(index="date", columns="ticker", values=value_col)
        piv = piv.sort_index().reindex(sorted(piv.columns), axis=1)
        return piv

    # ensure sorted ticker columns
    wide = df.set_index("date").sort_index()
    non_date = [c for c in wide.columns if str(c).lower() != "date"]
    if not non_date:
        raise ValueError(f"[{name}] wide-format has no ticker columns.")
    return wide.reindex(sorted(non_date), axis=1)

def _read_edges_long(path_or_df, name: str = "edges") -> pd.DataFrame:
    e = _read_any(path_or_df, name) if isinstance(path_or_df, (str, os.PathLike)) else path_or_df
    e = _normalize_dates(e)
    need = {"date","src","dst"}
    if not need.issubset(set(e.columns)):
        raise ValueError(f"[edges] need columns date, src, dst; got {e.columns.tolist()}")
    out = e[["date","src","dst"]].copy()
    out["src"] = out["src"].astype(int)
    out["dst"] = out["dst"].astype(int)
    return out.sort_values(["date","src","dst"]).reset_index(drop=True)

# betas
_DEFAULT_BETAS_COLS = [
    "hy_oas_dzbps", "baa10y_dzbps", "dgs10_dzbps", "usd_lret_z", "wti_lret_z",
    "brent_lret_z", "energy_lret_z", "vix_dz", "curve2s10s_dzbps", "epu_dz", "gpr_dz",
]

def _read_betas_multifactor(path_or_df, name: str = "betas", cols: Optional[List[str]] = None) -> pd.DataFrame:
    df = _read_any(path_or_df, name) if isinstance(path_or_df, (str, os.PathLike)) else path_or_df
    df = _normalize_dates(df)
    if "ticker" not in df.columns:
        raise ValueError(f"[{name}] needs a 'ticker' column; got {df.columns.tolist()}")

    use_cols = cols[:] if cols else _DEFAULT_BETAS_COLS[:]
    missing = [c for c in use_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] missing requested factor columns: {missing}")

    # build one wide matrix per factor, then concat on axis=1 with keys=factors
    wides = []
    keys = []
    for f in use_cols:
        piv = df.pivot(index="date", columns="ticker", values=f)
        piv = piv.sort_index().reindex(sorted(piv.columns), axis=1)
        wides.append(piv)
        keys.append(f)

    out = pd.concat(wides, axis=1, keys=keys)
    out = out.reindex(index=out.index).sort_index(axis=1)
    out = out.fillna(method="ffill").fillna(0.0)
    return out

def _intersect_align(
    mu: pd.DataFrame,
    r_sim: pd.DataFrame,
    r_real: pd.DataFrame,
    betas: Optional[pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    # tickers from simple wide frames
    base_cols = [set(mu.columns), set(r_sim.columns), set(r_real.columns)]
    common_tickers = sorted(set.intersection(*base_cols))
    if not common_tickers:
        raise ValueError("No common tickers across mu/r_sim/r_real.")

    mu    = mu.reindex(columns=common_tickers)
    r_sim = r_sim.reindex(columns=common_tickers)
    r_real= r_real.reindex(columns=common_tickers)

    # dates intersection (base three first)
    idxs = [mu.index, r_sim.index, r_real.index]
    common_dates = idxs[0]
    for idx in idxs[1:]:
        common_dates = common_dates.intersection(idx)
    if len(common_dates) == 0:
        raise ValueError("No common dates across mu/r_sim/r_real.")

    mu    = mu.reindex(common_dates).sort_index().fillna(0.0)
    r_sim = r_sim.reindex(common_dates).sort_index().fillna(0.0)
    r_real= r_real.reindex(common_dates).sort_index().fillna(0.0)

    # align betas
    if betas is not None:
        if not isinstance(betas.columns, pd.MultiIndex) or betas.columns.nlevels != 2:
            raise ValueError("[betas] must be MultiIndex (factor, ticker).")
        betas_tickers = sorted(set(betas.columns.get_level_values(1)))
        keep_tickers = sorted(set(common_tickers).intersection(betas_tickers))
        if not keep_tickers:
            raise ValueError("No common tickers between base matrices and betas.")

        betas = betas.loc[common_dates]
        betas = betas.loc[:, betas.columns.get_level_values(1).isin(keep_tickers)]
        betas = betas.fillna(method="ffill").fillna(0.0)
        common_dates2 = mu.index.intersection(betas.index)
        mu    = mu.reindex(common_dates2)
        r_sim = r_sim.reindex(common_dates2)
        r_real= r_real.reindex(common_dates2)
        betas = betas.reindex(common_dates2)

    return mu, r_sim, r_real, betas

def _slice(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if df is None:
        return df
    out = df
    if start:
        out = out[out.index >= pd.to_datetime(start)]
    if end:
        out = out[out.index <= pd.to_datetime(end)]
    return out

def _slice_edges(edges: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    e = edges
    if start:
        e = e[e["date"] >= pd.to_datetime(start)]
    if end:
        e = e[e["date"] <= pd.to_datetime(end)]
    return e.reset_index(drop=True)

def load_dataset_npz_or_csvs(cfg: Dict) -> Dict[str, pd.DataFrame]:
    data = cfg.get("data", {})
    dates = cfg.get("dates", {})

    # core matrices
    mu_train = _wide_or_pivot(data.get("pred_train_cs"), name="pred_cs")
    mu_rem = _wide_or_pivot(data.get("pred_rem_cs"), name="pred_cs")
    mu = pd.concat([mu_train, mu_rem]).sort_index()

    r_sim = _wide_or_pivot(
        data.get("r_sim"),
        name="simret",
        value_col=data.get("simret_value_col", "r_tilde_with_alpha"),
    )
    realized_path = data.get("realized")
    r_real = _wide_or_pivot(realized_path, name="realized", value_col="ret")
    is_dead_wide = None

    df_real_long = _read_any(realized_path, "realized_flags")
    df_real_long = _normalize_dates(df_real_long)
    if {"ticker", "is_dead"}.issubset(df_real_long.columns):
        is_dead_wide = df_real_long.pivot(
            index="date", columns="ticker", values="is_dead"
        )
        is_dead_wide = (
            is_dead_wide.sort_index()
            .reindex(sorted(is_dead_wide.columns), axis=1)
            .fillna(0.0)
        )

    # betas (multi-factor)
    betas = None
    if data.get("betas"):
        betas_cols = data.get("betas_cols", _DEFAULT_BETAS_COLS)
        betas = _read_betas_multifactor(data.get("betas"), name="betas", cols=betas_cols)

    # edges (node ids)
    edges_train = _read_edges_long(data.get("edges_train"), name="edges")
    edges_rem = _read_edges_long(data.get("edges_rem"), name="edges")
    edges = (
        pd.concat([edges_train, edges_rem], ignore_index=True)
        .sort_values(["date", "src", "dst"])
        .reset_index(drop=True)
    )

    # enforce common tickers & dates
    mu, r_sim, r_real, betas = _intersect_align(mu, r_sim, r_real, betas)
    is_dead = None
    if is_dead_wide is not None:
        is_dead = (
            is_dead_wide
            .reindex(index=mu.index, columns=mu.columns)
            .fillna(0.0)
        )

    # node order (from mu columns)
    node_order: List[str] = list(mu.columns)
    ticker_to_id = {t:i for i,t in enumerate(node_order)}

    # keep edges within range and on common dates
    n_nodes = len(node_order)
    edges = edges[edges["src"].between(0, n_nodes-1) & edges["dst"].between(0, n_nodes-1)].copy()
    edges = edges[edges["date"].isin(mu.index)].reset_index(drop=True)

    # splits
    tr_start = dates.get("train_start") or data.get("train_start")
    tr_end   = dates.get("train_end")   or data.get("train_end")
    val_start = dates.get("val_start")  or data.get("val_start")
    val_end   = dates.get("val_end")    or data.get("val_end")
    te_start = dates.get("test_start")  or data.get("test_start")
    te_end   = dates.get("test_end")    or data.get("test_end")

    is_dead_train = _slice(is_dead, tr_start, tr_end) if is_dead is not None else None
    is_dead_val   = _slice(is_dead, val_start, val_end) if is_dead is not None else None
    is_dead_test  = _slice(is_dead, te_start, te_end) if is_dead is not None else None

    out = {
        # unsliced
        "mu": mu, "r_sim": r_sim, "r_real": r_real, "betas": betas, "edges": edges,
        "tickers": node_order, "ticker_to_id": ticker_to_id,
        "is_dead": is_dead,

        # train
        "mu_train":     _slice(mu,    tr_start, tr_end),
        "r_sim_train":  _slice(r_sim, tr_start, tr_end),
        "r_real_train": _slice(r_real,tr_start, tr_end),
        "edges_train":  _slice_edges(edges, tr_start, tr_end),
        "is_dead_train": is_dead_train,
    }

    if val_start is not None and val_end is not None:
        out["mu_val"]      = _slice(mu,    val_start, val_end)
        out["r_sim_val"]   = _slice(r_sim, val_start, val_end)
        out["r_real_val"]  = _slice(r_real,val_start, val_end)
        out["edges_val"]   = _slice_edges(edges, val_start, val_end)
        out["is_dead_val"] = is_dead_val

    # test
    out.update({
        "mu_test":      _slice(mu,    te_start, te_end),
        "r_sim_test":   _slice(r_sim, te_start, te_end),
        "r_real_test":  _slice(r_real,te_start, te_end),
        "edges_test":   _slice_edges(edges, te_start, te_end),
        "is_dead_test": is_dead_test,
    })

    if betas is not None:
        out["betas_train"] = _slice(betas, tr_start, tr_end)
        out["betas_val"]   = _slice(betas, val_start, val_end)
        out["betas_test"]  = _slice(betas, te_start, te_end)

    return out

