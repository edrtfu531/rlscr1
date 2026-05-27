# macro_scenario_sim.py

import argparse
import json
import math
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import yaml
from pathlib import Path


def _try_read_table(path: Path) -> pd.DataFrame:
    """Read csv, with graceful fallbacks."""
    try:
        return pd.read_csv(path, parse_dates=["date"])
    except Exception:
        return pd.read_csv(path)

def _safe_to_csv(df: pd.DataFrame, out: Path) -> Path:
    df.to_csv(out)
    return out


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    df = df.sort_index()
    return df


def _alias_columns(panel: pd.DataFrame, needed: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    alias_map = {
        "ust_2y_10y_dzbps": ["curve2s10s_dzbps", "ust_2s10s_dzbps", "slope_2s10s_dzbps", "slope_2s10s"],
        "wti_lret_z": ["brent_lret_z", "oil_lret_z"],
        "dgs10_dzbps": ["ust10y_dzbps", "gs10_dzbps"],
        "hy_oas_dzbps": ["hyoas_dzbps", "hy_oas_z"],
        "vix_dz": ["vix_z", "vix_deltaz", "vix_chg_z"],
        "usd_lret_z": ["dxy_lret_z", "usdxy_lret_z"],
    }

    panel = panel.copy()
    missing = []
    for d in needed:
        if d not in panel.columns:
            found = False
            for alt in alias_map.get(d, []):
                if alt in panel.columns:
                    panel[d] = panel[alt]
                    found = True
                    break
            if not found:
                missing.append(d)
    cols = [c for c in needed if c in panel.columns]
    return panel[cols], missing


# Dataclasses
@dataclass
class ScenarioConfig:
    panel_path: Path
    mapping_yaml: Path
    out_dir: Path
    horizon: int = 10
    n_paths: int = 1000
    ridge_alpha: float = 10.0
    seed: int = 123
    start_from_date: Optional[str] = None  # first simulated day
    fit_end_date: Optional[str] = None
    allow_missing_dims: bool = False
    channels_day0: Optional[Dict[str, float]] = None
    manual_deltas_csv: Optional[Path] = None
    # Pulse kinds per dim
    pulse_kind_by_dim: Optional[Dict[str, str]] = None
    pulse_params: Optional[Dict[str, Dict[str, float]]] = None


@dataclass
class ScenarioOutputs:
    scores: Path
    meta: Path
    figures: List[Path]


# Core functions
def load_mapping(mapping_yaml: Path) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    m = yaml.safe_load(open(mapping_yaml, "r", encoding="utf-8"))
    dims = list(m["dims"])
    channels_map = {k: {kk: float(vv) for kk, vv in (v or {}).items()} for k, v in m["channels"].items()}
    return dims, channels_map


def load_panel(panel_path: Path, dims: List[str], allow_missing: bool) -> Tuple[pd.DataFrame, List[str]]:
    df = _try_read_table(panel_path)
    df = _ensure_datetime_index(df)
    df2, missing = _alias_columns(df, dims)
    if missing and not allow_missing:
        raise ValueError(f"Panel is missing required dims (use --allow_missing_dims to zero-fill): {missing}")

    for d in missing:
        df2[d] = 0.0
    df2 = df2.reindex(columns=dims).dropna()
    return df2, missing


def make_delta_schedule_from_channels(
    channels_day0: Dict[str, float],
    channels_map: Dict[str, Dict[str, float]],
    dims: List[str],
    horizon: int,
    pulse_kind_by_dim: Optional[Dict[str, str]] = None,
    pulse_params: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    
    if not channels_day0:
        raise ValueError("channels_day0 is empty; provide at least one channel: amplitude pair.")
    # Δ0
    delta0 = pd.Series(0.0, index=dims)
    for ch, amp in channels_day0.items():
        w = channels_map.get(ch, {})
        for d, wgt in w.items():
            if d in delta0.index:
                delta0[d] += amp * float(wgt)
    # Defaults: exponential half-life per dim (more realistic tails)
    default_hl = {
        "hy_oas_dzbps": 3.0,
        "vix_dz": 2.0,
        "dgs10_dzbps": 3.0,
        "ust_2y_10y_dzbps": 3.0,
        "usd_lret_z": 2.0,
        "wti_lret_z": 2.5,
    }
    if pulse_kind_by_dim is None:
        pulse_kind_by_dim = {d: "exp" for d in dims}
    if pulse_params is None:
        pulse_params = {}
    for d in dims:
        pulse_kind_by_dim.setdefault(d, "exp")
        pulse_params.setdefault(d, {}).setdefault("half_life", float(default_hl.get(d, 2.5)))

    # Build Δ_h
    H = horizon
    deltas = np.zeros((H, len(dims)))
    deltas[0, :] = delta0.values
    for h in range(1, H):
        for j, d in enumerate(dims):
            kind = pulse_kind_by_dim.get(d, "exp")
            if kind == "fast":
                deltas[h, j] = 0.0
            elif kind == "sticky":
                deltas[h, j] = 0.5 * delta0.iloc[j] if h <= 4 else 0.0
            elif kind == "exp":
                hl = float(pulse_params.get(d, {}).get("half_life", 2.5))
                lam = 0.5 ** (h / hl)
                deltas[h, j] = lam * delta0.iloc[j]
            else:
                deltas[h, j] = 0.0
    deltas_df = pd.DataFrame(deltas, columns=dims)
    deltas_df.index.name = "horizon_day"
    return deltas_df, delta0


def fit_ridge_var(panel: pd.DataFrame, alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Fit 1-lag ridge VAR with intercept via centering.
    X = panel.values[:-1, :]
    Y = panel.values[1:, :]
    x_bar = X.mean(axis=0)
    y_bar = Y.mean(axis=0)
    Xc = X - x_bar
    Yc = Y - y_bar
    D = X.shape[1]
    XtX = Xc.T @ Xc
    A_hat = np.linalg.solve(XtX + float(alpha) * np.eye(D), Xc.T @ Yc)  # D×D
    c_hat = y_bar - A_hat @ x_bar  # intercept
    E = Yc - Xc @ A_hat
    Sigma_e = np.cov(E.T, bias=False)

    if panel.shape[0] <= D + 5:
        raise ValueError(
            f"Not enough rows to estimate Sigma_x reliably: "
            f"{panel.shape[0]} rows for {D} dims."
        )
    Sigma_x = np.cov(panel.values.T, bias=False)
    return A_hat, c_hat, Sigma_e, Sigma_x


def run_scenario(cfg: ScenarioConfig) -> ScenarioOutputs:
    dims, channels_map = load_mapping(cfg.mapping_yaml)
    panel, missing = load_panel(cfg.panel_path, dims, allow_missing=cfg.allow_missing_dims)
    if panel.shape[0] < 20:
        raise ValueError("Panel too short after cleaning; need at least ~20 rows to fit VAR.")

    # Construct delta schedule
    if cfg.manual_deltas_csv:
        deltas_df = pd.read_csv(cfg.manual_deltas_csv, index_col=0)
        deltas_df = deltas_df[dims]
        delta0 = deltas_df.iloc[0, :]
    else:
        if not cfg.channels_day0:
            raise ValueError("Provide channels_day0 or manual_deltas_csv.")
        deltas_df, delta0 = make_delta_schedule_from_channels(
            cfg.channels_day0, channels_map, dims, cfg.horizon, cfg.pulse_kind_by_dim, cfg.pulse_params
        )

    # Determine start_from and fit_end, choose x0, and fit VAR on panel_fit
    if cfg.start_from_date:
        start_dt = pd.to_datetime(cfg.start_from_date)
        # x0 date = last available day before start_dt
        hist_idx = panel.index[panel.index < start_dt]
        if hist_idx.empty:
            raise ValueError("start_from_date is too early; no prior row in panel to anchor x0.")
        x0_date = hist_idx.max()
        fit_end = pd.to_datetime(cfg.fit_end_date) if cfg.fit_end_date else x0_date
        if fit_end > x0_date:
            fit_end = x0_date
        panel_fit = panel.loc[:fit_end]
        if panel_fit.shape[0] < 20:
            raise ValueError("Not enough rows in panel_fit to estimate VAR after applying fit_end/start_from constraints.")
        A_hat, c_hat, Sigma_e, Sigma_x = fit_ridge_var(panel_fit, cfg.ridge_alpha)
        x0 = panel.loc[x0_date].values.copy()
        fut_dates = pd.bdate_range(start_dt, periods=cfg.horizon)
    else:
        panel_fit = panel
        A_hat, c_hat, Sigma_e, Sigma_x = fit_ridge_var(panel_fit, cfg.ridge_alpha)
        x0 = panel.iloc[-1].values.copy()
        fut_dates = pd.bdate_range(panel.index[-1] + pd.Timedelta(days=1), periods=cfg.horizon)

    
    # Build scenario scores: dose (s_t), total/base state stress, and marginal over baseline
    D = Sigma_x.shape[0]
    try:
        inv = np.linalg.inv(Sigma_x + 1e-6 * np.eye(D))
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(Sigma_x + 1e-3 * np.eye(D))

    # dose size today
    s_dose = [float(np.sqrt(d @ inv @ d)) for d in deltas_df.values]

    x_base, x_tot = x0.copy(), x0.copy()
    s_base, s_tot = [], []
    for d in deltas_df.values:
        x_base = A_hat @ x_base
        x_tot  = A_hat @ x_tot + d
        s_base.append(float(np.sqrt(x_base @ inv @ x_base)))
        s_tot.append(float(np.sqrt(x_tot  @ inv @ x_tot )))

    scores = pd.DataFrame({
        "s_t": s_dose, 
        "s_state_total": s_tot,
        "s_state_base":  s_base,
        "s_state_marg":  np.maximum(np.array(s_tot) - np.array(s_base), 0.0)
    }, index=fut_dates)

    # Save
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = _safe_to_csv(scores, cfg.out_dir / "scenario_scores.csv")

    meta = cfg.out_dir / "whatif_meta.json"
    with open(meta, "w", encoding="utf-8") as f:
        json.dump(
            {
                **asdict(cfg),
                "panel_missing_dims_filled_as_zero": missing,
                "state_dims": dims,
                "n_history_rows_total": int(panel.shape[0]),
                "panel_fit_rows": int(panel_fit.shape[0]),
                "x0_date": str(x0_date) if cfg.start_from_date else str(panel.index[-1]),
                "effective_fit_end": str(panel_fit.index[-1]),
                "first_sim_date": str(fut_dates[0].date()),
            },
            f,
            default=str,
            indent=2,
        )

    return ScenarioOutputs(
        scores=scores_path,
        meta=meta,
        figures=[],
    )


# CLI
def parse_args() -> ScenarioConfig:
    p = argparse.ArgumentParser(description="Dynamic macro what-if simulator (ridge VAR + MC).")
    p.add_argument("--panel", required=True, type=Path)
    p.add_argument("--mapping", required=True, type=Path)
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--n_paths", type=int, default=1000)
    p.add_argument("--ridge_alpha", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--start_from", type=str, default=None, help="YYYY-MM-DD first simulated business day")
    p.add_argument("--fit_end", type=str, default=None, help="YYYY-MM-DD cap history for VAR fit (<= start_from - 1B)")
    p.add_argument("--allow_missing_dims", action="store_true", help="Fill missing dims with zeros (logged in meta)")
    p.add_argument("--channels_day0", type=str, default=None, help='JSON dict, e.g. {"channel_14":0.8}')
    p.add_argument("--manual_deltas_csv", type=Path, default=None)
    p.add_argument("--pulse_cfg", type=Path, default=None, help="YAML with pulse_kind_by_dim and pulse_params")

    args = p.parse_args()

    pulse_kind_by_dim = None
    pulse_params = None
    if args.pulse_cfg and args.pulse_cfg.exists():
        y = yaml.safe_load(open(args.pulse_cfg, "r", encoding="utf-8"))
        pulse_kind_by_dim = y.get("pulse_kind_by_dim")
        pulse_params = y.get("pulse_params")

    channels_day0 = None
    if args.channels_day0:
        channels_day0 = json.loads(args.channels_day0)

    return ScenarioConfig(
        panel_path=args.panel,
        mapping_yaml=args.mapping,
        out_dir=args.out_dir,
        horizon=args.horizon,
        n_paths=args.n_paths,
        ridge_alpha=args.ridge_alpha,
        seed=args.seed,
        start_from_date=args.start_from,
        fit_end_date=args.fit_end,
        allow_missing_dims=bool(args.allow_missing_dims),
        channels_day0=channels_day0,
        manual_deltas_csv=args.manual_deltas_csv,
        pulse_kind_by_dim=pulse_kind_by_dim,
        pulse_params=pulse_params,
    )


if __name__ == "__main__":
    cfg = parse_args()
    outs = run_scenario(cfg)
    

