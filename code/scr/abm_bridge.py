# abm_bridge.py


import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yaml
from macro_scenario_sim import ScenarioConfig, run_scenario

# IO helpers
def _try_read_table(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, parse_dates=["date"])
    except Exception:
        return pd.read_csv(path)

def load_mapping(mapping_path: Path) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    m = yaml.safe_load(open(mapping_path, "r", encoding="utf-8"))
    dims = list(m["dims"])
    channels_map = {k: {kk: float(vv) for kk, vv in (v or {}).items()} for k, v in m["channels"].items() if v}
    return dims, channels_map

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

def sigma_x_from_panel(panel_path, dims, allow_missing, fit_end=None):
    df = _try_read_table(panel_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]); df = df.set_index("date")
    df = df.sort_index()
    df2, missing = _alias_columns(df, dims)
    for d in missing: df2[d] = 0.0
    if fit_end:
        df2 = df2.loc[:pd.to_datetime(fit_end)]
    df2 = df2.reindex(columns=dims).dropna()
    return np.cov(df2.values.T, bias=False) if len(df2) > 5 else np.eye(len(dims))

def first_bdays_between(start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    """First business day of each month in [start, end]."""
    dates = []
    cur = pd.Timestamp(start.year, start.month, 1)
    if cur < start:
        cur = (cur + pd.offsets.MonthBegin(1))
    while cur <= end:
        fb = pd.bdate_range(cur, cur, freq="B")
        if len(fb) == 0:
            fb = pd.bdate_range(cur, periods=1)
        dates.append(fb[0])
        cur = cur + pd.offsets.MonthBegin(1)
    return dates


def weekly_injections_between(start, end, weekday, every_n_weeks=1):
    all_bdays = pd.bdate_range(start, end)
    hits = [d for d in all_bdays if d.weekday() == weekday]
    if every_n_weeks > 1:
        hits = hits[::every_n_weeks]
    return hits


def load_calendar(calendar_path: Path) -> list:
    items = []
    p = Path(calendar_path)
    if p.suffix.lower() in [".yaml", ".yml"]:
        data = yaml.safe_load(open(p, "r", encoding="utf-8")) or []
        for it in data:
            d = pd.to_datetime(it.get("date"))
            ch = {k: float(v) for k, v in (it.get("channels") or {}).items()}
            s = it.get("score_target", None)
            items.append({"date": d, "channels": ch, "score_target": float(s) if s is not None else None})
    else:
        df = pd.read_csv(p)
        if "date" not in df.columns or "channel" not in df.columns or "amplitude" not in df.columns:
            raise ValueError("CSV calendar must have: date, channel, amplitude[, score_target]")
        df["date"] = pd.to_datetime(df["date"])
        s_map = df.groupby("date")["score_target"].first() if "score_target" in df.columns else {}
        for d, grp in df.groupby("date"):
            ch = {str(r["channel"]): float(r["amplitude"]) for _, r in grp.iterrows()}
            s = s_map[d] if isinstance(s_map, pd.Series) and d in s_map else None
            items.append({"date": d, "channels": ch, "score_target": float(s) if s is not None else None})
    return sorted(items, key=lambda r: r["date"])


# Scenario construction (Δ and pulses)
def build_W(dims: List[str], channels_map: Dict[str, Dict[str, float]]) -> Tuple[np.ndarray, List[str]]:
    ch_names = sorted(channels_map.keys())
    D, C = len(dims), len(ch_names)
    W = np.zeros((D, C), dtype=float)
    for j, ch in enumerate(ch_names):
        for i, d in enumerate(dims):
            W[i, j] = float(channels_map[ch].get(d, 0.0))
    keep = np.where(np.abs(W).sum(axis=0) > 0)[0]
    W = W[:, keep]
    ch_names = [ch_names[j] for j in keep]
    return W, ch_names

def build_delta0_from_channels(channels_day0: Dict[str, float],
                               channels_map: Dict[str, Dict[str, float]],
                               dims: List[str]) -> np.ndarray:
    delta0 = np.zeros(len(dims), dtype=float)
    for ch, amp in (channels_day0 or {}).items():
        for i, d in enumerate(dims):
            w = channels_map.get(ch, {}).get(d, 0.0)
            delta0[i] += float(amp) * float(w)
    return delta0

def apply_pulse_to_schedule(schedule: pd.DataFrame,
                            inj_idx: int,
                            delta0_vec: np.ndarray,
                            dims: List[str],
                            pulse_kind_by_dim: Dict[str, str],
                            pulse_params: Dict[str, dict]):
    # Add Δ at inject day and add tails per pulse rules.
    H, _ = schedule.shape
    schedule.iloc[inj_idx, :] += delta0_vec
    for j, d in enumerate(dims):
        kind = pulse_kind_by_dim.get(d, "exp")
        if kind == "fast":
            continue
        elif kind == "sticky":
            for k in range(1, 5):
                h = inj_idx + k
                if h < H:
                    schedule.iat[h, j] += 0.5 * delta0_vec[j]
        elif kind == "exp":
            hl = float(pulse_params.get(d, {}).get("half_life", 2.5))
            for k in range(1, 30):
                h = inj_idx + k
                if h < H:
                    lam = 0.5 ** (k / hl)
                    schedule.iat[h, j] += lam * delta0_vec[j]

# Objectives (ranking)
def load_objectives_yaml(path: Optional[Path]) -> Dict[str, Dict[str, float]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    y = yaml.safe_load(open(p, "r", encoding="utf-8")) or {}
    return {k: {kk: float(vv) for kk, vv in (v or {}).items()} for k, v in y.items()}


def objective_weights(dims: List[str], name: str, obj_table: Optional[Dict[str, Dict[str, float]]] = None) -> np.ndarray:
    # Weights for ranking.
    w = np.zeros(len(dims))
    idx = {d: i for i, d in enumerate(dims)}
    tables = obj_table or {}
    if not tables:
        tables = {
            "risk_off": {"hy_oas_dzbps": +1.0, "vix_dz": +1.0, "usd_lret_z": +0.2, "wti_lret_z": -0.2, "ust_2y_10y_dzbps": -0.2},
            "risk_on":  {"hy_oas_dzbps": -1.0, "vix_dz": -1.0, "usd_lret_z": -0.2, "wti_lret_z": +0.3, "ust_2y_10y_dzbps": +0.2},
            "hawkish":  {"dgs10_dzbps": +0.5, "ust_2y_10y_dzbps": -0.3, "usd_lret_z": +0.3, "hy_oas_dzbps": +0.3, "vix_dz": +0.4},
            "dovish":   {"dgs10_dzbps": -0.5, "ust_2y_10y_dzbps": +0.3, "usd_lret_z": -0.3, "hy_oas_dzbps": -0.2, "vix_dz": -0.3},
        }
    for k, v in tables.get(name, {}).items():
        if k in idx:
            w[idx[k]] = v
    return w

# State-aware ranking (auto_select)
def fit_ridge_var_from_panel(panel_path: Path,
                             dims: List[str],
                             fit_end: Optional[str],
                             ridge_alpha: float = 10.0) -> Tuple[np.ndarray, pd.DataFrame]:
    # Fit a 1-lag ridge VAR for ranking (history up to fit_end).
    df = _try_read_table(panel_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]); df = df.set_index("date")
    df = df.sort_index()
    df2, _ = _alias_columns(df, dims)
    for d in dims:
        if d not in df2.columns:
            df2[d] = 0.0
    if fit_end:
        df2 = df2.loc[:pd.to_datetime(fit_end)]
    df2 = df2.reindex(columns=dims).dropna()
    if df2.shape[0] < 20:
        raise ValueError("Not enough rows to fit VAR for auto_select.")
    X = df2.values[:-1, :]
    Y = df2.values[1:, :]
    D = X.shape[1]

    # center to emulate intercept
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    XtX = Xc.T @ Xc
    A = np.linalg.solve(XtX + float(ridge_alpha) * np.eye(D), Xc.T @ Yc)
    return A, df2

def make_pulse_sequence(delta0: np.ndarray,
                        dims: List[str],
                        H: int,
                        pulse_kind_by_dim: Dict[str, str],
                        pulse_params: Dict[str, dict]) -> np.ndarray:
    D = len(dims)
    seq = np.zeros((H, D))
    seq[0, :] = delta0
    for h in range(1, H):
        for j, d in enumerate(dims):
            kind = pulse_kind_by_dim.get(d, "exp")
            if kind == "fast":
                seq[h, j] = 0.0
            elif kind == "sticky":
                seq[h, j] = 0.5 * delta0[j] if h <= 4 else 0.0
            elif kind == "exp":
                hl = float(pulse_params.get(d, {}).get("half_life", 2.5))
                seq[h, j] = (0.5 ** (h / hl)) * delta0[j]
            else:
                seq[h, j] = 0.0
    return seq


def channel_impact_score(A: np.ndarray,
                         delta0: np.ndarray,
                         dims: List[str],
                         H: int,
                         w: np.ndarray,
                         x_init: Optional[np.ndarray] = None,
                         pulse_kind_by_dim: Optional[Dict[str, str]] = None,
                         pulse_params: Optional[Dict[str, dict]] = None) -> float:
    # Response score over H days from x_init.
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    delta0 = np.nan_to_num(delta0, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.zeros(len(dims)) if x_init is None else np.nan_to_num(x_init, nan=0.0, posinf=0.0, neginf=0.0)

    seq = make_pulse_sequence(delta0, dims, H, pulse_kind_by_dim or {}, pulse_params or {})
    seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)

    score = 0.0
    for h in range(H):
        x = A @ x + seq[h, :]
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        score += float(np.dot(w, x))
    return float(score)


# Dynamic pulses
def derive_dynamic_pulses(x_init: Optional[np.ndarray],
                          dims: List[str],
                          base_kinds: Dict[str, str],
                          base_params: Dict[str, dict],
                          cfg: Optional[Path]) -> Tuple[Dict[str, str], Dict[str, dict]]:
    kinds = dict(base_kinds)
    params = {k: dict(v) for k, v in (base_params or {}).items()} if base_params else {}
    rules = None
    if cfg and Path(cfg).exists():
        y = yaml.safe_load(open(cfg, "r", encoding="utf-8")) or {}
        rules = y.get("rules", None)
    idx = {d: i for i, d in enumerate(dims)}
    def val(d): return float(x_init[idx[d]]) if (x_init is not None and d in idx) else 0.0

    # Built-ins if no rules supplied: extend tails during stress
    if not rules:
        if abs(val("vix_dz")) > 1.0 or abs(val("hy_oas_dzbps")) > 1.0:
            kinds["vix_dz"] = "exp";    params.setdefault("vix_dz", {})["half_life"] = 3.0
            kinds["hy_oas_dzbps"] = "exp"; params.setdefault("hy_oas_dzbps", {})["half_life"] = 3.0
        if abs(val("usd_lret_z")) > 0.7 or abs(val("wti_lret_z")) > 0.7:
            kinds["usd_lret_z"] = "exp"; params.setdefault("usd_lret_z", {})["half_life"] = 2.0
            kinds["wti_lret_z"] = "exp"; params.setdefault("wti_lret_z", {})["half_life"] = 2.0
        if abs(val("dgs10_dzbps")) > 0.05 or abs(val("ust_2y_10y_dzbps")) > 0.05:
            kinds["dgs10_dzbps"] = "exp";     params.setdefault("dgs10_dzbps", {})["half_life"] = 3.0
            kinds["ust_2y_10y_dzbps"] = "exp"; params.setdefault("ust_2y_10y_dzbps", {})["half_life"] = 3.0
        return kinds, params

    # Rule-driven YAML
    for rule in rules:
        cond = all(abs(val(dim)) >= float(th) for dim, th in (rule.get("if") or {}).items())
        if cond:
            for d, spec in (rule.get("set") or {}).items():
                kinds[d] = spec.get("kind", kinds.get(d, "none"))
                if "half_life" in spec:
                    params.setdefault(d, {})["half_life"] = float(spec["half_life"])
    return kinds, params


# Sizing controls
def regime_severity_target(x_init: Optional[np.ndarray],
                           Sigma_x: np.ndarray,
                           base: float, beta: float,
                           smin: float, smax: float,
                           fallback: Optional[float]) -> Optional[float]:
    if x_init is None:
        return fallback
    D = Sigma_x.shape[0]
    try:
        inv = np.linalg.inv(Sigma_x + 1e-6 * np.eye(D))
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(Sigma_x + 1e-3 * np.eye(D))
    r = float(np.sqrt(x_init.T @ inv @ x_init))  # regime index
    target = base + beta * r
    return max(smin, min(smax, target))


def scale_delta0_to_score(delta0_vec: np.ndarray,
                          Sigma_x: np.ndarray,
                          target_s: Optional[float]) -> np.ndarray:
    if target_s is None:
        return delta0_vec
    D = Sigma_x.shape[0]
    try:
        inv = np.linalg.inv(Sigma_x + 1e-6 * np.eye(D))
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(Sigma_x + 1e-3 * np.eye(D))
    s0 = float(np.sqrt(delta0_vec.T @ inv @ delta0_vec))
    if s0 <= 0:
        return delta0_vec
    return delta0_vec * (target_s / s0)


# CLI
def parse_channels_arg(s: str) -> Dict[str, float]:
    out = {}
    if not s:
        return out
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" not in tok:
            raise ValueError(f"Bad channel token: {tok} (expected name:amp)")
        name, amp = tok.split(":", 1)
        out[name.strip()] = float(amp)
    return out


def main():
    ap = argparse.ArgumentParser(description="ABM bridge for macro scenario simulator (state-aware, no hardcoding).")
    ap.add_argument("--panel", required=True, type=Path)
    ap.add_argument("--mapping", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)

    # Simulation backbone
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--n_paths", type=int, default=1000)
    ap.add_argument("--ridge_alpha", type=float, default=10.0)  # passed to macro_scenario_sim
    ap.add_argument("--seed", type=int, default=123)

    # Window
    ap.add_argument("--start_from", type=str, default=None, help="YYYY-MM-DD first simulated business day")
    ap.add_argument("--fit_end", type=str, default=None, help="YYYY-MM-DD VAR fit cap (<= start_from - 1B)")
    ap.add_argument("--end_date", type=str, default=None, help="YYYY-MM-DD inclusive end date; overrides --horizon")

    # Selection modes
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--calendar", type=Path, help="YAML/CSV dated injections; may include per-date score_target")
    g.add_argument("--channels", type=str, help="Explicit once-only: 'channel_14:+0.8,channel_6:+0.5'")
    g.add_argument("--auto_select", action="store_true", help="Pick top channels per date from current conditions")

    # Cadence
    ap.add_argument("--repeat", type=str, default=None, choices=["monthly", "weekly"],
                    help="Monthly = first business day each month; Weekly = chosen weekday")
    ap.add_argument("--weekly_day", type=str, default=None, help="mon|tue|wed|thu|fri; default = weekday of --start_from")
    ap.add_argument("--every_n_weeks", type=int, default=1)

    # Auto-select controls
    ap.add_argument("--select_objectives", type=str, default="risk_off",
                    help="comma list of objectives to consider: risk_off,risk_on,hawkish,dovish, or YAML names")
    ap.add_argument("--select_top_k", type=int, default=3)
    ap.add_argument("--select_horizon_days", type=int, default=5)
    ap.add_argument("--select_var_alpha", type=float, default=10.0, help="ridge alpha for ranking VAR fit")
    ap.add_argument("--objectives_yaml", type=Path, default=None, help="YAML mapping of objective → {dim: weight}")

    # Control
    ap.add_argument("--scale_mode", type=str, default="constant", choices=["constant", "auto"],
                    help="constant: use --scale_to_score; auto: base + beta * regime(x_init)")
    ap.add_argument("--scale_to_score", type=float, default=None, help="Global s0 target (constant mode)")
    ap.add_argument("--scale_base", type=float, default=0.8, help="auto mode base severity")
    ap.add_argument("--scale_beta", type=float, default=0.2, help="auto mode slope vs regime")
    ap.add_argument("--scale_min", type=float, default=0.4)
    ap.add_argument("--scale_max", type=float, default=1.6)

    
    # Risk-index and gate
    ap.add_argument("--append_index_csv", type=Path, default=None,
                    help="If set, append a row with daily forward risk metrics (peak/mean) to this CSV.")
    ap.add_argument("--index_metric", type=str, default="peak", choices=["peak", "mean"],
                    help="Which metric to use for gating when computing risk_scale (default: peak).")
    ap.add_argument("--gate_alpha", type=float, default=0.5, help="Gating aggressiveness α in risk_scale = clip(1 - α * (val/q), floor, 1).")
    ap.add_argument("--gate_floor", type=float, default=0.6, help="Minimum allowed risk scale.")
    ap.add_argument("--gate_quant", type=float, default=0.95, help="Quantile q for gating baseline (default: 0.95).")
    ap.add_argument("--gate_window", type=int, default=252, help="Lookback window for quantile (business days).")
    
    # Pulses
    ap.add_argument("--pulse_cfg", type=Path, default=None, help="YAML with pulse_kind_by_dim / pulse_params")
    ap.add_argument("--dynamic_pulses", action="store_true",
                    help="Adapt pulse kinds/half-lives per injection date from current conditions (--pulse_dyn_cfg for rules)")
    ap.add_argument("--pulse_dyn_cfg", type=Path, default=None, help="YAML rules for dynamic pulses")
    ap.add_argument("--allow_missing_dims", action="store_true", help="Fill missing dims with zeros")
    args = ap.parse_args()

    # Load mapping & covariance
    dims, channels_map = load_mapping(args.mapping)
    Sigma_x = sigma_x_from_panel(args.panel, dims,
                                 allow_missing=bool(args.allow_missing_dims),
                                 fit_end=args.fit_end)

    # Default pulses (exponential half-lives per dim)
    pulse_kind_by_dim = {d: "exp" for d in dims}
    pulse_params = {
        "hy_oas_dzbps": {"half_life": 3.0},
        "vix_dz": {"half_life": 2.0},
        "dgs10_dzbps": {"half_life": 3.0},
        "ust_2y_10y_dzbps": {"half_life": 3.0},
        "usd_lret_z": {"half_life": 2.0},
        "wti_lret_z": {"half_life": 2.5},
    }
    
    if args.pulse_cfg and args.pulse_cfg.exists():
        y = yaml.safe_load(open(args.pulse_cfg, "r", encoding="utf-8"))
        pulse_kind_by_dim = y.get("pulse_kind_by_dim", pulse_kind_by_dim)
        user_params = y.get("pulse_params", {})
        for k, v in user_params.items():
            pulse_params.setdefault(k, {}).update(v)

    # Build future index & horizon
    if args.end_date and args.start_from:
        start_dt = pd.to_datetime(args.start_from)
        end_dt = pd.to_datetime(args.end_date)
        fut_idx = pd.bdate_range(start_dt, end_dt)
        horizon = len(fut_idx)
    else:
        if args.horizon is None:
            args.horizon = 10
        horizon = args.horizon
        fut_idx = pd.bdate_range(pd.to_datetime(args.start_from), periods=horizon) if args.start_from else pd.bdate_range(pd.Timestamp.today(), periods=horizon)

    # Δ schedule we will compile
    schedule = pd.DataFrame(0.0, index=pd.Index(range(horizon), name="horizon_day"), columns=dims)
    selections_log: List[dict] = []

    # Full panel for x_init 
    panel_full = _try_read_table(args.panel)
    if "date" in panel_full.columns:
        panel_full["date"] = pd.to_datetime(panel_full["date"]); panel_full = panel_full.set_index("date")
    panel_full = panel_full.sort_index()
    panel_full, _ = _alias_columns(panel_full, dims)
    for d in dims:
        if d not in panel_full.columns:
            panel_full[d] = 0.0
    panel_full = panel_full.reindex(columns=dims)

    # Objectives
    objectives_table = load_objectives_yaml(args.objectives_yaml)

    def target_severity(x_init: Optional[np.ndarray]) -> Optional[float]:
        if args.scale_mode == "constant":
            return args.scale_to_score
        return regime_severity_target(x_init, Sigma_x, args.scale_base, args.scale_beta, args.scale_min, args.scale_max, args.scale_to_score)

    # Mode 1: Calendar
    if args.calendar:
        if not args.start_from or (not args.end_date and not args.horizon):
            raise ValueError("--calendar requires --start_from and either --end_date or --horizon")
        cal = load_calendar(args.calendar)
        date_index = pd.bdate_range(pd.to_datetime(args.start_from),
                                    pd.to_datetime(args.end_date)) if args.end_date else fut_idx

        for item in cal:
            inj_date = item["date"]
            if inj_date < date_index[0] or inj_date > date_index[-1]:
                continue
            arr = (date_index == inj_date).nonzero()[0]
            if len(arr) == 0:
                inj_date2 = pd.bdate_range(inj_date, periods=1)[0]
                arr = (date_index == inj_date2).nonzero()[0]
                if len(arr) == 0:
                    continue
                pos = int(arr[0])
            else:
                pos = int(arr[0])

            # Current state x_init = last day before injection
            prev = pd.bdate_range(end=inj_date, periods=2)[0]
            x_init = panel_full.loc[prev, dims].values.astype(float) if prev in panel_full.index else None

            # Dynamic pulses per date
            kinds, params = (pulse_kind_by_dim, pulse_params)
            if args.dynamic_pulses:
                kinds, params = derive_dynamic_pulses(x_init, dims, pulse_kind_by_dim, pulse_params, args.pulse_dyn_cfg)

            # Build Δ0 from channels and scale
            delta0 = build_delta0_from_channels(item["channels"], channels_map, dims)
            s_target = item.get("score_target", None)
            if s_target is None:
                s_target = target_severity(x_init)
            delta0 = scale_delta0_to_score(delta0, Sigma_x, s_target)

            apply_pulse_to_schedule(schedule, pos, delta0, dims, kinds, params)
            selections_log.append({
                "date": str(inj_date.date()),
                "mode": "calendar",
                "channels": item["channels"],
                "score_target": s_target
            })

    # Mode 2: Explicit channels
    elif args.channels:
        channels_day0 = parse_channels_arg(args.channels)
        # Determine all injection dates
        if args.repeat == "monthly":
            inj_dates = first_bdays_between(fut_idx[0], fut_idx[-1])
        elif args.repeat == "weekly":
            wd_map = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4}
            if args.weekly_day:
                key = args.weekly_day.strip().lower()[:3]
                if key not in wd_map:
                    raise ValueError("--weekly_day must be mon|tue|wed|thu|fri")
                target_wd = wd_map[key]
            else:
                target_wd = fut_idx[0].weekday();  target_wd = min(target_wd, 4)
            inj_dates = weekly_injections_between(fut_idx[0], fut_idx[-1], target_wd, args.every_n_weeks)
        else:
            inj_dates = [fut_idx[0]]

        for d in inj_dates:
            pos = int(np.where(fut_idx == d)[0][0])
            prev = pd.bdate_range(end=d, periods=2)[0]
            row = panel_full.loc[prev, dims] if prev in panel_full.index else None
            x_init = row.fillna(0.0).values.astype(float) if row is not None else None
            kinds, params = (pulse_kind_by_dim, pulse_params)
            if args.dynamic_pulses:
                kinds, params = derive_dynamic_pulses(x_init, dims, pulse_kind_by_dim, pulse_params, args.pulse_dyn_cfg)
            delta0 = build_delta0_from_channels(channels_day0, channels_map, dims)
            s_target = target_severity(x_init)
            delta0 = scale_delta0_to_score(delta0, Sigma_x, s_target)
            apply_pulse_to_schedule(schedule, pos, delta0, dims, kinds, params)
            selections_log.append({
                "date": str(d.date()),
                "mode": "channels",
                "channels": channels_day0,
                "score_target": s_target
            })

    # Mode 3: Auto-select (state-aware, per date)
    else:
        if not args.start_from or (not args.end_date and not args.horizon):
            raise ValueError("--auto_select requires --start_from and either --end_date or --horizon")

        # Fit VAR (ranking only) up to fit_end
        A, df_fit = fit_ridge_var_from_panel(args.panel, dims, args.fit_end, ridge_alpha=args.select_var_alpha)
        W, ch_names = build_W(dims, channels_map)
        delta0_by_ch = {name: W[:, j].copy() for j, name in enumerate(ch_names)}

        # Choose injection dates
        if args.repeat == "monthly":
            inj_dates = first_bdays_between(fut_idx[0], fut_idx[-1])
        elif args.repeat == "weekly":
            wd_map = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4}
            if args.weekly_day:
                key = args.weekly_day.strip().lower()[:3]
                if key not in wd_map:
                    raise ValueError("--weekly_day must be mon|tue|wed|thu|fri")
                target_wd = wd_map[key]
            else:
                target_wd = fut_idx[0].weekday();  target_wd = min(target_wd, 4)
            inj_dates = weekly_injections_between(fut_idx[0], fut_idx[-1], target_wd, args.every_n_weeks)
        else:
            inj_dates = list(fut_idx)

        # Prepare objectives
        obj_names = [s.strip() for s in (args.select_objectives or "risk_off").split(",") if s.strip()]
        w_list = [(name, objective_weights(dims, name, objectives_table)) for name in obj_names]

        for d in inj_dates:
            pos = int(np.where(fut_idx == d)[0][0])
            Hrem = max(1, min(args.select_horizon_days, horizon - pos))

            # Current state x_init = last day before date from full panel
            prev = pd.bdate_range(end=d, periods=2)[0]
            x_init = panel_full.loc[prev, dims].values.astype(float) if prev in panel_full.index else None

            # Dynamic pulses
            local_kinds, local_params = (pulse_kind_by_dim, pulse_params)
            if args.dynamic_pulses:
                local_kinds, local_params = derive_dynamic_pulses(x_init, dims, pulse_kind_by_dim, pulse_params, args.pulse_dyn_cfg)

            # Rank channels by best objective score
            scored = []
            for name, v in delta0_by_ch.items():
                best = -1e18
                for _, w in w_list:
                    s = channel_impact_score(A, v, dims, Hrem, w, x_init=x_init,
                                             pulse_kind_by_dim=local_kinds, pulse_params=local_params)
                    if s > best:
                        best = s
                scored.append((best, name))
            scored.sort(reverse=True, key=lambda t: t[0])
            chosen = scored[:max(1, args.select_top_k)]

            # Combine chosen Δ0 equally, then size
            combo = np.zeros(len(dims))
            for _, name in chosen:
                combo += delta0_by_ch[name]
            s_target = target_severity(x_init)
            combo = scale_delta0_to_score(combo, Sigma_x, s_target)

            apply_pulse_to_schedule(schedule, pos, combo, dims, local_kinds, local_params)
            selections_log.append({
                "date": str(d.date()),
                "mode": "auto_select",
                "chosen": [{"channel": n, "rank_score": float(s)} for s, n in chosen],
                "score_target": s_target,
                "objectives": obj_names
            })

    # Write manual schedule and run simulator
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manual_csv = args.out_dir / "manual_deltas.csv"
    schedule.to_csv(manual_csv)

    cfg = ScenarioConfig(
        panel_path=args.panel,
        mapping_yaml=args.mapping,
        out_dir=args.out_dir,
        horizon=horizon,
        n_paths=args.n_paths,
        ridge_alpha=args.ridge_alpha,
        seed=args.seed,
        start_from_date=args.start_from,
        fit_end_date=args.fit_end,
        allow_missing_dims=bool(args.allow_missing_dims),
        channels_day0=None, 
        manual_deltas_csv=manual_csv,
        pulse_kind_by_dim=pulse_kind_by_dim,
        pulse_params=pulse_params,
    )
    outs = run_scenario(cfg)

    # Log selections
    try:
        if selections_log:
            with open(args.out_dir / "selections_log.json", "w", encoding="utf-8") as f:
                json.dump(selections_log, f, indent=2)
    except Exception:
        pass

    
    try:
        scores_path = None
        if isinstance(outs, dict):
            scores_path = outs.get("scores", None)
        else:
            scores_path = getattr(outs, "scores", None)

        sc = None
        if scores_path:
            try:
                sc = _try_read_table(Path(scores_path))
            except Exception:
                sc = None
        if sc is None:
            cand = args.out_dir / "scenario_scores.csv"
            if cand.exists(): sc = _try_read_table(cand)
        if (sc is None or sc.empty) and (args.out_dir / "scenario_scores.csv").exists():
            sc = _try_read_table(args.out_dir / "scenario_scores.csv")
        if sc is None or sc.empty:
            raise FileNotFoundError("scenario_scores not found")

        # Normalize stress column
        if "s_state_total" not in sc.columns:
            for alt in ["s_state", "s_tot"]:
                if alt in sc.columns:
                    sc = sc.rename(columns={alt: "s_state_total"})
                    break
            else:
                cols = [c for c in sc.columns if c.lower().endswith("state_total")]
                if cols: sc = sc.rename(columns={cols[0]: "s_state_total"})
                else: raise KeyError("s_state_total not found in scenario_scores")

        # Build the index for the run
        if args.end_date and args.start_from:
            fut_idx = pd.bdate_range(pd.to_datetime(args.start_from), pd.to_datetime(args.end_date))
        else:
            H = int(args.horizon or len(sc))
            start_dt = pd.to_datetime(args.start_from) if args.start_from else pd.Timestamp.today().normalize()
            fut_idx = pd.bdate_range(start_dt, periods=H)

        # Compute forward 5D peak/mean for each day via rolling window on the path
        s = pd.Series(sc["s_state_total"].values, index=fut_idx)
        win = max(1, int(args.select_horizon_days or 5))
        risk_peak_5d_series = s.rolling(window=win, min_periods=1).max()
        risk_mean_5d_series = s.rolling(window=win, min_periods=1).mean()
        

        daily_rows = pd.DataFrame({
            "date": risk_peak_5d_series.index,
            "risk_peak_5d": risk_peak_5d_series.values,
            "risk_mean_5d": risk_mean_5d_series.values,
            "mode": ("calendar" if args.calendar else ("channels" if args.channels else "auto_select")),
            "horizon": int(len(fut_idx))
        }).dropna(subset=["risk_peak_5d","risk_mean_5d"])

        
        # Update risk index CSV with all days
        if args.append_index_csv:
            idx_path = Path(args.append_index_csv)
            idx_path.parent.mkdir(parents=True, exist_ok=True)
            hist = pd.read_csv(idx_path, parse_dates=["date"]) if idx_path.exists() else pd.DataFrame(columns=[
                "date","risk_peak_5d","risk_mean_5d","mode","horizon","risk_scale","gate_quantile","gate_q_value"
            ])
            hist["date"] = pd.to_datetime(hist["date"]) if not hist.empty else hist.get("date")
            metric_name = "risk_peak_5d" if (args.index_metric or "peak") == "peak" else "risk_mean_5d"

            # Append rows one by one to respect no-lookahead gating
            hist = hist.sort_values("date")
            for _, row in daily_rows.sort_values("date").iterrows():
                d = pd.to_datetime(row["date"])
                prior = hist[hist["date"] < d]
                if len(prior) >= max(5, int(0.1*args.gate_window)):
                    tail = prior.tail(args.gate_window)
                    q_val = float(tail[metric_name].quantile(args.gate_quant))
                    if not np.isfinite(q_val) or q_val <= 0:
                        risk_scale = float(args.gate_floor)
                    else:
                        val = float(row[metric_name])
                        risk_scale = float(np.clip(1.0 - args.gate_alpha * (val / q_val), args.gate_floor, 1.0))
                else:
                    risk_scale, q_val = 1.0, np.nan

                upd = dict(row)
                upd["risk_scale"] = risk_scale
                upd["gate_quantile"] = args.gate_quant
                upd["gate_q_value"] = q_val
                hist = pd.concat([hist[hist["date"] != d], pd.DataFrame([upd])], ignore_index=True).sort_values("date")

            hist.to_csv(idx_path, index=False)
            print(f"[ok] risk index updated → {idx_path}")
            
    except Exception as e:
        print(f"[WARN] Post-run index/gating failed: {e}")

    # Δx per day
    deltas_df = schedule.copy()
    deltas_df.index = fut_idx
    deltas_df = deltas_df.reindex(columns=dims)

    # Severity series (s_state_total)
    s = pd.Series(sc["s_state_total"].values, index=fut_idx)

    cols_extra = {}
    for c in ["s_state_base","s_state_marg"]:
        if c in sc.columns:
            cols_extra[c] = pd.Series(sc[c].values, index=fut_idx)

    win = max(1, int(args.select_horizon_days or 5))

    # Forward-looking = reverse time, do past-looking rolling, reverse back
    rev = s.iloc[::-1]
    risk_peak_5d_series = rev.rolling(window=win, min_periods=1).max().iloc[::-1]
    risk_mean_5d_series = rev.rolling(window=win, min_periods=1).mean().iloc[::-1]

    risk_scale_map = {}
    gate_q_map = {}
    gate_qv_map = {}
    if args.append_index_csv and Path(args.append_index_csv).exists():
        _hist = pd.read_csv(args.append_index_csv, parse_dates=["date"])
        _hist = _hist.sort_values("date")
        risk_scale_map = dict(zip(_hist["date"].dt.normalize(), _hist["risk_scale"]))
        if "gate_quantile" in _hist.columns:
            gate_q_map = dict(zip(_hist["date"].dt.normalize(), _hist["gate_quantile"]))
        if "gate_q_value" in _hist.columns:
            gate_qv_map = dict(zip(_hist["date"].dt.normalize(), _hist["gate_q_value"]))

    tmp = deltas_df.copy()
    tmp.index = pd.to_datetime(tmp.index).tz_localize(None).normalize()
    if tmp.index.name != "date":
        tmp.index.name = "date"

    context = tmp.reset_index()
    context["date"] = pd.to_datetime(context["date"]).dt.normalize()

    ordered = ["date"] + [d for d in dims if d in context.columns]
    context = context.loc[:, ordered + [c for c in context.columns if c not in ordered]]

    context["s_state_total"] = context["date"].map(s.to_dict())
    for k, ser in cols_extra.items():
        context[k] = context["date"].map(ser.to_dict())

    context["risk_peak_5d"] = context["date"].map(risk_peak_5d_series.to_dict())
    context["risk_mean_5d"] = context["date"].map(risk_mean_5d_series.to_dict())
    context["risk_scale"]   = context["date"].map(risk_scale_map) if risk_scale_map else np.nan
    context["gate_quantile"]= context["date"].map(gate_q_map)     if gate_q_map else np.nan
    context["gate_q_value"] = context["date"].map(gate_qv_map)    if gate_qv_map else np.nan
    context["mode"]         = ("calendar" if args.calendar else ("channels" if args.channels else "auto_select"))
    context["horizon"]      = int(len(fut_idx))

    out_csv  = args.out_dir / "scenario_context.csv"
    context.to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()

