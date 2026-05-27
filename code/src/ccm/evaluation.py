# evaluation.py

import os
import numpy as np
import pandas as pd
import torch
from typing import Optional
from functools import lru_cache
from policy import make_weights
from ccm_trainer import build_features_t
from diagnostics import sim_pac_gap_and_proxy


def _compute_turnover_and_cost(weights: np.ndarray, gamma: float):
    T, N = weights.shape
    prev = np.zeros(N, dtype=float)
    turnover = np.zeros(T, dtype=float)
    cost = np.zeros(T, dtype=float)

    if gamma <= 0:
        return turnover, cost

    for t in range(T):
        dw = weights[t] - prev
        turnover[t] = 0.5 * np.sum(np.abs(dw))
        cost[t] = gamma * turnover[t]
        prev = weights[t]

    return turnover, cost


def _compute_pnl_series(returns, weights):
    return np.sum(returns * weights, axis=1)

def _get_L(L_container, t, pack, split, N):
    if L_container is None:
        return np.eye(N)
    if isinstance(L_container, (list, tuple)):
        return L_container[t]
    return L_container[t] if getattr(L_container, "ndim", 0) == 3 else L_container

@lru_cache(maxsize=4)
def _load_vix_csv_series(path: str) -> pd.Series:
    df = pd.read_csv(path)
    date_col = "Date" if "Date" in df.columns else ("DATE" if "DATE" in df.columns else df.columns[0])
    candidates = ["Close", "VIX", "Adj Close", "AdjClose", "PX_LAST"]
    val_col = next((c for c in candidates if c in df.columns), df.columns[-1])
    df[date_col] = pd.to_datetime(df[date_col])
    return (df[[date_col, val_col]].dropna().drop_duplicates(subset=[date_col])
              .set_index(date_col)[val_col].astype(float).sort_index())

def _get_vix_aligned_to_dates(dates: pd.DatetimeIndex, path: str = "../../data/macro/VIX.csv") -> np.ndarray:
    ser = _load_vix_csv_series(path)
    aligned = ser.reindex(dates).ffill()
    lagged = aligned.shift(1)
    if len(lagged) > 0 and pd.isna(lagged.iloc[0]):
        lagged.iloc[0] = aligned.iloc[0]
    lagged = lagged.ffill()
    return lagged.to_numpy(dtype=float).reshape(-1)

# Evaluator
def _evaluate_rl_split(pack, state, cfg, split: str):
    # deterministic policy mean -> action -> projected weights.
    
    assert split in ("train", "val", "test")
    from ccm_trainer import project_action_to_weights_torch

    policy = state["policy"]
    policy.eval()

    tdevice = next(policy.parameters()).device
    tdtype  = next(policy.parameters()).dtype

    caps = state.get("caps", {}) or {}
    if "l1" not in caps or "box" not in caps or "turnover" not in caps:
        raise ValueError(f"state['caps'] missing required keys. Got: {list(caps.keys())}")

    # load split data
    mu = np.asarray(pack[f"mu_{split}"], float)
    mu_t = torch.tensor(mu, device=tdevice, dtype=tdtype)

    dates   = pd.to_datetime(pack.get(f"dates_{split}"))
    tickers = pack["tickers"]

    rreal    = pack.get(f"r_real_{split}", None)
    rsim     = pack.get(f"r_sim_{split}", None)
    risk_arr = pack.get(f"risk_scale_{split}", None)
    is_dead  = pack.get(f"is_dead_{split}", None)

    gamma_tc = float((cfg.get("constraints") or {}).get("transaction_cost_gamma", 0.0))

    vix_arr = _get_vix_aligned_to_dates(dates)

    rl_cfg = (cfg.get("rl", {}) or {})
    vix_threshold = float(rl_cfg.get("vix_threshold", 27.0))
    vix_jump_pts  = float(rl_cfg.get("vix_jump_pts", 11.0))
    vix_jump_pct  = float(rl_cfg.get("vix_jump_pct", 0.0))
    vix_l1_mult   = float(rl_cfg.get("vix_l1_mult", 0.2))
    vix_to_mult   = float(rl_cfg.get("vix_turnover_mult", 0.25))

    shock_days = int(rl_cfg.get("vix_shock_days", 10))
    alpha_hedge = float(rl_cfg.get("vix_hedge_alpha", 0.5))
    hedge_scale = float(rl_cfg.get("vix_hedge_scale", 1.0))

    def _is_vix_spike_eval(vix_arr_local: Optional[np.ndarray], i: int) -> bool:
        if vix_arr_local is None or i <= 0:
            return False
        v_now  = float(vix_arr_local[i])
        v_prev = float(vix_arr_local[i - 1])
        if not np.isfinite(v_now) or not np.isfinite(v_prev):
            return False

        jump_pts = v_now - v_prev
        jump_pct = jump_pts / max(abs(v_prev), 1e-6)

        high_enough = (v_now >= vix_threshold)
        jump_big = (jump_pts >= vix_jump_pts) or (vix_jump_pct > 0 and jump_pct >= vix_jump_pct)
        return high_enough and jump_big

    T, N = mu.shape
    prev_w_np = None
    rows = []
    pnl_real = []

    shock_until = -1

    for i in range(T):
        L_np = _get_L(pack.get(f"L_{split}"), i, pack, split, N)
        L_t  = torch.tensor(np.asarray(L_np, float), device=tdevice, dtype=tdtype)

        scale = float(risk_arr[i]) if risk_arr is not None else 1.0

        mu_i = mu_t[i]
        prev_w_t = (
            torch.zeros(N, device=tdevice, dtype=tdtype)
            if prev_w_np is None
            else torch.tensor(prev_w_np, device=tdevice, dtype=tdtype)
        )

        # same features as training
        phi = build_features_t(mu_i, L_t, prev_w_t, scale)

        # deterministic mean action
        with torch.no_grad():
            out = policy(phi.unsqueeze(0))
            mu_out = out[0] if isinstance(out, (tuple, list)) else out
            a = mu_out.squeeze(0)
        
        caps_step = dict(caps)

        spike_today = _is_vix_spike_eval(vix_arr, i)
        if spike_today:
            shock_until = max(shock_until, i + shock_days)

        shock_active = (i <= shock_until)

        if shock_active:
            caps_step["l1"] = float(caps["l1"]) * vix_l1_mult
            caps_step["turnover"] = float(caps["turnover"]) * vix_to_mult

        kappa_smooth = float(caps_step.get("kappa_smooth", caps.get("kappa_smooth", 0.0)))

        # project action -> weights
        with torch.no_grad():
            w_t_torch, _ = project_action_to_weights_torch(
                a_t=a,
                L_t=L_t,
                caps=caps_step,
                prev_w=(prev_w_t if i > 0 else None),
                kappa_smooth=kappa_smooth,
                theta_scale=1.0,
                dtype=tdtype,
            )

            if shock_active and alpha_hedge > 0 :
                m = torch.ones_like(w_t_torch)
                m = m / (m.abs().sum() + 1e-12)
                w_t_torch = (1.0 - alpha_hedge) * w_t_torch + alpha_hedge * (-hedge_scale * m)
                box_b = float(caps_step["box"])
                l1_B  = float(caps_step["l1"])

                w_t_torch = torch.clamp(w_t_torch, -box_b, box_b)
                l1_now = w_t_torch.abs().sum()
                if l1_now > 1e-12:
                    w_t_torch = w_t_torch * (l1_B / l1_now)

        w_t = w_t_torch.detach().cpu().numpy()
        
        if is_dead is not None:
            dead_mask = is_dead[i] >= 0.5
            if np.any(dead_mask):
                w_t = w_t.copy()
                w_t[dead_mask] = 0.0

        rows.append(w_t)
        prev_w_np = w_t

        if rreal is not None:
            pnl_real.append(float(np.dot(w_t, np.asarray(rreal[i], float))))

    # Save weights
    outdir = cfg["output"]["dir"]
    os.makedirs(outdir, exist_ok=True)

    W = np.vstack(rows) if rows else np.zeros((0, N), dtype=float)
    df_w = pd.DataFrame(W, index=dates, columns=tickers)
    # compute sim/real series
    sim_returns_rl = None
    real_returns_rl = None

    if rsim is not None and len(rows) > 0:
        R_sim = np.asarray(rsim, float)
        if R_sim.shape[:2] != W.shape[:2]:
            raise ValueError(f"Shape mismatch: R_sim {R_sim.shape} vs W {W.shape}")
        sim_returns_rl = np.sum(W * R_sim, axis=1)

    if len(pnl_real) > 0:
        real_returns_rl = np.asarray(pnl_real, float)

    turnover_rl, cost_rl = _compute_turnover_and_cost(W, gamma_tc)
    if sim_returns_rl is not None:
        sim_returns_rl = sim_returns_rl - cost_rl
    if real_returns_rl is not None:
        real_returns_rl = real_returns_rl - cost_rl

    if real_returns_rl is not None:
        pd.Series(real_returns_rl, index=dates, name=f"ret_{split}_rl").to_csv(
            os.path.join(outdir, f"portfolio_returns_{split}_rl.csv")
        )

    # PAC proxy (use net returns)
    if sim_returns_rl is not None and real_returns_rl is not None:
        n_rollouts = int((cfg.get("train", {}) or {}).get("scenarios", 64))
        pac = sim_pac_gap_and_proxy(
            sim_portfolio_returns=sim_returns_rl,
            real_portfolio_returns=real_returns_rl,
            weights=W,
            n_rollouts=n_rollouts
        )
        
        logs_dir = os.path.join(outdir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        pd.DataFrame([{"split": split, **pac}]).to_csv(
            os.path.join(logs_dir, f"scr_ppo_full_sim_pac_{split}.csv"), index=False
        )
        
        pd.DataFrame({"turnover": turnover_rl, "cost": cost_rl}, index=dates).to_csv(
            os.path.join(logs_dir, f"portfolio_turnover_{split}_rl.csv"),
            index=True
        )


    return {"weights_" + split: df_w}

def evaluate(pack, state, cfg):
    _evaluate_rl_split(pack, state, cfg, split="train")
    _evaluate_rl_split(pack, state, cfg, split="val")
    return _evaluate_rl_split(pack, state, cfg, split="test")
