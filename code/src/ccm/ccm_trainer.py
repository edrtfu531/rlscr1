# ccm_trainer.py

from __future__ import annotations
import os, random
import numpy as np
import pandas as pd
from functools import lru_cache
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from policy import make_weights  # scalar path
from simulator_sampler import sample as sample_scenarios
from diagnostics import contraction_residuals_ccm, graph_bias_variance_metrics

# Feature builder
def build_features_t(mu_t, L_t, prev_w_t, risk_scale_t, standardize=False):
    def _z(x):
        m, s = x.mean(), x.std()
        return (x - m) / (s + 1e-6)

    Lmu = L_t @ mu_t if (L_t is not None and getattr(L_t, "ndim", 0) == 2) else torch.zeros_like(mu_t)
    Lw  = L_t @ prev_w_t if (L_t is not None and getattr(L_t, "ndim", 0) == 2) else torch.zeros_like(prev_w_t)

    if standardize:
        mu_t, Lmu, prev_w_t, Lw = _z(mu_t), _z(Lmu), _z(prev_w_t), _z(Lw)

    rs = torch.full_like(mu_t, float(risk_scale_t))
    return torch.cat([mu_t, Lmu, prev_w_t, Lw, rs], dim=-1)

# utils
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def softplus_pos_param(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x) + 1e-8

def torch_project_box(w: torch.Tensor, box_b: float) -> torch.Tensor:
    return torch.clamp(w, -box_b, box_b)

def torch_project_l1_scaling(w: torch.Tensor, l1_budget_B: float, eps: float = 1e-12) -> torch.Tensor:
    l1 = torch.sum(torch.abs(w))
    scale = torch.clamp(l1_budget_B / torch.clamp(l1, min=eps), max=1.0)
    return w * scale

def torch_turnover_throttle(w: torch.Tensor, prev_w: torch.Tensor, tau_turnover: float, eps: float = 1e-12):
    delta = w - prev_w
    tv = torch.sum(torch.abs(delta))
    scale = torch.clamp(tau_turnover / torch.clamp(tv, min=eps), max=1.0)
    return prev_w + scale * delta, tv

def torch_graph_smooth(mu_t: torch.Tensor, L_t: torch.Tensor, theta: torch.Tensor, kappa: torch.Tensor, eps: float = 1e-8):
    N = mu_t.shape[-1]
    I = torch.eye(N, device=mu_t.device, dtype=mu_t.dtype)
    A = I + kappa * L_t + eps * I
    b = theta * mu_t
    return torch.linalg.solve(A, b)

def torch_topk_mask(x: torch.Tensor, k: int) -> torch.Tensor:
    if k is None or k <= 0 or k >= x.numel():
        return torch.ones_like(x, dtype=torch.bool)
    _, idx = torch.topk(torch.abs(x), k)
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[idx] = True
    return mask

def torch_topk_longshort(w: torch.Tensor, k_long: int, k_short: int) -> torch.Tensor:
    if (k_long is None or k_long <= 0) and (k_short is None or k_short <= 0):
        return w
    pos = torch.relu(w)
    neg = torch.relu(-w)

    if k_long and k_long > 0:
        pos = pos * torch_topk_mask(pos, k_long)
    if k_short and k_short > 0:
        neg = neg * torch_topk_mask(neg, k_short)

    return pos - neg

def torch_fill_to_l1_then_box(w: torch.Tensor, l1_B: float, box_b: float, eps: float = 1e-12) -> torch.Tensor:
    l1 = torch.sum(torch.abs(w))
    w = w * (l1_B / torch.clamp(l1, min=eps))
    w = torch.clamp(w, -box_b, box_b)
    l1 = torch.sum(torch.abs(w))
    if l1 > l1_B + 1e-12:
        w = w * (l1_B / torch.clamp(l1, min=eps))
    return w

def project_action_to_weights_torch(
    a_t: torch.Tensor,
    L_t: torch.Tensor,
    caps: dict,
    prev_w: torch.Tensor | None,
    kappa_smooth: float,
    theta_scale: float = 1.0,
    dtype: torch.dtype = torch.float64,
):
    device = a_t.device
    theta = torch.tensor(float(max(0.0, theta_scale)), device=device, dtype=dtype)
    kappa = torch.tensor(float(max(0.0, kappa_smooth)), device=device, dtype=dtype)

    # smooth raw action through graph operator
    raw = theta * a_t
    w = torch_graph_smooth(raw, L_t, theta=torch.tensor(1.0, device=device, dtype=dtype), kappa=kappa)

    # top-k optional
    k_abs   = int(caps.get("top_k_abs", 0) or 0)
    k_long  = int(caps.get("top_k_long", 0) or 0)
    k_short = int(caps.get("top_k_short", 0) or 0)

    if k_long > 0 or k_short > 0:
        w = torch_topk_longshort(w, k_long=k_long, k_short=k_short)
    elif k_abs > 0:
        w = w * torch_topk_mask(w, k_abs)

    box_b = float(caps.get("box", 0.15))
    l1_B  = float(caps.get("l1", 1.0))

    w = torch_fill_to_l1_then_box(w, l1_B=l1_B, box_b=box_b)

    if prev_w is None:
        turnover = torch.sum(torch.abs(w)) * 0.0
    else:
        w, turnover = torch_turnover_throttle(w, prev_w, tau_turnover=float(caps.get("turnover", 0.0)))

    return w, turnover

def entropic_tail_penalty_torch(pnl_s: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    eta = torch.clamp(eta, min=1e-6)
    z = -eta * pnl_s
    lme = torch.logsumexp(z, dim=0) - torch.log(torch.tensor(pnl_s.shape[0], device=pnl_s.device, dtype=pnl_s.dtype))
    return lme / eta

def graph_penalty_np_or_torch(w, L, gamma: float):
    if gamma is None or gamma <= 0:
        if isinstance(w, np.ndarray):
            return 0.0
        return (w * 0.0).sum()
    if isinstance(w, np.ndarray):
        return float(gamma * float(w @ (L @ w)))
    return gamma * (w @ (L @ w))

def _get_L(L_container, t: int, pack: dict, split: str, N: int):
    if L_container is None:
        return np.eye(N)
    if isinstance(L_container, (list, tuple)):
        return L_container[t]
    return L_container[t] if getattr(L_container, "ndim", 0) == 3 else L_container

def _risk_scale_at(pack: dict, split: str, t: int) -> float | None:
    key = f"risk_scale_{split}"
    rs = pack.get(key)
    if rs is None:
        return None
    return float(rs[t])

@lru_cache(maxsize=4)
def _load_vix_csv(path: str) -> pd.Series:
    df = pd.read_csv(path)
    date_col = "Date" if "Date" in df.columns else ("DATE" if "DATE" in df.columns else df.columns[0])

    candidates = ["Close", "VIX", "Adj Close", "AdjClose", "PX_LAST"]
    val_col = next((c for c in candidates if c in df.columns), df.columns[-1])

    df[date_col] = pd.to_datetime(df[date_col])
    ser = (
        df[[date_col, val_col]]
        .dropna()
        .drop_duplicates(subset=[date_col])
        .set_index(date_col)[val_col]
        .astype(float)
        .sort_index()
    )
    return ser

def _get_vix_aligned(pack: dict, split: str, path: str = "../../data/macro/VIX.csv") -> np.ndarray | None:
    dates_key = f"dates_{split}"
    if dates_key not in pack or pack[dates_key] is None:
        return None

    dates = pd.to_datetime(pack[dates_key])
    vix_ser = _load_vix_csv(path)

    vix_aligned = vix_ser.reindex(dates).ffill()

    vix_lag = vix_aligned.shift(1)
    if len(vix_lag) > 0 and pd.isna(vix_lag.iloc[0]):
        vix_lag.iloc[0] = vix_aligned.iloc[0]
    vix_lag = vix_lag.ffill()

    return vix_lag.to_numpy(dtype=float).reshape(-1)

def train(pack: dict, cfg: dict, device: str = "cpu", dtype=torch.float64):
    if bool(cfg.get("train", {}).get("use_rl", False)) or bool(cfg.get("rl", {}).get("use_rl", False)):
        return train_rl(pack, cfg, device=device, dtype=dtype)

# RL trainer (PPO-style)
class TrainableEta(nn.Module):
    def __init__(self, init_eta: float = 8.0, eps: float = 1e-3):
        super().__init__()
        raw0 = np.log(np.exp(init_eta) - 1.0) if init_eta > 0 else 0.0
        self.raw = nn.Parameter(torch.tensor(raw0, dtype=torch.float32))
        self.eps = eps

    def forward(self, eta_min: float = 1.0, eta_max: float = 40.0):
        eta = F.softplus(self.raw) + self.eps
        return torch.clamp(eta, min=eta_min, max=eta_max)


def _sample_R_t_for_day(t: int, pack: dict, cfg: dict) -> np.ndarray:
    rsim_all = pack.get("r_sim_train", None)
    betas_all = pack.get("betas_train", None)

    def _day(arr):
        if arr is None:
            return None
        arr = np.asarray(arr)
        if arr.ndim >= 1:
            return arr[t]
        return arr

    rsim_day = _day(rsim_all)

    try:
        R_t = sample_scenarios(rsim_all, betas_all, t, cfg)
    except TypeError:
        R_t = rsim_day[None, :] if (rsim_day is not None and rsim_day.ndim == 1) else rsim_day

    return np.asarray(R_t, float)


def _reward_from_samples_entropic(
    R_t_np: np.ndarray,
    w_t_np: np.ndarray,
    eta: torch.Tensor,
    risk_scale: float | None,
    device: str,
    dtype: torch.dtype,
):
    R_t = torch.from_numpy(R_t_np).to(device=device, dtype=dtype)
    w_t = torch.from_numpy(w_t_np).to(device=device, dtype=dtype)
    pnl_s = (R_t @ w_t)
    if risk_scale is not None:
        pnl_s = pnl_s * pnl_s.new_tensor(float(risk_scale))
    mean_pnl = pnl_s.mean()
    tail_pen = entropic_tail_penalty_torch(pnl_s, eta)
    return mean_pnl, tail_pen, (mean_pnl - tail_pen)

@torch.no_grad()
def _bootstrap_next_value(
    critic,
    next_obs: torch.Tensor,
    batches: int,
    build_features_t_fn,
    L_t_t: torch.Tensor,
    w_t: torch.Tensor,
    rscale_next: float,
    critic_include_w: bool,
):
    vals = []
    mu_next = next_obs
    for _ in range(max(1, int(batches))):
        phi_next = build_features_t_fn(mu_next, L_t_t, w_t, rscale_next)
        if critic_include_w:
            v = critic(phi_next.unsqueeze(0), w_t.unsqueeze(0)).squeeze(0)
        else:
            v = critic(phi_next.unsqueeze(0)).squeeze(0)
        vals.append(v)
    return torch.stack(vals).mean()

def build_next_obs_cf(
    curr_obs_t: torch.Tensor,
    R_t_np: np.ndarray,
    risk_scale: float | None = None,
    mode: str = "sim_mean",
    alpha: float = 0.25,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
):
    if mode == "identity":
        return curr_obs_t
    if mode == "sim_mean":
        Rm = torch.from_numpy(R_t_np).to(device=device, dtype=dtype).mean(0)
        if risk_scale is not None:
            Rm = Rm * float(risk_scale)
        return (1.0 - alpha) * curr_obs_t + alpha * Rm
    return curr_obs_t

def train_rl(pack: dict, cfg: dict, device: str = "cpu", dtype=torch.float64):
    from policy import ActorMLP, CriticMLP

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    mu_tr_np = pack["mu_train"]
    mu_tr = torch.tensor(mu_tr_np, device=device, dtype=dtype)
    is_dead_tr = pack.get("is_dead_train", None)

    T, N = mu_tr.shape

    rl_cfg = cfg.get("rl", {}) or cfg.get("train", {}) or {}
    epochs = int(rl_cfg.get("epochs", 50))
    ppo_clip = float(rl_cfg.get("ppo_clip", 0.2))
    lam_gae = float(rl_cfg.get("gae_lambda", 0.95))
    gamma_V = float(rl_cfg.get("gamma_V", 0.985))
    ppo_epochs = int(rl_cfg.get("ppo_epochs", 4))

    dro_cfg = rl_cfg.get("dro", {}) if "dro" in rl_cfg else rl_cfg
    train_eta = bool(dro_cfg.get("train_eta", True))
    eta_init = float(dro_cfg.get("eta_init", 8.0))
    eta_min = float(dro_cfg.get("eta_min", 1.0))
    eta_max = float(dro_cfg.get("eta_max", 40.0))
    epsilon = float(dro_cfg.get("epsilon", 0.0))
    lambda_rho = float(rl_cfg.get("lambda_rho", 0.5))

    use_cf_boot = bool(rl_cfg.get("use_counterfactual_bootstrap", True))
    boot_batches = int(rl_cfg.get("bootstrap_num_batches", 2))
    beta_cf = float(rl_cfg.get("beta_cf", 0.5))
    cf_mode = str(rl_cfg.get("cf_build", "sim_mean"))
    cf_alpha = float(rl_cfg.get("cf_alpha", 0.25))

    gamma_tc = float((cfg.get("constraints") or {}).get("transaction_cost_gamma", 0.0))

    graph_cfg = cfg.get("graph", {}) or {}
    kappa_default = float(graph_cfg.get("kappa", rl_cfg.get("kappa", 0.05)))
    gamma_lap = float(rl_cfg.get("laplacian_gamma", 1.0))
    lam_turn = float(rl_cfg.get("turnover_penalty", 0.1))

    caps = {
        "l1": float(rl_cfg.get("caps", {}).get("l1", 1.0)),
        "box": float(rl_cfg.get("caps", {}).get("box", 0.15)),
        "turnover": float(rl_cfg.get("caps", {}).get("turnover", 0.4)),
        "kappa_smooth": float(rl_cfg.get("caps", {}).get("kappa_smooth", kappa_default)),
    }

    vix_threshold = float(rl_cfg.get("vix_threshold", 27.0))
    vix_jump_pts  = float(rl_cfg.get("vix_jump_pts", 11.0))
    vix_jump_pct  = float(rl_cfg.get("vix_jump_pct", 0.0))
    vix_l1_mult   = float(rl_cfg.get("vix_l1_mult", 0.2))
    vix_to_mult   = float(rl_cfg.get("vix_turnover_mult", 0.25))
    vix_tr = _get_vix_aligned(pack, "train")  # (T,) or None

    def _is_vix_spike(vix_arr: Optional[np.ndarray], i: int) -> bool:
        if vix_arr is None or i <= 0:
            return False
        v_now  = float(vix_arr[i])
        v_prev = float(vix_arr[i - 1])
        if not np.isfinite(v_now) or not np.isfinite(v_prev):
            return False

        jump_pts = v_now - v_prev
        jump_pct = jump_pts / max(abs(v_prev), 1e-6)

        high_enough = (v_now >= vix_threshold)
        jump_big = (jump_pts >= vix_jump_pts) or (vix_jump_pct > 0 and jump_pct >= vix_jump_pct)
        return high_enough and jump_big
    
    model_cfg = cfg.get("model", {}) or {}
    hidden = int(model_cfg.get("hidden", 256))
    critic_include_w = bool(model_cfg.get("critic_include_w", False))
    act_sigma = float(model_cfg.get("act_sigma", 0.05))

    optim_cfg = cfg.get("optim", {}) or {}
    lr_actor = float(optim_cfg.get("lr_actor", 3e-4))
    lr_critic = float(optim_cfg.get("lr_critic", 3e-4))
    max_grad = float(optim_cfg.get("max_grad", 1.0))

    feat_dim = 5 * N
    actor = ActorMLP(feat_dim=feat_dim, n_assets=N, hidden=hidden).to(device=device, dtype=dtype)
    critic = CriticMLP(
        feat_dim=feat_dim,
        hidden=hidden,
        include_w=critic_include_w,
        n_assets=(N if critic_include_w else 0),
    ).to(device=device, dtype=dtype)

    opt_actor = torch.optim.Adam(actor.parameters(), lr=lr_actor)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=lr_critic)

    eta_module = TrainableEta(init_eta=eta_init).to(device)
    opt_eta = torch.optim.Adam([p for p in eta_module.parameters() if p.requires_grad], lr=1e-3) if train_eta else None

    diag_rows = []
    graph_diag_rows = []
    
    for epoch in range(epochs):
        phis, actions, logps = [], [], []
        values, rewards_ccm, vnext_boot = [], [], []
        ws = []

        prev_w_t = torch.zeros(N, device=device, dtype=dtype)
        shock_days = int(rl_cfg.get("vix_shock_days", 10))
        shock_until = -1

        for t in range(T):
            mu_t = mu_tr[t]

            L_t_np = _get_L(pack.get("L_train"), t, pack, "train", N)
            L_t_np = np.asarray(L_t_np) if L_t_np is not None else np.eye(N)
            L_t_t = torch.tensor(L_t_np, device=device, dtype=dtype)

            rscale_val = _risk_scale_at(pack, "train", t)
            rscale = float(rscale_val) if rscale_val is not None else 1.0

            phi_t = build_features_t(mu_t, L_t_t, prev_w_t, rscale)
            phis.append(phi_t)

            a_mu = actor(phi_t.unsqueeze(0)).squeeze(0)
            
            a_samp = a_mu + act_sigma * torch.randn_like(a_mu)
            logp_t = torch.distributions.Normal(a_mu, act_sigma).log_prob(a_samp).sum()

            caps_t = dict(caps)

            spike_today = _is_vix_spike(vix_tr, t)
            if spike_today:
                shock_until = max(shock_until, t + shock_days)

            shock_active = (t <= shock_until)
        
            if shock_active:
                caps_t["l1"] = float(caps["l1"]) * vix_l1_mult
                caps_t["turnover"] = float(caps["turnover"]) * vix_to_mult

            kappa_smooth = float(caps_t.get("kappa_smooth", kappa_default))
            prev_for_proj = None if t == 0 else prev_w_t

            w_t, turnover_t = project_action_to_weights_torch(
                a_t=a_samp,
                L_t=L_t_t,
                caps=caps_t,
                prev_w=prev_for_proj,
                kappa_smooth=kappa_smooth,
                theta_scale=1.0,
                dtype=dtype,
            )

            if shock_active:
                alpha_hedge = float(rl_cfg.get("vix_hedge_alpha", 0.5))
                hedge_scale = float(rl_cfg.get("vix_hedge_scale", 1.0))

                if alpha_hedge > 0:
                    m = torch.ones_like(w_t)
                    m = m / (m.abs().sum() + 1e-12)
                    w_t = (1 - alpha_hedge) * w_t + alpha_hedge * (-hedge_scale * m)

                    box_b = float(caps_t.get("box", 0.15))
                    l1_B  = float(caps_t.get("l1", 1.0))
                    w_t = torch.clamp(w_t, -box_b, box_b)
                    l1_now = w_t.abs().sum()
                    if l1_now > 1e-12:
                        w_t = w_t * (l1_B / l1_now)

                    if prev_for_proj is not None:
                        w_t, turnover_t = torch_turnover_throttle(
                            w_t, prev_for_proj, tau_turnover=float(caps_t.get("turnover", 0.0))
                        )

            tau_cap = caps_t.get("turnover", None)
            if tau_cap is not None and tau_cap > 0 and t > 0:
                delta = w_t - prev_w_t
                l1_now = torch.sum(torch.abs(delta))
                if l1_now > float(tau_cap) + 1e-12:
                    alpha = float(tau_cap) / float(l1_now + 1e-12)
                    w_t = prev_w_t + alpha * delta
                    turnover_t = torch.sum(torch.abs(w_t - prev_w_t))

            # delistings
            if is_dead_tr is not None:
                dead_mask_np = (is_dead_tr[t] >= 0.5)
                if np.any(dead_mask_np):
                    dead_mask = torch.from_numpy(dead_mask_np.astype(bool)).to(device=device)
                    w_t = w_t.masked_fill(dead_mask, 0.0)
                    turnover_t = torch.sum(torch.abs(w_t - prev_w_t))

            ws.append(w_t.detach())

            # diagnostics
            diag_cfg = cfg.get("diagnostics", {}) or {}
            if (t % 10 == 0) or (t == T - 1):
                mu_pen = float(cfg.get("rl", {}).get("laplacian_gamma", 1.0))
                probes = int(diag_cfg.get("trace_probes", 64))
                w_t_np_for_diag = w_t.detach().cpu().numpy()
                gdiag = graph_bias_variance_metrics(L_t_np, w_t_np_for_diag, mu_pen=mu_pen, probes=probes)
                graph_diag_rows.append({"epoch": int(epoch + 1), "t": int(t), **gdiag})

            # turnover penalties
            tau = caps_t.get("turnover", None)
            if tau is not None and tau > 0:
                excess = torch.relu(turnover_t - float(tau))
                turn_excess = lam_turn * excess
            else:
                turn_excess = turnover_t.new_zeros(())

            turn_cost = 0.5 * gamma_tc * turnover_t

            prev_w_t = w_t.detach()

            # critic value
            if critic_include_w:
                v_t = critic(phi_t.unsqueeze(0), w_t.unsqueeze(0)).squeeze(0)
            else:
                v_t = critic(phi_t.unsqueeze(0)).squeeze(0)

            # reward from scenarios
            R_t_np = _sample_R_t_for_day(t, pack, cfg)
            eta_t = eta_module(eta_min=eta_min, eta_max=eta_max)

            mean_pnl_t, tail_pen_t, _ = _reward_from_samples_entropic(
                R_t_np, w_t.detach().cpu().numpy(), eta_t, risk_scale=rscale, device=device, dtype=dtype
            )

            pen_g_t = gamma_lap * (w_t @ (L_t_t @ w_t))
            dro_dual_full = lambda_rho * (tail_pen_t + eta_t * mean_pnl_t.new_tensor(epsilon))

            r_t = (mean_pnl_t - dro_dual_full) - pen_g_t - turn_excess - turn_cost

            # counterfactual bootstrap
            if use_cf_boot:
                next_obs = build_next_obs_cf(
                    mu_t, R_t_np, risk_scale=rscale,
                    mode=cf_mode, alpha=cf_alpha,
                    device=device, dtype=dtype
                )
                Vnext_cf = _bootstrap_next_value(
                    critic=critic,
                    next_obs=next_obs,
                    batches=boot_batches,
                    build_features_t_fn=build_features_t,
                    L_t_t=L_t_t,
                    w_t=w_t,
                    rscale_next=rscale,
                    critic_include_w=critic_include_w,
                )
            else:
                Vnext_cf = mean_pnl_t.new_zeros(())

            Vclassic = mean_pnl_t.new_zeros(())
            if t + 1 < T:
                mu_tp1 = mu_tr[t + 1]
                L_tp1_np = _get_L(pack.get("L_train"), t + 1, pack, "train", N)
                L_tp1_np = np.asarray(L_tp1_np) if L_tp1_np is not None else np.eye(N)
                L_tp1_t = torch.tensor(L_tp1_np, device=device, dtype=dtype)
                rscale_p1_val = _risk_scale_at(pack, "train", t + 1)
                rscale_p1 = float(rscale_p1_val) if rscale_p1_val is not None else 1.0
                phi_tp1 = build_features_t(mu_tp1, L_tp1_t, w_t, rscale_p1)

                if critic_include_w:
                    Vclassic = critic(phi_tp1.unsqueeze(0), w_t.unsqueeze(0)).squeeze(0)
                else:
                    Vclassic = critic(phi_tp1.unsqueeze(0)).squeeze(0)

            Vnext = beta_cf * Vnext_cf + (1.0 - beta_cf) * Vclassic

            actions.append(a_samp.detach())
            logps.append(logp_t.detach())
            values.append(v_t.detach())
            rewards_ccm.append(r_t.detach())
            vnext_boot.append(Vnext.detach())

            if opt_eta is not None:
                opt_eta.zero_grad()
                tail_pen_t.backward()
                nn.utils.clip_grad_norm_(eta_module.parameters(), max_norm=5.0)
                opt_eta.step()

        # stack rollout
        phis = torch.stack(phis)
        actions = torch.stack(actions)
        logps = torch.stack(logps)
        values = torch.stack(values)
        rewards_ccm = torch.stack(rewards_ccm)
        vnext_boot = torch.stack(vnext_boot)

        ws = torch.stack(ws)
        old_ws = ws.detach()

        # contraction residual diagnostics
        V_batch_np = values.detach().cpu().numpy().reshape(-1)
        r_batch_np = rewards_ccm.detach().cpu().numpy().reshape(-1)
        Vnext_batch_np = vnext_boot.detach().cpu().numpy().reshape(-1)
        gammaV_val = float(cfg.get("rl", {}).get("gamma_V", 0.99))

        diag_ctr = contraction_residuals_ccm(
            V_t=V_batch_np, r_t=r_batch_np,
            V_next_mean=Vnext_batch_np, gamma_V=gammaV_val
        )
        diag_rows.append({"epoch": int(epoch + 1), **diag_ctr})

        # GAE
        with torch.no_grad():
            Tlen = rewards_ccm.shape[0]
            adv = torch.zeros(Tlen, device=device, dtype=dtype)
            lastgaelam = torch.tensor(0.0, device=device, dtype=dtype)
            for t in reversed(range(Tlen)):
                delta = rewards_ccm[t] + gamma_V * vnext_boot[t] - values[t]
                lastgaelam = delta + gamma_V * lam_gae * lastgaelam
                adv[t] = lastgaelam
            returns = adv + values

        old_phis = phis.detach()
        old_actions = actions.detach()
        old_logps = logps.detach()
        old_adv = adv.detach()
        old_returns = returns.detach()

        target_kl = 0.015

        for i_opt in range(ppo_epochs):
            a_mu_now = actor(old_phis)
            base_now = torch.distributions.Normal(a_mu_now, act_sigma)
            dist_now = torch.distributions.Independent(base_now, 1)
            logp_now = dist_now.log_prob(old_actions)

            adv_norm = (old_adv - old_adv.mean()) / (old_adv.std() + 1e-8)
            ratio = torch.exp(logp_now - old_logps)

            with torch.no_grad():
                log_ratio = logp_now - old_logps
                approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean()
            if approx_kl > 1.5 * target_kl:
                break

            entropy = dist_now.entropy().mean()
            loss_pi = -(
                torch.min(ratio * adv_norm, torch.clamp(ratio, 1 - ppo_clip, 1 + ppo_clip) * adv_norm)
            ).mean() - 1e-3 * entropy

            # critic loss
            if critic_include_w:
                v_pred = critic(old_phis, old_ws)
            else:
                v_pred = critic(old_phis)

            loss_v = 0.5 * (v_pred - old_returns).pow(2).mean()


            opt_actor.zero_grad()
            loss_pi.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), max_grad)
            opt_actor.step()

            opt_critic.zero_grad()
            loss_v.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), max_grad)
            opt_critic.step()

    outdir = cfg.get("output", {}).get("dir", "./outputs")
    logs_dir = os.path.join(outdir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    if len(diag_rows):
        pd.DataFrame(diag_rows).to_csv(os.path.join(logs_dir, "scr_ppo_full_contraction_diag.csv"), index=False)
    

    return {
        "actor": actor,
        "critic": critic,
        "policy": actor,
        "caps": caps,
        "kappa": float(caps.get("kappa_smooth", kappa_default)),
        "theta": 1.0,
        "eta": float(eta_module().detach().cpu()),
    }
