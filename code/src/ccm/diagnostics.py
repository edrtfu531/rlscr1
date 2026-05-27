# diagnostics.py

from __future__ import annotations
import numpy as np
def _hutchinson_trace_of_inv(I_plus_muL: np.ndarray, probes: int = 64, rng: np.random.RandomState | None = None) -> float:
    # Estimate tr((I + μL)^(-1)) using Hutchinson’s estimator with Rademacher probes.
    rng = rng or np.random.RandomState(0)
    N = I_plus_muL.shape[0]
    acc = 0.0
    for _ in range(probes):
        z = rng.choice([-1.0, 1.0], size=N).astype(float)
        x = np.linalg.solve(I_plus_muL, z)   # (I+μL)^{-1} z
        acc += float(z @ x)
    return acc / float(probes)

# Contraction diagnostic
def contraction_residuals_ccm(
    V_t: np.ndarray,
    r_t: np.ndarray,
    V_next_mean: np.ndarray,
    gamma_V: float
) -> dict:
    # Residual of the SC²B Bellman equation: R + γ E V(next) - V(current).
    resid = r_t + gamma_V * V_next_mean - V_t
    return {
        "resid_sup": float(np.max(np.abs(resid))),
        "resid_l2": float(np.linalg.norm(resid) / np.sqrt(len(resid) + 1e-12)),
        "resid_mean_abs": float(np.mean(np.abs(resid)))
    }

# Graph bias–variance diagnostic
def graph_bias_variance_metrics(
    L_t: np.ndarray, 
    w_t: np.ndarray,
    mu_pen: float,
    probes: int = 64
) -> dict:
    N = L_t.shape[0]
    I_plus_muL = np.eye(N) + mu_pen * L_t
    tr_inv = _hutchinson_trace_of_inv(I_plus_muL, probes=probes)

    # λ2(L): second-smallest eigenvalue; robustly via smallest few eigs
    evals = np.linalg.eigvalsh(L_t)
    evals = np.sort(evals)
    lambda2 = float(evals[1]) if len(evals) > 1 else 0.0

    w = w_t.reshape(-1)
    smooth = float(w @ (L_t @ w))
    wnorm = float(np.linalg.norm(w))

    return {
        "trace_inv_I_muL": float(tr_inv),
        "lambda2": lambda2,
        "w_L_w": smooth,
        "w_l2": wnorm
    }

def sim_pac_gap_and_proxy(
    sim_portfolio_returns: np.ndarray,   # daily returns from simulator
    real_portfolio_returns: np.ndarray,  # daily realized returns
    weights: np.ndarray,
    n_rollouts: int
) -> dict:
    # Computes the Sim-to-Real gap and PAC proxy bound continuously over time.
    
    T = len(sim_portfolio_returns)
    assert T == len(real_portfolio_returns), "Lengths must match"
    
    # 1. Cumulative Means (Running Average of Returns)
    time_steps = np.arange(1, T + 1)
    
    # J_n(t) = Mean sim return from day 0 to t
    J_n_series = np.cumsum(sim_portfolio_returns) / time_steps
    
    # J(t) = Mean realized return from day 0 to t
    J_series = np.cumsum(real_portfolio_returns) / time_steps
    
    # Gap(t) = |J_n(t) - J(t)|
    gap_series = np.abs(J_n_series - J_series)
    
    w_padded = np.vstack([np.zeros((1, weights.shape[1])), weights])
    daily_turnover = np.sum(np.abs(np.diff(w_padded, axis=0)), axis=1) # Result is length T
    
    # B_t = Cumulative sum of turnover up to time t
    Bt_series = np.cumsum(daily_turnover)
    
    term_sim_noise = np.sqrt(1.0 / max(1, n_rollouts))
    term_turnover  = np.sqrt(Bt_series / time_steps)
    
    proxy_series = term_sim_noise + term_turnover

    return {
        "J_n_sim_cumulative": J_n_series,        # Plot Line 1
        "J_real_cumulative": J_series,           # Plot Line 2
        "abs_gap_cumulative": gap_series,        # Plot Gap
        "proxy_bound_cumulative": proxy_series,  # Plot Bound (Upper Limit)
        "B_T_cumulative": Bt_series              # Optional diagnostic
    }