# simulator_sampler.py

import numpy as np

def _knn_beta_indices(betas_train, beta_t, k=50):
    if betas_train is None or beta_t is None:
        return None
    b = np.asarray(betas_train, float)
    v = np.asarray(beta_t, float).reshape(1, -1)
    B = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    V = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    sims = (B @ V.T).ravel()
    idx = np.argsort(-sims)[:min(k, len(sims))]
    return idx

def sample(r_sim_train, betas_train, t, cfg):

    S    = int(cfg["train"].get("scenarios", 64))
    roll = int(cfg["train"].get("roll_window", 60))
    mode = cfg["train"].get("sampler", {}).get("mode", "knn_beta")
    k    = int(cfg["train"].get("sampler", {}).get("k", 50))

    R = np.asarray(r_sim_train, float)
    T, N = R.shape
    if T == 0:
        return np.zeros((S, N), dtype=float)

    t0 = max(0, t - roll)
    pool = np.arange(t0, t + 1)

    betas_window = None
    betas_t = None
    if betas_train is not None:
        B = np.asarray(betas_train)
        betas_window = B[t0:t+1]
        betas_t = B[t]


    if mode == "knn_beta" and betas_window is not None and betas_t is not None:
        idx = _knn_beta_indices(betas_window, betas_t, k=k)
        if idx is not None and len(idx) > 0:
            pool = t0 + idx

    # pool must be leak-safe
    if pool.size == 0:
        pool = np.array([t0], dtype=int)
    assert pool.max() <= t, "Sampler pool includes future rows (look-ahead)!"

    ridx = np.random.choice(pool, size=S, replace=True)
    return R[ridx, :]

