# constraints.py

import numpy as np

def project_l1_box(w, l1_cap=1.0, box_cap=0.15, eps=1e-12):
    w = np.clip(w, -box_cap, box_cap)
    s = np.sum(np.abs(w))
    if s > l1_cap + eps and s > eps:
        w = w * (l1_cap / s)
    return w

def turnover(w, prev_w):
    return float(np.sum(np.abs(w - prev_w)))

def throttle_turnover(w, prev_w, tau):
    if tau is None or tau <= 0: return w
    dw = w - prev_w
    l1 = np.sum(np.abs(dw))
    if l1 <= tau or l1 <= 1e-12:
        return w
    scale = tau / l1
    return prev_w + dw * scale

def _project_l1_ball(x, B):
    # Duchi et al. (2008) L1 projection
    u = np.abs(x)
    if u.sum() <= B: return x
    s = np.sort(u)[::-1]
    cssv = np.cumsum(s)
    rho = np.nonzero(s * np.arange(1, len(s)+1) > (cssv - B))[0][-1]
    theta = (cssv[rho] - B) / (rho + 1.0)
    return np.sign(x) * np.maximum(u - theta, 0.0)

def project_weights(w_raw, w_prev, l1_budget, box_b, turnover_cap):
    w = np.clip(w_raw, -box_b, box_b)
    w = _project_l1_ball(w, l1_budget)

    if np.sum(np.abs(w - w_prev)) > turnover_cap:
        d = w - w_prev
        d = _project_l1_ball(d, turnover_cap)
        w = w_prev + d
    return w


