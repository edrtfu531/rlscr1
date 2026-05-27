# policy.py

import numpy as np
from constraints import project_l1_box, throttle_turnover
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorMLP(nn.Module):
    """
    Actor maps features phi_t -> raw action a_t (R^N).
    LN -> MLP -> output.
    """
    def __init__(self, feat_dim: int, n_assets: int, hidden: int = 256):
        super().__init__()
        self.ln = nn.LayerNorm(feat_dim)
        self.fc1 = nn.Linear(feat_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc_out = nn.Linear(hidden, n_assets)
        nn.init.uniform_(self.fc_out.weight, -0.01, 0.01)
        nn.init.constant_(self.fc_out.bias, 0.0)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        x = self.ln(phi)
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        return self.fc_out(x)

class CriticMLP(nn.Module):
    """
    Critic that maps features -> scalar V.
    """
    def __init__(self, feat_dim: int, hidden: int = 256, include_w: bool = False, n_assets: int = 0):
        super().__init__()
        in_dim = feat_dim + (n_assets if include_w else 0)
        self.ln = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.v_out = nn.Linear(hidden, 1)

    def forward(self, phi: torch.Tensor, w_opt: torch.Tensor = None) -> torch.Tensor:
        x = phi if w_opt is None else torch.cat([phi, w_opt], dim=-1)
        x = self.ln(x)
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        return self.v_out(x).squeeze(-1)

def graph_smooth(raw, L, kappa):
    N = raw.shape[0]
    I = np.eye(N, dtype=float)
    M = I + float(max(0.0, kappa)) * L
    try:
        w = np.linalg.solve(M, raw)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(M) @ raw
    return w

def make_weights(mu_t, L_t, theta, kappa, caps, prev_w=None):
    raw = float(max(0.0, theta)) * mu_t
    w = graph_smooth(raw, L_t, float(max(0.0, kappa)))
    w = project_l1_box(w, l1_cap=caps.get("l1",1.0), box_cap=caps.get("box",0.15))
    if prev_w is None:
        return w
    w = throttle_turnover(w, prev_w, tau=caps.get("turnover", None))
    return w

class GaussianPolicy(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, log_std_init: float=-1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
        self.log_std = nn.Parameter(torch.ones(out_dim) * log_std_init)

    def forward(self, x):
        mu = self.net(x)
        std = torch.exp(self.log_std).clamp_min(1e-4)
        return mu, std

    def sample(self, x):
        mu, std = self.forward(x)
        eps = torch.randn_like(mu)
        a = mu + std * eps
        # log prob of Gaussian with diag cov
        logp = -0.5 * (((a - mu) / std)**2 + 2*self.log_std + torch.log(torch.tensor(2*3.141592653589793)))
        logp = logp.sum(-1)
        return a, logp, mu, std

    def log_prob(self, x, a):
        mu, std = self.forward(x)
        logp = -0.5 * (((a - mu) / std)**2 + 2*self.log_std + torch.log(torch.tensor(2*3.141592653589793)))
        return logp.sum(-1)

class ValueNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def project_action_to_weights(action_np, L_t, caps, prev_w, kappa: float, theta: float):
    raw = float(max(0.0, theta)) * action_np
    w = graph_smooth(raw, L_t, float(max(0.0, kappa)))
    w = project_l1_box(w, l1_cap=caps.get("l1",1.0), box_cap=caps.get("box",0.15))
    if prev_w is not None:
        w = throttle_turnover(w, prev_w, tau=caps.get("turnover", None))
    return w

