# graph_utils.py

import pandas as pd, numpy as np

def _find_edge_columns(df_edges_day: pd.DataFrame):
    cols = list(df_edges_day.columns)
    if all(c in cols for c in ["src", "dst"]):
        return "src", "dst"
    
    cands = [c for c in cols if c.lower() in ("source","src","u","from")]
    candt = [c for c in cols if c.lower() in ("target","dst","v","to")]
    if not cands or not candt:
        raise ValueError(f"Cannot find edge endpoints in columns {cols}")
    return cands[0], candt[0]

def _lap_from_edges(n: int, edges: np.ndarray) -> np.ndarray:
    if n <= 0:
        return np.eye(max(1,n))
    W = np.zeros((n,n), dtype=float)
    for (i,j) in edges:
        if i==j: 
            continue
        if 0 <= i < n and 0 <= j < n:
            W[i,j] = 1.0
            W[j,i] = 1.0
    d = np.sum(W, axis=1)
    return np.diag(d) - W

def algebraic_connectivity(L: np.ndarray) -> float:
    # Return λ2(L), the second-smallest eigenvalue of a symmetric Laplacian.
    if L.size == 0:
        return 0.0

    ev = np.linalg.eigvalsh((L + L.T) * 0.5)
    ev = np.sort(ev)
    if len(ev) < 2:
        return 0.0
    return float(ev[1])

def build_laplacian_series(df_edges: pd.DataFrame, tickers: list[str], dates_index: pd.Index):
    """
    Build a dict date->Laplacian aligned to dates_index.
    Edge-universe guard:
      - Drops edges whose endpoints aren't in today's ticker universe
    """
    df = df_edges.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.set_index("date").sort_index()

    # Node mapping
    N = len(tickers)

    out = {}
    for d in dates_index:
        try:
            day = df.loc[d]
        except KeyError:
            out[d] = np.eye(N)
            continue
        if isinstance(day, pd.Series):
            day = day.to_frame().T

        src_col, dst_col = _find_edge_columns(day)
        e = day[[src_col, dst_col]].to_numpy(dtype=int)

        mask = (e[:,0] >= 0) & (e[:,0] < N) & (e[:,1] >= 0) & (e[:,1] < N)
        e = e[mask]
        L = _lap_from_edges(N, e)
        out[d] = L

    return out

