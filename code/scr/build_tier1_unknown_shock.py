# build_tier1_unknown_shock.py

import argparse, math
from pathlib import Path
from typing import List, Tuple, Union, Optional, Iterable
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSCanonical
from sklearn.decomposition import PCA
from sklearn.covariance import MinCovDet
from scipy.stats import chi2

def load_calendar(returns_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(returns_csv, parse_dates=["date"])
    df["date"] = df["date"].dt.tz_localize(None).dt.normalize()
    cal = df[["date"]].drop_duplicates().sort_values("date").reset_index(drop=True)
    return cal

def load_returns_matrix(returns_csv: Path) -> pd.DataFrame:
    # Pivot to date x ticker daily returns.
    df = pd.read_csv(returns_csv, parse_dates=["date"])
    df["date"] = df["date"].dt.tz_localize(None).dt.normalize()
    piv = df.pivot_table(index="date", columns="ticker", values="ret", aggfunc="first").sort_index()
    piv.columns = [f"ret_{c}" for c in piv.columns]
    return piv.reset_index()

def load_csv(fp: Union[str, Path],
             date_candidates: Optional[Iterable[str]] = None) -> pd.DataFrame:
    if date_candidates is None:
        date_candidates = (
            "date","Date","DATE","observation_date","timestamp","Timestamp","time","Time","TIME",
            "dt","DT","month","Month","MONTH","DATE ","Unnamed: 0","Unnamed: 0_level_0","index","level_0"
        )

    df = pd.read_csv(fp)

    # Standardize to 'date'
    date_col = None
    for c in df.columns:
        if c.lower() == "date":
            date_col = c
            break

    if date_col is None:
        for c in date_candidates:
            if c in df.columns:
                date_col = c
                break

    if date_col is None:
        best_col, best_hits = None, -1
        for c in df.columns:
            try:
                parsed = pd.to_datetime(df[c], errors="coerce")
                hits = parsed.notna().sum()
                if hits > max(5, int(0.3 * len(parsed))):
                    if hits > best_hits:
                        best_col, best_hits = c, hits
            except Exception:
                continue
        date_col = best_col

    if date_col is None:
        raise ValueError(f"{fp} does not have a recognizable date column.")

    if date_col != "date":
        df = df.rename(columns={date_col: "date"})

    # Normalize the date column
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().all():
        try:
            df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
        except Exception:
            pass

    df = df.dropna(subset=["date"]).copy()
    if isinstance(df["date"].dtype, pd.DatetimeTZDtype):
        df["date"] = df["date"].dt.tz_convert(None)
    df["date"] = df["date"].dt.tz_localize(None, nonexistent="NaT", ambiguous="NaT") if hasattr(df["date"].dt, "tz_localize") else df["date"]
    df["date"] = df["date"].dt.normalize()
    df = df[~df["date"].duplicated(keep="last")].sort_values("date").reset_index(drop=True)
    
    return df


# Feature assembly
def build_views(cal: pd.DataFrame, returns_mat: pd.DataFrame, shock: pd.DataFrame, fedsig: pd.DataFrame):
    # Prefix for clarity
    shock = shock.add_prefix("shock_").rename(columns={"shock_date":"date"})
    fedsig = fedsig.add_prefix("fedsig_").rename(columns={"fedsig_date":"date"})

    df = cal.merge(returns_mat, on="date", how="left") \
            .merge(shock, on="date", how="left") \
            .merge(fedsig, on="date", how="left") \
            .sort_values("date").reset_index(drop=True)

    # Column groups
    ret_cols = [c for c in df.columns if c.startswith("ret_")]
    mac_cols = [c for c in df.columns if c.startswith("shock_") and c != "shock_date"]
    txt_cols = [c for c in df.columns if c.startswith("fedsig_emb_")] \
             + [c for c in ["fedsig_stance_score","fedsig_p_hike","fedsig_p_hold","fedsig_p_cut",
                            "fedsig_finbert_sent_pos","fedsig_finbert_sent_neg","fedsig_finbert_sent_neu"]
                if c in df.columns]

    if not ret_cols: raise ValueError("No return columns (ret_*) found.")
    if not mac_cols: raise ValueError("No macro columns (shock_*) found.")
    if not txt_cols: raise ValueError("No text features from fedsignal (emb_*/stance/sent).")

    df[ret_cols] = df[ret_cols].fillna(0.0)

    # Text (already next-day safe) & Macro shocks: slow-moving → forward-fill on trading calendar
    df[txt_cols] = df[txt_cols].ffill()
    df[mac_cols] = df[mac_cols].ffill()

    df[txt_cols] = df[txt_cols].fillna(0.0)
    df[mac_cols] = df[mac_cols].fillna(0.0)

    X = df[ret_cols + mac_cols].to_numpy(dtype=np.float32)
    S = df[txt_cols].to_numpy(dtype=np.float32)
    return df[["date"]], X, S, ret_cols, mac_cols, txt_cols


# Universal Shock Space U_t
def fit_pls_canonical(X_train, S_train, k):
    # Standardize each view on train only
    sx = StandardScaler(with_mean=True, with_std=True).fit(X_train)
    ss = StandardScaler(with_mean=True, with_std=True).fit(S_train)
    Xz = sx.transform(X_train)
    Sz = ss.transform(S_train)

    k_eff = max(1, min(k, Xz.shape[1], Sz.shape[1]))
    pls = PLSCanonical(n_components=k_eff).fit(Xz, Sz)
    return pls, sx, ss, k_eff

def transform_U(pls, sx, ss, X_all, S_all, k_eff):
    Xz = sx.transform(X_all)
    Sz = ss.transform(S_all)
    # PLSCanonical returns scores (latent) for each view
    T_x, T_s = pls.transform(Xz, Sz)
    # U as the average of aligned scores
    U = 0.5 * (T_x[:, :k_eff] + T_s[:, :k_eff])
    return U.astype(np.float32), T_x.astype(np.float32), T_s.astype(np.float32)

# Novelty scores
def pca_reconstruction_error(U_train, U_all, var_keep=0.9):
    # choose #components to keep target variance on train
    pca = PCA(n_components=min(U_train.shape[0], U_train.shape[1])).fit(U_train)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    r = int(np.searchsorted(cumsum, var_keep) + 1)
    r = max(1, min(r, U_train.shape[1]))
    pca = PCA(n_components=r).fit(U_train)

    Uh = pca.inverse_transform(pca.transform(U_all))
    rec_err = np.sum((U_all - Uh)**2, axis=1)
    # z-score using train stats
    mu, sd = rec_err[:len(U_train)].mean(), rec_err[:len(U_train)].std(ddof=1) + 1e-9
    rec_err_z = (rec_err - mu) / sd
    return rec_err_z.astype(np.float32), r

def mahalanobis_tail_prob(U_train, U_all):
    mcd = MinCovDet().fit(U_train)
    md2 = mcd.mahalanobis(U_all)   # squared Mahalanobis
    df = U_train.shape[1]
    # tail probability under chi-square
    p_tail = 1.0 - chi2.cdf(md2, df=df)
    return p_tail.astype(np.float32)

def page_hinkley(x, delta=0.005, lam=1.0):
    x = np.asarray(x, dtype=np.float64)
    mean = 0.0
    m_t = 0.0
    ph = np.zeros_like(x)
    for t, xi in enumerate(x):
        mean += (xi - mean) / (t+1)
        m_t = min(0.0, m_t + xi - mean - delta)
        ph[t] = -m_t
    # Normalize
    ph = (ph - ph.min()) / (ph.max() - ph.min() + 1e-9)
    return ph.astype(np.float32)

# Episode clustering (DP-means style)
def dp_means_assign(U, is_shock, lam=2.0):
    """
    Online DP-means on U_t for t where is_shock==1.
    lam: distance^2 threshold to start a new cluster.
    """
    centers = []
    counts = []
    ids = np.zeros(U.shape[0], dtype=np.int32)
    next_id = 1
    for t in range(U.shape[0]):
        if not is_shock[t]:
            ids[t] = 0
            continue
        
        u_t = U[t]
        if len(centers) == 0:
            centers.append(u_t.copy())
            counts.append(1) 
            ids[t] = next_id
            next_id += 1
        else:
            d2 = np.array([np.sum((u_t - c)**2) for c in centers])
            j = int(np.argmin(d2))
            if d2[j] > lam:
                centers.append(u_t.copy())
                counts.append(1)
                ids[t] = next_id
                next_id += 1
            else:
                # Update center with incremental mean
                counts[j] += 1
                eta = 1.0 / counts[j]
                centers[j] = (1.0 - eta) * centers[j] + eta * u_t
                ids[t] = j + 1
    return ids



# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--returns",    type=Path, required=True)
    ap.add_argument("--shock",      type=Path, required=True)
    ap.add_argument("--fedsig",     type=Path, required=True)
    ap.add_argument("--train_end",  type=str,  required=True)
    ap.add_argument("--k_u",        type=int,  default=8)
    ap.add_argument("--out",        type=Path, default=Path("data/tier1_unknown_shock.csv"))
    args = ap.parse_args()

    # Load
    cal = load_calendar(args.returns)
    R   = load_returns_matrix(args.returns)
    train_end = pd.Timestamp(args.train_end)

    ret_mat = R.set_index("date")
    train_mask = ret_mat.index <= train_end
    ret_train = ret_mat[train_mask]

    valid_ret_cols = [
        c for c in ret_train.columns
        if ret_train[c].notna().mean() >= 0.9
    ]

    if len(valid_ret_cols) == 0:
        raise ValueError("No tickers with sufficient coverage in training window.")

    ret_mat = ret_mat[valid_ret_cols]
    R = ret_mat.reset_index()

    Sx  = load_csv(args.shock)
    Fs  = load_csv(args.fedsig)

    # Assemble aligned views
    base_dates, X, S, ret_cols, mac_cols, txt_cols = build_views(cal, R, Sx, Fs)

    # Expanding-window training by year to handle non-stationarity
    train_end = pd.Timestamp(args.train_end)
    mask_train = (base_dates["date"] <= train_end).to_numpy()
    if mask_train.sum() < 50:
        raise ValueError("Training period too short for stable fits. Adjust --train_end.")

    finite_all = np.isfinite(X).all(axis=1) & np.isfinite(S).all(axis=1)
    mask_train_valid = mask_train & finite_all

    if mask_train_valid.sum() < 50:
        raise ValueError("Training set has too few finite rows after cleaning. "
                         "Check start dates / warm-up windows.")

    pls, sx, ss, k_eff = fit_pls_canonical(X[mask_train_valid], S[mask_train_valid], k=args.k_u)
    U, T_x, T_s = transform_U(pls, sx, ss, X, S, k_eff)

    # Novelty scores
    rec_err_z, r_dim = pca_reconstruction_error(U[mask_train], U)
    maha_p = mahalanobis_tail_prob(U[mask_train], U)

    # Combine into raw Z (higher = more surprising)
    maha_s = -np.log(maha_p + 1e-12)
    maha_s = (maha_s - np.mean(maha_s[mask_train])) / (np.std(maha_s[mask_train]) + 1e-9)
    Z_raw = 0.6 * (1/(1+np.exp(-rec_err_z))) + 0.4 * maha_s

    # Add a change statistic on Z_raw (Page-Hinkley)
    change_stat = page_hinkley(Z_raw, delta=0.01, lam=1.0)

    # Blend with change_stat
    Z = 0.7*Z_raw + 0.3*change_stat
    Z_train = Z[mask_train]
    hi = float(np.quantile(Z_train, 0.995))
    lo = float(np.quantile(Z_train, 0.985))
    if not np.isfinite(hi) or not np.isfinite(lo) or hi <= lo:
        hi, lo = np.nanpercentile(Z_train, [99.5, 98.5])

    # Shock mask with hysteresis (stable episodes)
    is_shock = np.zeros_like(Z, dtype=np.int32)
    p_train = maha_p[mask_train]
    tau = float(np.quantile(p_train, 0.005))  # bottom 0.5% tail prob in train
    hard_flag = (maha_p <= tau).astype(np.int32)


    # Hysteresis as a separate mask
    hys = np.zeros_like(Z, dtype=np.int32)
    active = False
    for t, z in enumerate(Z):
        if not active and z >= hi:
            active = True
        if active:
            hys[t] = 1
        if active and z <= lo:
            active = False

    # Final
    is_shock = np.maximum(hys, hard_flag).astype(np.int32)

    episode_id = dp_means_assign(U, is_shock, lam=2.0)

    # Output
    out = base_dates.copy()
    # latent coords
    for j in range(k_eff):
        out[f"U_{j+1}"] = U[:, j]
        
    # components & score
    out["recon_err_z"] = rec_err_z
    out["maha_p"] = maha_p
    out["change_stat"] = change_stat
    out["Z"] = Z
    out["is_shock"] = is_shock
    out["episode_id"] = episode_id

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

if __name__ == "__main__":
    main()

