# rolling_mapper_online.py

import argparse, os
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from pandas.tseries.offsets import BDay
import warnings
warnings.filterwarnings("ignore")

def _read_returns(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    lc = {c.lower(): c for c in df.columns}

    if {"date","ticker","ret"}.issubset(lc.keys()):
        out = df.rename(columns={lc["date"]:"date", lc["ticker"]:"ticker", lc["ret"]:"ret"})[["date","ticker","ret"]]
        out["date"] = pd.to_datetime(out["date"], utc=True).dt.tz_convert(None).dt.normalize()
        out["ret"]  = pd.to_numeric(out["ret"], errors="coerce")
        return out.dropna(subset=["date","ticker","ret"]).sort_values(["ticker","date"])

    elif {"date","ticker","close"}.issubset(lc.keys()):
        df[lc["date"]]  = pd.to_datetime(df[lc["date"]], utc=True).dt.tz_convert(None).dt.normalize()
        df[lc["close"]] = pd.to_numeric(df[lc["close"]], errors="coerce")
        df = df.dropna(subset=[lc["date"], lc["ticker"], lc["close"]]).sort_values([lc["ticker"], lc["date"]])
        df["ret"] = df.groupby(lc["ticker"])[lc["close"]].pct_change()
        out = df[[lc["date"], lc["ticker"], "ret"]].rename(columns={lc["date"]:"date", lc["ticker"]:"ticker"})
        return out.dropna(subset=["ret"]).sort_values(["ticker","date"])

    else:
        raise ValueError("returns must have ['date','ticker','ret'] or ['date','ticker','close']")

def _read_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None).dt.normalize()
        df = df.set_index("date")
    else:
        idx = pd.to_datetime(df.index, utc=True).tz_convert(None).normalize()
        df = df.set_index(idx)
    return df.sort_index()



def _harmonize(df, *, expect_long_channels=False, fill_inactive=False):
    if df is None:
        return pd.DataFrame()
    df = df.copy()

    if "date" in df.columns:
        s = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
        if expect_long_channels and "channel" in df.columns and ("S" in df.columns or "value" in df.columns):
            val = "S" if "S" in df.columns else "value"
            df = df.assign(date=s).pivot_table(index="date", columns="channel", values=val, aggfunc="last")
        else:
            df = df.set_index(s).drop(columns=[c for c in ["date"] if c in df.columns])
    else:
        df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)

    # Sort & unique index
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["__".join(map(str, tup)) for tup in df.columns.to_flat_index()]
    if not df.columns.is_unique:
        df = df.groupby(level=0, axis=1).last()

    if fill_inactive:
        df = df.fillna(0.0)

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", required=False, default=None, help="Dynamic episode channels (Tier 2)")
    ap.add_argument("--base_shocks", required=True, help="Baseline shocks (A4, e.g., shock_indices.csv)")
    ap.add_argument("--returns", required=True, help="Returns CSV")
    ap.add_argument("--window", type=int, default=90)
    ap.add_argument("--lambda_decay", type=float, default=0.98)
    ap.add_argument("--alpha", type=float, default=1e-3, help="ElasticNet regularization strength")
    ap.add_argument("--l1_ratio", type=float, default=0.3, help="ElasticNet mixing [0=ridge,1=lasso]")
    ap.add_argument("--outdir", required=True)

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    S_base_raw = _read_df(args.base_shocks)
    S_chan_raw = _read_df(args.channels) if args.channels else None
    S_base = _harmonize(S_base_raw, expect_long_channels=False, fill_inactive=False)
    S_chan = _harmonize(S_chan_raw, expect_long_channels=True,  fill_inactive=True)

    rets = _read_returns(args.returns)
    rets_idx = pd.DatetimeIndex(sorted(rets['date'].unique()))
    idx = rets_idx

    S_base = S_base.reindex(idx).ffill()
    S_chan = S_chan.reindex(idx).fillna(0.0)
    S = pd.concat([S_base.loc[idx], S_chan.loc[idx]], axis=1, sort=False).sort_index()

    all_dates = S.index
    betas_rec, rtilde_rec = [], []

    for tk, grp in rets.groupby("ticker"):
        # align to S dates
        y_all = grp.set_index("date")["ret"].reindex(all_dates)
        X_all = S.reindex(all_dates)
        
        for t_idx in range(len(all_dates)):
            t = all_dates[t_idx]
            t_end = t
            t_start = t_end - pd.tseries.offsets.BDay(args.window-1)

            X_win = X_all.loc[t_start:t_end].shift(1)
            y_win = y_all.loc[t_start:t_end]
            df_win = pd.concat([y_win, X_win], axis=1).dropna(how="any")

            if len(df_win) < max(30, int(0.7*args.window)):
                continue

            y = df_win.iloc[:,0].values
            X = df_win.iloc[:,1:].values
            
            # Standardize features inside window
            scaler = StandardScaler(with_mean=True, with_std=True)
            Xs = scaler.fit_transform(X)

            sd = scaler.scale_
            keep = sd > 1e-12
            Xs_fit = Xs[:, keep]

            if Xs_fit.shape[1] == 0:
                continue

            # Exponential decay weights
            n = len(y)
            w = np.array([args.lambda_decay**(n-1-i) for i in range(n)], dtype=float)

            enet = ElasticNet(alpha=args.alpha, l1_ratio=args.l1_ratio,
                              fit_intercept=True, random_state=42, max_iter=2000)
            enet.fit(Xs_fit, y, sample_weight=w)

            # Back-transform on kept columns
            mu = scaler.mean_[keep]
            sd = scaler.scale_[keep]
            beta_std = enet.coef_
            beta_kept = beta_std / (sd + 1e-12)
            alpha = float(enet.intercept_ - np.dot(mu, beta_kept))

            # Record betas
            beta_full = np.zeros(len(S.columns), dtype=float)
            beta_full[keep] = beta_kept

            rec = {"date": t, "ticker": tk, "alpha": alpha}
            for c, b in zip(S.columns, beta_full):
                rec[c] = b
            betas_rec.append(rec)

            # Prediction using full S_t but only the kept columns contribute
            st = S.loc[t, S.columns].values
            rtilde_wo = float(np.dot(st, beta_full))
            rtilde_w  = alpha + rtilde_wo
            rtilde_rec.append({"date": t + BDay(1), "ticker": tk,
                               "r_tilde": rtilde_wo, "r_tilde_with_alpha": rtilde_w})

    betas = pd.DataFrame(betas_rec)
    if not betas.empty:
        betas["date"] = pd.to_datetime(betas["date"]).dt.normalize()
        betas = betas.set_index(["date","ticker"]).sort_index()
    rtilde = pd.DataFrame(rtilde_rec)
    if not rtilde.empty:
        rtilde["date"] = pd.to_datetime(rtilde["date"]).dt.normalize()
        rtilde = rtilde.sort_values(["date","ticker"]).reset_index(drop=True)

    betas.to_csv(os.path.join(args.outdir, "betas_online.csv"))
    rtilde.to_csv(os.path.join(args.outdir, "r_tilde_online.csv"))

if __name__ == "__main__":
    main()


