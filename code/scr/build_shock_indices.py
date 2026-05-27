# build_shock_indices.py

import argparse, os, warnings
from typing import Optional
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

DATE_CANDS = ["date","Date","DATE","observation_date","time","Time","TIME","dt",
              "timestamp","Timestamp","month","Month","MONTH","DATE "]
VALUE_CANDS = ["value","Value","VALUE","close","Close","PRICE","Price","price",
               "INDEX","Index","index","VIX","VIXCLS"]

def ensure_dir(fp: str):
    abs_fp = os.path.abspath(fp)
    os.makedirs(os.path.dirname(abs_fp), exist_ok=True)

# Calendar from returns.csv 
def load_returns_calendar(path: str) -> pd.DatetimeIndex:
    df = pd.read_csv(path)
    dcol = None
    for c in df.columns:
        if c.lower() == "date":
            dcol = c; break
    if dcol is None:
        for c in df.columns:
            try:
                if pd.to_datetime(df[c], errors="coerce").notna().any():
                    dcol = c; break
            except Exception:
                continue
    if dcol is None:
        raise ValueError("Could not find a date column in returns.csv")
    dates = pd.to_datetime(df[dcol], errors="coerce").dropna().sort_values().unique()
    return pd.DatetimeIndex(dates)

# Loaders
def _pick_date_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if c in DATE_CANDS:
            return c
    for c in df.columns:
        s = pd.to_datetime(df[c], errors="coerce")
        if s.notna().sum() > max(5, int(0.3*len(s))):
            return c
    return None

def _pick_value_col(df: pd.DataFrame, prefer: Optional[str]=None) -> Optional[str]:
    cols = list(df.columns)
    if prefer and prefer in cols:
        return prefer
    for cand in VALUE_CANDS:
        if cand in cols:
            return cand
    for c in cols:
        if c in DATE_CANDS: continue
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    if len(cols) >= 2:
        return cols[1]
    return None

def load_series_flexible(fp: str, calendar: pd.DatetimeIndex, prefer_value: Optional[str]=None) -> pd.Series:
    df = pd.read_csv(fp)
    dcol = _pick_date_col(df)
    if dcol is not None:
        vcol = _pick_value_col(df, prefer=prefer_value) or (df.columns[1] if len(df.columns) >= 2 else None)
        if vcol is None:
            raise ValueError(f"No value column found in {fp}")
        s = pd.to_numeric(df[vcol], errors="coerce")
        idx = pd.to_datetime(df[dcol], errors="coerce")
        ser = pd.Series(s.values, index=idx, name=os.path.basename(fp).split(".")[0])
        ser = ser.dropna()
        ser = ser[~ser.index.duplicated(keep="last")].sort_index()
        return ser.reindex(calendar).ffill()
    
    vcol = _pick_value_col(df)
    if vcol is None:
        raise ValueError(f"Could not find a value column in {fp}")
    vals = pd.to_numeric(df[vcol], errors="coerce")
    if len(vals) >= len(calendar):
        vals = vals.iloc[-len(calendar):].reset_index(drop=True)
        return pd.Series(vals.values, index=calendar, name=os.path.basename(fp).split(".")[0])
    pad = len(calendar) - len(vals)
    return pd.Series([np.nan]*pad + list(vals.values), index=calendar, name=os.path.basename(fp).split(".")[0])

def load_gpr_from_export(fp: str) -> pd.Series:
    df = pd.read_csv(fp)
    dcol = None
    for c in ["month", "date", "Date", "observation_date"]:
        if c in df.columns:
            dcol = c; break
    if dcol is None:
        raise ValueError("Could not find a date/month column in data_gpr_export.csv.")
    
    idx = pd.to_datetime(df[dcol], errors="coerce")
    prefer_cols = ["GPR","GPR_new","GPRT","GPRA","GPRH"]
    val_col = None
    for c in prefer_cols:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            val_col = c; break
    if val_col is None:
        for c in df.columns:
            if c.lower() in ["month","date","year","unnamed: 0"]:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                val_col = c; break
    if val_col is None:
        raise ValueError("Could not detect a numeric GPR column in data_gpr_export.csv.")
    
    ser = pd.Series(pd.to_numeric(df[val_col], errors="coerce").values, index=idx, name="GPR")
    ser = ser.dropna()
    ser = ser[~ser.index.duplicated(keep="last")].sort_index()
    ser.index = ser.index.to_period('M').to_timestamp()
    return ser

# Transforms
def delta(s: pd.Series) -> pd.Series:
    return s.diff()

def delta_bps(levels: pd.Series) -> pd.Series:
    return levels.diff() * 100.0  # percent → bps

def log_return(price: pd.Series) -> pd.Series:
    return np.log(price).diff()

def rolling_z(signal: pd.Series, window: int, min_periods: int) -> pd.Series:
    mu = signal.rolling(window=window, min_periods=min_periods).mean()
    sd = signal.rolling(window=window, min_periods=min_periods).std(ddof=0)
    z = (signal - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan)

# Main build
def main():
    ap = argparse.ArgumentParser(description="Build macro shocks aligned to returns.csv calendar (no lookahead).")
    ap.add_argument("--returns_csv", required=True, help="Path to returns.csv with a 'date' column")
    ap.add_argument("--macro_dir", required=True, help="Directory with macro CSVs")
    ap.add_argument("--out_csv", required=True, help="CSV output path")
    ap.add_argument("--window", type=int, default=252, help="Rolling window for z-scores (default 252)")
    ap.add_argument("--min_periods", type=int, default=60, help="Min periods for rolling z (default 60)")
    args = ap.parse_args()

    if args.out_csv:
        args.out_csv = os.path.abspath(args.out_csv)
    warnings.simplefilter("ignore", category=FutureWarning)

    # Warm-up calendar so rolling stats are stable
    returns_cal = load_returns_calendar(args.returns_csv)
    warmup_days = max(400, args.window + 50)
    start_date = returns_cal.min()
    warmup_cal = pd.bdate_range(end=start_date - pd.tseries.offsets.BDay(1), periods=warmup_days)
    cal = warmup_cal.union(returns_cal)

    def path_for(name: str) -> Optional[str]:
        for pat in [name, name.lower(), name.upper()]:
            ext = ".csv"
            p = os.path.join(args.macro_dir, pat+ext) if not pat.endswith(ext) else os.path.join(args.macro_dir, pat)
            if os.path.exists(p):
                return p
        return None

    panel = pd.DataFrame(index=cal)

    # HY OAS → Δ(bps) → z
    p = path_for("BAMLH0A0HYM2")
    if p:
        hy = load_series_flexible(p, cal)
        panel["hy_oas_dzbps"] = rolling_z(delta_bps(hy), args.window, args.min_periods)

    # BAA10Y (corporate 10y) → Δ(bps) → z
    p = path_for("BAA10Y")
    if p:
        baa = load_series_flexible(p, cal)
        panel["baa10y_dzbps"] = rolling_z(delta_bps(baa), args.window, args.min_periods)

    # DGS10 → Δbps z-score
    p = path_for("dgs10_dzbps")
    if p:
        dgs10_dz = load_series_flexible(p, cal)
        panel["dgs10_dzbps"] = pd.to_numeric(dgs10_dz, errors="coerce")
    else:
        p = path_for("DGS10")
        if p:
            dgs10_lvl = load_series_flexible(p, cal)
            panel["dgs10_dzbps"] = rolling_z(delta_bps(dgs10_lvl), args.window, args.min_periods)

    # USD broad (DTWEXBGS) → log return → z
    p = path_for("DTWEXBGS")
    if p:
        usd = load_series_flexible(p, cal)
        panel["usd_lret_z"] = rolling_z(log_return(usd), args.window, args.min_periods)

    # Oil (WTI/Brent) → log return → z (+ energy avg)
    p_wti = path_for("DCOILWTICO")
    p_brent = path_for("DCOILBRENTEU")
    wti_lret = brent_lret = None
    if p_wti:
        wti = load_series_flexible(p_wti, cal)
        wti_lret = log_return(wti)
        panel["wti_lret_z"] = rolling_z(wti_lret, args.window, args.min_periods)
    if p_brent:
        brent = load_series_flexible(p_brent, cal)
        brent_lret = log_return(brent)
        panel["brent_lret_z"] = rolling_z(brent_lret, args.window, args.min_periods)
    if wti_lret is not None and brent_lret is not None:
        panel["energy_lret_z"] = rolling_z((wti_lret + brent_lret) / 2.0, args.window, args.min_periods)

    # VIX (level) → Δ → z
    p = path_for("VIX") or path_for("VIXCLS")
    if p:
        vix = load_series_flexible(p, cal)
        panel["vix_dz"] = rolling_z(delta(vix), args.window, args.min_periods)

    # 2s10s curve → Δ(bps) → z
    p = path_for("dgs_2y_10y")
    if p:
        curve = load_series_flexible(p, cal, prefer_value="Spread_10Y_2Y")
        panel["curve2s10s_dzbps"] = rolling_z(delta_bps(curve), args.window, args.min_periods)

    # EPU (daily or monthly) → FFill→daily, shift +1d (publication), Δ → z
    p = path_for("EPU")
    if p:
        epu_raw = load_series_flexible(p, cal)
        epu_shifted = epu_raw.shift(1)
        panel["epu_dz"] = rolling_z(delta(epu_shifted), args.window, args.min_periods)

    # GPR monthly from data_gpr_export → FFill→daily, shift +1d (publication), Δ → z
    p = path_for("data_gpr_export")
    if p:
        gpr_m = load_gpr_from_export(p)
        gpr_d = gpr_m.reindex(cal, method="ffill").shift(1)  # avoid same-day leak
        panel["gpr_dz"] = rolling_z(delta(gpr_d), args.window, args.min_periods)

    # Equity proxy: avg(SPX, NASDAQ) log-return -> z
    p = path_for("spx_nasdaq")
    if p:
        try:
            spx = load_series_flexible(p, cal, prefer_value="SPX")
            ndx = load_series_flexible(p, cal, prefer_value="NASDAQ")
            if spx.notna().sum() > 0 and ndx.notna().sum() > 0:
                eq_lret = (np.log(spx).diff() + np.log(ndx).diff()) / 2.0
                panel["equity_lret_z"] = rolling_z(eq_lret, args.window, args.min_periods)
            else:
                print("[warn] spx_nasdaq.csv found but SPX or NASDAQ series is empty; skipping equity_lret_z.")
        except Exception as e:
            print(f"[warn] Failed to build equity_lret_z from spx_nasdaq.csv: {e}")


    # One alignment shift (no look-ahead)
    lag_cols = ["hy_oas_dzbps","dgs10_dzbps","curve2s10s_dzbps",
            "usd_lret_z","wti_lret_z","brent_lret_z","energy_lret_z","vix_dz",
            "equity_lret_z", "baa10y_dzbps"]
    for c in lag_cols:
        if c in panel.columns:
            panel[c] = panel[c].shift(1)

    panel = panel.reindex(returns_cal)

    # Save
    if args.out_csv:
        ensure_dir(args.out_csv)
        dfcsv = panel.copy()
        dfcsv.index.name = "date"
        dfcsv.reset_index().to_csv(args.out_csv, index=False, date_format="%Y-%m-%d")

if __name__ == "__main__":
    main()

