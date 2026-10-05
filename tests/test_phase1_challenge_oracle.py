"""SupplyGuard Phase 1 Empirical Verification Oracle.
Independent verification harness written by Challenger 1 to stress-test
empirical calculations and findings from notebooks/01_EDA.ipynb.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import acf, grangercausalitytests

DATA_PATH = Path("data/raw/SCRM_timeSeries_2018_train.csv")
DATE_COL = "Timestamp"
DATE_FORMAT = "%m/%d/%Y %I:%M:%S %p"
GAP_MAX = pd.Timedelta(minutes=6)
NODE_TARGET_COLS = ["RI_Supplier1", "RI_Manufacturer1", "RI_Distributor1", "RI_Retailer1"]
FEATURE_COLS = NODE_TARGET_COLS + ["Total_Cost"]

def load_and_preprocess_data():
    print(f"Loading {DATA_PATH}...")
    df_raw = pd.read_csv(DATA_PATH)
    raw_len = len(df_raw)
    df_raw[DATE_COL] = pd.to_datetime(df_raw[DATE_COL], format=DATE_FORMAT)
    df = df_raw.sort_values(DATE_COL, kind="stable").reset_index(drop=True)
    num_dups = df.duplicated(subset=[DATE_COL]).sum()
    df = df.drop_duplicates(subset=[DATE_COL], keep="first").reset_index(drop=True)
    clean_len = len(df)
    
    # Gap segmentation
    time_diffs = df[DATE_COL].diff()
    gap_mask = time_diffs > GAP_MAX
    df["segment_id"] = gap_mask.cumsum()
    num_gaps = int(gap_mask.sum())
    num_segments = df["segment_id"].nunique()
    
    seg_lengths = df.groupby("segment_id").size()
    largest_seg_id = int(seg_lengths.idxmax())
    max_seg_len = int(seg_lengths.max())
    
    print(f"Raw rows: {raw_len:,}, Duplicates dropped: {num_dups:,}, Clean rows: {clean_len:,}")
    print(f"Gaps > 6 min: {num_gaps:,}, Contiguous segments: {num_segments:,}")
    print(f"Largest segment ID: {largest_seg_id}, length: {max_seg_len:,} rows")
    
    return df, largest_seg_id

def compute_persistence_method_a(df, horizons):
    """Method A: Worker notebook method (group[col].dropna().values)."""
    results = {}
    for h_name, h_step in horizons.items():
        y_true_all = {col: [] for col in NODE_TARGET_COLS}
        y_pred_all = {col: [] for col in NODE_TARGET_COLS}
        
        for seg_id, group in df.groupby("segment_id"):
            if len(group) > h_step + 1:
                for col in NODE_TARGET_COLS:
                    series_vals = group[col].dropna().values
                    if len(series_vals) > h_step:
                        y_true_all[col].append(series_vals[h_step:])
                        y_pred_all[col].append(series_vals[:-h_step])
        
        scores = {}
        for col in NODE_TARGET_COLS:
            yt = np.concatenate(y_true_all[col])
            yp = np.concatenate(y_pred_all[col])
            scores[col] = float(r2_score(yt, yp))
        scores["Mean"] = float(np.mean([scores[col] for col in NODE_TARGET_COLS]))
        results[h_step] = scores
    return results

def compute_persistence_method_b(df, horizons):
    """Method B: Strict temporal step lag without dropping NaNs beforehand.
    Aligns group[col].values[h:] with group[col].values[:-h] and masks invalid pairs.
    """
    results = {}
    for h_name, h_step in horizons.items():
        y_true_all = {col: [] for col in NODE_TARGET_COLS}
        y_pred_all = {col: [] for col in NODE_TARGET_COLS}
        
        for seg_id, group in df.groupby("segment_id"):
            if len(group) > h_step:
                for col in NODE_TARGET_COLS:
                    vals = group[col].values
                    yt = vals[h_step:]
                    yp = vals[:-h_step]
                    valid = ~(np.isnan(yt) | np.isnan(yp))
                    if np.sum(valid) > 0:
                        y_true_all[col].append(yt[valid])
                        y_pred_all[col].append(yp[valid])
        
        scores = {}
        for col in NODE_TARGET_COLS:
            yt = np.concatenate(y_true_all[col])
            yp = np.concatenate(y_pred_all[col])
            scores[col] = float(r2_score(yt, yp))
        scores["Mean"] = float(np.mean([scores[col] for col in NODE_TARGET_COLS]))
        results[h_step] = scores
    return results

def compute_persistence_method_c(df, horizons):
    """Method C: Segment with bounded ffill(limit=5) and bfill(limit=5) as per Plan A."""
    results = {}
    for h_name, h_step in horizons.items():
        y_true_all = {col: [] for col in NODE_TARGET_COLS}
        y_pred_all = {col: [] for col in NODE_TARGET_COLS}
        
        for seg_id, group in df.groupby("segment_id"):
            if len(group) > h_step:
                filled = group[NODE_TARGET_COLS].ffill(limit=5).bfill(limit=5)
                for col in NODE_TARGET_COLS:
                    vals = filled[col].values
                    yt = vals[h_step:]
                    yp = vals[:-h_step]
                    valid = ~(np.isnan(yt) | np.isnan(yp))
                    if np.sum(valid) > 0:
                        y_true_all[col].append(yt[valid])
                        y_pred_all[col].append(yp[valid])
        
        scores = {}
        for col in NODE_TARGET_COLS:
            yt = np.concatenate(y_true_all[col])
            yp = np.concatenate(y_pred_all[col])
            scores[col] = float(r2_score(yt, yp))
        scores["Mean"] = float(np.mean([scores[col] for col in NODE_TARGET_COLS]))
        results[h_step] = scores
    return results

def compute_naive_global_persistence(df, horizons):
    """Stress test: Naive shift across the entire dataset ignoring gaps!"""
    results = {}
    for h_name, h_step in horizons.items():
        scores = {}
        for col in NODE_TARGET_COLS:
            yt = df[col].iloc[h_step:].values
            yp = df[col].iloc[:-h_step].values
            valid = ~(np.isnan(yt) | np.isnan(yp))
            scores[col] = float(r2_score(yt[valid], yp[valid]))
        scores["Mean"] = float(np.mean([scores[col] for col in NODE_TARGET_COLS]))
        results[h_step] = scores
    return results

def compute_cross_correlations(df, largest_seg_id, max_lags=20):
    """Check contemporaneous correlation and lead/lag cross-correlation."""
    print("\n--- Correlation Analysis ---")
    # Full dataset contemporaneous correlation
    corr_full = df[FEATURE_COLS].corr()
    print("Full Dataset Contemporaneous Correlation:")
    print(corr_full.round(4))
    
    # Largest segment contemporaneous correlation (raw and with ffill)
    largest_seg = df[df["segment_id"] == largest_seg_id].copy().reset_index(drop=True)
    corr_largest_raw = largest_seg[FEATURE_COLS].corr()
    print("\nLargest Segment (Raw) Contemporaneous Correlation:")
    print(corr_largest_raw.round(4))
    
    largest_seg_filled = largest_seg.copy()
    largest_seg_filled[FEATURE_COLS] = largest_seg_filled[FEATURE_COLS].ffill(limit=5).bfill(limit=5)
    corr_largest_filled = largest_seg_filled[FEATURE_COLS].corr()
    print("\nLargest Segment (Filled) Contemporaneous Correlation:")
    print(corr_largest_filled.round(4))
    
    # Lead-lag cross-correlation on largest segment (filled)
    adjacent_pairs = [
        ("RI_Supplier1", "RI_Manufacturer1", "S -> M"),
        ("RI_Manufacturer1", "RI_Distributor1", "M -> D"),
        ("RI_Distributor1", "RI_Retailer1", "D -> R"),
        ("RI_Supplier1", "RI_Distributor1", "S -> D (Physical index order)"),
        ("RI_Manufacturer1", "RI_Retailer1", "M -> R (Non-adjacent)")
    ]
    
    lags = list(range(-max_lags, max_lags + 1))
    cross_corr_results = {}
    
    for col_x, col_y, label in adjacent_pairs:
        x_vals = largest_seg_filled[col_x].values
        y_vals = largest_seg_filled[col_y].values
        corrs = []
        for k in lags:
            if k > 0:
                c = np.corrcoef(x_vals[:-k], y_vals[k:])[0, 1]
            elif k < 0:
                c = np.corrcoef(x_vals[-k:], y_vals[:k])[0, 1]
            else:
                c = np.corrcoef(x_vals, y_vals)[0, 1]
            corrs.append(c)
        corrs = np.array(corrs)
        max_c = np.max(np.abs(corrs))
        argmax_k = lags[np.argmax(np.abs(corrs))]
        cross_corr_results[label] = {
            "lags": lags,
            "corrs": corrs,
            "max_abs_corr": max_c,
            "peak_lag": argmax_k,
            "lag_0_corr": corrs[lags.index(0)]
        }
        print(f"Pair '{label}': max |r| = {max_c:.4f} at lag {argmax_k}, lag 0 r = {corrs[lags.index(0)]:.4f}")
        
    return corr_full, corr_largest_raw, corr_largest_filled, cross_corr_results

def run_all():
    df, largest_seg_id = load_and_preprocess_data()
    
    horizons = {
        "2 min (H=1)": 1,
        "10 min (H=5)": 5,
        "20 min (H=10)": 10,
        "60 min (H=30)": 30
    }
    
    print("\n================== PERSISTENCE R² EVALUATION ==================")
    print("\n--- Method A: Worker Notebook dropna() ---")
    r2_a = compute_persistence_method_a(df, horizons)
    for h, scores in r2_a.items():
        print(f"H={h:2d} ({(h*2):2d} min): S={scores['RI_Supplier1']:.4f}, M={scores['RI_Manufacturer1']:.4f}, D={scores['RI_Distributor1']:.4f}, R={scores['RI_Retailer1']:.4f} | Mean={scores['Mean']:.4f}")

    print("\n--- Method B: Strict Step Lag (NaN-masked, no collapse) ---")
    r2_b = compute_persistence_method_b(df, horizons)
    for h, scores in r2_b.items():
        print(f"H={h:2d} ({(h*2):2d} min): S={scores['RI_Supplier1']:.4f}, M={scores['RI_Manufacturer1']:.4f}, D={scores['RI_Distributor1']:.4f}, R={scores['RI_Retailer1']:.4f} | Mean={scores['Mean']:.4f}")

    print("\n--- Method C: Bounded ffill/bfill(limit=5) ---")
    r2_c = compute_persistence_method_c(df, horizons)
    for h, scores in r2_c.items():
        print(f"H={h:2d} ({(h*2):2d} min): S={scores['RI_Supplier1']:.4f}, M={scores['RI_Manufacturer1']:.4f}, D={scores['RI_Distributor1']:.4f}, R={scores['RI_Retailer1']:.4f} | Mean={scores['Mean']:.4f}")

    print("\n--- Stress Test: Naive Global Shift (Gap-unaware) ---")
    r2_naive = compute_naive_global_persistence(df, horizons)
    for h, scores in r2_naive.items():
        print(f"H={h:2d} ({(h*2):2d} min): S={scores['RI_Supplier1']:.4f}, M={scores['RI_Manufacturer1']:.4f}, D={scores['RI_Distributor1']:.4f}, R={scores['RI_Retailer1']:.4f} | Mean={scores['Mean']:.4f}")

    corr_full, corr_l_raw, corr_l_filled, cross_corr = compute_cross_correlations(df, largest_seg_id)

    print("\n================== GATE 1 HYPOTHESIS CHECKS ==================")
    # Check 1: Manufacturer R² >= 0.99 at 2 min
    print("\n[Check 1: Manufacturer R² >= 0.99 at 2 min]")
    for name, r2_dict in [("Method A", r2_a), ("Method B", r2_b), ("Method C", r2_c)]:
        val = r2_dict[1]["RI_Manufacturer1"]
        passed = val >= 0.99
        print(f"  {name}: {val:.4f} >= 0.99 -> {passed}")

    # Check 2: Max Node R² < 0.99 at 10 min
    print("\n[Check 2: Max Node R² < 0.99 at 10 min across all nodes]")
    for name, r2_dict in [("Method A", r2_a), ("Method B", r2_b), ("Method C", r2_c)]:
        max_val = max(r2_dict[5][col] for col in NODE_TARGET_COLS)
        passed = max_val < 0.99
        print(f"  {name}: max={max_val:.4f} < 0.99 -> {passed} (M={r2_dict[5]['RI_Manufacturer1']:.4f}, S={r2_dict[5]['RI_Supplier1']:.4f}, D={r2_dict[5]['RI_Distributor1']:.4f}, R={r2_dict[5]['RI_Retailer1']:.4f})")

    # Check 3: Cross-correlation between adjacent echelons < 0.40
    print("\n[Check 3: Cross-correlations between adjacent echelons < 0.40]")
    for pair in ["S -> M", "M -> D", "D -> R"]:
        info = cross_corr[pair]
        passed = info["max_abs_corr"] < 0.40
        print(f"  Pair {pair}: max_abs={info['max_abs_corr']:.4f} < 0.40 -> {passed} (peak at lag {info['peak_lag']}, lag 0={info['lag_0_corr']:.4f})")

if __name__ == "__main__":
    run_all()
