#!/usr/bin/env python3
"""SupplyGuard Dataset Acquisition & Verification Script.

Downloads and validates the raw SCRM time series dataset according to
Plan A Phase 0 specifications.
"""
import os
import hashlib
import requests
import pandas as pd
import numpy as np

# Pinned mirror commit hash from GitHub webintellectual/Supply-Chain-Stability-Classifier (main branch)
PINNED_COMMIT = "698ec038f7410a426655d73bff990699ead8808c"
DATA_URL = (
    f"https://raw.githubusercontent.com/webintellectual/Supply-Chain-Stability-Classifier/"
    f"{PINNED_COMMIT}/Dataset/SCRM_timeSeries_2018_train.csv"
)

RAW_DIR = os.path.join("data", "raw")
RAW_PATH = os.path.join(RAW_DIR, "SCRM_timeSeries_2018_train.csv")
REQUIRED_COLS = [
    "Timestamp",
    "RI_Supplier1",
    "RI_Manufacturer1",
    "RI_Distributor1",
    "RI_Retailer1",
    "Total_Cost",
]
DATE_FORMAT = "%m/%d/%Y %I:%M:%S %p"


def download_dataset(url: str, dest_path: str):
    """Download the raw dataset if not present locally."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        print(f"[INFO] Raw dataset already exists at {dest_path}")
        return

    print(f"[INFO] Downloading dataset from {url} ...")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024 * 1024  # 1 MB
    downloaded = 0

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"  Downloaded {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({percent:.1f}%)", end="\r")
    print(f"\n[INFO] Download completed: {dest_path}")


def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def compute_max_consecutive_nulls(series: pd.Series) -> int:
    """Compute maximum run length of consecutive NaNs."""
    is_null = series.isna().astype(int)
    if is_null.sum() == 0:
        return 0
    # Group consecutive identical boolean values
    blocks = (is_null != is_null.shift()).cumsum()
    runs = is_null.groupby(blocks).sum()
    return int(runs.max())


def validate_dataset(filepath: str):
    """Validate dataset structure, timestamps, gaps, and null profiles."""
    print("=" * 70)
    print("DATASET INTEGRITY & PROFILE REPORT (Plan A Phase 0)")
    print("=" * 70)

    # 1. Checksum
    sha = compute_sha256(filepath)
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"File Path        : {filepath}")
    print(f"File Size        : {file_size_mb:.2f} MB")
    print(f"SHA-256 Checksum : {sha}")
    print(f"Pinned Commit    : {PINNED_COMMIT}")

    # 2. Schema check
    df = pd.read_csv(filepath)
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    assert not missing_cols, f"Missing required columns: {missing_cols}"
    print(f"Row Count        : {len(df):,} (Expected: 649,999)")

    # 3. Null Profiling (Never crash on real ~5% nulls!)
    null_shares = df[REQUIRED_COLS].isna().mean()
    print("\nNull Value Shares:")
    for col, share in null_shares.items():
        print(f"  {col:<18}: {share * 100:6.2f}% ({df[col].isna().sum():,} nulls)")

    max_null_share = df[REQUIRED_COLS[1:]].isna().mean().max()
    # Plan A Policy: Real data has ~5.1% RI_Distributor1 and ~5.5% Total_Cost. Assert bounded < 10%
    assert max_null_share < 0.10, f"Unusually high null share detected: {max_null_share:.2%}"

    # Null run lengths
    max_null_dist = compute_max_consecutive_nulls(df["RI_Distributor1"])
    max_null_cost = compute_max_consecutive_nulls(df["Total_Cost"])
    print(f"\nMax Consecutive Null Runs:")
    print(f"  RI_Distributor1   : {max_null_dist:,} rows")
    print(f"  Total_Cost        : {max_null_cost:,} rows")

    # 4. Timestamps
    ts = pd.to_datetime(df["Timestamp"], format=DATE_FORMAT, errors="coerce")
    unparseable = int(ts.isna().sum())
    print(f"\nTimestamp Diagnostics:")
    print(f"  Format Applied    : {DATE_FORMAT}")
    print(f"  Unparseable Count : {unparseable} (must be 0)")
    assert unparseable == 0, f"Detected {unparseable} unparseable timestamps!"

    print(f"  Time Span         : {ts.min()} -> {ts.max()}")
    print(f"  Sorted Monotonic  : {ts.is_monotonic_increasing}")
    dup_stamps = int(ts.duplicated().sum())
    print(f"  Duplicate Stamps  : {dup_stamps:,}")

    # 5. Sampling Intervals and Gap Diagnostics
    ts_sorted = ts.sort_values().reset_index(drop=True)
    deltas = ts_sorted.diff().dropna()
    print(f"\nSampling Interval Statistics:")
    print(f"  Median Step       : {deltas.median()}")
    print(f"  Min Step          : {deltas.min()}")
    print(f"  Max Gap           : {deltas.max()}")

    # Top 5 largest gaps
    largest_gaps = deltas.nlargest(5)
    print("\nTop 5 Largest Time Gaps:")
    for idx, gap in largest_gaps.items():
        t_before = ts_sorted.iloc[idx - 1]
        t_after = ts_sorted.iloc[idx]
        print(f"  Gap: {gap} (from {t_before} to {t_after})")

    print("=" * 70)
    print("STATUS: Dataset validation PASSED all Phase 0 criteria.")
    print("=" * 70)
    return {
        "sha256": sha,
        "rows": len(df),
        "span": f"{ts.min()} to {ts.max()}",
        "median_step": str(deltas.median()),
        "max_gap": str(deltas.max()),
        "duplicate_stamps": dup_stamps,
        "max_null_dist": max_null_dist,
    }


def main():
    download_dataset(DATA_URL, RAW_PATH)
    validate_dataset(RAW_PATH)


if __name__ == "__main__":
    main()
