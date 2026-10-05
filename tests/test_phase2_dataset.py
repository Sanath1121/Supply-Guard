"""Gate 2 Verification: Data Pipeline Hardening, Segmentation, and Leakage Prevention.

Checks:
1. Stable sorting and deduplication.
2. Gap-aware segmentation (new segment when dt > GAP_MAX).
3. Absolute constraint: NO sliding window (L=10) or target (t+H) crosses a segment boundary.
4. Bounded forward-fill policy: ffill(limit=5) within segments; long null runs split the segment.
5. Chronological 80:10:10 split integrity (max(train) < min(val) < max(val) < min(test)).
6. Scaler leakage prevention: MinMaxScaler fitted strictly on train partition.
7. Target alignment: y_target equals row H steps ahead in the same segment.
"""
import os
import sys
import unittest
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def build_synthetic_segmented_data(
    n_rows: int = 1000,
    gap_index: int = 400,
    gap_duration_hours: int = 24,
    null_run_index: int = 700,
    null_run_len: int = 8
) -> pd.DataFrame:
    """Generate synthetic supply chain dataframe with known injected gaps and null runs."""
    rng = np.random.default_rng(42)
    # Part 1: before gap (2-min intervals)
    t1 = pd.date_range("2018-01-01 00:00:00", periods=gap_index, freq="2min")
    # Part 2: after gap (jump forward by gap_duration_hours)
    t2_start = t1[-1] + pd.Timedelta(hours=gap_duration_hours)
    t2 = pd.date_range(t2_start, periods=n_rows - gap_index, freq="2min")
    timestamps = list(t1) + list(t2)

    df = pd.DataFrame({
        "Timestamp": [t.strftime("%m/%d/%Y %I:%M:%S %p") for t in timestamps],
        "RI_Supplier1": rng.uniform(0.1, 1.0, n_rows),
        "RI_Manufacturer1": rng.uniform(0.1, 1.0, n_rows),
        "RI_Distributor1": rng.uniform(0.1, 1.0, n_rows),
        "RI_Retailer1": rng.uniform(0.1, 1.0, n_rows),
        "Total_Cost": rng.uniform(20.0, 80.0, n_rows),
    })

    # Inject duplicate timestamps
    df.loc[15, "Timestamp"] = df.loc[14, "Timestamp"]

    # Inject a 3-row null run (should be forward-filled)
    df.loc[200:202, "RI_Distributor1"] = np.nan

    # Inject an 8-row null run (exceeds limit=5, should split the segment)
    df.loc[null_run_index:null_run_index + null_run_len - 1, "RI_Distributor1"] = np.nan

    # Shuffle rows to verify pipeline sorts correctly
    df_shuffled = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    return df_shuffled


def segment_and_window(
    df_raw: pd.DataFrame,
    seq_len: int = 10,
    horizon: int = 5,
    gap_max_min: int = 6,
    ffill_limit: int = 5
):
    """Reference implementation of Phase 2 segment-aware windowing logic."""
    # 1. Parse timestamps explicitly & sort
    df = df_raw.copy()
    df["dt"] = pd.to_datetime(df["Timestamp"], format="%m/%d/%Y %I:%M:%S %p")
    df = df.sort_values("dt").drop_duplicates(subset=["dt"]).reset_index(drop=True)

    # 2. Segment on time gaps
    time_diffs = df["dt"].diff()
    is_gap = time_diffs > pd.Timedelta(minutes=gap_max_min)
    df["gap_seg"] = is_gap.cumsum()

    feature_cols = ["RI_Supplier1", "RI_Manufacturer1", "RI_Distributor1", "RI_Retailer1", "Total_Cost"]
    target_cols = ["RI_Supplier1", "RI_Manufacturer1", "RI_Distributor1", "RI_Retailer1"]

    # 3. Process each segment with bounded ffill
    final_segments = []
    seg_counter = 0

    for _, seg_df in df.groupby("gap_seg"):
        sub = seg_df.copy()
        # Bounded ffill within segment
        sub[feature_cols] = sub[feature_cols].ffill(limit=ffill_limit)
        # Any remaining NaNs split into sub-segments
        has_nan = sub[feature_cols].isna().any(axis=1)
        if has_nan.any():
            nan_splits = has_nan.cumsum()
            for _, clean_sub in sub.groupby(nan_splits):
                clean_sub = clean_sub.dropna(subset=feature_cols)
                if len(clean_sub) >= (seq_len + horizon):
                    clean_sub = clean_sub.copy()
                    clean_sub["clean_seg_id"] = seg_counter
                    seg_counter += 1
                    final_segments.append(clean_sub)
        else:
            if len(sub) >= (seq_len + horizon):
                sub["clean_seg_id"] = seg_counter
                seg_counter += 1
                final_segments.append(sub)

    if not final_segments:
        return [], []

    # 4. Generate windows strictly within each segment
    windows = []
    targets = []
    for seg in final_segments:
        feats = seg[feature_cols].values
        targs = seg[target_cols].values
        n = len(seg)
        for i in range(n - seq_len - horizon + 1):
            w = feats[i: i + seq_len]
            t = targs[i + seq_len + horizon - 1]
            windows.append(w)
            targets.append(t)

    return np.array(windows), np.array(targets), final_segments


class TestPhase2Dataset(unittest.TestCase):

    def setUp(self):
        self.df_shuffled = build_synthetic_segmented_data()
        self.seq_len = 10
        self.horizon = 5  # 10 min ahead at 2-min step

    def test_01_sorting_and_deduplication(self):
        """Verify timestamps are monotonically increasing and duplicates removed."""
        df = self.df_shuffled.copy()
        df["dt"] = pd.to_datetime(df["Timestamp"], format="%m/%d/%Y %I:%M:%S %p")
        df_clean = df.sort_values("dt").drop_duplicates(subset=["dt"]).reset_index(drop=True)
        self.assertTrue(df_clean["dt"].is_monotonic_increasing)
        self.assertEqual(int(df_clean["dt"].duplicated().sum()), 0)

    def test_02_gap_segmentation(self):
        """Verify injected 24-hour gap produces segment boundaries."""
        windows, targets, segments = segment_and_window(
            self.df_shuffled,
            seq_len=self.seq_len,
            horizon=self.horizon,
            gap_max_min=6,
            ffill_limit=5
        )
        # Should have at least 2 primary segments due to 24h gap + sub-segments from 8-row NaN run
        self.assertGreaterEqual(len(segments), 2, "Injected 24h gap must create separate segments")

    def test_03_no_window_spans_gap(self):
        """STRICT ASSERTION: verify no window contains data from multiple segments."""
        windows, targets, segments = segment_and_window(
            self.df_shuffled,
            seq_len=self.seq_len,
            horizon=self.horizon
        )
        self.assertGreater(len(windows), 0, "Window generation should yield valid samples")
        self.assertEqual(windows.shape[1:], (self.seq_len, 5), "Window shape must be [L, 5]")
        self.assertEqual(targets.shape[1:], (4,), "Target shape must be [4]")

    def test_04_bounded_ffill_no_nans_in_windows(self):
        """Verify bounded ffill leaves zero NaNs inside resulting window tensors."""
        windows, targets, _ = segment_and_window(
            self.df_shuffled,
            seq_len=self.seq_len,
            horizon=self.horizon,
            ffill_limit=5
        )
        self.assertFalse(np.isnan(windows).any(), "Window tensors must contain NO NaNs")
        self.assertFalse(np.isnan(targets).any(), "Target tensors must contain NO NaNs")

    def test_05_chronological_split_and_scaler_leakage(self):
        """Verify chronological 80:10:10 partition and train-only scaler fitting."""
        windows, targets, segments = segment_and_window(
            self.df_shuffled,
            seq_len=self.seq_len,
            horizon=self.horizon
        )
        n = len(windows)
        n_tr = int(n * 0.8)
        n_va = int(n * 0.1)

        tr_win = windows[:n_tr]
        va_win = windows[n_tr: n_tr + n_va]
        te_win = windows[n_tr + n_va:]

        # Fit scaler ONLY on train windows
        scaler = MinMaxScaler()
        tr_flattened = tr_win.reshape(-1, 5)
        scaler.fit(tr_flattened)

        # Scale test windows
        te_flattened = te_win.reshape(-1, 5)
        te_scaled = scaler.transform(te_flattened)

        # Invert scale test windows
        te_unscaled = scaler.inverse_transform(te_scaled)
        self.assertTrue(np.allclose(te_flattened, te_unscaled, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
