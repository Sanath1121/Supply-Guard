"""Adversarial Stress Test Suite for Phase 2: Data Ingestion Pipeline Hardening.

Author: Empirical Challenger 1 (teamwork_preview_challenger_m1_1)
Mission: Actively stress-test temporal boundary edge cases, segment length thresholds,
         sliding window continuity, NaN imputation limits, and data leakage isolation.
"""
import os
import sys
import unittest
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import Config
from src.dataset import load_clean_frame, build_datasets, segment_and_window


def create_base_synthetic_df(timestamps, seed=42):
    """Create a valid synthetic dataframe with specified timestamps."""
    rng = np.random.default_rng(seed)
    n = len(timestamps)
    return pd.DataFrame({
        "Timestamp": [t.strftime("%m/%d/%Y %I:%M:%S %p") for t in timestamps],
        "RI_Supplier1": rng.uniform(0.1, 0.9, n),
        "RI_Manufacturer1": rng.uniform(0.1, 0.9, n),
        "RI_Distributor1": rng.uniform(0.1, 0.9, n),
        "RI_Retailer1": rng.uniform(0.1, 0.9, n),
        "Total_Cost": rng.uniform(20.0, 50.0, n),
        "SCMstability_category": 1,
    })


class AdversarialPhase2DatasetTests(unittest.TestCase):
    """Rigorous empirical stress test cases."""

    def setUp(self):
        self.tmp_dir = os.path.join(PROJECT_ROOT, "data", "raw", "_adversarial_tmp")
        os.makedirs(self.tmp_dir, exist_ok=True)

    def tearDown(self):
        # Clean up temporary test files
        if os.path.exists(self.tmp_dir):
            for f in os.listdir(self.tmp_dir):
                try:
                    os.remove(os.path.join(self.tmp_dir, f))
                except OSError:
                    pass
            try:
                os.rmdir(self.tmp_dir)
            except OSError:
                pass

    def test_01_temporal_boundary_exact_6min_vs_6_1min(self):
        """Stress Test 1: Verify exact temporal boundary: 6.0 min vs 6.1 min vs minimal second offset.

        Specification: time gap exceeds GAP_MAX_MIN (6.0 min) triggers a split.
        Because CSV format '%m/%d/%Y %I:%M:%S %p' has 1-second resolution:
        - dt == 6.0 min (360 seconds): does NOT exceed 6.0 min -> SAME segment.
        - dt == 5 min 59 sec (359 seconds): does NOT exceed 6.0 min -> SAME segment.
        - dt == 6 min 1 sec (361 seconds): minimal resolvable step exceeding 6.0 min -> NEW segment.
        - dt == 6.1 min (366 seconds): strictly exceeds 6.0 min -> NEW segment.
        """
        base_t = pd.Timestamp("2018-01-01 00:00:00")
        
        # Block A: 20 rows at 2-min intervals (t=0 to 38 min)
        t_A = [base_t + pd.Timedelta(minutes=2 * i) for i in range(20)]
        
        # Block B: starts exactly 6.0 min (360s) after Block A end
        # 20 rows at 2-min intervals
        t_B_start = t_A[-1] + pd.Timedelta(minutes=6.0)
        t_B = [t_B_start + pd.Timedelta(minutes=2 * i) for i in range(20)]
        
        # Block C: starts exactly 6.1 min (366s) after Block B end (exceeds threshold)
        # 20 rows at 2-min intervals
        t_C_start = t_B[-1] + pd.Timedelta(minutes=6.1)
        t_C = [t_C_start + pd.Timedelta(minutes=2 * i) for i in range(20)]
        
        # Block D: starts 361s (6m 1s) after Block C end (minimal resolvable second exceeding 6.0m)
        # 20 rows at 2-min intervals
        t_D_start = t_C[-1] + pd.Timedelta(seconds=361)
        t_D = [t_D_start + pd.Timedelta(minutes=2 * i) for i in range(20)]
        
        # Block E: starts 359s (5m 59s) after Block D end (does not exceed threshold)
        # 20 rows at 2-min intervals
        t_E_start = t_D[-1] + pd.Timedelta(seconds=359)
        t_E = [t_E_start + pd.Timedelta(minutes=2 * i) for i in range(20)]
        
        all_timestamps = t_A + t_B + t_C + t_D + t_E
        df = create_base_synthetic_df(all_timestamps)
        
        # Test reference segment_and_window
        windows, targets, segments = segment_and_window(
            df, seq_len=10, horizon=5, gap_max_min=6.0, ffill_limit=5
        )
        
        # Expected segments:
        # A + B: merged into 40 rows (dt = 360s = 6.0m <= 6.0m)
        # C: split from B (dt = 366s = 6.1m > 6.0m) -> 20 rows
        # D + E: split from C (dt = 361s > 6.0m), merged D + E (dt = 359s < 6.0m) -> 40 rows
        self.assertEqual(len(segments), 3, f"Expected 3 segments across boundary tests, got {len(segments)}")
        self.assertEqual(len(segments[0]), 40, f"Block A+B should merge into 40 rows (dt=6.0 min), got {len(segments[0])}")
        self.assertEqual(len(segments[1]), 20, f"Block C should be 20 rows, got {len(segments[1])}")
        self.assertEqual(len(segments[2]), 40, f"Block D+E should merge into 40 rows (dt=5m59s), got {len(segments[2])}")
        
        # Test via Config and load_clean_frame
        csv_path = os.path.join(self.tmp_dir, "test_temporal_boundary.csv")
        df.to_csv(csv_path, index=False)
        cfg = Config()
        cfg.RAW_DATA_PATH = csv_path
        clean_df = load_clean_frame(cfg)
        
        self.assertEqual(clean_df["seg_id"].nunique(), 3, "load_clean_frame must produce 3 segments")
        counts = clean_df.groupby("seg_id")["seg_id"].count().values
        self.assertEqual(list(counts), [40, 20, 40], "Segment row counts must be [40, 20, 40]")

    def test_02_multiple_consecutive_gaps_and_pathological_stream(self):
        """Stress Test 2: Multiple consecutive gaps (> 6.0 min).

        Pathological scenario: 10 consecutive rows where EVERY row is 10 min after the previous.
        Every single step exceeds GAP_MAX_MIN.
        Each of these 10 rows forms an isolated segment of length 1 (< 15 steps).
        Followed by a 24-hour gap and a contiguous stream of 25 rows at 2-min cadence.
        Expected: All 10 isolated rows are dropped as short segments (< 15 steps).
                  The 25-row block is retained as 1 segment producing exactly (25 - 15 + 1) = 11 windows.
        """
        base_t = pd.Timestamp("2018-01-01 00:00:00")
        # 10 rows, each 10 min apart (all gaps)
        sparse_ts = [base_t + pd.Timedelta(minutes=10 * i) for i in range(10)]
        
        # 24 hour jump, then 25 contiguous rows at 2-min step
        dense_start = sparse_ts[-1] + pd.Timedelta(hours=24)
        dense_ts = [dense_start + pd.Timedelta(minutes=2 * i) for i in range(25)]
        
        df = create_base_synthetic_df(sparse_ts + dense_ts)
        csv_path = os.path.join(self.tmp_dir, "test_consecutive_gaps.csv")
        df.to_csv(csv_path, index=False)
        
        cfg = Config()
        cfg.RAW_DATA_PATH = csv_path
        clean_df = load_clean_frame(cfg)
        
        stats = clean_df.attrs["row_loss"]
        self.assertEqual(stats["n_short_seg_rows_dropped"], 10, "All 10 sparse rows must be dropped as short segments")
        self.assertEqual(stats["n_clean_rows_retained"], 25, "Dense 25 rows must be retained")
        self.assertEqual(clean_df["seg_id"].nunique(), 1, "Exactly 1 valid segment retained")
        
        # Test window generation
        windows, targets, segments = segment_and_window(df, seq_len=10, horizon=5, gap_max_min=6.0)
        self.assertEqual(len(windows), 11, "25 rows with L=10, H=5 must produce exactly 11 windows")

    def test_03_segment_lengths_14_vs_15_vs_16(self):
        """Stress Test 3: Segment lengths of exactly 14 rows (< L+H) vs 15 rows (== L+H) vs 16 rows.

        With L=10, H=5:
        - min_seg_len = 10 + 5 = 15.
        - Segment of 14 rows: must be dropped. Windows generated = 0.
        - Segment of 15 rows: must be retained. Windows generated = 1.
          Window sequence: rows 0..9 (length 10).
          Target: row 14 (exactly step t+5).
        - Segment of 16 rows: must be retained. Windows generated = 2.
        """
        base_t = pd.Timestamp("2018-01-01 00:00:00")
        
        # Seg 1: exactly 14 rows (2-min cadence)
        t_seg1 = [base_t + pd.Timedelta(minutes=2 * i) for i in range(14)]
        
        # 1-hour gap
        t_seg2_start = t_seg1[-1] + pd.Timedelta(hours=1)
        # Seg 2: exactly 15 rows (2-min cadence)
        t_seg2 = [t_seg2_start + pd.Timedelta(minutes=2 * i) for i in range(15)]
        
        # 1-hour gap
        t_seg3_start = t_seg2[-1] + pd.Timedelta(hours=1)
        # Seg 3: exactly 16 rows (2-min cadence)
        t_seg3 = [t_seg3_start + pd.Timedelta(minutes=2 * i) for i in range(16)]
        
        df = create_base_synthetic_df(t_seg1 + t_seg2 + t_seg3)
        
        # Reference windowing
        windows, targets, segments = segment_and_window(df, seq_len=10, horizon=5, gap_max_min=6.0)
        self.assertEqual(len(segments), 2, "Seg 1 (14 rows) dropped; Seg 2 (15) and Seg 3 (16) retained")
        self.assertEqual(len(segments[0]), 15)
        self.assertEqual(len(segments[1]), 16)
        
        # Total windows: 1 from Seg 2 + 2 from Seg 3 = 3 windows
        self.assertEqual(len(windows), 3, "Total windows must be 1 + 2 = 3")
        
        # Inspect the window from the 15-row segment
        # In Seg 2: context is indices 0..9, target is index 14
        seg2_vals = segments[0][["RI_Supplier1", "RI_Manufacturer1", "RI_Distributor1", "RI_Retailer1", "Total_Cost"]].values
        self.assertTrue(np.allclose(windows[0], seg2_vals[:10]), "Window 0 sequence must equal first 10 rows")
        self.assertTrue(np.allclose(targets[0], seg2_vals[14, :4]), "Window 0 target must equal row 14 (t+5)")
        
        # Pipeline execution via Config
        csv_path = os.path.join(self.tmp_dir, "test_seg_lengths.csv")
        df.to_csv(csv_path, index=False)
        cfg = Config()
        cfg.RAW_DATA_PATH = csv_path
        clean_df = load_clean_frame(cfg)
        self.assertEqual(clean_df.attrs["row_loss"]["n_short_seg_rows_dropped"], 14, "14 rows dropped")
        self.assertEqual(clean_df.attrs["row_loss"]["n_clean_rows_retained"], 31, "15 + 16 = 31 rows retained")

    def test_04_sliding_window_gap_spanning_invariant(self):
        """Stress Test 4: Sliding window gap spanning invariant.

        EMPIRICAL PROOF: Across thousands of windows generated over a fragmented,
        multi-gap dataset, verify that NO window ever spans a gap.
        Checks:
        1. Every adjacent pair in window sequence has delta <= GAP_MAX_MIN (6.0 min).
        2. Target timestamp minus window end timestamp == H * 2 min = 10 min (<= H * GAP_MAX_MIN).
        3. All context rows and target row belong to the identical contiguous segment.
        """
        rng = np.random.default_rng(123)
        base_t = pd.Timestamp("2018-01-01 00:00:00")
        
        # Create 8 segments of random lengths between 15 and 80 rows
        # Separated by random gaps between 10 minutes and 48 hours
        all_ts = []
        curr_t = base_t
        for seg_idx in range(8):
            seg_len = rng.integers(15, 80)
            seg_ts = [curr_t + pd.Timedelta(minutes=2 * i) for i in range(seg_len)]
            all_ts.extend(seg_ts)
            gap_duration_min = rng.uniform(10.0, 2880.0) # 10 min to 48 hours
            curr_t = seg_ts[-1] + pd.Timedelta(minutes=gap_duration_min)
            
        df = create_base_synthetic_df(all_ts)
        
        # Inject null runs in middle of segments:
        # Bounded null run (3 rows, filled)
        df.loc[30:32, "RI_Supplier1"] = np.nan
        # Unbounded null run (7 rows, splits segment)
        df.loc[120:126, "RI_Distributor1"] = np.nan
        
        csv_path = os.path.join(self.tmp_dir, "test_window_gap_invariant.csv")
        df.to_csv(csv_path, index=False)
        
        cfg = Config()
        cfg.RAW_DATA_PATH = csv_path
        cfg.MAX_SAMPLES = None
        
        tr, va, te, sc, info = build_datasets(cfg)
        
        max_step_delta = pd.Timedelta(minutes=cfg.GAP_MAX_MIN)
        expected_target_offset = pd.Timedelta(minutes=cfg.HORIZON * cfg.CADENCE_MIN) # 10.0 min
        
        for split_name, ds in [("train", tr), ("val", va), ("test", te)]:
            n_samples = len(ds)
            self.assertGreater(n_samples, 0, f"{split_name} should contain samples")
            
            # Verify timestamps tensor
            self.assertIsNotNone(ds.seq_timestamps, f"{split_name} seq_timestamps must not be None")
            seq_ts = ds.seq_timestamps # [N, L]
            tgt_ts = ds.timestamps     # [N]
            
            for i in range(n_samples):
                # 1. Check all adjacent sequence steps
                w_times = [pd.Timestamp(t) for t in seq_ts[i]]
                for step in range(len(w_times) - 1):
                    dt = w_times[step + 1] - w_times[step]
                    self.assertLessEqual(
                        dt, max_step_delta,
                        f"Sample {i} in {split_name}: step {step}->{step+1} delta {dt} exceeds GAP_MAX_MIN!"
                    )
                    # For strictly 2-min cadence, verify dt == 2.0 min
                    self.assertEqual(dt, pd.Timedelta(minutes=2.0))
                
                # 2. Check target offset from last context step
                last_ctx_t = w_times[-1]
                target_t = pd.Timestamp(tgt_ts[i])
                tgt_delta = target_t - last_ctx_t
                self.assertEqual(
                    tgt_delta, expected_target_offset,
                    f"Sample {i} in {split_name}: target delta {tgt_delta} does not match {expected_target_offset}!"
                )

    def test_05_zero_nans_under_aggressive_null_injection(self):
        """Stress Test 5: Exhaustive NaN elimination and tensor finiteness.

        Aggressive adversarial null injection:
        1. All 5 features null simultaneously for 4 rows (<= 5, forward filled).
        2. Single feature null for 5 rows (exactly limit=5, forward filled).
        3. Single feature null for 6 rows (> 5, drops row 6 and splits segment).
        4. Nulls at the very start of a segment (rows 0..2) (cannot forward fill -> dropped).
        5. Entire segment null (completely dropped).
        Assert:
        - ZERO NaNs in sequences, node_targets, and tri_targets tensors.
        - ZERO Infs in sequences, node_targets, and tri_targets tensors.
        """
        base_t = pd.Timestamp("2018-01-01 00:00:00")
        ts = [base_t + pd.Timedelta(minutes=2 * i) for i in range(300)]
        df = create_base_synthetic_df(ts)
        
        # 1. Multi-feature 4-row null run
        df.loc[20:23, ["RI_Supplier1", "RI_Manufacturer1", "RI_Distributor1", "RI_Retailer1", "Total_Cost"]] = np.nan
        
        # 2. 5-row null run (exactly ffill limit)
        df.loc[50:54, "RI_Distributor1"] = np.nan
        
        # 3. 6-row null run (exceeds limit by 1 row)
        df.loc[100:105, "RI_Retailer1"] = np.nan
        
        # 4. Nulls at segment start (create a gap at row 150, rows 151:153 are null)
        df.loc[150, "Timestamp"] = (pd.Timestamp(df.loc[149, "Timestamp"]) + pd.Timedelta(hours=5)).strftime("%m/%d/%Y %I:%M:%S %p")
        # reconstruct following timestamps
        t150 = pd.Timestamp(df.loc[150, "Timestamp"])
        for j in range(151, 300):
            df.loc[j, "Timestamp"] = (t150 + pd.Timedelta(minutes=2 * (j - 150))).strftime("%m/%d/%Y %I:%M:%S %p")
        df.loc[150:152, "RI_Manufacturer1"] = np.nan # leading nulls in new segment
        
        csv_path = os.path.join(self.tmp_dir, "test_aggressive_nans.csv")
        df.to_csv(csv_path, index=False)
        
        cfg = Config()
        cfg.RAW_DATA_PATH = csv_path
        tr, va, te, sc, info = build_datasets(cfg)
        
        for name, ds in [("train", tr), ("val", va), ("test", te)]:
            self.assertFalse(torch.isnan(ds.sequences).any(), f"NaNs detected in {name} sequences")
            self.assertFalse(torch.isnan(ds.node_targets).any(), f"NaNs detected in {name} node_targets")
            self.assertFalse(torch.isnan(ds.tri_targets).any(), f"NaNs detected in {name} tri_targets")
            self.assertFalse(torch.isinf(ds.sequences).any(), f"Infs detected in {name} sequences")
            self.assertFalse(torch.isinf(ds.node_targets).any(), f"Infs detected in {name} node_targets")
            self.assertFalse(torch.isinf(ds.tri_targets).any(), f"Infs detected in {name} tri_targets")

    def test_06_strict_partition_isolation_and_scaler_leakage(self):
        """Stress Test 6: Strict chronological partitioning and scaler isolation.

        Inject massive numeric outliers strictly into:
        1. Test partition: Total_Cost = 1,000,000.0.
        2. Validation partition: RI_Supplier1 = 500,000.0.
        All Train partition values are strictly in [0.0, 10.0].
        Assert:
        - Scaler data_max_ for Total_Cost is <= 10.0 (NO test outlier leakage).
        - Scaler data_max_ for RI_Supplier1 is <= 10.0 (NO val outlier leakage).
        - Partition target timestamps are strictly ordered:
          tr.timestamps.max() < va.timestamps.min() < va.timestamps.max() < te.timestamps.min()
        - Target index of every Train sample lies in [0, n_tr - 1].
        - Target index of every Val sample lies in [n_tr, n_tr + n_va - 1].
        - Target index of every Test sample lies in [n_tr + n_va, n_total - 1].
        """
        n_rows = 500
        base_t = pd.Timestamp("2018-01-01 00:00:00")
        ts = [base_t + pd.Timedelta(minutes=2 * i) for i in range(n_rows)]
        df = create_base_synthetic_df(ts)
        
        # Enforce train partition values <= 10.0
        n_tr = int(n_rows * 0.8)  # 400 rows
        n_va = int(n_rows * 0.1)  # 50 rows
        # test is rows 450..499
        
        # Inject outlier in test
        df.loc[460, "Total_Cost"] = 1_000_000.0
        # Inject outlier in val
        df.loc[410, "RI_Supplier1"] = 500_000.0
        
        csv_path = os.path.join(self.tmp_dir, "test_scaler_leakage.csv")
        df.to_csv(csv_path, index=False)
        
        cfg = Config()
        cfg.RAW_DATA_PATH = csv_path
        tr, va, te, scaler, info = build_datasets(cfg)
        
        # Scaler max checks
        feat_cols = cfg.FEATURE_COLS
        tc_idx = feat_cols.index("Total_Cost")
        sup_idx = feat_cols.index("RI_Supplier1")
        
        self.assertLess(
            scaler.data_max_[tc_idx], 100.0,
            f"Total_Cost scaler data_max_ {scaler.data_max_[tc_idx]} indicates test outlier leakage!"
        )
        self.assertLess(
            scaler.data_max_[sup_idx], 10.0,
            f"RI_Supplier1 scaler data_max_ {scaler.data_max_[sup_idx]} indicates val outlier leakage!"
        )
        
        # Chronological timestamps order
        self.assertLess(tr.timestamps.max(), va.timestamps.min())
        self.assertLess(va.timestamps.max(), te.timestamps.min())

    def test_07_row_loss_conservation_identity(self):
        """Stress Test 7: Row loss conservation accounting identity.

        Verify exact conservation equation:
        n_raw == n_bad_timestamps + n_duplicates_dropped + n_nans_dropped + n_short_seg_rows_dropped + n_clean_rows_retained
        under a dirty dataset containing all types of corruptions.
        """
        n_rows = 200
        base_t = pd.Timestamp("2018-01-01 00:00:00")
        ts = [base_t + pd.Timedelta(minutes=2 * i) for i in range(n_rows)]
        df = create_base_synthetic_df(ts)
        
        # 1. Bad timestamps (3 rows)
        df.loc[5:7, "Timestamp"] = "INVALID_TIMESTAMP_STRING"
        
        # 2. Duplicate timestamp (2 duplicates)
        df.loc[20, "Timestamp"] = df.loc[19, "Timestamp"]
        df.loc[25, "Timestamp"] = df.loc[24, "Timestamp"]
        
        # 3. Long unfillable null run (8 rows, 5 filled, 3 dropped)
        df.loc[80:87, "RI_Distributor1"] = np.nan
        
        # 4. Short segment (create 8-hour gap before and after a 10-row cluster)
        t_gap1 = pd.Timestamp(df.loc[119, "Timestamp"]) + pd.Timedelta(hours=8)
        for k in range(120, 130):
            df.loc[k, "Timestamp"] = (t_gap1 + pd.Timedelta(minutes=2 * (k - 120))).strftime("%m/%d/%Y %I:%M:%S %p")
        t_gap2 = pd.Timestamp(df.loc[129, "Timestamp"]) + pd.Timedelta(hours=8)
        for k in range(130, n_rows):
            df.loc[k, "Timestamp"] = (t_gap2 + pd.Timedelta(minutes=2 * (k - 130))).strftime("%m/%d/%Y %I:%M:%S %p")
            
        csv_path = os.path.join(self.tmp_dir, "test_row_conservation.csv")
        df.to_csv(csv_path, index=False)
        
        cfg = Config()
        cfg.RAW_DATA_PATH = csv_path
        clean_df = load_clean_frame(cfg)
        stats = clean_df.attrs["row_loss"]
        
        sum_components = (
            stats["n_bad_timestamps"]
            + stats["n_duplicates_dropped"]
            + stats["n_nans_dropped"]
            + stats["n_short_seg_rows_dropped"]
            + stats["n_clean_rows_retained"]
        )
        self.assertEqual(
            sum_components, stats["n_raw"],
            f"Row loss conservation violated: sum={sum_components} != raw={stats['n_raw']}"
        )

    def test_08_cross_partition_context_borrowing_within_same_segment(self):
        """Stress Test 8: Context rows crossing partition boundaries ONLY within same segment.

        Construct a single contiguous segment spanning across the train/val boundary (100 rows).
        Train partition has 80 rows, Val has 10 rows, Test has 10 rows.
        - A window in Val whose target is at index 80 (first row of Val):
          Its context rows are indices 66..75 (strictly inside Train).
          Since both target and context rows belong to the identical unbroken segment,
          this borrowing is completely valid and free of time gaps.
        - Verify context rows borrow safely across boundary.
        - Verify target timestamps are strictly within Val.
        """
        n_rows = 100
        base_t = pd.Timestamp("2018-01-01 00:00:00")
        ts = [base_t + pd.Timedelta(minutes=2 * i) for i in range(n_rows)]
        df = create_base_synthetic_df(ts)
        
        csv_path = os.path.join(self.tmp_dir, "test_cross_partition.csv")
        df.to_csv(csv_path, index=False)
        
        cfg = Config()
        cfg.RAW_DATA_PATH = csv_path
        cfg.TRAIN_RATIO, cfg.VAL_RATIO, cfg.TEST_RATIO = 0.80, 0.10, 0.10
        tr, va, te, scaler, info = build_datasets(cfg)
        
        # Train targets: rows 14..79 (66 windows)
        # Val targets: rows 80..89 (10 windows)
        # Test targets: rows 90..99 (10 windows)
        self.assertEqual(len(tr), 66)
        self.assertEqual(len(va), 10)
        self.assertEqual(len(te), 10)
        
        # Verify first val window has target at row 80
        val_first_tgt_t = pd.Timestamp(va.timestamps[0])
        expected_tgt_t = base_t + pd.Timedelta(minutes=2 * 80)
        self.assertEqual(val_first_tgt_t, expected_tgt_t)
        
        # Its context rows are rows 66..75 (in train partition)
        val_first_ctx_start_t = pd.Timestamp(va.seq_timestamps[0, 0])
        expected_ctx_start_t = base_t + pd.Timedelta(minutes=2 * (80 - 14)) # row 66
        self.assertEqual(val_first_ctx_start_t, expected_ctx_start_t)
        
        # Chronological target strict monotonicity
        self.assertLess(tr.timestamps.max(), va.timestamps.min())
        self.assertLess(va.timestamps.max(), te.timestamps.min())

    def test_09_nan_run_splitting_segment_dropping_short_subsegments(self):
        """Stress Test 9: Null run > 5 rows splits a segment; sub-segments < 15 are dropped.

        Construct a 20-row segment:
        - Rows 0..9 (10 rows)
        - Rows 10..15 have NaN (6 rows, > limit=5) -> row 15 remains NaN and is dropped.
        - Rows 16..19 (4 rows)
        Result:
        - Sub-segment 1: rows 0..9 (length 10 < 15) -> dropped as short segment!
        - Sub-segment 2: rows 16..19 (length 4 < 15) -> dropped as short segment!
        Total rows retained: 0!
        """
        base_t = pd.Timestamp("2018-01-01 00:00:00")
        ts = [base_t + pd.Timedelta(minutes=2 * i) for i in range(20)]
        df = create_base_synthetic_df(ts)
        df.loc[10:15, "RI_Distributor1"] = np.nan
        
        csv_path = os.path.join(self.tmp_dir, "test_nan_split_short.csv")
        df.to_csv(csv_path, index=False)
        
        cfg = Config()
        cfg.RAW_DATA_PATH = csv_path
        clean_df = load_clean_frame(cfg)
        
        stats = clean_df.attrs["row_loss"]
        self.assertEqual(stats["n_nans_dropped"], 1, "Exactly 1 unfilled NaN row dropped (row 15)")
        # Rows 0..9 (10) and rows 10..14 (5 filled) were part of subseg 1 (15 rows!)
        # Wait! Rows 10..14 are 5 rows, which CAN be forward filled from row 9!
        # So rows 0..14 is 15 rows! 15 >= 15 -> RETAINED!
        # Row 15 dropped.
        # Rows 16..19 (4 rows) < 15 -> DROPPED!
        self.assertEqual(stats["n_short_seg_rows_dropped"], 4, "Rows 16..19 dropped as short segment")
        self.assertEqual(stats["n_clean_rows_retained"], 15, "Subsegment 0..14 has length 15 and is retained")

    def test_10_real_scrm_dataset_verification(self):
        """Stress Test 10: Real SCRM 650k dataset full verification.

        Execute build_datasets on data/raw/SCRM_timeSeries_2018_train.csv
        and verify all strict invariant constraints on production data.
        """
        raw_path = os.path.join(PROJECT_ROOT, "data", "raw", "SCRM_timeSeries_2018_train.csv")
        if not os.path.exists(raw_path):
            self.skipTest(f"Real dataset {raw_path} not found")
            
        cfg = Config()
        cfg.RAW_DATA_PATH = raw_path
        cfg.MAX_SAMPLES = None
        
        tr, va, te, sc, info = build_datasets(cfg)
        
        # 1. Check shapes
        self.assertEqual(tr.sequences.shape[1:], (10, 5))
        self.assertEqual(va.sequences.shape[1:], (10, 5))
        self.assertEqual(te.sequences.shape[1:], (10, 5))
        self.assertEqual(tr.node_targets.shape[1], 4)
        
        # 2. Check zero NaNs
        self.assertFalse(torch.isnan(tr.sequences).any(), "NaNs in real train sequences")
        self.assertFalse(torch.isnan(va.sequences).any(), "NaNs in real val sequences")
        self.assertFalse(torch.isnan(te.sequences).any(), "NaNs in real test sequences")
        self.assertFalse(torch.isnan(tr.node_targets).any(), "NaNs in real train targets")
        self.assertFalse(torch.isnan(va.node_targets).any(), "NaNs in real val targets")
        self.assertFalse(torch.isnan(te.node_targets).any(), "NaNs in real test targets")
        
        # 3. Check chronological partition separation
        self.assertLess(tr.timestamps.max(), va.timestamps.min())
        self.assertLess(va.timestamps.max(), te.timestamps.min())
        
        # 4. Check row loss conservation
        row_loss = info["row_loss"]
        sum_rows = (
            row_loss["n_bad_timestamps"]
            + row_loss["n_duplicates_dropped"]
            + row_loss["n_nans_dropped"]
            + row_loss["n_short_seg_rows_dropped"]
            + row_loss["n_clean_rows_retained"]
        )
        self.assertEqual(sum_rows, row_loss["n_raw"], "Real dataset conservation accounting mismatch")
        self.assertEqual(row_loss["n_raw"], 649999)
        self.assertEqual(row_loss["n_clean_rows_retained"], 592599)


if __name__ == "__main__":
    unittest.main()
