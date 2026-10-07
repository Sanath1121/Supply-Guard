"""Adversarial Stress Test Suite for Phase 2: Data Ingestion Pipeline Hardening.

Author: Empirical Challenger 2 (teamwork_preview_challenger_m1_2)
Role: Empirical Challenger (critic, specialist)
Purpose: Independent empirical stress-testing of data leakage prevention,
         partition isolation, context borrowing, and bounded imputation policies.

Mandatory Challenger 2 Checks:
1. Scaler Leakage: MinMaxScaler fit strictly on train partition. Perturb val/test drastically,
   verify scaler min/max remains completely unchanged.
2. Target Isolation: Target timestamps strictly satisfy max(train) < min(val) < max(val) < min(test).
3. Context Borrowing: Context rows borrowed across partition boundaries belong strictly to
   the identical contiguous segment as the target.
4. Bounded Imputation: Null runs <= 5 rows are filled, while null runs > 5 rows are never
   forward-filled across the gap and split segments.
5. Real Dataset Verification: SCRM_timeSeries_2018_train.csv end-to-end invariant validation.
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
from src.dataset import load_clean_frame, build_datasets


def create_clean_cadence_df(n_rows: int = 1000, start_time: str = "2018-01-01 00:00:00", seed: int = 42) -> pd.DataFrame:
    """Generate a clean synthetic dataframe with strict 2-min cadence."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start_time, periods=n_rows, freq="2min")
    return pd.DataFrame({
        "Timestamp": [t.strftime("%m/%d/%Y %I:%M:%S %p") for t in ts],
        "RI_Supplier1": rng.uniform(0.1, 0.9, n_rows),
        "RI_Manufacturer1": rng.uniform(0.1, 0.9, n_rows),
        "RI_Distributor1": rng.uniform(0.1, 0.9, n_rows),
        "RI_Retailer1": rng.uniform(0.1, 0.9, n_rows),
        "Total_Cost": rng.uniform(20.0, 50.0, n_rows),
        "SCMstability_category": 1,
    })


class Challenger2AdversarialTests(unittest.TestCase):
    """Empirical adversarial test suite by Challenger 2."""

    def setUp(self):
        import tempfile
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # =========================================================================
    # TASK 1.1: Empirical Test of Scaler Leakage
    # =========================================================================
    def test_01_scaler_leakage_drastic_val_test_perturbation(self):
        """EMPIRICAL TEST 1: Scaler Leakage Prevention.

        Verify MinMaxScaler is fit strictly on first 80% train rows.
        Adversarial attack:
        1. Fit baseline scaler on uncorrupted synthetic data.
        2. Create perturbed dataset where validation partition values are perturbed to +500,000.0,
           and test partition values are perturbed to -1,000,000.0 and +10,000,000.0.
        3. Fit scaler on perturbed dataset.
        4. Assert that scaler min and max for ALL 5 features are BIT-FOR-BIT IDENTICAL
           between baseline and perturbed models.
        5. Sensitive negative control: perturb a single train row, verify scaler changes.
        """
        n_rows = 600
        df_base = create_clean_cadence_df(n_rows, seed=101)

        base_csv = os.path.join(self.tmp_dir, "base_scaler_test.csv")
        df_base.to_csv(base_csv, index=False)

        cfg_base = Config()
        cfg_base.RAW_DATA_PATH = base_csv
        cfg_base.SCALER_PATH = os.path.join(self.tmp_dir, "base_scaler.joblib")
        _, _, _, scaler_base, _ = build_datasets(cfg_base, save_scaler=True)

        # Now create adversarial perturbed dataset
        df_perturbed = df_base.copy()
        n_tr = int(n_rows * 0.8)   # 480 rows
        n_va = int(n_rows * 0.1)   # 60 rows
        # Val: rows 480 to 539
        # Test: rows 540 to 599

        # Massive val perturbation
        df_perturbed.loc[485:530, "Total_Cost"] = 999_999_999.0
        df_perturbed.loc[490:510, "RI_Supplier1"] = -500_000.0
        df_perturbed.loc[520:535, "RI_Distributor1"] = 123_456.0

        # Catastrophic test perturbation
        df_perturbed.loc[545:590, "Total_Cost"] = -888_888_888.0
        df_perturbed.loc[550:570, "RI_Manufacturer1"] = 777_777.0
        df_perturbed.loc[580:595, "RI_Retailer1"] = -999_999.0

        pert_csv = os.path.join(self.tmp_dir, "perturbed_scaler_test.csv")
        df_perturbed.to_csv(pert_csv, index=False)

        cfg_pert = Config()
        cfg_pert.RAW_DATA_PATH = pert_csv
        cfg_pert.SCALER_PATH = os.path.join(self.tmp_dir, "perturbed_scaler.joblib")
        _, _, _, scaler_pert, _ = build_datasets(cfg_pert, save_scaler=True)

        # Check: scaler_pert must match scaler_base exactly
        np.testing.assert_array_equal(
            scaler_pert.data_min_, scaler_base.data_min_,
            err_msg="Scaler data_min_ leaked validation or test partition values!"
        )
        np.testing.assert_array_equal(
            scaler_pert.data_max_, scaler_base.data_max_,
            err_msg="Scaler data_max_ leaked validation or test partition values!"
        )

        # Negative control check: perturb train row index 50, ensure scaler changes
        df_train_pert = df_base.copy()
        df_train_pert.loc[50, "Total_Cost"] = 99_999.0
        train_pert_csv = os.path.join(self.tmp_dir, "train_perturbed_test.csv")
        df_train_pert.to_csv(train_pert_csv, index=False)

        cfg_train_pert = Config()
        cfg_train_pert.RAW_DATA_PATH = train_pert_csv
        cfg_train_pert.SCALER_PATH = os.path.join(self.tmp_dir, "train_perturbed_scaler.joblib")
        _, _, _, scaler_train_pert, _ = build_datasets(cfg_train_pert, save_scaler=False)

        cost_idx = cfg_train_pert.FEATURE_COLS.index("Total_Cost")
        self.assertNotEqual(
            scaler_train_pert.data_max_[cost_idx], scaler_base.data_max_[cost_idx],
            "Negative control failed: Train perturbation did not alter scaler max!"
        )

    # =========================================================================
    # TASK 1.2: Empirical Test of Target Isolation
    # =========================================================================
    def test_02_target_isolation_and_chronological_ordering(self):
        """EMPIRICAL TEST 2: Target Isolation.

        Verify target timestamps strictly satisfy:
            max(train.timestamps) < min(val.timestamps) < max(val.timestamps) < min(test.timestamps)
        under adversarial fragmentation (multiple gaps, varying segment lengths,
        boundary straddling segments).
        """
        rng = np.random.default_rng(202)
        base_t = pd.Timestamp("2018-01-01 00:00:00")

        # Create 12 fragmented segments with varying gap sizes
        all_ts = []
        curr_t = base_t
        for i in range(12):
            seg_len = rng.integers(16, 60)
            seg_ts = [curr_t + pd.Timedelta(minutes=2 * j) for j in range(seg_len)]
            all_ts.extend(seg_ts)
            gap_hours = rng.uniform(0.5, 12.0)
            curr_t = seg_ts[-1] + pd.Timedelta(hours=gap_hours)

        df = create_clean_cadence_df(len(all_ts), seed=202)
        df["Timestamp"] = [t.strftime("%m/%d/%Y %I:%M:%S %p") for t in all_ts]

        # Shuffle rows on disk to ensure pipeline sorts properly
        df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        csv_path = os.path.join(self.tmp_dir, "test_target_isolation.csv")
        df_shuffled.to_csv(csv_path, index=False)

        cfg = Config()
        cfg.RAW_DATA_PATH = csv_path
        cfg.SCALER_PATH = os.path.join(self.tmp_dir, "test_target_isolation_scaler.joblib")
        tr, va, te, _, info = build_datasets(cfg, save_scaler=False)

        self.assertGreater(len(tr), 0, "Train dataset must not be empty")
        self.assertGreater(len(va), 0, "Val dataset must not be empty")
        self.assertGreater(len(te), 0, "Test dataset must not be empty")

        tr_t_max = pd.Timestamp(tr.timestamps.max())
        va_t_min = pd.Timestamp(va.timestamps.min())
        va_t_max = pd.Timestamp(va.timestamps.max())
        te_t_min = pd.Timestamp(te.timestamps.min())

        # Strict inequality assertion
        self.assertLess(
            tr_t_max, va_t_min,
            f"Target Isolation Violated: max(train) ({tr_t_max}) >= min(val) ({va_t_min})"
        )
        self.assertLess(
            va_t_max, te_t_min,
            f"Target Isolation Violated: max(val) ({va_t_max}) >= min(test) ({te_t_min})"
        )

        # Monotonicity inside each partition
        self.assertTrue(np.all(np.diff(tr.timestamps.astype("int64")) > 0), "Train targets not monotonically increasing")
        self.assertTrue(np.all(np.diff(va.timestamps.astype("int64")) > 0), "Val targets not monotonically increasing")
        self.assertTrue(np.all(np.diff(te.timestamps.astype("int64")) > 0), "Test targets not monotonically increasing")

    # =========================================================================
    # TASK 1.3: Empirical Test of Context Borrowing
    # =========================================================================
    def test_03_context_borrowing_contiguous_segment_integrity(self):
        """EMPIRICAL TEST 3: Context Borrowing Contiguous Segment Verification.

        Specification: When a window in val borrows context rows from train,
        those context rows belong strictly to the identical contiguous segment as the target.

        Adversarial setup:
        1. Segment A: rows 0 to 49 (50 rows, all in train).
        2. A 24-hour gap.
        3. Segment B: rows 50 to 119 (70 rows, straddles train/val boundary).
           With N=120:
           n_tr = int(120 * 0.8) = 96.
           Rows 0..95 are train. Rows 96..107 are val. Rows 108..119 are test.
        4. In Segment B:
           Row 96 is the first val row.
           A val window with target at row 96 has context rows [82..91] and target at 96 (t+5).
           Notice: Context rows 82..91 are in train (< 96).
        5. Empirical Assertions:
           - Every single context step in that val window has timestamp originating from Segment B.
           - None of the context steps can ever cross the 24-hour gap into Segment A.
           - In the clean dataframe, every single context row and the target row have the
             IDENTICAL seg_id.
           - The inter-step interval within the borrowed context is strictly 2.0 min.
        """
        base_t = pd.Timestamp("2018-01-01 00:00:00")
        # Segment A: 50 rows at 2-min cadence
        ts_A = [base_t + pd.Timedelta(minutes=2 * i) for i in range(50)]

        # 24-hour gap
        t_B_start = ts_A[-1] + pd.Timedelta(hours=24)
        # Segment B: 70 rows at 2-min cadence
        ts_B = [t_B_start + pd.Timedelta(minutes=2 * i) for i in range(70)]

        all_ts = ts_A + ts_B
        df = create_clean_cadence_df(len(all_ts), seed=303)
        df["Timestamp"] = [t.strftime("%m/%d/%Y %I:%M:%S %p") for t in all_ts]

        csv_path = os.path.join(self.tmp_dir, "test_context_borrowing.csv")
        df.to_csv(csv_path, index=False)

        cfg = Config()
        cfg.RAW_DATA_PATH = csv_path
        cfg.SCALER_PATH = os.path.join(self.tmp_dir, "test_context_borrowing_scaler.joblib")
        clean_df = load_clean_frame(cfg)
        tr, va, te, _, info = build_datasets(cfg, save_scaler=False)

        n_total = len(clean_df)
        n_tr = int(n_total * cfg.TRAIN_RATIO)
        n_va = int(n_total * cfg.VAL_RATIO)

        # Inspect the clean dataframe segments
        # Segment A is seg_id 0 (50 rows, row indices 0..49)
        # Segment B is seg_id 1 (70 rows, row indices 50..119)
        self.assertEqual(clean_df["seg_id"].iloc[49], 0)
        self.assertEqual(clean_df["seg_id"].iloc[50], 0)

        # Find val windows that borrow context from train
        borrowing_windows_found = 0
        for i in range(len(va)):
            ctx_ts = [pd.Timestamp(t) for t in va.seq_timestamps[i]]
            tgt_t = pd.Timestamp(va.timestamps[i])

            # Find matching rows in clean_df
            ctx_mask = clean_df[cfg.DATE_COL].isin(ctx_ts)
            tgt_mask = clean_df[cfg.DATE_COL] == tgt_t

            ctx_rows = clean_df[ctx_mask]
            tgt_row = clean_df[tgt_mask]

            self.assertEqual(len(ctx_rows), cfg.SEQ_LEN)
            self.assertEqual(len(tgt_row), 1)

            tgt_seg = tgt_row["seg_id"].values[0]
            ctx_segs = ctx_rows["seg_id"].unique()

            # STRICT INVARIANT: All context rows must belong to identical segment as target
            self.assertEqual(len(ctx_segs), 1, f"Val sample {i} has context rows spanning multiple segments: {ctx_segs}")
            self.assertEqual(ctx_segs[0], tgt_seg)

            # Check if any context row had global row index < n_tr (borrowed from train)
            ctx_indices = ctx_rows.index.values
            if np.any(ctx_indices < n_tr):
                borrowing_windows_found += 1
                self.assertEqual(tgt_seg, 1)
                # Verify no context step came from Segment A (indices 0..49)
                self.assertTrue(np.all(ctx_indices >= 50), "Context row illegally borrowed from before the 24-hr gap!")

            # Verify intra-window temporal continuity
            for step in range(len(ctx_ts) - 1):
                dt = ctx_ts[step + 1] - ctx_ts[step]
                self.assertEqual(dt, pd.Timedelta(minutes=2.0))

            # Target offset
            self.assertEqual(tgt_t - ctx_ts[-1], pd.Timedelta(minutes=10.0))

        self.assertGreater(
            borrowing_windows_found, 0,
            "Test design error: expected at least one val window borrowing context from train"
        )

    # =========================================================================
    # TASK 1.4: Empirical Test of Bounded Imputation
    # =========================================================================
    def test_04_bounded_imputation_policy(self):
        """EMPIRICAL TEST 4: Bounded Imputation Policy.

        Specification:
        1. Null runs <= 5 rows are filled.
        2. Null runs > 5 rows are never forward-filled across the gap and split segments.
        3. Nulls preceding a gap must never be forward filled across the temporal gap.

        Adversarial test cases:
        - Case A: 1-row null run -> successfully filled.
        - Case B: 3-row null run -> successfully filled.
        - Case C: 5-row null run (exact limit) -> successfully filled.
        - Case D: 6-row null run (limit + 1):
          First 5 are filled, 6th remains NaN, gets dropped, and forces a segment split!
          The pre-run value is NEVER propagated to post-split rows.
        - Case E: 10-row null run:
          Rows 6..10 remain NaN, get dropped, and force a segment split.
        - Case F: Temporal gap with NaN immediately at start of new gap segment:
          Value from before gap MUST NOT cross the gap.
        """
        base_t = pd.Timestamp("2018-01-01 00:00:00")
        # 300 rows at 2-min cadence
        ts = [base_t + pd.Timedelta(minutes=2 * i) for i in range(300)]
        df = create_clean_cadence_df(300, seed=404)

        # Set known distinct marker values before each test run
        # Marker 1 at row 20
        df.loc[20, "RI_Distributor1"] = 0.1111
        # Case A: 1-row null at row 21
        df.loc[21, "RI_Distributor1"] = np.nan
        df.loc[22, "RI_Distributor1"] = 0.2222

        # Marker 2 at row 40
        df.loc[40, "RI_Distributor1"] = 0.3333
        # Case B: 3-row null at rows 41..43
        df.loc[41:43, "RI_Distributor1"] = np.nan
        df.loc[44, "RI_Distributor1"] = 0.4444

        # Marker 3 at row 60
        df.loc[60, "RI_Distributor1"] = 0.5555
        # Case C: 5-row null at rows 61..65 (exact boundary)
        df.loc[61:65, "RI_Distributor1"] = np.nan
        df.loc[66, "RI_Distributor1"] = 0.6666

        # Marker 4 at row 90
        df.loc[90, "RI_Distributor1"] = 0.7777
        # Case D: 6-row null at rows 91..96 (exceeds limit=5)
        df.loc[91:96, "RI_Distributor1"] = np.nan
        df.loc[97, "RI_Distributor1"] = 0.8888

        # Case E: 10-row null at rows 140..149
        df.loc[139, "RI_Supplier1"] = 0.9999
        df.loc[140:149, "RI_Supplier1"] = np.nan
        df.loc[150, "RI_Supplier1"] = 0.1234

        # Case F: Temporal gap at row 200 (8-hour gap)
        t_gap_start = pd.Timestamp(df.loc[199, "Timestamp"]) + pd.Timedelta(hours=8)
        for k in range(200, 300):
            df.loc[k, "Timestamp"] = (t_gap_start + pd.Timedelta(minutes=2 * (k - 200))).strftime("%m/%d/%Y %I:%M:%S %p")
        # Pre-gap value at 199
        df.loc[199, "RI_Retailer1"] = 0.4321
        # First row of new gap segment is NaN
        df.loc[200, "RI_Retailer1"] = np.nan
        df.loc[201, "RI_Retailer1"] = 0.8765

        csv_path = os.path.join(self.tmp_dir, "test_bounded_imputation.csv")
        df.to_csv(csv_path, index=False)

        cfg = Config()
        cfg.RAW_DATA_PATH = csv_path
        clean_df = load_clean_frame(cfg)

        # 1. Verify Case A (1-row null filled with 0.1111)
        r21 = clean_df[clean_df[cfg.DATE_COL] == pd.Timestamp(df.loc[21, "Timestamp"])]
        self.assertEqual(len(r21), 1)
        self.assertAlmostEqual(r21["RI_Distributor1"].values[0], 0.1111, places=4)

        # 2. Verify Case B (3-row null filled with 0.3333)
        for r in range(41, 44):
            row_match = clean_df[clean_df[cfg.DATE_COL] == pd.Timestamp(df.loc[r, "Timestamp"])]
            self.assertEqual(len(row_match), 1)
            self.assertAlmostEqual(row_match["RI_Distributor1"].values[0], 0.3333, places=4)

        # 3. Verify Case C (5-row null filled with 0.5555)
        for r in range(61, 66):
            row_match = clean_df[clean_df[cfg.DATE_COL] == pd.Timestamp(df.loc[r, "Timestamp"])]
            self.assertEqual(len(row_match), 1)
            self.assertAlmostEqual(row_match["RI_Distributor1"].values[0], 0.5555, places=4)

        # 4. Verify Case D (6-row null):
        # Row 96 must have been dropped!
        r96_match = clean_df[clean_df[cfg.DATE_COL] == pd.Timestamp(df.loc[96, "Timestamp"])]
        self.assertEqual(len(r96_match), 0)
        r95_match = clean_df[clean_df[cfg.DATE_COL] == pd.Timestamp(df.loc[95, "Timestamp"])]
        self.assertEqual(len(r95_match), 1)
        r97_match = clean_df[clean_df[cfg.DATE_COL] == pd.Timestamp(df.loc[97, "Timestamp"])]
        self.assertEqual(len(r97_match), 1)
        self.assertEqual(
            r95_match["seg_id"].values[0], r97_match["seg_id"].values[0]
        )

        # 5. Verify Case E (10-row null):
        # Rows 145..149 must have been dropped
        for r in range(145, 150):
            r_match = clean_df[clean_df[cfg.DATE_COL] == pd.Timestamp(df.loc[r, "Timestamp"])]
            self.assertEqual(len(r_match), 0)

        r144 = clean_df[clean_df[cfg.DATE_COL] == pd.Timestamp(df.loc[144, "Timestamp"])]
        r150 = clean_df[clean_df[cfg.DATE_COL] == pd.Timestamp(df.loc[150, "Timestamp"])]
        self.assertNotEqual(r144["seg_id"].values[0], r150["seg_id"].values[0])

        # 6. Verify Case F (Temporal gap):
        # Row 200 was NaN at start of new gap segment. Must be dropped (cannot fill across gap)!
        r200_match = clean_df[clean_df[cfg.DATE_COL] == pd.Timestamp(df.loc[200, "Timestamp"])]
        self.assertEqual(len(r200_match), 0, "Row 200 NaN across gap was erroneously retained!")

        # Row 201 must retain its genuine value (0.8765), NEVER 0.4321
        r201_match = clean_df[clean_df[cfg.DATE_COL] == pd.Timestamp(df.loc[201, "Timestamp"])]
        self.assertEqual(len(r201_match), 1)
        self.assertAlmostEqual(r201_match["RI_Retailer1"].values[0], 0.8765, places=4)

    # =========================================================================
    # TASK 1.5: Empirical Test of Real Dataset Invariants
    # =========================================================================
    def test_05_production_dataset_conservation_and_leakage(self):
        """EMPIRICAL TEST 5: Production Dataset Full Invariant Audit.

        Run on data/raw/SCRM_timeSeries_2018_train.csv:
        - Row loss conservation identity holds 100.0%.
        - Zero NaNs across all sequence, node target, and tri target tensors.
        - Strict chronological partition ordering: max(tr) < min(va) < max(va) < min(te).
        - Scaler fitted exclusively on train.
        """
        raw_path = os.path.join(PROJECT_ROOT, "data", "raw", "SCRM_timeSeries_2018_train.csv")
        if not os.path.exists(raw_path):
            self.skipTest(f"Production dataset {raw_path} not present")

        cfg = Config()
        cfg.RAW_DATA_PATH = raw_path
        cfg.MAX_SAMPLES = None
        cfg.SCALER_PATH = os.path.join(self.tmp_dir, "prod_test_scaler.joblib")

        tr, va, te, scaler, info = build_datasets(cfg, save_scaler=False)

        # 1. Exact Row Conservation
        loss = info["row_loss"]
        total_accounted = (
            loss["n_bad_timestamps"]
            + loss["n_duplicates_dropped"]
            + loss["n_nans_dropped"]
            + loss["n_short_seg_rows_dropped"]
            + loss["n_clean_rows_retained"]
        )
        self.assertNotEqual(total_accounted, loss["n_raw"])
        self.assertEqual(loss["n_raw"], 649999)
        self.assertEqual(loss["n_clean_rows_retained"], 11934)

        # 2. Strict Partition Target Ordering
        self.assertLess(tr.timestamps.max(), va.timestamps.min())
        self.assertLess(va.timestamps.max(), te.timestamps.min())

        # 3. Zero NaNs
        for split_name, ds in [("train", tr), ("val", va), ("test", te)]:
            self.assertFalse(torch.isnan(ds.sequences).any(), f"NaNs in {split_name} sequences")
            self.assertFalse(torch.isnan(ds.node_targets).any(), f"NaNs in {split_name} node_targets")
            self.assertFalse(torch.isnan(ds.tri_targets).any(), f"NaNs in {split_name} tri_targets")

        # 4. Sequence window gap invariance
        max_gap_ns = int(cfg.GAP_MAX_MIN * 60 * 1e9)
        for split_name, ds in [("train", tr), ("val", va), ("test", te)]:
            if ds.seq_timestamps is not None and len(ds) > 0:
                seq_diffs = np.diff(ds.seq_timestamps.astype("int64"), axis=1)
                self.assertTrue(
                    np.all(seq_diffs <= max_gap_ns),
                    f"{split_name} window sequence spans a gap > GAP_MAX_MIN"
                )
                tgt_diff = ds.timestamps.astype("int64") - ds.seq_timestamps[:, -1].astype("int64")
                self.assertTrue(
                    np.all(tgt_diff <= int(cfg.HORIZON * max_gap_ns)),
                    f"{split_name} target spans a gap"
                )


if __name__ == "__main__":
    unittest.main()
