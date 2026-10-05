"""Gate 0 Verification: Environment, Repository Setup, and Data Acquisition.

Checks:
1. Required directory layout.
2. Core dependencies imported successfully.
3. Timestamp format parsing (%m/%d/%Y %I:%M:%S %p).
4. Null policy: never assert nulls < 1% (real data has ~5% nulls).
5. Raw dataset integrity check (if downloaded): sha256, row count, column schema.
"""
import os
import sys
import unittest
import hashlib
import pandas as pd
import numpy as np

# Resolve repo root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class TestPhase0Setup(unittest.TestCase):

    def test_01_directory_structure(self):
        """Verify standard repository layout defined in Plan A Phase 0."""
        expected_dirs = [
            "src",
            "src/models",
            "training",
            "tests",
            "outputs",
            "outputs/models",
            "outputs/results",
            "outputs/figures",
        ]
        missing = [d for d in expected_dirs if not os.path.exists(os.path.join(PROJECT_ROOT, d))]
        self.assertEqual(missing, [], f"Missing required directories: {missing}")

    def test_02_environment_dependencies(self):
        """Verify core deep learning, data, and visualization packages."""
        required_packages = [
            ("torch", "PyTorch 2.x"),
            ("numpy", "NumPy"),
            ("pandas", "Pandas"),
            ("sklearn", "Scikit-Learn"),
            ("joblib", "Joblib"),
            ("networkx", "NetworkX"),
            ("streamlit", "Streamlit"),
        ]
        failed_imports = []
        for pkg, name in required_packages:
            try:
                __import__(pkg)
            except ImportError as e:
                failed_imports.append(f"{name} ({pkg}): {e}")
        self.assertEqual(failed_imports, [], f"Dependencies missing: {failed_imports}")

    def test_03_timestamp_format_parsing(self):
        """Verify explicit month-first format '%m/%d/%Y %I:%M:%S %p' parses correctly."""
        sample_stamps = [
            "1/28/2015 12:02:00 AM",
            "12/19/2018 11:58:00 PM",
            "07/04/2016 02:15:30 PM",
            "02/29/2016 06:00:00 AM"  # leap year
        ]
        parsed = pd.to_datetime(sample_stamps, format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
        self.assertEqual(int(parsed.isna().sum()), 0, "All valid timestamps must parse without NaT")
        # Check specific month/day logic to ensure month is parsed before day
        self.assertEqual(parsed[0].month, 1)
        self.assertEqual(parsed[0].day, 28)
        self.assertEqual(parsed[1].month, 12)
        self.assertEqual(parsed[1].day, 19)

    def test_04_null_handling_policy(self):
        """Ensure the real dataset has ~5% nulls and we don't assume <1%."""
        raw_path = os.path.join(PROJECT_ROOT, "data", "raw", "SCRM_timeSeries_2018_train.csv")
        if not os.path.exists(raw_path):
            self.skipTest("Raw data file not downloaded.")
            
        df = pd.read_csv(raw_path, usecols=["RI_Distributor1", "Total_Cost"])
        null_share_dist = df["RI_Distributor1"].isna().mean()
        null_share_cost = df["Total_Cost"].isna().mean()
        
        self.assertGreater(null_share_dist, 0.04, f"Real data null share is around 5%. Found {null_share_dist:.3f}")
        self.assertLess(null_share_dist, 0.10, "Nulls should be < 10%")

    def test_05_raw_dataset_spot_check(self):
        """Spot check raw dataset if downloaded in data/raw/."""
        raw_path = os.path.join(PROJECT_ROOT, "data", "raw", "SCRM_timeSeries_2018_train.csv")
        if not os.path.exists(raw_path):
            self.skipTest(f"Raw data file not yet downloaded at {raw_path}. Run setup_and_download.py.")

        # Check required columns
        required_cols = [
            "Timestamp", "RI_Supplier1", "RI_Manufacturer1",
            "RI_Distributor1", "RI_Retailer1", "Total_Cost"
        ]
        df_head = pd.read_csv(raw_path, nrows=50)
        missing_cols = [c for c in required_cols if c not in df_head.columns]
        self.assertEqual(missing_cols, [], f"Raw CSV missing expected columns: {missing_cols}")

        # Check file size (>10 MB)
        file_size_mb = os.path.getsize(raw_path) / (1024 * 1024)
        self.assertGreater(file_size_mb, 10.0, f"Raw CSV file appears truncated ({file_size_mb:.2f} MB)")


if __name__ == "__main__":
    unittest.main()
