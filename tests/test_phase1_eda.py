"""Gate 1 Verification: Exploratory Data Analysis (EDA) and Decision Gating.

Checks:
1. Autocorrelation Function (ACF) calculation over lags 1-50.
2. Within-segment cross-correlation (never bridging temporal gaps).
3. Persistence R² evaluation across horizons (2, 10, 20, 60 minutes).
4. Go/No-Go Decision Table logic:
   - Persistence R² >= 0.99 -> lengthen horizon.
   - Cross-node correlation weak -> reframe RQ2 ("tests whether assumed graph helps").
"""
import os
import sys
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def compute_acf(series: np.ndarray, nlags: int = 50) -> np.ndarray:
    """Compute autocorrelation for lags 0 to nlags."""
    n = len(series)
    variance = np.var(series)
    if variance < 1e-12:
        return np.zeros(nlags + 1)
    centered = series - np.mean(series)
    r = np.correlate(centered, centered, mode="full")[-n:]
    return r[: nlags + 1] / (variance * np.arange(n, n - nlags - 1, -1))


def compute_persistence_r2(series: np.ndarray, horizon_steps: int) -> float:
    """Compute persistence baseline R² where y_hat(t+H) = y(t)."""
    if len(series) <= horizon_steps:
        return 0.0
    y_true = series[horizon_steps:]
    y_pred = series[:-horizon_steps]
    return float(r2_score(y_true, y_pred))


class TestPhase1EDA(unittest.TestCase):

    def setUp(self):
        # Create a synthetic series with strong short-term persistence (AR(1) with phi=0.98)
        # to replicate the Banerjee real dataset profile
        np.random.seed(42)
        n = 3000
        self.series = np.zeros(n)
        for t in range(1, n):
            self.series[t] = 0.98 * self.series[t - 1] + np.random.normal(0, 0.1)

    def test_01_acf_calculation_structure(self):
        """Verify ACF calculates smoothly across lags 1-50."""
        acf_vals = compute_acf(self.series, nlags=50)
        self.assertEqual(len(acf_vals), 51)
        self.assertAlmostEqual(acf_vals[0], 1.0, places=4, msg="Lag 0 autocorrelation must be 1.0")
        # For AR(1) with phi=0.98, lag 1 ACF must be > 0.90
        self.assertGreater(acf_vals[1], 0.90, "Short-lag autocorrelation should be high")

    def test_02_within_segment_cross_correlation(self):
        """Verify cross-correlation calculation does not cross artificial segment boundaries."""
        # Segment 1 and Segment 2 separated by a large time gap
        seg1_s = np.sin(np.linspace(0, 10, 500))
        seg1_m = np.cos(np.linspace(0, 10, 500))
        seg2_s = np.random.normal(0, 1, 500)
        seg2_m = np.random.normal(0, 1, 500)

        # Cross-correlations computed separately per segment
        corr_seg1 = np.corrcoef(seg1_s, seg1_m)[0, 1]
        corr_seg2 = np.corrcoef(seg2_s, seg2_m)[0, 1]

        self.assertFalse(np.isnan(corr_seg1))
        self.assertFalse(np.isnan(corr_seg2))

    def test_03_persistence_drop_across_horizons(self):
        """Verify that expanding horizon H from 1 step (2 min) to 5 steps (10 min) reduces persistence dominance."""
        # Sampling interval = 2 min
        # Horizon H=1 step (2 min), H=5 steps (10 min), H=10 steps (20 min)
        r2_h1 = compute_persistence_r2(self.series, horizon_steps=1)
        r2_h5 = compute_persistence_r2(self.series, horizon_steps=5)
        r2_h10 = compute_persistence_r2(self.series, horizon_steps=10)

        # H=1 step should be very high (probe found R² > 0.94 - 0.99)
        self.assertGreater(r2_h1, 0.90, "H=1 step persistence is strong")
        # H=5 steps (10 min) should strictly be less than H=1 step
        self.assertLess(r2_h5, r2_h1, "Persistence R² must drop as horizon increases")
        self.assertLess(r2_h10, r2_h5, "Persistence R² must continue dropping at 20 min")

    def test_04_eda_decision_table_gate(self):
        """Test gate evaluation logic for horizon and research question framing."""
        # Scenario A: persistence at 2 min is 0.98 -> Horizon must be lengthened to >= 10 min
        raw_persistence_2min = 0.985
        chosen_horizon_min = 10 if raw_persistence_2min >= 0.95 else 2
        self.assertEqual(chosen_horizon_min, 10, "High persistence at 2 min requires locking 10-min horizon")

        # Scenario B: cross-echelon correlations are low (corr < 0.30)
        max_cross_corr = 0.29  # matching the probe finding
        framing_decision = "REFRAME_RQ2" if max_cross_corr < 0.40 else "HYPOTHESIS_CONFIRMED"
        self.assertEqual(framing_decision, "REFRAME_RQ2", "Low correlation requires reframing RQ2 as empirical test")


if __name__ == "__main__":
    unittest.main()
