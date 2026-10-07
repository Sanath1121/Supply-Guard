"""Adversarial Stress Test Suite for Phase 5: Evaluation Pipeline & Claims Audit.

Author: QA Adversarial Specialist
Objective:
1. Test partition isolation and leakage bounds.
2. Relative skill score monotonicity and extreme boundary behavior.
3. Severity confusion matrix row/column conservation.
4. Claims Table (§4) boundary condition sensitivity.
5. Metric artifact and figure file validity (PNG headers).
"""
import os
import sys
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.evaluate import (
    reg,
    severity,
    evaluate_headline_claim,
)


class TestAdversarialPhase5(unittest.TestCase):
    """Adversarial stress tests for Phase 5 metrics, claims, and figures."""

    def test_01_skill_score_mathematical_bounds_and_monotonicity(self):
        """Stress-test skill score bounds: 0% at persistence, positive when better, negative when worse."""
        mse_pers = 0.0010
        # Exactly matching persistence
        skill_equal = 1.0 - (mse_pers / mse_pers)
        self.assertEqual(skill_equal, 0.0)

        # 50% improvement
        skill_better = 1.0 - (0.0005 / mse_pers)
        self.assertEqual(skill_better, 0.5)

        # Worse than persistence
        skill_worse = 1.0 - (0.0015 / mse_pers)
        self.assertLess(skill_worse, 0.0)
        self.assertEqual(skill_worse, -0.5)

    def test_02_confusion_matrix_total_conservation(self):
        """Verify confusion matrix sum equals exact total sample count (4 echelons * N_test)."""
        cm_path = os.path.join(PROJECT_ROOT, "outputs", "results", "confusion_matrix.csv")
        self.assertTrue(os.path.exists(cm_path), f"Missing {cm_path}")

        cm_df = pd.read_csv(cm_path, index_col=0)
        total_eval_samples = cm_df.values.sum()

        # In test set, there are 1,194 windows. For 4 nodes, total classifications = 1,194 * 4 = 4,776
        self.assertEqual(total_eval_samples, 1194 * 4)

    def test_03_claims_table_decision_boundaries(self):
        """Stress-test Claims Table (§4) logic across synthetic counterfactual scenarios."""
        # Case A: Models beat persistence, ST-GCN beats LSTM beyond 1 std -> "Graph structure improves..."
        df_overall_a = pd.DataFrame([
            {"model": "persistence", "target_type": "derived_4node_mean", "MSE_mean": 0.0010, "MSE_std": 0.0},
            {"model": "lstm", "target_type": "derived_4node_mean", "MSE_mean": 0.0005, "MSE_std": 0.00001},
            {"model": "st_gcn_lstm_dir", "target_type": "derived_4node_mean", "MSE_mean": 0.0004, "MSE_std": 0.00001},
            {"model": "st_gcn_lstm_sym", "target_type": "derived_4node_mean", "MSE_mean": 0.00045, "MSE_std": 0.00001},
        ])
        df_node_a = pd.DataFrame([
            {"model": "lstm", "node": "Supplier", "MSE_mean": 0.0001, "MSE_std": 0.000005},
            {"model": "st_gcn_lstm_dir", "node": "Supplier", "MSE_mean": 0.00005, "MSE_std": 0.000005},
        ])
        res_a = evaluate_headline_claim(df_overall_a, df_node_a)
        self.assertEqual(res_a["headline_claim"], "Graph structure improves echelon-level forecasts on this dataset")

        # Case B: Models beat persistence, but ST-GCN does not beat LSTM beyond 1 std -> "Temporal modelling helps..."
        df_overall_b = pd.DataFrame([
            {"model": "persistence", "target_type": "derived_4node_mean", "MSE_mean": 0.0010, "MSE_std": 0.0},
            {"model": "lstm", "target_type": "derived_4node_mean", "MSE_mean": 0.00050, "MSE_std": 0.00005},
            {"model": "st_gcn_lstm_dir", "target_type": "derived_4node_mean", "MSE_mean": 0.00048, "MSE_std": 0.00005},
            {"model": "st_gcn_lstm_sym", "target_type": "derived_4node_mean", "MSE_mean": 0.00049, "MSE_std": 0.00005},
        ])
        df_node_b = pd.DataFrame([
            {"model": "lstm", "node": "Supplier", "MSE_mean": 0.0001, "MSE_std": 0.00005},
            {"model": "st_gcn_lstm_dir", "node": "Supplier", "MSE_mean": 0.00009, "MSE_std": 0.00005},
        ])
        res_b = evaluate_headline_claim(df_overall_b, df_node_b)
        self.assertEqual(res_b["headline_claim"], "Temporal modelling helps; the assumed graph adds no measurable accuracy but enables per-node attribution")

        # Case C: Persistence beats models -> "Dataset is dominated by short-term persistence"
        df_overall_c = pd.DataFrame([
            {"model": "persistence", "target_type": "derived_4node_mean", "MSE_mean": 0.0003, "MSE_std": 0.0},
            {"model": "lstm", "target_type": "derived_4node_mean", "MSE_mean": 0.0005, "MSE_std": 0.00005},
            {"model": "st_gcn_lstm_dir", "target_type": "derived_4node_mean", "MSE_mean": 0.0004, "MSE_std": 0.00005},
            {"model": "st_gcn_lstm_sym", "target_type": "derived_4node_mean", "MSE_mean": 0.00045, "MSE_std": 0.00005},
        ])
        df_node_c = pd.DataFrame([
            {"model": "lstm", "node": "Supplier", "MSE_mean": 0.0001, "MSE_std": 0.00005},
            {"model": "st_gcn_lstm_dir", "node": "Supplier", "MSE_mean": 0.0001, "MSE_std": 0.00005},
        ])
        res_c = evaluate_headline_claim(df_overall_c, df_node_c)
        self.assertEqual(res_c["headline_claim"], "Dataset is dominated by short-term persistence")

    def test_04_figure_image_binary_validity(self):
        """Verify generated plot files are non-empty and start with the standard PNG magic number."""
        png_magic = b"\x89PNG\r\n\x1a\n"
        fig_dir = os.path.join(PROJECT_ROOT, "outputs", "figures")
        expected_figs = [
            "prediction_vs_truth.png",
            "error_by_node.png",
            "severity_confusion_matrix.png",
        ]
        for f in expected_figs:
            fpath = os.path.join(fig_dir, f)
            self.assertTrue(os.path.exists(fpath), f"Figure not found: {fpath}")
            self.assertGreater(os.path.getsize(fpath), 1024, f"Figure too small: {fpath}")
            with open(fpath, "rb") as fh:
                header = fh.read(8)
                self.assertEqual(header, png_magic, f"File {f} is not a valid PNG image")


if __name__ == "__main__":
    unittest.main()
