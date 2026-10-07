import os
import sys
import unittest
import torch
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import Config
from src.models.st_gcn_lstm import build_model
from src.explainability import RiskExplainer, upstream_share, narrate


class TestPhase6Explainability(unittest.TestCase):
    """Gate 6 verification suite for production explainability pipeline."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        np.random.seed(42)
        cls.cfg = Config()
        cls.cfg.SEQ_LEN = 10
        cls.cfg.HORIZON = 5

        # Production checkpoint and training-mean baseline vector
        cls.model = build_model("st_gcn_lstm_dir", cls.cfg).eval()
        ckpt_path = os.path.join(PROJECT_ROOT, "outputs", "models", "st_gcn_lstm_dir_seed42.pt")
        if os.path.exists(ckpt_path):
            cls.model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))

        # Dataset training-mean baseline vector in scaled space
        cls.baseline = torch.tensor([0.8350868, 0.4200690, 0.3848433, 0.5827187, 0.4981064], dtype=torch.float32)
        cls.explainer = RiskExplainer(cls.model, cls.baseline, steps=64)

        # Standard test sample with non-trivial variation
        cls.sample_x = cls.baseline.unsqueeze(0).repeat(10, 1) + torch.randn(10, 5) * 0.1

    def test_01_integrated_gradients_completeness(self):
        """Verify Integrated Gradients satisfies the Completeness Axiom within 0.05 on production model."""
        res = self.explainer.explain(self.sample_x, target_node=3, residual_delta=False)
        self.assertEqual(res["attribution"].shape, (10, 5))
        self.assertLess(abs(res["completeness_gap"]), 0.05)
        self.assertIn("feature_importance", res)
        self.assertIn("time_importance", res)

    def test_02_delta_attribution_prevents_persistence_bias(self):
        """Verify Delta-attribution isolates network adjustment from y_t with completeness gap < 0.05."""
        res_full = self.explainer.explain(self.sample_x, target_node=3, residual_delta=False)
        res_delta = self.explainer.explain(self.sample_x, target_node=3, residual_delta=True)

        self.assertLess(abs(res_delta["completeness_gap"]), 0.05)
        self.assertEqual(res_delta["attribution"].shape, (10, 5))
        self.assertIsNotNone(res_delta["feature_importance"])

    def test_03_deletion_test_validation(self):
        """Verify feature deletion: masking top-attributed feature degrades forecast more than least-attributed."""
        res = self.explainer.explain(self.sample_x, target_node=3, residual_delta=False)
        feat_imp = np.array(res["feature_importance"])
        top_f = int(np.argmax(feat_imp))
        min_f = int(np.argmin(feat_imp))

        with torch.no_grad():
            orig_pred = self.model(self.sample_x.unsqueeze(0))[0, 3].item()

            x_top_deleted = self.sample_x.clone()
            x_top_deleted[:, top_f] = self.baseline[top_f]
            pred_top_deleted = self.model(x_top_deleted.unsqueeze(0))[0, 3].item()
            diff_top = abs(orig_pred - pred_top_deleted)

            x_min_deleted = self.sample_x.clone()
            x_min_deleted[:, min_f] = self.baseline[min_f]
            pred_min_deleted = self.model(x_min_deleted.unsqueeze(0))[0, 3].item()
            diff_min = abs(orig_pred - pred_min_deleted)

        self.assertGreater(
            diff_top,
            diff_min,
            f"Deleting top feature {top_f} (diff {diff_top:.4f}) must impact output more than min feature {min_f} (diff {diff_min:.4f})"
        )

    def test_04_upstream_share_bounds_and_structure(self):
        """Verify upstream share metric obeys [0, 1] bounds and hierarchical chain topology."""
        res_retailer = self.explainer.explain(self.sample_x, target_node=3, residual_delta=True)
        up_share_ret = upstream_share(res_retailer)
        self.assertGreaterEqual(up_share_ret, 0.0)
        self.assertLessEqual(up_share_ret, 1.0)

        # Supplier node (target=0) has no upstream echelons in S -> M -> D -> R
        res_supplier = self.explainer.explain(self.sample_x, target_node=0, residual_delta=True)
        up_share_sup = upstream_share(res_supplier)
        self.assertEqual(up_share_sup, 0.0)

    def test_05_narrative_integrity_and_causal_claim_prohibition(self):
        """Verify narrate() produces valid English and strictly forbids the phrase 'root cause'."""
        res = self.explainer.explain(self.sample_x, target_node=3, residual_delta=True)
        narrative = narrate(res, seq_len=self.cfg.SEQ_LEN)

        self.assertIsInstance(narrative, str)
        self.assertIn("Forecast Retailer risk", narrative)
        self.assertIn("Most influential input:", narrative)
        self.assertIn("Most influential time step:", narrative)
        self.assertIn("Upstream echelons contribute", narrative)

        # AGENTS.md Rule 5: Attribution is sensitivity, NEVER assert root cause
        self.assertNotIn(
            "root cause",
            narrative.lower(),
            "Violation of AGENTS.md Rule 5: 'root cause' phrasing is forbidden in explainability outputs."
        )

    def test_06_production_attribution_artifact(self):
        """Verify production artifact outputs/results/attribution_examples.csv satisfies all Gate 6 requirements."""
        csv_path = os.path.join(PROJECT_ROOT, "outputs", "results", "attribution_examples.csv")
        self.assertTrue(os.path.exists(csv_path), f"Missing Gate 6 artifact: {csv_path}")

        df = pd.read_csv(csv_path)
        self.assertGreaterEqual(len(df), 10, "Attribution examples CSV must contain at least 10 evaluation windows")

        required_cols = [
            "window_index", "target_node", "completeness_gap",
            "upstream_share_dir", "upstream_share_sym",
            "deletion_drop_top", "deletion_drop_rand",
            "deletion_test_passed", "narrative"
        ]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Missing required column in attribution examples CSV: {col}")

        # Check completeness gaps across all 10 windows
        max_gap = df["completeness_gap"].abs().max()
        self.assertLess(max_gap, 0.05, f"Completeness gap exceeded 0.05 in artifact: {max_gap}")

        # Check deletion test pass rate (must meet Gate 6 requirement >= 60%)
        pass_rate = df["deletion_test_passed"].mean()
        self.assertGreaterEqual(pass_rate, 0.60, f"Deletion test pass rate {pass_rate:.1%} fell below 60% threshold")

        # Check all narratives for absence of 'root cause'
        for text in df["narrative"]:
            self.assertNotIn("root cause", str(text).lower(), "Artifact narrative contains forbidden 'root cause' phrasing")


if __name__ == "__main__":
    unittest.main()
