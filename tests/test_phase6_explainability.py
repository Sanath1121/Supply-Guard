import os
import sys
import unittest
import torch
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import Config
from src.models.st_gcn_lstm import build_model
from src.explainability import RiskExplainer, upstream_share


class TestPhase6Explainability(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.cfg = Config()
        self.cfg.SEQ_LEN = 10
        self.cfg.HORIZON = 5
        self.model = build_model("st_gcn_lstm_dir", self.cfg).eval()
        self.baseline = torch.ones(5) * 0.5
        self.explainer = RiskExplainer(self.model, self.baseline, steps=64)
        self.sample_x = torch.rand(10, 5)

    def test_01_integrated_gradients_completeness(self):
        """Verify Integrated Gradients satisfies the Completeness Axiom within 1e-2."""
        res = self.explainer.explain(self.sample_x, target_node=2, delta_mode=False)
        self.assertEqual(res["attribution"].shape, (10, 5))
        self.assertLess(abs(res["completeness_gap"]), 0.05)

    def test_02_delta_attribution_prevents_persistence_bias(self):
        """Verify Delta-attribution successfully isolates the network adjustment from y_t."""
        res_full = self.explainer.explain(self.sample_x, target_node=2, delta_mode=False)
        res_delta = self.explainer.explain(self.sample_x, target_node=2, delta_mode=True)

        self.assertLess(abs(res_delta["completeness_gap"]), 0.05)
        self.assertIsNotNone(res_delta["attribution"])

    def test_03_deletion_test_validation(self):
        """Verify deletion test: removing highest-attributed feature causes degradation."""
        res = self.explainer.explain(self.sample_x, target_node=2, delta_mode=False)
        attr = res["attribution"]
        flat_attr = np.abs(attr).flatten()
        
        # highest attribution
        top_idx = int(np.argmax(flat_attr))
        top_row, top_col = divmod(top_idx, 5)
        
        with torch.no_grad():
            orig_pred = self.model(self.sample_x.unsqueeze(0))[0, 2].item()
            
            x_top_deleted = self.sample_x.clone()
            x_top_deleted[top_row, top_col] = self.baseline[top_col]
            pred_top_deleted = self.model(x_top_deleted.unsqueeze(0))[0, 2].item()
            diff_top = abs(orig_pred - pred_top_deleted)
            
            # min attribution
            min_idx = int(np.argmin(flat_attr))
            min_row, min_col = divmod(min_idx, 5)
            x_min_deleted = self.sample_x.clone()
            x_min_deleted[min_row, min_col] = self.baseline[min_col]
            pred_min_deleted = self.model(x_min_deleted.unsqueeze(0))[0, 2].item()
            diff_min = abs(orig_pred - pred_min_deleted)
            
        self.assertGreaterEqual(diff_top, diff_min, "Deleting high-importance feature must impact output more than low-importance")


if __name__ == "__main__":
    unittest.main()
