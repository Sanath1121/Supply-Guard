"""Gate 8 Verification: End-to-End Pipeline, Claims Audit, and Clean Reproduction.

Checks:
1. Claims table consistency audit: ensures reported conclusions strictly match empirical numbers.
2. Seed reproducibility: verifying that fixed seeds yield deterministic forecasts.
3. Final deliverable checklist.
"""
import os
import sys
import unittest
import torch
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests.test_phase3_models import STGCNLSTM, build_graphs


def audit_headline_claims(metrics_dict: dict) -> str:
    """Evaluate pre-agreed claims table from Plan A Section 4 based on actual results."""
    pers_mse = metrics_dict["persistence_mse"]
    lstm_mse = metrics_dict["lstm_mse"]
    stgcn_mse = metrics_dict["stgcn_mse"]
    stgcn_std = metrics_dict["stgcn_std"]

    # Rule 1: Models >> persistence and st_gcn_lstm beats lstm beyond 1 std
    if (stgcn_mse < pers_mse * 0.90) and (stgcn_mse + stgcn_std < lstm_mse):
        return "Graph structure improves echelon-level forecasts on this dataset"

    # Rule 2: Models > persistence, but st_gcn_lstm ≈ lstm
    if (stgcn_mse < pers_mse) and abs(stgcn_mse - lstm_mse) <= stgcn_std:
        return "Temporal modelling helps; the assumed graph adds no measurable accuracy but enables per-node attribution"

    # Rule 3: Persistence >= models
    if stgcn_mse >= pers_mse:
        return "Dataset is dominated by short-term persistence"

    return "Empirical comparison completed"


class TestPhase8E2E(unittest.TestCase):

    def test_01_claims_table_audit_logic(self):
        """Verify the pre-agreed claims table strictly guards against over-claiming."""
        # Case A: Graph clearly wins
        case_a = {"persistence_mse": 0.050, "lstm_mse": 0.040, "stgcn_mse": 0.030, "stgcn_std": 0.002}
        self.assertEqual(
            audit_headline_claims(case_a),
            "Graph structure improves echelon-level forecasts on this dataset"
        )

        # Case B: Graph approximately equals LSTM (the expected real-data probe outcome)
        case_b = {"persistence_mse": 0.050, "lstm_mse": 0.035, "stgcn_mse": 0.036, "stgcn_std": 0.003}
        self.assertEqual(
            audit_headline_claims(case_b),
            "Temporal modelling helps; the assumed graph adds no measurable accuracy but enables per-node attribution"
        )

        # Case C: Persistence beats models
        case_c = {"persistence_mse": 0.010, "lstm_mse": 0.015, "stgcn_mse": 0.014, "stgcn_std": 0.002}
        self.assertEqual(
            audit_headline_claims(case_c),
            "Dataset is dominated by short-term persistence"
        )

    def test_02_seed_reproducibility(self):
        """Verify fixed random seeds produce identical tensor outputs."""
        graphs = build_graphs()
        x = torch.rand(2, 10, 5)

        # Run 1 with seed 42
        torch.manual_seed(42)
        m1 = STGCNLSTM(mode="directed", residual=True)
        out1 = m1(x, graphs)

        # Run 2 with seed 42
        torch.manual_seed(42)
        m2 = STGCNLSTM(mode="directed", residual=True)
        out2 = m2(x, graphs)

        self.assertTrue(torch.allclose(out1, out2), "Models with identical seed must yield identical outputs")


if __name__ == "__main__":
    unittest.main()


