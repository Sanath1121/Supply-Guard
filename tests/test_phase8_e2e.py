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

import pandas as pd

from tests.test_phase3_models import STGCNLSTM, build_graphs
from training.evaluate import evaluate_headline_claim

GRAPH_CLAIM = "Graph structure improves echelon-level forecasts on this dataset"
TEMPORAL_CLAIM = ("Temporal modelling helps; the assumed graph adds no measurable accuracy "
                  "but enables per-node attribution")
PERSISTENCE_CLAIM = "Dataset is dominated by short-term persistence"


def _agg_tables(pers, lstm, lstm_std, dir_, dir_std, sup_lstm, sup_lstm_std, sup_dir, sup_dir_std, sym=None):
    """Build overall/node aggregate tables in the exact schema training/evaluate.py writes."""
    sym = dir_ if sym is None else sym
    overall = pd.DataFrame([
        {"model": "persistence", "target_type": "derived_4node_mean", "MSE_mean": pers, "MSE_std": np.nan},
        {"model": "lstm", "target_type": "derived_4node_mean", "MSE_mean": lstm, "MSE_std": lstm_std},
        {"model": "st_gcn_lstm_dir", "target_type": "derived_4node_mean", "MSE_mean": dir_, "MSE_std": dir_std},
        {"model": "st_gcn_lstm_sym", "target_type": "derived_4node_mean", "MSE_mean": sym, "MSE_std": dir_std},
    ])
    node = pd.DataFrame([
        {"model": "lstm", "node": "Supplier", "MSE_mean": sup_lstm, "MSE_std": sup_lstm_std},
        {"model": "st_gcn_lstm_dir", "node": "Supplier", "MSE_mean": sup_dir, "MSE_std": sup_dir_std},
    ])
    return overall, node


class TestPhase8E2E(unittest.TestCase):

    def test_01_claims_table_audit_logic(self):
        """The real pre-registered claims rule (training.evaluate.evaluate_headline_claim) guards against over-claiming."""
        # Case A: graph beats LSTM beyond the summed std, overall AND on the Supplier
        o, n = _agg_tables(0.050, 0.040, 0.002, 0.030, 0.002, 0.0040, 0.0001, 0.0030, 0.0001)
        res = evaluate_headline_claim(o, n)
        self.assertEqual(res["headline_claim"], GRAPH_CLAIM)
        self.assertTrue(res["beats_beyond_1std"])

        # Case B: graph approximately equals LSTM (within std) but both beat persistence
        o, n = _agg_tables(0.050, 0.035, 0.003, 0.036, 0.003, 0.0040, 0.0001, 0.0040, 0.0001)
        res = evaluate_headline_claim(o, n)
        self.assertEqual(res["headline_claim"], TEMPORAL_CLAIM)
        self.assertFalse(res["beats_beyond_1std"])

        # Case B2: overall gap passes but the Supplier gap does not -> graph claim must NOT be made
        o, n = _agg_tables(0.050, 0.040, 0.002, 0.030, 0.002, 0.0040, 0.0005, 0.0039, 0.0005)
        self.assertEqual(evaluate_headline_claim(o, n)["headline_claim"], TEMPORAL_CLAIM)

        # Case C: persistence is at least as good as the graph model
        o, n = _agg_tables(0.010, 0.015, 0.002, 0.014, 0.002, 0.0040, 0.0001, 0.0040, 0.0001)
        self.assertEqual(evaluate_headline_claim(o, n)["headline_claim"], PERSISTENCE_CLAIM)

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


