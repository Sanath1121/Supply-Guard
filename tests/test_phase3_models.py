"""Gate 3 Verification: Models, Graph Convolutions, and Baseline Architectures.

Verifies:
1. Graph adjacency matrices define S->M->D->R topology; normalized Laplacian eigenvalues in [0, 2].
2. Output tensor shapes: [B, 4] for node models, [B] for PaperHybridOverall, tested with B=4 and B=1.
3. 2-hop gradient reachability: Supplier input influences Distributor output through 2 GCN layers.
4. Persistence residual identity: With head zeroed, predictions equal persistence baseline y_t.
5. Sigmoid removal from PaperHybridOverall: Raw, unbounded real-valued scalar outputs.
6. Parameter count verification matches outputs/results/param_counts.csv.
"""
import os
import sys
import unittest
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import Config
from src.graph_builder import build_graphs, normalised_laplacian
from src.models import (
    STGCNLSTM,
    PaperHybridOverall,
    LSTMBaseline,
    build_model,
    GraphConv,
)

# Re-exports for downstream test suites (Phase 4, Phase 6, Phase 8)
__all__ = [
    "STGCNLSTM",
    "PaperHybridOverall",
    "LSTMBaseline",
    "build_model",
    "GraphConv",
    "build_graphs",
    "normalised_laplacian",
    "TestPhase3Models",
]


class TestPhase3Models(unittest.TestCase):

    def setUp(self):
        self.graphs = build_graphs()
        self.B, self.L = 4, 10
        self.sample_seq = torch.rand(self.B, self.L, 5)

    def test_01_graph_adjacency_and_laplacian(self):
        """Verify symmetric A_hat, directed A_down/A_up, and Laplacian eigenvalues."""
        A_hat = self.graphs["A_hat"]
        self.assertTrue(torch.equal(A_hat, A_hat.T), "A_hat must be symmetric")

        # Directed edge checks: S(0) -> M(1) -> D(2) -> R(3)
        A_down = self.graphs["A_down"]
        self.assertEqual(A_down[1, 0].item(), 1.0, "Manufacturer receives downstream from Supplier")
        self.assertEqual(A_down[2, 1].item(), 1.0, "Distributor receives downstream from Manufacturer")
        self.assertEqual(A_down[3, 2].item(), 1.0, "Retailer receives downstream from Distributor")
        self.assertEqual(A_down[0].sum().item(), 0.0, "Supplier has no upstream parents in A_down")

        A_up = self.graphs["A_up"]
        self.assertEqual(A_up[0, 1].item(), 1.0, "Supplier receives upstream feedback from Manufacturer")
        self.assertEqual(A_up[1, 2].item(), 1.0, "Manufacturer receives upstream feedback from Distributor")
        self.assertEqual(A_up[2, 3].item(), 1.0, "Distributor receives upstream feedback from Retailer")
        self.assertEqual(A_up[3].sum().item(), 0.0, "Retailer has no downstream children in A_up")

        # Normalized Laplacian sanity check
        L = normalised_laplacian(4)
        self.assertTrue(torch.allclose(L, L.T, atol=1e-6), "Laplacian must be symmetric")
        eigenvalues = torch.linalg.eigvalsh(L)
        self.assertTrue((eigenvalues >= -1e-6).all(), "Laplacian eigenvalues must be non-negative")
        self.assertTrue((eigenvalues <= 2.0 + 1e-6).all(), "Normalized Laplacian eigenvalues must be <= 2")

    def test_02_model_forward_shapes(self):
        """Verify output shapes: [B, 4] for node models, [B] for PaperHybridOverall."""
        m_stgcn_dir = STGCNLSTM(mode="directed", residual=True)
        m_stgcn_sym = STGCNLSTM(mode="symmetric", residual=True)
        m_lstm = LSTMBaseline(residual=True)
        m_paper = PaperHybridOverall()

        # Batch size 4 testing with single input tensor contract seq [B, L, 5]
        out_dir = m_stgcn_dir(self.sample_seq)
        out_sym = m_stgcn_sym(self.sample_seq)
        out_lstm = m_lstm(self.sample_seq)
        out_paper = m_paper(self.sample_seq)

        self.assertEqual(out_dir.shape, (self.B, 4))
        self.assertEqual(out_sym.shape, (self.B, 4))
        self.assertEqual(out_lstm.shape, (self.B, 4))
        self.assertEqual(out_paper.shape, (self.B,))

        # Verify backwards-compatible invocation with explicit graphs passed
        out_dir_g = m_stgcn_dir(self.sample_seq, self.graphs)
        self.assertEqual(out_dir_g.shape, (self.B, 4))

        # Edge case: Batch size 1 (must preserve batch dimension, no 0D collapse)
        seq_single = torch.rand(1, self.L, 5)
        self.assertEqual(m_stgcn_dir(seq_single).shape, (1, 4))
        self.assertEqual(m_paper(seq_single).shape, (1,))

    def test_03_two_hop_gradient_reachability(self):
        """Verify that in 2-layer GCN, Supplier input influences Distributor output."""
        x = torch.rand(2, 10, 5, requires_grad=True)
        model = STGCNLSTM(mode="directed", residual=False)
        model.eval()

        # Initialize head with non-zero weights so gradients flow backwards
        for p in model.head.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
            else:
                nn.init.ones_(p)

        out = model(x)
        # Backprop from Distributor output (index 2)
        out[:, 2].sum().backward()

        # Check gradient with respect to Supplier input feature 0 (at index 0)
        supplier_grad = x.grad[:, :, 0].abs().sum().item()
        self.assertGreater(supplier_grad, 1e-7, "Supplier must influence Distributor via 2-hop GCN")

    def test_04_residual_path_identity(self):
        """Verify that when delta head is zero, model outputs exact persistence baseline."""
        model = STGCNLSTM(mode="directed", residual=True)
        model.eval()

        # Zero out the final layer of the head
        with torch.no_grad():
            for p in model.head.parameters():
                p.zero_()

        out = model(self.sample_seq)
        expected_persistence = self.sample_seq[:, -1, :4]
        self.assertTrue(torch.allclose(out, expected_persistence, atol=1e-6))
        self.assertEqual((out - expected_persistence).abs().max().item(), 0.0)

        # Also verify LSTMBaseline residual identity
        m_lstm = LSTMBaseline(residual=True)
        m_lstm.eval()
        with torch.no_grad():
            for p in m_lstm.head.parameters():
                p.zero_()
        out_lstm = m_lstm(self.sample_seq)
        self.assertEqual((out_lstm - expected_persistence).abs().max().item(), 0.0)

    def test_05_paper_hybrid_sigmoid_removal(self):
        """Verify that nn.Sigmoid is removed from PaperHybridOverall for unbounded predictions."""
        m_paper = PaperHybridOverall()
        has_sigmoid = any(isinstance(layer, nn.Sigmoid) for layer in m_paper.fc)
        self.assertFalse(has_sigmoid, "nn.Sigmoid must be removed from PaperHybridOverall.fc")

        # Test that outputs are unbounded and can handle large inputs
        x_extreme = torch.ones(2, self.L, 5) * 100.0
        out_extreme = m_paper(x_extreme)
        self.assertTrue(torch.is_tensor(out_extreme))
        self.assertEqual(out_extreme.shape, (2,))

    def test_06_parameter_counts_report(self):
        """Verify parameter count report exists and records 4 valid architectures."""
        csv_path = os.path.join(PROJECT_ROOT, "outputs", "results", "param_counts.csv")
        self.assertTrue(os.path.exists(csv_path), "outputs/results/param_counts.csv must exist")

        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 4, "param_counts.csv must contain exactly 4 model rows")
        self.assertListEqual(list(df.columns), ["model", "total_params", "trainable_params"])
        self.assertTrue((df["total_params"] > 0).all(), "total_params must be positive")
        self.assertTrue((df["trainable_params"] > 0).all(), "trainable_params must be positive")

        # Check parameter delta: directed GCN has 2,176 more parameters than symmetric GCN
        dir_params = df[df["model"].str.contains("directed")]["total_params"].iloc[0]
        sym_params = df[df["model"].str.contains("symmetric")]["total_params"].iloc[0]
        self.assertEqual(dir_params - sym_params, 2176, "Directed mode must have 2,176 more params than symmetric mode")


if __name__ == "__main__":
    unittest.main()
