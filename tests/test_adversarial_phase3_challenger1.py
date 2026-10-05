"""Adversarial stress test suite for Phase 3: Model Architectures & Graph Topologies.

Author: Challenger 1 (teamwork_preview_challenger_p3_1)
Objective: Empirically stress-test Invariants 1 & 2:
1. S->M->D->R Topology, Nilpotency, Spectral Theorem on Laplacian & Kipf-Welling Adjacency,
   Degenerate and Perturbed Graph Topologies.
2. STGCNLSTM & PaperHybridOverall Shape Invariants across boundary conditions (B=1, B=7, B=128,
   L=1, 5, 10, 30, 50), 0D collapse prevention, and empirical proof of unbounded outputs.
"""
import copy
import os
import sys
import unittest
import torch
import torch.nn as nn

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


class TestAdversarialPhase3Challenger1(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.graphs = build_graphs()

    # =========================================================================
    # INVARIANT 1: Topology, Nilpotency, Spectral Properties & Perturbations
    # =========================================================================

    def test_01_dag_topology_and_nilpotency(self):
        """Verify S->M->D->R DAG topology, strict nilpotency, and directional flows."""
        A_dir = self.graphs["A_dir"]
        self.assertEqual(A_dir.shape, (4, 4))
        # Exact non-zero entries must be only (0,1), (1,2), (2,3)
        expected_edges = {(0, 1), (1, 2), (2, 3)}
        actual_edges = set((i, j) for i in range(4) for j in range(4) if A_dir[i, j] > 0)
        self.assertEqual(actual_edges, expected_edges, "A_dir must strictly have S->M->D->R edges")

        # DAG Nilpotency check: A_dir^4 must be strictly zero matrix
        A2 = torch.matmul(A_dir, A_dir)
        A3 = torch.matmul(A2, A_dir)
        A4 = torch.matmul(A3, A_dir)
        self.assertEqual(A2.sum().item(), 2.0, "2-hop paths in 4-node line graph must be 2")
        self.assertEqual(A3.sum().item(), 1.0, "3-hop path in 4-node line graph must be 1 (S->R)")
        self.assertEqual(A4.sum().item(), 0.0, "A_dir^4 must be strictly 0 for 4-node DAG")
        self.assertTrue(torch.all(A4 == 0.0), "Nilpotent index of 4-node DAG must be <= 4")

        # A_down and A_up directional properties and nilpotency
        A_down = self.graphs["A_down"]
        A_up = self.graphs["A_up"]
        self.assertEqual(torch.matmul(torch.matmul(A_down, A_down), torch.matmul(A_down, A_down)).sum().item(), 0.0)
        self.assertEqual(torch.matmul(torch.matmul(A_up, A_up), torch.matmul(A_up, A_up)).sum().item(), 0.0)

        # Row stochasticity: non-isolated rows must sum exactly to 1.0
        self.assertEqual(A_down[0].sum().item(), 0.0, "Supplier has no upstream in A_down")
        self.assertTrue(torch.allclose(A_down[1:].sum(dim=1), torch.ones(3)))
        self.assertEqual(A_up[3].sum().item(), 0.0, "Retailer has no downstream in A_up")
        self.assertTrue(torch.allclose(A_up[:3].sum(dim=1), torch.ones(3)))

    def test_02_laplacian_spectral_properties(self):
        """Verify spectral properties of Normalized Laplacian and Kipf-Welling A_hat."""
        L = normalised_laplacian(4)
        self.assertEqual(L.shape, (4, 4))
        # Symmetry
        self.assertTrue(torch.equal(L, L.T), "Laplacian must be strictly symmetric")

        # Trace identity: Tr(L) == 4.0
        self.assertAlmostEqual(torch.trace(L).item(), 4.0, places=6)

        # Harmonic nullspace: L @ (D^{1/2} 1) == 0
        degrees = torch.tensor([1.0, 2.0, 2.0, 1.0])
        d_sqrt = torch.sqrt(degrees)
        harmonic_prod = torch.matmul(L, d_sqrt)
        self.assertTrue(torch.allclose(harmonic_prod, torch.zeros(4), atol=1e-6),
                        "D^{1/2} 1 must be in nullspace of normalized Laplacian")

        # Eigenvalues of L must be strictly in [0, 2]
        evals_L = torch.linalg.eigvalsh(L)
        self.assertGreaterEqual(evals_L.min().item(), -1e-6)
        self.assertLessEqual(evals_L.max().item(), 2.0 + 1e-6)

        # Exact theoretical spectrum for P_4: {0, 0.5, 1.5, 2.0}
        expected_evals = torch.tensor([0.0, 0.5, 1.5, 2.0])
        self.assertTrue(torch.allclose(evals_L, expected_evals, atol=1e-5),
                        f"Expected {expected_evals}, got {evals_L}")

        # Bipartite symmetry about 1.0: lambda_i + lambda_{3-i} == 2.0
        for i in range(4):
            self.assertAlmostEqual((evals_L[i] + evals_L[3 - i]).item(), 2.0, places=5)

        # Kipf & Welling A_hat spectral radius rho <= 1.0
        A_hat = self.graphs["A_hat"]
        self.assertTrue(torch.equal(A_hat, A_hat.T), "A_hat must be symmetric")
        evals_A_hat = torch.linalg.eigvalsh(A_hat)
        self.assertGreaterEqual(evals_A_hat.min().item(), -1.0 - 1e-6)
        self.assertLessEqual(evals_A_hat.max().item(), 1.0 + 1e-6)
        self.assertAlmostEqual(evals_A_hat.max().item(), 1.0, places=5,
                               msg="Max eigenvalue of A_hat must be 1.0 (self-loop dampening)")

    def test_03_graph_perturbations_and_isolated_nodes(self):
        """Stress-test GraphConv and STGCNLSTM against disconnected and perturbed topologies."""
        # 1. Fully disconnected graph (all matrices 0)
        zero_graphs = {
            "A_hat": torch.zeros(4, 4),
            "A_down": torch.zeros(4, 4),
            "A_up": torch.zeros(4, 4),
        }
        conv_dir = GraphConv(2, 16, mode="directed")
        conv_sym = GraphConv(2, 16, mode="symmetric")
        x = torch.randn(8, 4, 2)

        out_zero_dir = conv_dir(x, zero_graphs)
        out_zero_sym = conv_sym(x, zero_graphs)
        self.assertTrue(torch.isfinite(out_zero_dir).all(), "Zero graph must not yield NaNs in directed conv")
        self.assertTrue(torch.isfinite(out_zero_sym).all(), "Zero graph must not yield NaNs in symmetric conv")
        self.assertEqual(out_zero_dir.shape, (8, 4, 16))
        self.assertEqual(out_zero_sym.shape, (8, 4, 16))

        # 2. Graph with an isolated node (node 3 completely disconnected)
        iso_graphs = {
            "A_hat": self.graphs["A_hat"].clone(),
            "A_down": self.graphs["A_down"].clone(),
            "A_up": self.graphs["A_up"].clone(),
        }
        iso_graphs["A_hat"][3, :] = 0; iso_graphs["A_hat"][:, 3] = 0; iso_graphs["A_hat"][3, 3] = 1.0
        iso_graphs["A_down"][3, :] = 0; iso_graphs["A_down"][:, 3] = 0
        iso_graphs["A_up"][3, :] = 0; iso_graphs["A_up"][:, 3] = 0

        m_stgcn = STGCNLSTM(mode="directed", residual=True)
        seq = torch.rand(3, 10, 5)
        out_iso = m_stgcn(seq, g=iso_graphs)
        self.assertEqual(out_iso.shape, (3, 4))
        self.assertTrue(torch.isfinite(out_iso).all(), "Isolated node must produce finite outputs")

        # 3. Continuous stochastic perturbation
        pert_graphs = {
            "A_hat": self.graphs["A_hat"] + torch.randn(4, 4) * 0.05,
            "A_down": self.graphs["A_down"] + torch.randn(4, 4) * 0.05,
            "A_up": self.graphs["A_up"] + torch.randn(4, 4) * 0.05,
        }
        out_pert = m_stgcn(seq, g=pert_graphs)
        self.assertEqual(out_pert.shape, (3, 4))
        self.assertTrue(torch.isfinite(out_pert).all(), "Perturbed graphs must produce finite outputs")

    # =========================================================================
    # INVARIANT 2: Model Shapes, Scalability & Unboundedness
    # =========================================================================

    def test_04_stgcn_lstm_shape_scalability_stress(self):
        """Challenge STGCNLSTM output shapes across batch sizes B and sequence lengths L."""
        test_cases = [
            # (B, L)
            (1, 1),
            (1, 5),
            (1, 10),
            (1, 30),
            (7, 5),
            (7, 10),
            (7, 30),
            (64, 10),
            (128, 5),
            (128, 30),
        ]
        modes = ["directed", "symmetric"]
        residuals = [True, False]

        for mode in modes:
            for residual in residuals:
                model = STGCNLSTM(mode=mode, residual=residual)
                model.eval()
                for B, L in test_cases:
                    seq = torch.rand(B, L, 5)
                    out = model(seq)
                    self.assertEqual(out.shape, (B, 4),
                                     f"Failed for mode={mode}, res={residual}, B={B}, L={L}: got {out.shape}")
                    self.assertTrue(torch.isfinite(out).all(), f"NaNs detected for B={B}, L={L}")

    def test_05_paper_hybrid_overall_scalar_shape_preservation(self):
        """Challenge PaperHybridOverall to preserve 1D batch dimension [B], preventing 0D scalar collapse."""
        model = PaperHybridOverall()
        model.eval()

        # Critical edge case: B=1 must return shape [1], NEVER 0D scalar ()
        seq_single = torch.rand(1, 10, 5)
        out_single = model(seq_single)
        self.assertEqual(out_single.shape, torch.Size([1]), "B=1 must strictly have shape [1]")
        self.assertEqual(out_single.dim(), 1, "B=1 must be a 1D tensor, not a 0D scalar")

        # Large and prime batch scaling
        batch_sizes = [1, 2, 7, 13, 64, 128]
        seq_lengths = [2, 5, 10, 30]
        for B in batch_sizes:
            for L in seq_lengths:
                seq = torch.rand(B, L, 5)
                out = model(seq)
                self.assertEqual(out.shape, torch.Size([B]), f"Shape mismatch for B={B}, L={L}: got {out.shape}")
                self.assertEqual(out.dim(), 1)
                self.assertTrue(torch.isfinite(out).all())

        # Backward pass gradient verification with B=1
        model.train()
        seq_single_grad = torch.rand(1, 10, 5, requires_grad=True)
        out_grad = model(seq_single_grad)
        loss = out_grad.sum()
        loss.backward()
        self.assertIsNotNone(seq_single_grad.grad)
        self.assertEqual(seq_single_grad.grad.shape, (1, 10, 5))
        self.assertTrue(torch.isfinite(seq_single_grad.grad).all())

    def test_06_paper_hybrid_overall_unboundedness_empirical_proof(self):
        """Empirically prove PaperHybridOverall is unbounded (produces values < 0 and > 1)."""
        # Architectural inspection: assert nn.Sigmoid is completely absent
        model = PaperHybridOverall()
        for name, module in model.named_modules():
            self.assertNotIsInstance(module, nn.Sigmoid, f"nn.Sigmoid found in {name}")

        # Empirically verify that across model seeds/inputs, outputs can be < 0 and > 1
        observed_negative = False
        observed_greater_than_one = False

        for seed in range(15):
            torch.manual_seed(seed)
            m = PaperHybridOverall()
            m.eval()
            x_test = torch.randn(20, 10, 5) * 20.0
            out = m(x_test)
            if (out < 0.0).any():
                observed_negative = True
            if (out > 1.0).any():
                observed_greater_than_one = True
            if observed_negative and observed_greater_than_one:
                break

        self.assertTrue(observed_negative,
                        "PaperHybridOverall must be capable of producing negative values (unbounded below)")
        self.assertTrue(observed_greater_than_one,
                        "PaperHybridOverall must be capable of producing values > 1.0 (unbounded above)")


    def test_07_residual_persistence_under_extreme_input_ranges(self):
        """Verify residual identity holds across extreme input ranges when head is zeroed."""
        model = STGCNLSTM(mode="directed", residual=True)
        model.eval()

        with torch.no_grad():
            for p in model.head.parameters():
                p.zero_()

        # Test extreme ranges: [-1000, 1000]
        x_extreme = (torch.rand(10, 15, 5) - 0.5) * 2000.0
        out_extreme = model(x_extreme)
        expected_persistence = x_extreme[:, -1, :4]

        max_abs_diff = (out_extreme - expected_persistence).abs().max().item()
        self.assertEqual(max_abs_diff, 0.0,
                         f"Residual identity failed on extreme inputs: max abs diff = {max_abs_diff}")

    def test_08_multi_hop_gradient_isolation_and_reachability(self):
        """Adversarially probe gradient reachability across echelons."""
        x = torch.rand(4, 10, 5, requires_grad=True)
        model = STGCNLSTM(mode="directed", residual=False)
        model.eval()

        for p in model.head.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
            else:
                nn.init.ones_(p)

        out = model(x)
        # Backprop from Distributor (node 2)
        out[:, 2].sum().backward()

        # Supplier (node 0) must receive gradient via 2-hop downstream convolution
        grad_supplier = x.grad[:, :, 0].abs().sum().item()
        self.assertGreater(grad_supplier, 1e-7, "Supplier must influence Distributor")

        # Now test with disconnected graph: gradients between distinct nodes must vanish
        x_iso = torch.rand(4, 10, 5, requires_grad=True)
        zero_graphs = {
            "A_hat": torch.zeros(4, 4),
            "A_down": torch.zeros(4, 4),
            "A_up": torch.zeros(4, 4),
        }
        out_iso = model(x_iso, g=zero_graphs)
        out_iso[:, 2].sum().backward()
        # With zero graph edges, node 0 input feature cannot propagate to node 2 representation!
        # (Only node 2's own features propagate through self_lin and LSTM)
        grad_supplier_isolated = x_iso.grad[:, :, 0].abs().sum().item()
        self.assertEqual(grad_supplier_isolated, 0.0,
                         "With zero graph, Supplier input must have ZERO gradient to Distributor output")


if __name__ == "__main__":
    unittest.main()
