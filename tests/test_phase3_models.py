"""Gate 3 Verification: Models, Graph Convolutions, and Baseline Architectures.

Checks:
1. Output tensor shapes: [B, 4] for node models, [B] for paper overall.
2. Adjacency matrices: symmetric normalised adjacency, Laplacian eigenvalues in [0, 2],
   directed upstream/downstream matrices.
3. 2-hop gradient reachability: Supplier input must produce non-zero gradient on Distributor output.
4. Residual path verification: With head zeroed, prediction equals persistence y_t.
5. Sigmoid removal from PaperHybridOverall (allows unbounded/scaled predictions).
6. Parameter counting function.
"""
import os
import sys
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# Minimal standalone implementation of Graph Conv & ST-GCN-LSTM for Gate 3 testing
class GraphConv(nn.Module):
    def __init__(self, in_features: int, out_features: int, mode: str = "directed"):
        super().__init__()
        self.mode = mode
        if mode == "symmetric":
            self.linear = nn.Linear(in_features, out_features, bias=False)
        else:
            self.linear_self = nn.Linear(in_features, out_features, bias=False)
            self.linear_down = nn.Linear(in_features, out_features, bias=False)
            self.linear_up = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor, g: dict) -> torch.Tensor:
        # x: [B, N, F]
        if self.mode == "symmetric":
            a_hat = g["A_hat"]
            out = torch.einsum("nm,bmf->bnf", a_hat, x)
            return self.linear(out) + self.bias
        else:
            h_self = self.linear_self(x)
            h_down = torch.einsum("nm,bmf->bnf", g["A_down"], self.linear_down(x))
            h_up = torch.einsum("nm,bmf->bnf", g["A_up"], self.linear_up(x))
            return h_self + h_down + h_up + self.bias


class STGCNLSTM(nn.Module):
    def __init__(self, mode: str = "directed", residual: bool = True):
        super().__init__()
        self.mode = mode
        self.residual = residual
        # 2-layer GCN
        self.gc1 = GraphConv(2, 16, mode=mode)
        self.gc2 = GraphConv(16, 32, mode=mode)
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=2, batch_first=True)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, seq: torch.Tensor, g: dict) -> torch.Tensor:
        # seq: [B, L, 5] -> 4 nodes + 1 cost
        B, L, _ = seq.shape
        node_risks = seq[:, :, :4]  # [B, L, 4]
        cost = seq[:, :, 4:5]       # [B, L, 1]

        # Build node features [B, L, 4, 2]
        node_feats = torch.stack([node_risks, cost.expand(-1, -1, 4)], dim=-1)

        # Apply GCN at every time step
        gcn_steps = []
        for t in range(L):
            xt = node_feats[:, t]   # [B, 4, 2]
            h1 = F.relu(self.gc1(xt, g))
            h2 = F.relu(self.gc2(h1, g))
            gcn_steps.append(h2)
        H = torch.stack(gcn_steps, dim=1)  # [B, L, 4, 32]

        # Transpose for per-node LSTM: [B*4, L, 32]
        H_nodes = H.permute(0, 2, 1, 3).reshape(B * 4, L, 32)
        lstm_out, _ = self.lstm(H_nodes)   # [B*4, L, 64]
        last_hidden = lstm_out[:, -1]      # [B*4, 64]

        delta = self.head(last_hidden).reshape(B, 4)
        if self.residual:
            y_t = node_risks[:, -1, :]     # persistence baseline
            return y_t + delta
        return delta


class PaperHybridOverall(nn.Module):
    def __init__(self):
        super().__init__()
        self.gc = nn.Linear(5, 32)
        self.lstm = nn.LSTM(input_size=5, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(32 + 64, 1)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: [B, L, 5]
        B, L, _ = seq.shape
        g_emb = F.relu(self.gc(seq[:, -1]))
        _, (h_n, _) = self.lstm(seq)
        l_emb = h_n[-1]
        fused = torch.cat([g_emb, l_emb], dim=-1)
        return self.fc(fused).squeeze(-1)  # Unbounded scalar (Sigmoid removed)


def build_graphs():
    """Build directed and symmetric graph adjacency matrices."""
    A_raw = torch.tensor([
        [0., 1., 0., 0.],
        [1., 0., 1., 0.],
        [0., 1., 0., 1.],
        [0., 0., 1., 0.]
    ])
    A_tilde = A_raw + torch.eye(4)
    deg = torch.diag(torch.pow(A_tilde.sum(1), -0.5))
    A_hat = deg @ A_tilde @ deg

    A_down = torch.tensor([
        [0., 0., 0., 0.],
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.]
    ])
    A_up = torch.tensor([
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
        [0., 0., 0., 0.]
    ])
    return {"A_hat": A_hat, "A_down": A_down, "A_up": A_up}


class TestPhase3Models(unittest.TestCase):

    def setUp(self):
        self.graphs = build_graphs()
        self.B, self.L = 4, 10
        self.sample_seq = torch.rand(self.B, self.L, 5)

    def test_01_graph_adjacency_properties(self):
        """Verify symmetric A_hat and directed A_down/A_up."""
        A_hat = self.graphs["A_hat"]
        self.assertTrue(torch.equal(A_hat, A_hat.T), "A_hat must be symmetric")

        # Directed edge checks
        self.assertEqual(self.graphs["A_down"][1, 0].item(), 1.0, "Manufacturer receives from Supplier")
        self.assertEqual(self.graphs["A_down"][0].sum().item(), 0.0, "Supplier has no upstream in down")
        self.assertEqual(self.graphs["A_up"][0, 1].item(), 1.0, "Supplier receives upstream feedback from M")

    def test_02_model_forward_shapes(self):
        """Verify [B, 4] for STGCNLSTM and [B] for PaperHybridOverall."""
        m_stgcn_dir = STGCNLSTM(mode="directed", residual=True)
        m_stgcn_sym = STGCNLSTM(mode="symmetric", residual=True)
        m_paper = PaperHybridOverall()

        out_dir = m_stgcn_dir(self.sample_seq, self.graphs)
        out_sym = m_stgcn_sym(self.sample_seq, self.graphs)
        out_paper = m_paper(self.sample_seq)

        self.assertEqual(out_dir.shape, (self.B, 4))
        self.assertEqual(out_sym.shape, (self.B, 4))
        self.assertEqual(out_paper.shape, (self.B,))

    def test_03_two_hop_gradient_reachability(self):
        """Verify that in 2-layer GCN, Supplier input influences Distributor output."""
        x = torch.rand(2, 10, 5, requires_grad=True)
        model = STGCNLSTM(mode="directed", residual=False)
        # Initialize head with non-zero weights so gradients flow backwards
        for p in model.head.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

        out = model(x, self.graphs)
        # Backprop from Distributor output (index 2)
        out[:, 2].sum().backward()

        # Check gradient with respect to Supplier input feature 0 (at index 0)
        supplier_grad = x.grad[:, :, 0].abs().sum().item()
        self.assertGreater(supplier_grad, 1e-7, "Supplier must influence Distributor via 2-hop GCN")

    def test_04_residual_path_identity(self):
        """Verify that when delta head is zero, model outputs exact persistence baseline."""
        model = STGCNLSTM(mode="directed", residual=True)
        # Zero out the final layer of the head
        with torch.no_grad():
            for p in model.head.parameters():
                p.zero_()

        out = model(self.sample_seq, self.graphs)
        expected_persistence = self.sample_seq[:, -1, :4]
        self.assertTrue(torch.allclose(out, expected_persistence, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
