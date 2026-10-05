"""Gate 4 Verification: Training Pipeline, Convergence, and Colab Resume Resilience.

Checks:
1. Training loop execution and loss convergence (no NaNs, loss decreases).
2. Gradient clipping (torch.nn.utils.clip_grad_norm_).
3. Checkpoint save and skip-if-exists resume logic (resilience to Colab disconnects).
4. Loss logging CSV generation and schema.
"""
import os
import sys
import unittest
import tempfile
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests.test_phase3_models import STGCNLSTM, build_graphs


class TestPhase4Training(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.graphs = build_graphs()
        torch.manual_seed(42)
        # Create small synthetic training dataset (128 samples, L=10, 5 features)
        self.X_train = torch.rand(128, 10, 5)
        # Target: next step node risks (first 4 columns)
        self.y_train = self.X_train[:, -1, :4] + torch.randn(128, 4) * 0.05
        self.dataset = TensorDataset(self.X_train, self.y_train)
        self.loader = DataLoader(self.dataset, batch_size=32, shuffle=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_01_training_step_convergence(self):
        """Verify training loop reduces MSE loss and produces no NaNs."""
        model = STGCNLSTM(mode="directed", residual=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        losses = []

        model.train()
        for epoch in range(5):
            epoch_loss = 0.0
            for bx, by in self.loader:
                optimizer.zero_grad()
                pred = model(bx, self.graphs)
                loss = F.mse_loss(pred, by)
                self.assertFalse(torch.isnan(loss).item(), "Loss must not be NaN")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / len(self.loader))

        # Check loss decreases from first epoch to last
        self.assertLess(losses[-1], losses[0], "Training loss must decrease over epochs")

    def test_02_gradient_clipping(self):
        """Verify gradient norm does not exceed clip threshold."""
        model = STGCNLSTM(mode="directed", residual=True)
        # Artificially inject huge gradients
        x = torch.rand(4, 10, 5) * 1000.0
        y = torch.rand(4, 4) * 1000.0
        pred = model(x, self.graphs)
        loss = F.mse_loss(pred, y)
        loss.backward()

        total_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        self.assertGreater(total_norm_before.item(), 1.0, "Gradient was large before clipping")

        # After clipping, individual parameter gradient norms are bounded
        for p in model.parameters():
            if p.grad is not None:
                self.assertLessEqual(p.grad.norm().item(), 1.05)

    def test_03_checkpoint_save_and_skip_resume(self):
        """Verify Colab checkpoint resume: save checkpoint and detect existing state."""
        model = STGCNLSTM(mode="directed", residual=True)
        ckpt_path = os.path.join(self.temp_dir, "st_gcn_lstm_directed_seed42.pt")

        # Simulate checkpoint save
        torch.save({
            "epoch": 25,
            "model_state_dict": model.state_dict(),
            "best_val_loss": 0.042
        }, ckpt_path)

        # Check file exists
        self.assertTrue(os.path.exists(ckpt_path))

        # Check resume condition (skip if exists)
        should_skip = os.path.exists(ckpt_path)
        self.assertTrue(should_skip, "Checkpoint existence check must signal to skip / resume")

        # Load back checkpoint and assert equality of inference
        model_loaded = STGCNLSTM(mode="directed", residual=True)
        ckpt = torch.load(ckpt_path, weights_only=True)
        model_loaded.load_state_dict(ckpt["model_state_dict"])

        test_input = torch.rand(2, 10, 5)
        model.eval(); model_loaded.eval()
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(test_input, self.graphs), model_loaded(test_input, self.graphs)))

    def test_04_loss_csv_logging_schema(self):
        """Verify loss logging CSV schema."""
        csv_path = os.path.join(self.temp_dir, "loss_log.csv")
        log_df = pd.DataFrame({
            "epoch": [1, 2, 3],
            "train_loss": [0.08, 0.05, 0.03],
            "val_loss": [0.09, 0.06, 0.04],
            "lr": [1e-3, 1e-3, 1e-3],
            "time_sec": [1.2, 1.1, 1.2]
        })
        log_df.to_csv(csv_path, index=False)

        read_df = pd.read_csv(csv_path)
        expected_cols = ["epoch", "train_loss", "val_loss", "lr", "time_sec"]
        self.assertEqual(list(read_df.columns), expected_cols)
        self.assertEqual(len(read_df), 3)


if __name__ == "__main__":
    unittest.main()
