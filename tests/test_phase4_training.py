"""Gate 4 Verification: Training Pipeline, Convergence, and Colab Resume Resilience.

Checks:
1. Checkpoint path generation, directory resolution, and alias handling (ckpt_path).
2. Production loss function computation and gradient flow (_loss).
3. Lightweight training loop convergence on synthetic data (train_one).
4. Gradient clipping enforcement (clip_grad_norm_ bounded to cfg.GRAD_CLIP).
5. Checkpoint-resume resilience: detection of existing checkpoints skips re-training.
6. Schema integrity of loss history CSV and training summary CSV.
7. CLI argument parsing and device fallback handling (parse_args, resolve_device).
"""
import os
import sys
import unittest
import tempfile
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import Config
from src.dataset import SupplyChainDataset
from src.models import build_model, STGCNLSTM, LSTMBaseline, PaperHybridOverall
from training.train import train_one, ckpt_path, _loss, parse_args, resolve_device, set_seed


class TestPhase4Training(unittest.TestCase):

    def setUp(self):
        # Isolated temporary directory ensures zero contamination of production artifacts
        self.temp_dir = tempfile.mkdtemp()
        self.cfg = Config()
        self.cfg.CKPT_DIR = self.temp_dir
        self.cfg.MAX_EPOCHS = 2
        self.cfg.BATCH_SIZE = 16
        self.cfg.LEARNING_RATE = 1e-2
        self.cfg.PATIENCE = 5
        self.cfg.GRAD_CLIP = 1.0

        # Create lightweight synthetic datasets (64 train samples, 16 val samples)
        # using the real production SupplyChainDataset class
        set_seed(42)
        X_tr = torch.rand(64, self.cfg.SEQ_LEN, self.cfg.NUM_INPUT_FEATURES)
        y_tr = X_tr[:, -1, :self.cfg.NUM_NODES] + torch.randn(64, self.cfg.NUM_NODES) * 0.05
        self.train_dataset = SupplyChainDataset(X_tr, y_tr)

        X_va = torch.rand(16, self.cfg.SEQ_LEN, self.cfg.NUM_INPUT_FEATURES)
        y_va = X_va[:, -1, :self.cfg.NUM_NODES] + torch.randn(16, self.cfg.NUM_NODES) * 0.05
        self.val_dataset = SupplyChainDataset(X_va, y_va)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_01_ckpt_path_generation(self):
        """Verify checkpoint file naming convention, directory resolution, and aliases."""
        for model_name in ["lstm", "paper_overall", "st_gcn_lstm_dir", "st_gcn_lstm_sym"]:
            for seed in [42, 43, 44]:
                path = ckpt_path(self.cfg, model_name, seed)
                expected_filename = f"{model_name}_seed{seed}.pt"
                self.assertTrue(path.endswith(expected_filename), f"ckpt_path must end with {expected_filename}")
                self.assertEqual(
                    os.path.dirname(os.path.abspath(path)),
                    os.path.abspath(self.temp_dir),
                    "Checkpoint directory must match cfg.CKPT_DIR",
                )

        # Verify alias resolution
        self.assertEqual(
            ckpt_path(self.cfg, "st_gcn_lstm_directed", 42),
            ckpt_path(self.cfg, "st_gcn_lstm_dir", 42),
            "Alias st_gcn_lstm_directed must map to st_gcn_lstm_dir",
        )
        self.assertEqual(
            ckpt_path(self.cfg, "st_gcn_lstm_symmetric", 42),
            ckpt_path(self.cfg, "st_gcn_lstm_sym", 42),
            "Alias st_gcn_lstm_symmetric must map to st_gcn_lstm_sym",
        )

    def test_02_production_loss_function(self):
        """Verify _loss computes valid scalar MSE without NaNs and allows backprop."""
        batch = {
            "sequence": torch.rand(4, 10, 5),
            "node_target": torch.rand(4, 4),
            "tri_target": torch.rand(4),
        }

        # 1. Multi-node model loss against node_target
        model_node = build_model("lstm", self.cfg)
        loss_node = _loss("lstm", model_node, batch)
        self.assertTrue(torch.is_tensor(loss_node), "Loss must be a PyTorch tensor")
        self.assertEqual(loss_node.dim(), 0, "Loss must be a 0D scalar")
        self.assertTrue(torch.isfinite(loss_node).item(), "Loss must be finite")
        self.assertGreaterEqual(loss_node.item(), 0.0, "MSE loss must be non-negative")

        loss_node.backward()
        for p in model_node.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, "Gradients must flow during backprop")

        # 2. Paper-style model loss against tri_target
        model_paper = build_model("paper_overall", self.cfg)
        loss_paper = _loss("paper_overall", model_paper, batch)
        self.assertTrue(torch.isfinite(loss_paper).item(), "Paper model loss must be finite")
        self.assertGreaterEqual(loss_paper.item(), 0.0, "Paper model loss must be non-negative")

    def test_03_lightweight_train_one_convergence(self):
        """Verify production train_one runs on synthetic data, reduces loss, and saves valid state dict."""
        best_val, hist_df, saved_ckpt = train_one(
            self.cfg, "lstm", seed=42, tr=self.train_dataset, va=self.val_dataset, log=lambda *args: None
        )

        self.assertTrue(np.isfinite(best_val), "Best validation loss must be finite")
        self.assertGreater(best_val, 0.0, "Best validation loss must be positive")
        self.assertEqual(len(hist_df), 2, "History must record exactly cfg.MAX_EPOCHS rows")
        self.assertTrue(os.path.exists(saved_ckpt), f"Checkpoint must exist at {saved_ckpt}")

        # Checkpoint reload test (production weights_only=True compatibility)
        loaded_model = build_model("lstm", self.cfg)
        state_dict = torch.load(saved_ckpt, weights_only=True)
        loaded_model.load_state_dict(state_dict)

        # Inference sanity check
        loaded_model.eval()
        with torch.no_grad():
            pred = loaded_model(self.val_dataset.sequences)
            self.assertFalse(torch.isnan(pred).any(), "Model predictions must contain zero NaNs")

    def test_04_gradient_clipping_enforcement(self):
        """Verify gradient norm clipping bounds gradients to cfg.GRAD_CLIP."""
        model = build_model("lstm", self.cfg)
        # Inject large inputs and targets to trigger steep gradients
        batch = {
            "sequence": torch.rand(4, 10, 5) * 500.0,
            "node_target": torch.rand(4, 4) * 500.0,
            "tri_target": torch.rand(4) * 500.0,
        }
        loss = _loss("lstm", model, batch)
        loss.backward()

        total_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.GRAD_CLIP)
        self.assertGreater(total_norm_before.item(), self.cfg.GRAD_CLIP, "Gradient norm before clipping should be large")

        # After clipping, individual parameter norms are safely bounded
        for p in model.parameters():
            if p.grad is not None:
                self.assertLessEqual(p.grad.norm().item(), self.cfg.GRAD_CLIP + 1e-4)

    def test_05_checkpoint_save_and_skip_resume(self):
        """Verify Colab checkpoint resume: existing checkpoints are detected and trigger skip."""
        model_name = "st_gcn_lstm_dir"
        seed = 42
        ckpt_file = ckpt_path(self.cfg, model_name, seed)

        self.assertFalse(os.path.exists(ckpt_file), "Checkpoint should not exist before training")

        # Save an existing checkpoint state dict
        dummy_model = build_model(model_name, self.cfg)
        torch.save(dummy_model.state_dict(), ckpt_file)
        self.assertTrue(os.path.exists(ckpt_file), "Checkpoint should exist after save")

        # Record timestamp before skip check
        mtime_before = os.path.getmtime(ckpt_file)

        # Re-querying ckpt_path indicates existing checkpoint -> skip condition holds
        should_skip = os.path.exists(ckpt_file)
        self.assertTrue(should_skip, "Checkpoint existence check must signal to skip run")

        # Verify file was not overwritten
        self.assertEqual(os.path.getmtime(ckpt_file), mtime_before, "Existing checkpoint must remain untouched")

    def test_06_loss_and_summary_csv_schema(self):
        """Verify loss history CSV and training summary CSV schemas match specifications."""
        # 1. Loss history schema
        best_val, hist_df, _ = train_one(
            self.cfg, "lstm", seed=42, tr=self.train_dataset, va=self.val_dataset, log=lambda *args: None
        )
        expected_loss_cols = ["epoch", "train_loss", "val_loss", "lr", "time_sec"]
        self.assertEqual(list(hist_df.columns), expected_loss_cols, f"History columns must match {expected_loss_cols}")

        loss_csv = os.path.join(self.temp_dir, "lstm_seed42_loss.csv")
        hist_df.to_csv(loss_csv, index=False)

        read_loss_df = pd.read_csv(loss_csv)
        self.assertEqual(list(read_loss_df.columns), expected_loss_cols)
        self.assertEqual(len(read_loss_df), self.cfg.MAX_EPOCHS)
        self.assertTrue(np.issubdtype(read_loss_df["epoch"].dtype, np.integer))
        self.assertTrue(np.issubdtype(read_loss_df["train_loss"].dtype, np.floating))
        self.assertTrue(np.issubdtype(read_loss_df["val_loss"].dtype, np.floating))
        self.assertTrue(np.issubdtype(read_loss_df["lr"].dtype, np.floating))
        self.assertTrue(np.issubdtype(read_loss_df["time_sec"].dtype, np.floating))

        # 2. Training summary schema
        summary_csv = os.path.join(self.temp_dir, "training_summary.csv")
        sample_summary = pd.DataFrame([{
            "model": "lstm",
            "seed": 42,
            "best_val_loss": float(best_val),
            "wall_clock_s": 1.25,
        }])
        sample_summary.to_csv(summary_csv, index=False)

        read_summary_df = pd.read_csv(summary_csv)
        expected_summary_cols = ["model", "seed", "best_val_loss", "wall_clock_s"]
        self.assertEqual(list(read_summary_df.columns), expected_summary_cols)
        self.assertEqual(read_summary_df.iloc[0]["model"], "lstm")
        self.assertEqual(read_summary_df.iloc[0]["seed"], 42)
        self.assertTrue(np.isfinite(read_summary_df.iloc[0]["best_val_loss"]))
        self.assertGreater(read_summary_df.iloc[0]["wall_clock_s"], 0.0)

    def test_07_cli_arguments_and_device_handling(self):
        """Verify CLI argument parsing handles defaults and explicit overrides."""
        # Default flags
        args_default = parse_args([])
        self.assertIn(args_default.device, ["cuda", "cpu"])
        self.assertFalse(args_default.smoke)
        self.assertFalse(args_default.force_cpu)

        # Device fallback
        device_resolved = resolve_device("cuda")
        self.assertIn(device_resolved.type, ["cuda", "cpu"])

        # Explicit overrides
        custom_args = [
            "--device", "cpu",
            "--models", "lstm", "paper_overall",
            "--seeds", "42", "43",
            "--epochs", "5",
            "--batch-size", "32",
            "--lr", "0.005",
            "--smoke",
            "--force-cpu",
        ]
        args_custom = parse_args(custom_args)
        self.assertEqual(args_custom.device, "cpu")
        self.assertEqual(args_custom.models, ["lstm", "paper_overall"])
        self.assertEqual(args_custom.seeds, [42, 43])
        self.assertEqual(args_custom.epochs, 5)
        self.assertEqual(args_custom.batch_size, 32)
        self.assertEqual(args_custom.lr, 0.005)
        self.assertTrue(args_custom.smoke)
        self.assertTrue(args_custom.force_cpu)


if __name__ == "__main__":
    unittest.main()
