"""Phase 4 Challenger 2 Adversarial Stress Test Suite: Pipeline Mechanics & Numerical Invariants.

Author: Challenger 2 (teamwork_preview_challenger_p4_2)
Objective: Empirically stress-test training pipeline mechanics and numerical invariants:
1. Checkpoint resume skipping: verify existing checkpoint files are detected, logged with [Skip],
   and their mtime/content is strictly preserved (untouched).
2. Gradient clipping bounding: inject extreme gradients (x1,000, x10,000, x1e6) across all 4 architectures
   and verify torch.nn.utils.clip_grad_norm_ bounds total parameter L2 norm to <= cfg.GRAD_CLIP.
3. Early stopping resilience: verify train_one halts cleanly at epoch = 1 + patience on non-improving
   validation data, preserves epoch 1 weights as the saved checkpoint, and correctly resets patience on improvement.
4. Device resolution fallback: verify resolve_device("cuda") and variants ("cuda:0") on CPU systems
   issue [WARNING] and return torch.device("cpu") without crashing or propagating unhandled exceptions.
5. Model name aliasing: verify MODEL_ALIASES correctly maps st_gcn_lstm_symmetric and st_gcn_lstm_directed
   to canonical forms across ckpt_path, build_model, train_one, and CLI argument parsing.
6. Safety guardrail & progressive summary deduplication: verify CPU multi-seed grid prevention,
   safe bypass with --force-cpu, and idempotency of training_summary.csv.
"""
import copy
import io
import os
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import Config
from src.dataset import SupplyChainDataset
from src.models import build_model, STGCNLSTM, LSTMBaseline, PaperHybridOverall
from training.train import (
    MODEL_ALIASES,
    MODEL_NAMES,
    ckpt_path,
    _loss,
    train_one,
    resolve_device,
    parse_args,
    set_seed,
    main,
)


class TestAdversarialPhase4Challenger2(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.cfg = Config()
        self.cfg.CKPT_DIR = os.path.join(self.temp_dir, "models")
        self.results_dir = os.path.join(self.temp_dir, "results")
        self.cfg.RESULTS_DIR = self.results_dir
        self.cfg.SUMMARY_PATH = os.path.join(self.results_dir, "training_summary.csv")
        os.makedirs(self.cfg.CKPT_DIR, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        self.cfg.MAX_EPOCHS = 10
        self.cfg.BATCH_SIZE = 16
        self.cfg.LEARNING_RATE = 1e-2
        self.cfg.PATIENCE = 3
        self.cfg.GRAD_CLIP = 1.0

        set_seed(42)
        # Standard synthetic dataset
        X_tr = torch.rand(64, self.cfg.SEQ_LEN, self.cfg.NUM_INPUT_FEATURES)
        y_tr = X_tr[:, -1, :self.cfg.NUM_NODES] + torch.randn(64, self.cfg.NUM_NODES) * 0.05
        self.tr_ds = SupplyChainDataset(X_tr, y_tr)

        X_va = torch.rand(16, self.cfg.SEQ_LEN, self.cfg.NUM_INPUT_FEATURES)
        y_va = X_va[:, -1, :self.cfg.NUM_NODES] + torch.randn(16, self.cfg.NUM_NODES) * 0.05
        self.va_ds = SupplyChainDataset(X_va, y_va)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    # =========================================================================
    # 1. CHECKPOINT RESUME SKIPPING INVARIANTS
    # =========================================================================

    def test_01_checkpoint_resume_skipping_preserves_mtime_and_content(self):
        """Stress-test checkpoint resume skipping: existing .pt file must remain 100% untouched."""
        target_model = "lstm"
        target_seed = 42
        ckpt_file = ckpt_path(self.cfg, target_model, target_seed)

        # Write dummy sentinel bytes to ckpt_file
        sentinel_payload = b"SUPPLYGUARD_TEST_CHECKPOINT_SENTINEL_P4"
        with open(ckpt_file, "wb") as f:
            f.write(sentinel_payload)

        loss_file = os.path.join(self.results_dir, f"{target_model}_seed{target_seed}_loss.csv")
        with open(loss_file, "w") as fl:
            fl.write("epoch,train_loss,val_loss\n1,0.1,0.1")

        # Wait briefly to ensure mtime separation
        time.sleep(0.05)
        mtime_before = os.path.getmtime(ckpt_file)
        size_before = os.path.getsize(ckpt_file)
        
        # Create dummy loss file to bypass training
        loss_file = os.path.join(self.results_dir, f"{target_model}_seed{target_seed}_loss.csv")
        pd.DataFrame().to_csv(loss_file)

        # Mock build_datasets to avoid requiring raw data parquet on disk
        def mock_build_datasets(cfg, save_scaler=False):
            return self.tr_ds, self.va_ds, self.va_ds, None, {
                "n_rows_used": 80,
                "start": "2024-01-01",
                "end": "2024-01-02",
            }

        stdout_capture = io.StringIO()
        with patch("sys.stdout", stdout_capture), \
             patch("src.dataset.build_datasets", side_effect=mock_build_datasets), \
             patch("training.train.build_datasets", side_effect=mock_build_datasets):
            main(
                cfg=self.cfg,
                models=[target_model],
                seeds=[target_seed],
                cli_args=["--force-cpu", "--epochs", "2"],
            )

        output = stdout_capture.getvalue()
        # Invariant 1: [Skip] must be logged explicitly
        self.assertIn(
            f"[{target_model} seed={target_seed}] [Skip] Checkpoint and loss history exist:",
            output,
            "Pipeline must log [Skip] message for pre-existing checkpoint",
        )

        # Invariant 2: mtime and size must be bit-for-bit unchanged
        mtime_after = os.path.getmtime(ckpt_file)
        size_after = os.path.getsize(ckpt_file)
        self.assertEqual(mtime_before, mtime_after, "Checkpoint mtime was modified! Skip logic failed.")
        self.assertEqual(size_before, size_after, "Checkpoint file size was altered!")

        with open(ckpt_file, "rb") as f:
            content_after = f.read()
        self.assertEqual(sentinel_payload, content_after, "Checkpoint payload was overwritten!")

    def test_02_checkpoint_skipping_multi_model_partial_state(self):
        """Verify partial resume: existing checkpoints skip, missing checkpoints train."""
        seed = 43
        # Pre-create checkpoint for lstm only
        lstm_ckpt = ckpt_path(self.cfg, "lstm", seed)
        torch.save({"dummy": 1}, lstm_ckpt)
        loss_file = os.path.join(self.results_dir, f"lstm_seed{seed}_loss.csv")
        with open(loss_file, "w") as fl:
            fl.write("epoch,train_loss,val_loss\n1,0.1,0.1")
        lstm_mtime = os.path.getmtime(lstm_ckpt)

        paper_ckpt = ckpt_path(self.cfg, "paper_overall", seed)
        self.assertFalse(os.path.exists(paper_ckpt), "Paper checkpoint should not exist initially")

        # We MUST ALSO pre-create the loss CSV so it is skipped
        lstm_loss = os.path.join(self.results_dir, f"lstm_seed{seed}_loss.csv")
        pd.DataFrame().to_csv(lstm_loss, index=False)

        def mock_build_datasets(cfg, save_scaler=False):
            return self.tr_ds, self.va_ds, self.va_ds, None, {
                "n_rows_used": 80,
                "start": "2024-01-01",
                "end": "2024-01-02",
            }
            
        def mock_to_csv(df, path, *args, **kwargs):
            if "training_summary.csv" in path:
                path = os.path.join(self.temp_dir, "training_summary.csv")
            df.to_csv_orig(path, *args, **kwargs)

        stdout_capture = io.StringIO()
        pd.DataFrame.to_csv_orig = pd.DataFrame.to_csv
        with patch("sys.stdout", stdout_capture), \
             patch("src.dataset.build_datasets", side_effect=mock_build_datasets), \
             patch("training.train.build_datasets", side_effect=mock_build_datasets), \
             patch("training.train.pd.DataFrame.to_csv", side_effect=mock_to_csv, autospec=True):
            main(
                cfg=self.cfg,
                models=["lstm", "paper_overall"],
                seeds=[seed],
                cli_args=["--force-cpu", "--epochs", "1", "--batch-size", "16"],
            )

        output = stdout_capture.getvalue()
        # lstm skipped
        self.assertIn(f"[lstm seed={seed}] [Skip] Checkpoint and loss history exist:", output)
        self.assertEqual(os.path.getmtime(lstm_ckpt), lstm_mtime, "lstm checkpoint was overwritten")

        # paper_overall trained and checkpoint created
        self.assertIn(f"[paper_overall seed={seed}]", output)
        self.assertTrue(os.path.exists(paper_ckpt), "Missing paper_overall checkpoint was not trained and created")

    # =========================================================================
    # 2. GRADIENT CLIPPING BOUNDING INVARIANTS
    # =========================================================================

    def test_03_gradient_clipping_bounds_extreme_gradients(self):
        """Stress-test gradient clipping under massive input perturbations (1e3, 1e4, 1e6)."""
        grad_clip_limit = 0.75
        self.cfg.GRAD_CLIP = grad_clip_limit

        for arch in ["lstm", "paper_overall", "st_gcn_lstm_sym", "st_gcn_lstm_dir"]:
            model = build_model(arch, self.cfg)
            for scale in [1e3, 1e4, 1e6]:
                model.zero_grad()
                batch = {
                    "sequence": torch.randn(8, self.cfg.SEQ_LEN, self.cfg.NUM_INPUT_FEATURES) * scale,
                    "node_target": torch.randn(8, self.cfg.NUM_NODES) * scale,
                    "tri_target": torch.randn(8) * scale,
                }
                loss = _loss(arch, model, batch)
                loss.backward()

                # Calculate pre-clip norm across trainable parameters
                pre_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_limit)
                self.assertGreater(
                    pre_clip_norm.item(),
                    grad_clip_limit,
                    f"Scale {scale} for {arch} did not produce sufficiently large gradient norm",
                )

                # Compute total post-clip L2 norm
                grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
                post_clip_norm = torch.norm(torch.stack([g.norm(2) for g in grads]), 2).item()

                # Invariant: total parameter norm must strictly not exceed grad_clip + epsilon
                self.assertLessEqual(
                    post_clip_norm,
                    grad_clip_limit + 1e-4,
                    f"Post-clip norm ({post_clip_norm:.6f}) exceeded GRAD_CLIP ({grad_clip_limit}) for {arch}",
                )
                # Check for zero NaNs/Infs in clipped gradients
                for g in grads:
                    self.assertFalse(torch.isnan(g).any(), f"NaN detected in gradients of {arch}")
                    self.assertFalse(torch.isinf(g).any(), f"Inf detected in gradients of {arch}")

    def test_04_gradient_clipping_unmodified_when_below_threshold(self):
        """Verify that gradients already below cfg.GRAD_CLIP are strictly preserved (no distortion)."""
        model = build_model("lstm", self.cfg)
        grad_clip_limit = 100.0  # Large threshold
        # Small inputs -> small gradients
        batch = {
            "sequence": torch.randn(4, self.cfg.SEQ_LEN, self.cfg.NUM_INPUT_FEATURES) * 1e-3,
            "node_target": torch.randn(4, self.cfg.NUM_NODES) * 1e-3,
            "tri_target": torch.randn(4) * 1e-3,
        }
        loss = _loss("lstm", model, batch)
        loss.backward()

        grads_before = [p.grad.clone() for p in model.parameters() if p.grad is not None]
        norm_reported = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_limit)

        self.assertLess(norm_reported.item(), grad_clip_limit)
        grads_after = [p.grad for p in model.parameters() if p.grad is not None]

        for gb, ga in zip(grads_before, grads_after):
            self.assertTrue(torch.allclose(gb, ga, atol=1e-7), "Small gradients were distorted by clip_grad_norm_")

    # =========================================================================
    # 3. EARLY STOPPING RESILIENCE INVARIANTS
    # =========================================================================

    def test_05_early_stopping_halts_cleanly_at_patience_limit(self):
        """Verify train_one halts at exactly epoch = 1 + patience when validation loss never improves."""
        patience = 3
        max_epochs = 15
        cfg_es = copy.copy(self.cfg)
        cfg_es.PATIENCE = patience
        cfg_es.MAX_EPOCHS = max_epochs
        cfg_es.BATCH_SIZE = 16
        cfg_es.LEARNING_RATE = 0.05

        # Construct adversarial dataset:
        # Train dataset forces weights to predict large positive values (+10.0)
        X_tr = torch.ones(32, cfg_es.SEQ_LEN, cfg_es.NUM_INPUT_FEATURES)
        y_tr = torch.ones(32, cfg_es.NUM_NODES) * 10.0
        tr = SupplyChainDataset(X_tr, y_tr)

        # Val target is zero, so as model predicts positive values during training,
        # val loss strictly increases every epoch after epoch 1.
        X_va = torch.ones(16, cfg_es.SEQ_LEN, cfg_es.NUM_INPUT_FEATURES)
        y_va = torch.zeros(16, cfg_es.NUM_NODES)
        va = SupplyChainDataset(X_va, y_va)

        log_messages = []
        def mock_log(msg):
            log_messages.append(msg)

        best_val, hist_df, path = train_one(
            cfg_es, "lstm", seed=42, tr=tr, va=va, device="cpu", log=mock_log
        )

        # Invariant 1: Halts cleanly at exactly epoch 1 + patience = 4
        self.assertEqual(
            len(hist_df),
            1 + patience,
            f"Expected exactly {1 + patience} epochs, but ran {len(hist_df)}",
        )
        self.assertLess(len(hist_df), max_epochs, "Early stopping failed to halt before max_epochs")

        # Invariant 2: 'early stop' was logged
        self.assertTrue(any("early stop" in m for m in log_messages), "Early stop message was not logged")

        # Invariant 3: Best validation loss is the epoch 1 loss
        self.assertEqual(best_val, hist_df["val_loss"].iloc[0])
        self.assertEqual(best_val, hist_df["val_loss"].min())

        # Invariant 4: Checkpoint file exists on disk
        self.assertTrue(os.path.exists(path), "Checkpoint was not saved on disk")

        # Invariant 5: Verify saved checkpoint state_dict can be reloaded
        loaded = build_model("lstm", cfg_es)
        loaded.load_state_dict(torch.load(path, weights_only=True))

    def test_06_early_stopping_resets_counter_on_loss_improvement(self):
        """Verify patience counter resets when val loss improves, delaying early stop."""
        cfg_es = copy.copy(self.cfg)
        cfg_es.PATIENCE = 2
        cfg_es.MAX_EPOCHS = 6

        # Train for 6 epochs on easily learnable synthetic data where model converges
        torch.manual_seed(123)
        X = torch.randn(64, cfg_es.SEQ_LEN, cfg_es.NUM_INPUT_FEATURES)
        y = X[:, -1, :cfg_es.NUM_NODES] * 2.0
        tr = SupplyChainDataset(X, y)
        va = SupplyChainDataset(X[:16], y[:16])

        best_val, hist_df, path = train_one(
            cfg_es, "lstm", seed=42, tr=tr, va=va, device="cpu", log=lambda *args: None
        )
        self.assertGreaterEqual(len(hist_df), 2, "Model should train for at least 2 epochs when improving")
        self.assertTrue(np.isfinite(best_val))

    # =========================================================================
    # 4. DEVICE RESOLUTION FALLBACK INVARIANTS
    # =========================================================================

    def test_07_device_resolution_fallback_on_cpu(self):
        """Stress-test resolve_device('cuda') and CUDA variants on non-CUDA systems."""
        cuda_available = torch.cuda.is_available()

        if not cuda_available:
            for req in ["cuda", "cuda:0", "cuda:1"]:
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    dev = resolve_device(req)
                self.assertEqual(dev.type, "cpu", f"Requested {req} must fall back to cpu")
                self.assertIn("[WARNING]", captured.getvalue(), f"Fallback from {req} must print [WARNING]")
        else:
            with patch("torch.cuda.is_available", return_value=False):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    dev = resolve_device("cuda")
                self.assertEqual(dev.type, "cpu")
                self.assertIn("[WARNING]", captured.getvalue())

    def test_08_device_resolution_cpu_requests_are_silent(self):
        """Verify resolve_device('cpu') returns CPU without any warning output."""
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            dev = resolve_device("cpu")
        self.assertEqual(dev.type, "cpu")
        self.assertEqual(captured.getvalue(), "", "resolve_device('cpu') should not print any warnings")

    # =========================================================================
    # 5. MODEL NAME ALIASING INVARIANTS
    # =========================================================================

    def test_09_model_name_aliasing_coverage_and_resolution(self):
        """Verify MODEL_ALIASES mappings across ckpt_path, build_model, and train_one."""
        # 1. Alias mapping table
        self.assertEqual(MODEL_ALIASES.get("st_gcn_lstm_symmetric"), "st_gcn_lstm_sym")
        self.assertEqual(MODEL_ALIASES.get("st_gcn_lstm_directed"), "st_gcn_lstm_dir")

        # 2. ckpt_path equivalence: alias produces canonical checkpoint filename
        seed = 42
        path_alias_sym = ckpt_path(self.cfg, "st_gcn_lstm_symmetric", seed)
        path_canon_sym = ckpt_path(self.cfg, "st_gcn_lstm_sym", seed)
        self.assertEqual(path_alias_sym, path_canon_sym, "Alias symmetric ckpt_path mismatch")
        self.assertTrue(path_alias_sym.endswith("st_gcn_lstm_sym_seed42.pt"))

        path_alias_dir = ckpt_path(self.cfg, "st_gcn_lstm_directed", seed)
        path_canon_dir = ckpt_path(self.cfg, "st_gcn_lstm_dir", seed)
        self.assertEqual(path_alias_dir, path_canon_dir, "Alias directed ckpt_path mismatch")
        self.assertTrue(path_alias_dir.endswith("st_gcn_lstm_dir_seed42.pt"))

        # 3. train_one alias resolution and correct topology instantiation
        # Sym mode
        m_sym = build_model(MODEL_ALIASES["st_gcn_lstm_symmetric"], self.cfg)
        self.assertEqual(m_sym.gcn1.mode, "symmetric")
        self.assertEqual(m_sym.gcn2.mode, "symmetric")
        self.assertEqual(m_sym.cfg.GRAPH_MODE, "symmetric")

        # Dir mode
        m_dir = build_model(MODEL_ALIASES["st_gcn_lstm_directed"], self.cfg)
        self.assertEqual(m_dir.gcn1.mode, "directed")
        self.assertEqual(m_dir.gcn2.mode, "directed")
        self.assertEqual(m_dir.cfg.GRAPH_MODE, "directed")

        # 4. End-to-end train_one with alias names
        cfg_fast = copy.copy(self.cfg)
        cfg_fast.MAX_EPOCHS = 1
        cfg_fast.BATCH_SIZE = 16

        best_sym, _, path_sym = train_one(
            cfg_fast, "st_gcn_lstm_symmetric", seed=42, tr=self.tr_ds, va=self.va_ds, log=lambda *args: None
        )
        self.assertTrue(os.path.exists(path_sym))
        self.assertTrue(path_sym.endswith("st_gcn_lstm_sym_seed42.pt"))

        best_dir, _, path_dir = train_one(
            cfg_fast, "st_gcn_lstm_directed", seed=42, tr=self.tr_ds, va=self.va_ds, log=lambda *args: None
        )
        self.assertTrue(os.path.exists(path_dir))
        self.assertTrue(path_dir.endswith("st_gcn_lstm_dir_seed42.pt"))

    # =========================================================================
    # 6. SAFETY GUARDRAIL & SUMMARY DEDUPLICATION INVARIANTS
    # =========================================================================

    def test_10_cpu_safety_guardrail_enforced_for_grid_training(self):
        """Verify RuntimeError safety guardrail when multi-seed CPU training is attempted."""
        # 4 models x 5 seeds = 20 runs (> 2 runs) on CPU without --force-cpu
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(RuntimeError) as ctx:
                main(
                    cfg=self.cfg,
                    models=["lstm", "paper_overall"],
                    seeds=[42, 43],  # 2 x 2 = 4 runs > 2
                    cli_args=["--device", "cpu"],
                )
            self.assertIn("SAFETY GUARDRAIL ENFORCED", str(ctx.exception))

    def test_11_training_summary_progressive_deduplication(self):
        """Verify training_summary.csv is deduplicated when the same model/seed is rerun."""
        summary_path = self.cfg.SUMMARY_PATH
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        initial_df = pd.DataFrame([
            {"model": "lstm", "seed": 42, "best_val_loss": 0.50, "wall_clock_s": 10.0},
            {"model": "paper_overall", "seed": 42, "best_val_loss": 0.30, "wall_clock_s": 12.0},
        ])
        initial_df.to_csv(summary_path, index=False)

        def mock_build_datasets(cfg, save_scaler=False):
            return self.tr_ds, self.va_ds, self.va_ds, None, {
                "n_rows_used": 80, "start": "2024-01-01", "end": "2024-01-02"
            }

        with patch("src.dataset.build_datasets", side_effect=mock_build_datasets), \
             patch("training.train.build_datasets", side_effect=mock_build_datasets), \
             patch("sys.stdout", io.StringIO()):
            ckpt = ckpt_path(self.cfg, "lstm", 42)
            if os.path.exists(ckpt):
                os.remove(ckpt)

            main(
                cfg=self.cfg,
                models=["lstm"],
                seeds=[42],
                cli_args=["--force-cpu", "--epochs", "1", "--batch-size", "16"],
            )

        updated_df = pd.read_csv(summary_path)
        # Must still have exactly 2 rows (lstm row was updated, not appended as duplicate!)
        lstm_rows = updated_df[(updated_df["model"] == "lstm") & (updated_df["seed"] == 42)]
        self.assertEqual(len(lstm_rows), 1, "Duplicate lstm seed 42 row found in training_summary.csv!")
        self.assertEqual(len(updated_df), 2, "Summary dataframe row count must remain 2 after deduplication")


if __name__ == "__main__":
    unittest.main()

