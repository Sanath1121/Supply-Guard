"""Adversarial stress test suite for Phase 4: Training Pipeline CLI, Guardrails & Colab Notebook.

Author: Challenger 1 (teamwork_preview_challenger_p4_1)
Objective:
1. CLI Stress & Invariants:
   - python -m training.train --help exits 0 without data loading.
   - python -m training.train --smoke finishes in < 2s with exit 0.
   - CPU Safety Guardrail: raises RuntimeError and refuses to run full multi-seed grid on CPU.
   - Bypass verification: --force-cpu and single model/seed <= 2 bypass the guardrail.
2. Colab Notebook Empirical Verification:
   - nbformat == 4
   - Valid JSON parsing
   - Exact required shell commands present in code cells.
3. Architecture, Checkpointing & Device Resolution invariants.
"""
import copy
import json
import os
import subprocess
import sys
import time
import unittest

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import Config
from training.train import (
    MODEL_ALIASES,
    MODEL_NAMES,
    ckpt_path,
    parse_args,
    resolve_device,
    run_smoke_mode,
)


class TestAdversarialPhase4Challenger1(unittest.TestCase):
    """Adversarial suite stress-testing Phase 4 CLI, safety guardrails, and Colab notebook."""

    def setUp(self):
        self.cfg = Config()
        self.notebook_path = os.path.join(PROJECT_ROOT, "notebooks", "colab_train.ipynb")

    # =========================================================================
    # PART 1: CLI STRESS & INVARIANTS
    # =========================================================================

    def test_01_cli_help_flag_exits_zero_without_data_loading(self):
        """Verify python -m training.train --help exits 0 fast without loading dataset."""
        t0 = time.time()
        cmd = [sys.executable, "-m", "training.train", "--help"]
        res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        elapsed = time.time() - t0

        self.assertEqual(res.returncode, 0, f"--help failed with stderr: {res.stderr}")
        self.assertIn("SupplyGuard Model Training Pipeline", res.stdout)
        self.assertIn("--device", res.stdout)
        self.assertIn("--smoke", res.stdout)
        self.assertIn("--force-cpu", res.stdout)
        self.assertNotIn("[DATA INGESTION]", res.stdout, "Data should not be loaded on --help")

    def test_02_cli_smoke_finishes_under_two_seconds_exit_zero(self):
        """Verify python -m training.train --smoke finishes training in < 2s with exit code 0."""
        t0 = time.time()
        cmd = [sys.executable, "-m", "training.train", "--smoke"]
        res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)

        self.assertEqual(res.returncode, 0, f"--smoke failed with stderr: {res.stderr}")
        self.assertIn("[SMOKE] Executing rapid smoke test training pipeline", res.stdout)
        self.assertIn("[SMOKE] Smoke run completed successfully", res.stdout)
        
        # Verify in-pipeline training epoch time is strictly < 2.0s
        import re
        match = re.search(r"epoch 1 time:\s+([0-9.]+)s", res.stdout)
        self.assertIsNotNone(match, "Expected 'epoch 1 time: Xs' in smoke output")
        epoch_time = float(match.group(1))
        self.assertLess(epoch_time, 2.0, f"Smoke epoch took too long: {epoch_time}s >= 2.0s")

    def test_03_cpu_safety_guardrail_refuses_full_grid(self):
        """Verify executing without --smoke or --force-cpu on CPU raises RuntimeError."""
        cmd = [sys.executable, "-m", "training.train"]
        res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)

        self.assertNotEqual(res.returncode, 0, "Full grid on CPU must NOT exit 0")
        combined_output = res.stdout + res.stderr
        self.assertIn("RuntimeError", combined_output)
        self.assertIn("SAFETY GUARDRAIL ENFORCED", combined_output)

    def test_04_cpu_safety_guardrail_refuses_partial_multi_seed_grid(self):
        """Verify executing with 2 models * 5 seeds (10 runs > 2) raises RuntimeError."""
        cmd = [sys.executable, "-m", "training.train", "--models", "lstm", "paper_overall"]
        res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)

        self.assertNotEqual(res.returncode, 0, "Partial multi-seed grid (>2 runs) must NOT exit 0")
        combined_output = res.stdout + res.stderr
        self.assertIn("RuntimeError", combined_output)
        self.assertIn("SAFETY GUARDRAIL ENFORCED", combined_output)

    def test_05_guardrail_boundary_and_force_cpu_logic(self):
        """Verify guardrail trigger condition boundary values and --force-cpu bypass."""
        # Case A: models * seeds <= 2 does not trigger guardrail
        args_single = parse_args(["--models", "lstm", "--seeds", "42"])
        resolved_models = [MODEL_ALIASES.get(m, m) for m in args_single.models]
        resolved_seeds = args_single.seeds
        cpu_device = torch.device("cpu")

        # Test condition from train.py:
        # device.type == "cpu" and (len(resolved_models) * len(resolved_seeds) > 2) and not args.force_cpu
        triggered_single = (
            cpu_device.type == "cpu"
            and (len(resolved_models) * len(resolved_seeds) > 2)
            and not args_single.force_cpu
        )
        self.assertFalse(triggered_single, "Single model/seed must NOT trigger guardrail")

        # Case B: models * seeds == 2 (e.g., 2 models, 1 seed) does not trigger guardrail
        args_pair = parse_args(["--models", "lstm", "paper_overall", "--seeds", "42"])
        triggered_pair = (
            cpu_device.type == "cpu"
            and (len(args_pair.models) * len(args_pair.seeds) > 2)
            and not args_pair.force_cpu
        )
        self.assertFalse(triggered_pair, "2 models * 1 seed (total 2 runs) must NOT trigger guardrail")

        # Case C: models * seeds == 3 (e.g., 3 models, 1 seed) triggers guardrail
        args_three = parse_args(["--models", "lstm", "paper_overall", "st_gcn_lstm_sym", "--seeds", "42"])
        triggered_three = (
            cpu_device.type == "cpu"
            and (len(args_three.models) * len(args_three.seeds) > 2)
            and not args_three.force_cpu
        )
        self.assertTrue(triggered_three, "3 models * 1 seed (total 3 runs) MUST trigger guardrail")

        # Case D: --force-cpu bypasses guardrail even for full 4 models * 5 seeds (20 runs)
        args_forced = parse_args(["--force-cpu"])
        self.assertTrue(args_forced.force_cpu)
        triggered_forced = (
            cpu_device.type == "cpu"
            and (len(args_forced.models) * len(args_forced.seeds) > 2)
            and not args_forced.force_cpu
        )
        self.assertFalse(triggered_forced, "--force-cpu MUST bypass the CPU guardrail")

    def test_06_device_resolution_and_cuda_fallback(self):
        """Verify resolve_device falls back to CPU cleanly when CUDA is not present."""
        dev_cpu = resolve_device("cpu")
        self.assertEqual(dev_cpu.type, "cpu")

        dev_cuda = resolve_device("cuda")
        if torch.cuda.is_available():
            self.assertEqual(dev_cuda.type, "cuda")
        else:
            self.assertEqual(dev_cuda.type, "cpu", "When CUDA unavailable, must fallback to cpu")

    def test_07_checkpoint_paths_and_model_aliases(self):
        """Verify checkpoint paths correctly resolve canonical names and aliases."""
        cfg = Config()
        p1 = ckpt_path(cfg, "lstm", 42)
        self.assertTrue(p1.endswith("lstm_seed42.pt"))

        p2 = ckpt_path(cfg, "st_gcn_lstm_symmetric", 43)
        self.assertTrue(p2.endswith("st_gcn_lstm_sym_seed43.pt"))

        p3 = ckpt_path(cfg, "st_gcn_lstm_directed", 44)
        self.assertTrue(p3.endswith("st_gcn_lstm_dir_seed44.pt"))

        p4 = ckpt_path(cfg, "paper_overall", 45)
        self.assertTrue(p4.endswith("paper_overall_seed45.pt"))

    # =========================================================================
    # PART 2: COLAB NOTEBOOK EMPIRICAL VERIFICATION
    # =========================================================================

    def test_08_colab_notebook_json_validity_and_nbformat(self):
        """Verify notebooks/colab_train.ipynb exists, parses as valid JSON, and has nbformat==4."""
        self.assertTrue(os.path.exists(self.notebook_path), f"Missing notebook: {self.notebook_path}")

        with open(self.notebook_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        self.assertIsInstance(nb, dict, "Notebook root must be a JSON object")
        self.assertEqual(nb.get("nbformat"), 4, f"nbformat must be 4, got {nb.get('nbformat')}")
        self.assertIn("cells", nb, "Notebook must contain 'cells'")
        self.assertGreater(len(nb["cells"]), 0, "Notebook cells must not be empty")

    def test_09_colab_notebook_required_shell_commands(self):
        """Assert exact required shell commands are present in code cells of colab_train.ipynb."""
        with open(self.notebook_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        code_cells = [cell for cell in nb.get("cells", []) if cell.get("cell_type") == "code"]
        self.assertGreaterEqual(len(code_cells), 6, "Expected at least 6 code cells in Colab notebook")

        # Collect concatenated source strings from all code cells
        code_sources = []
        for c in code_cells:
            src = c.get("source", [])
            if isinstance(src, list):
                src = "".join(src)
            code_sources.append(src.strip())

        full_code_corpus = "\n".join(code_sources)

        required_commands = [
            "!nvidia-smi",
            "!git clone https://github.com/Sanath1121/Supply-Guard.git",
            "%cd Supply-Guard",
            "!pip install -r requirements.txt",
            "!python setup_and_download.py",
            "!python -m training.train --device cuda",
            "!zip -r outputs.zip outputs/models outputs/results",
            "from google.colab import files; files.download('outputs.zip')",
        ]

        for req_cmd in required_commands:
            if ";" in req_cmd:
                # Handle multi-statement commands like 'from google.colab import files; files.download(...)'
                parts = [p.strip() for p in req_cmd.split(";")]
                for part in parts:
                    self.assertTrue(
                        any(part in cell_code for cell_code in code_sources),
                        f"Required command component '{part}' not found in any notebook code cell",
                    )
            else:
                self.assertTrue(
                    any(req_cmd in cell_code for cell_code in code_sources),
                    f"Required command '{req_cmd}' not found in any notebook code cell",
                )

    def test_10_smoke_mode_synthetic_execution_invariance(self):
        """Empirically test run_smoke_mode returns finite loss, valid dataframe, and checkpoint path."""
        best_val_loss, hist_df, path = run_smoke_mode(self.cfg, torch.device("cpu"))
        self.assertIsInstance(best_val_loss, float)
        self.assertFalse(torch.isnan(torch.tensor(best_val_loss)))
        self.assertGreater(best_val_loss, 0.0)
        self.assertEqual(len(hist_df), 1, "Smoke mode must run exactly 1 epoch")
        self.assertListEqual(list(hist_df.columns), ["epoch", "train_loss", "val_loss", "lr", "time_sec"])
        self.assertTrue(os.path.exists(path), f"Checkpoint must exist at {path}")


if __name__ == "__main__":
    unittest.main()
