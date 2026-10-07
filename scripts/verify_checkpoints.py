"""SupplyGuard Checkpoint Verification Script (NEW-1).

Empirically verifies all 20 trained PyTorch model checkpoints against
the recorded validation losses in outputs/results/training_summary.csv.

Usage:
    python scripts/verify_checkpoints.py
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import Config
from src.dataset import build_datasets
from src.models.st_gcn_lstm import build_model
from training.train import MODEL_ALIASES, ckpt_path


def verify_all_checkpoints(tolerance: float = 1e-4) -> bool:
    """Verify all 20 checkpoints by recomputing validation MSE on the validation split."""
    print("=" * 80)
    print("SUPPLYGUARD CHECKPOINT INTEGRITY VERIFICATION (20 RUNS)")
    print("=" * 80)

    cfg = Config()
    summary_path = os.path.join(PROJECT_ROOT, "outputs", "results", "training_summary.csv")
    if not os.path.exists(summary_path):
        print(f"[ERROR] Training summary not found: {summary_path}")
        return False

    summary_df = pd.read_csv(summary_path)
    print(f"Loaded training summary with {len(summary_df)} entries.")

    print("[DATA] Loading dataset validation partition...")
    t0 = time.time()
    _, va_ds, _, scaler, info = build_datasets(cfg, save_scaler=False)
    print(f"[DATA] Validation partition loaded: {len(va_ds)} windows in {time.time() - t0:.2f}s")

    va_loader = DataLoader(va_ds, batch_size=512, shuffle=False)

    results = []
    all_passed = True

    for _, row in summary_df.iterrows():
        model_name = row["model"]
        canonical_name = MODEL_ALIASES.get(model_name, model_name)
        seed = int(row["seed"])
        recorded_loss = float(row["best_val_loss"])

        model_file = ckpt_path(cfg, canonical_name, seed)
        if not os.path.exists(model_file):
            print(f"[MISSING] Checkpoint file missing: {model_file}")
            results.append({
                "model": canonical_name,
                "seed": seed,
                "recorded": recorded_loss,
                "recomputed": np.nan,
                "diff": np.nan,
                "status": "MISSING",
            })
            all_passed = False
            continue

        # Instantiate model and load checkpoint
        model = build_model(canonical_name, cfg)
        state_dict = torch.load(model_file, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()

        # Compute validation loss across batches
        total_loss, total_count = 0.0, 0
        with torch.no_grad():
            for batch in va_loader:
                seq = batch["sequence"]
                pred = model(seq)
                tgt = batch["tri_target"] if canonical_name == "paper_overall" else batch["node_target"]
                loss = F.mse_loss(pred, tgt)
                n = len(seq)
                total_loss += loss.item() * n
                total_count += n

        recomputed_loss = total_loss / max(total_count, 1)
        diff = abs(recomputed_loss - recorded_loss)
        passed = diff <= tolerance

        if not passed:
            all_passed = False

        status = "PASS" if passed else "FAIL"
        results.append({
            "model": canonical_name,
            "seed": seed,
            "recorded": recorded_loss,
            "recomputed": recomputed_loss,
            "diff": diff,
            "status": status,
        })

    res_df = pd.DataFrame(results)
    print("\n" + "-" * 80)
    print(f"{'Model':<18} | {'Seed':<5} | {'Recorded Val MSE':<16} | {'Recomputed Val MSE':<18} | {'Diff':<10} | {'Status'}")
    print("-" * 80)
    for _, r in res_df.iterrows():
        rec_str = f"{r['recorded']:.6f}" if not np.isnan(r['recorded']) else "N/A"
        recomp_str = f"{r['recomputed']:.6f}" if not np.isnan(r['recomputed']) else "N/A"
        diff_str = f"{r['diff']:.2e}" if not np.isnan(r['diff']) else "N/A"
        print(f"{r['model']:<18} | {r['seed']:<5} | {rec_str:<16} | {recomp_str:<18} | {diff_str:<10} | {r['status']}")
    print("-" * 80)

    if all_passed:
        print(f"\n[VERIFIED] ALL {len(res_df)} CHECKPOINTS MATCH TRAINING SUMMARY WITHIN TOLERANCE ({tolerance:.1e})!")
    else:
        print(f"\n[FAILURE] One or more checkpoints deviated from training summary!")

    return all_passed


if __name__ == "__main__":
    success = verify_all_checkpoints()
    sys.exit(0 if success else 1)
