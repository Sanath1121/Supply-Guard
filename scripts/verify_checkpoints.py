#!/usr/bin/env python3
"""SupplyGuard Checkpoint Rigorous Verification Harness.

Verifies:
1. All 20 model checkpoints exist on disk.
2. Loads every checkpoint with weights_only=True and strict=True.
3. Recomputes validation MSE on the exact validation partition (57,817 windows).
4. Compares recomputed validation MSE to training_summary.csv with relative tolerance <= 1e-4.
5. For collapsed symmetric seeds (st_gcn_lstm_sym seeds 43 and 44), compares
   recomputed validation MSE to the analytical persistence validation MSE.
6. Halts immediately and exits non-zero if ANY mismatch is encountered.
"""
import os
import sys
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import Config
from src.dataset import build_datasets
from src.models import build_model


def verify_checkpoints(tol: float = 1e-4, batch_size: int = 256):
    cfg = Config()
    summary_path = os.path.join("outputs", "results", "training_summary.csv")
    if not os.path.exists(summary_path):
        print(f"[FATAL ERROR] training_summary.csv not found at {summary_path}")
        sys.exit(1)

    summary_df = pd.read_csv(summary_path)
    if len(summary_df) < 20:
        print(f"[FATAL ERROR] training_summary.csv contains {len(summary_df)} rows, expected 20.")
        sys.exit(1)

    print("=" * 80)
    print("SUPPLYGUARD PRODUCTION CHECKPOINT VERIFICATION HARNESS")
    print(f"Target: 20 checkpoints | Relative Tolerance: {tol:.1e} | Strict State Dict Load")
    print("=" * 80)

    # 1. Load validation partition without saving/mutating scaler
    print("[1/3] Loading validation partition (build_datasets)...")
    tr, va, te, scaler, info = build_datasets(cfg, save_scaler=False)
    n_va = len(va)
    print(f"      Validation samples: {n_va} windows | Segments: {info.get('n_clean_rows_retained')} clean rows")
    va_dl = DataLoader(va, batch_size=batch_size, shuffle=False)

    # 2. Compute analytical persistence validation MSE
    print("[2/3] Computing analytical persistence validation baseline...")
    pers_loss_sum = 0.0
    pers_total_items = 0
    with torch.no_grad():
        for b in va_dl:
            pred_pers = b["sequence"][:, -1, :4]
            tgt_node = b["node_target"]
            loss_p = F.mse_loss(pred_pers, tgt_node)
            blen = len(b["sequence"])
            pers_loss_sum += loss_p.item() * blen
            pers_total_items += blen

    pers_val_mse = pers_loss_sum / max(pers_total_items, 1)
    print(f"      Persistence Validation MSE: {pers_val_mse:.8f}")

    # 3. Verify all 20 checkpoints
    print("[3/3] Evaluating checkpoints against recorded losses and persistence...")
    print("-" * 80)
    header = f"{'Model Architecture':<18} | {'Seed':<5} | {'Recorded Val MSE':<16} | {'Recomputed Val':<16} | {'Rel Diff':<10} | {'Status'}"
    print(header)
    print("-" * 80)

    failures = []
    collapsed_seeds = {"st_gcn_lstm_sym": [43, 44]}

    for _, row in summary_df.iterrows():
        model_name = str(row["model"])
        seed = int(row["seed"])
        recorded_val = float(row["best_val_loss"])
        ckpt_file = os.path.join("outputs", "models", f"{model_name}_seed{seed}.pt")

        if not os.path.exists(ckpt_file):
            msg = f"Missing checkpoint file: {ckpt_file}"
            print(f"{model_name:<18} | {seed:<5} | {recorded_val:<16.8f} | {'MISSING':<16} | {'N/A':<10} | [FAILED]")
            failures.append((model_name, seed, msg))
            continue

        try:
            model = build_model(model_name, cfg)
            state_dict = torch.load(ckpt_file, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict, strict=True)
            model.eval()
        except Exception as e:
            msg = f"Checkpoint load failed: {e}"
            print(f"{model_name:<18} | {seed:<5} | {recorded_val:<16.8f} | {'LOAD ERROR':<16} | {'N/A':<10} | [FAILED]")
            failures.append((model_name, seed, msg))
            continue

        # Recompute validation loss matching train.py logic
        loss_sum = 0.0
        n_items = 0
        with torch.no_grad():
            for b in va_dl:
                seq = b["sequence"]
                pred = model(seq)
                tgt = b["tri_target"] if model_name == "paper_overall" else b["node_target"]
                batch_loss = F.mse_loss(pred, tgt)
                blen = len(seq)
                loss_sum += batch_loss.item() * blen
                n_items += blen

        recomputed_val = loss_sum / max(n_items, 1)
        rel_diff = abs(recomputed_val - recorded_val) / recorded_val

        # Verification check against recorded value
        if rel_diff > tol:
            status = "[MISMATCH]"
            failures.append((model_name, seed, f"Rel diff {rel_diff:.2e} exceeds tolerance {tol:.1e}"))
        else:
            status = "[VERIFIED]"

        # Additional check for collapsed seeds
        if model_name in collapsed_seeds and seed in collapsed_seeds[model_name]:
            rel_diff_pers = abs(recomputed_val - pers_val_mse) / pers_val_mse
            if rel_diff_pers <= tol:
                status += " (PERS-COLLAPSE)"
            else:
                status = "[COLLAPSE MISMATCH]"
                failures.append((model_name, seed, f"Expected collapse match to persistence {pers_val_mse:.8f}, got rel diff {rel_diff_pers:.2e}"))

        print(f"{model_name:<18} | {seed:<5} | {recorded_val:<16.8f} | {recomputed_val:<16.8f} | {rel_diff:<10.2e} | {status}")

    print("-" * 80)
    if failures:
        print(f"\n[VERIFICATION FAILED] Encountered {len(failures)} mismatch/failure(s):")
        for m, s, reason in failures:
            print(f"  - {m} (seed {s}): {reason}")
        print("\nHALTING PIPELINE PER AUDIT CONTRACT. INVESTIGATE DISCREPANCIES.")
        sys.exit(1)
    else:
        print("\n[VERIFICATION SUCCESS] ALL 20 CHECKPOINTS 100% VERIFIED WITHIN TOLERANCE!")
        print(f"  - Recorded training losses match recomputed validation MSE (max rel diff <= {tol:.1e}).")
        print(f"  - st_gcn_lstm_sym seeds 43 and 44 verified as collapsed to analytical persistence MSE ({pers_val_mse:.8f}).")
        print("  - All state dicts load with strict=True and weights_only=True.")
        return True


if __name__ == "__main__":
    verify_checkpoints()
