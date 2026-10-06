"""Diagnostic script for investigating st_gcn_lstm_sym seed 43/44 collapse.

Read-only analysis:
1. State dictionary parameter weights & biases (head, gcn, lstm) across seeds 42-46.
2. Loss curves comparison across all 5 seeds.
3. Forward activation & delta norm analysis on validation data.
"""
import os
import sys
import torch
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import Config
from src.dataset import build_datasets
from src.models.st_gcn_lstm import build_model


def inspect_parameters():
    print("=" * 80)
    print("PART 1: PARAMETER WEIGHT AND BIAS INSPECTION (SEEDS 42-46)")
    print("=" * 80)

    seeds = [42, 43, 44, 45, 46]
    model_name = "st_gcn_lstm_sym"

    summary_rows = []

    for s in seeds:
        ckpt_path = os.path.join(PROJECT_ROOT, "outputs", "models", f"{model_name}_seed{s}.pt")
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        head0_w = state_dict["head.0.weight"]
        head0_b = state_dict["head.0.bias"]
        head3_w = state_dict["head.3.weight"]
        head3_b = state_dict["head.3.bias"]

        gcn1_w = state_dict["gcn1.lin.weight"]
        lstm_w_ih = state_dict["lstm.weight_ih_l0"]

        summary_rows.append({
            "seed": s,
            "head0_w_norm": float(torch.norm(head0_w)),
            "head0_b_norm": float(torch.norm(head0_b)),
            "head3_w_norm": float(torch.norm(head3_w)),
            "head3_w_max": float(torch.max(torch.abs(head3_w))),
            "head3_b_norm": float(torch.norm(head3_b)),
            "head3_b_max": float(torch.max(torch.abs(head3_b))),
            "gcn1_w_norm": float(torch.norm(gcn1_w)),
            "lstm_ih_norm": float(torch.norm(lstm_w_ih)),
        })

    df_params = pd.DataFrame(summary_rows)
    print(df_params.to_string(index=False))
    return df_params


def inspect_loss_curves():
    print("\n" + "=" * 80)
    print("PART 2: LOSS CURVE TRAJECTORY ANALYSIS")
    print("=" * 80)

    seeds = [42, 43, 44, 45, 46]
    for s in seeds:
        loss_csv = os.path.join(PROJECT_ROOT, "outputs", "results", f"st_gcn_lstm_sym_seed{s}_loss.csv")
        if os.path.exists(loss_csv):
            df_loss = pd.read_csv(loss_csv)
            n_epochs = len(df_loss)
            min_val_idx = df_loss["val_loss"].idxmin()
            best_epoch = df_loss.loc[min_val_idx, "epoch"]
            best_val = df_loss.loc[min_val_idx, "val_loss"]
            initial_tr = df_loss.loc[0, "train_loss"]
            initial_va = df_loss.loc[0, "val_loss"]
            final_tr = df_loss.loc[n_epochs - 1, "train_loss"]
            final_va = df_loss.loc[n_epochs - 1, "val_loss"]
            print(f"Seed {s}: {n_epochs} epochs | Best epoch: {best_epoch:.0f} (val: {best_val:.8f}) | "
                  f"Ep 1 (tr: {initial_tr:.6f}, va: {initial_va:.8f}) -> "
                  f"Ep {n_epochs} (tr: {final_tr:.6f}, va: {final_va:.8f})")


def inspect_activations_and_deltas():
    print("\n" + "=" * 80)
    print("PART 3: VALIDATION ACTIVATION AND DELTA MAGNITUDES")
    print("=" * 80)

    cfg = Config()
    cfg.MAX_SAMPLES = 5000  # Evaluate on first 5000 validation samples for speed
    _, va, _, _, _ = build_datasets(cfg, save_scaler=False)
    loader = torch.utils.data.DataLoader(va, batch_size=256, shuffle=False)

    batch = next(iter(loader))
    x_b = batch["sequence"]  # [B, 10, 5]
    y_last_b = x_b[:, -1, :4]  # [B, 4] persistence prediction

    for s in [42, 43, 44, 45, 46]:
        ckpt_path = os.path.join(PROJECT_ROOT, "outputs", "models", f"st_gcn_lstm_sym_seed{s}.pt")
        model = build_model("st_gcn_lstm_sym", cfg)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
        model.eval()

        with torch.no_grad():
            pred = model(x_b)  # [B, 4]
            # Since model outputs y_last + delta: delta = pred - y_last
            delta = pred - y_last_b

            delta_norm = torch.norm(delta, dim=-1).mean().item()
            delta_max = torch.max(torch.abs(delta)).item()
            pred_std = pred.std().item()

            print(f"Seed {s}: Mean ||Delta||: {delta_norm:.8f} | Max |Delta|: {delta_max:.8f} | Pred std: {pred_std:.6f}")


if __name__ == "__main__":
    inspect_parameters()
    inspect_loss_curves()
    inspect_activations_and_deltas()
