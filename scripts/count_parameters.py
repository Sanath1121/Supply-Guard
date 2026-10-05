#!/usr/bin/env python3
"""SupplyGuard Phase 3: Model Parameter Counting Utility.

Instantiates all 4 production architectures and computes total and trainable parameters:
1. lstm (LSTMBaseline)
2. paper_overall (PaperHybridOverall)
3. st_gcn_lstm (directed mode)
4. st_gcn_lstm (symmetric mode)

Outputs tabular results to outputs/results/param_counts.csv.
"""
import copy
import os
import sys
import pandas as pd
import torch

# Ensure repository root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import Config
from src.models import STGCNLSTM, LSTMBaseline, PaperHybridOverall, build_model


def count_parameters(model: torch.nn.Module):
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    cfg = Config()

    cfg_directed = copy.deepcopy(Config())
    cfg_directed.GRAPH_MODE = "directed"

    cfg_symmetric = copy.deepcopy(Config())
    cfg_symmetric.GRAPH_MODE = "symmetric"

    models = {
        "lstm": build_model("lstm", cfg),
        "paper_overall": build_model("paper_overall", cfg),
        "st_gcn_lstm (directed)": STGCNLSTM(cfg_directed),
        "st_gcn_lstm (symmetric)": STGCNLSTM(cfg_symmetric),
    }

    records = []
    print("=" * 70)
    print("SUPPLYGUARD MODEL PARAMETER COUNTS")
    print("=" * 70)
    print(f"{'Model':<28} | {'Total Params':>14} | {'Trainable Params':>16}")
    print("-" * 70)

    for name, model in models.items():
        total, trainable = count_parameters(model)
        records.append({
            "model": name,
            "total_params": total,
            "trainable_params": trainable,
        })
        print(f"{name:<28} | {total:>14,d} | {trainable:>16,d}")

    print("=" * 70)

    df = pd.DataFrame(records)
    out_dir = os.path.join(PROJECT_ROOT, "outputs", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "param_counts.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved parameter count report to: {out_csv}")
    return df


if __name__ == "__main__":
    main()
