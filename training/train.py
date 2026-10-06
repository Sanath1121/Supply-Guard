"""SupplyGuard Model Training Pipeline (Plan A Phase 4).

Features:
- Multi-architecture training: lstm, paper_overall, st_gcn_lstm_sym, st_gcn_lstm_dir
- Multi-seed execution: 42, 43, 44, 45, 46
- Hardware device routing: --device cuda (with automatic fallback to cpu)
- Gradient norm clipping: cfg.GRAD_CLIP (1.0)
- Validation early stopping: cfg.PATIENCE (10)
- Checkpoint-resume resilience: skips already trained checkpoints with [Skip] log
- Loss curve persistence: outputs/results/{model}_seed{seed}_loss.csv
- Progressive training summary: outputs/results/training_summary.csv
- Local CPU safety guardrail: prevents accidental multi-hour full CPU execution
- Rapid smoke verification: --smoke
"""
import argparse
import copy
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import Config
from src.dataset import SupplyChainDataset, build_datasets
from src.models.st_gcn_lstm import build_model

MODEL_NAMES = ["lstm", "paper_overall", "st_gcn_lstm_sym", "st_gcn_lstm_dir"]
MODEL_ALIASES = {
    "st_gcn_lstm_symmetric": "st_gcn_lstm_sym",
    "st_gcn_lstm_directed": "st_gcn_lstm_dir",
}


def parse_args(args=None):
    """Parse command line arguments for the training pipeline."""
    parser = argparse.ArgumentParser(
        description="SupplyGuard Model Training Pipeline (Plan A Phase 4)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Target device ('cuda' or 'cpu'). Automatically falls back to 'cpu' if CUDA is unavailable.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODEL_NAMES,
        help=f"List of model names to train. Default: {MODEL_NAMES}",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46],
        help="Random seeds for evaluation. Default: [42, 43, 44, 45, 46]",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Max epochs per run (overrides cfg.MAX_EPOCHS).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides cfg.BATCH_SIZE).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides cfg.LEARNING_RATE).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run fast smoke test (synthetic data, 1 epoch, 1 seed) to verify pipeline.",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Bypass CPU safety guardrail for local multi-seed execution.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite of existing model checkpoints and result CSVs.",
    )
    return parser.parse_args(args)



def resolve_device(requested_device: str) -> torch.device:
    """Resolve target device with automatic fallback to CPU if CUDA is unavailable."""
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[WARNING] Requested device '{requested_device}', but CUDA is unavailable. Falling back to 'cpu'.")
        return torch.device("cpu")
    return torch.device(requested_device)


def set_seed(seed: int):
    """Set random seed for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ckpt_path(cfg, name: str, seed: int) -> str:
    """Return destination path for model checkpoint, resolving model name aliases."""
    canonical_name = MODEL_ALIASES.get(name, name)
    return os.path.join(cfg.CKPT_DIR, f"{canonical_name}_seed{seed}.pt")


def _loss(model_name: str, model: torch.nn.Module, batch: dict, device: Optional[torch.device] = None) -> torch.Tensor:
    """Compute MSE loss for model predictions against targets, routing tensors to device."""
    seq = batch["sequence"].to(device) if device is not None else batch["sequence"]
    pred = model(seq)
    tgt = batch["tri_target"] if model_name == "paper_overall" else batch["node_target"]
    tgt = tgt.to(device) if device is not None else tgt
    return F.mse_loss(pred, tgt)


def train_one(cfg, name: str, seed: int, tr, va, device="cpu", log=print) -> Tuple[float, pd.DataFrame, str]:
    """Train a single model for a single seed with early stopping and gradient clipping."""
    set_seed(seed)
    dev = torch.device(device) if isinstance(device, str) else (device or torch.device("cpu"))
    canonical_name = MODEL_ALIASES.get(name, name)
    model = build_model(canonical_name, cfg).to(dev)

    lr = getattr(cfg, "LEARNING_RATE", 1e-3)
    batch_size = getattr(cfg, "BATCH_SIZE", 64)
    max_epochs = getattr(cfg, "MAX_EPOCHS", 50)
    patience = getattr(cfg, "PATIENCE", 10)
    grad_clip = getattr(cfg, "GRAD_CLIP", 1.0)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    tr_dl = DataLoader(tr, batch_size=batch_size, shuffle=True, generator=g)
    va_dl = DataLoader(va, batch_size=min(256, max(len(va), 1)), shuffle=False)

    best, bad, hist = float("inf"), 0, []
    path = ckpt_path(cfg, canonical_name, seed)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    for ep in range(1, max_epochs + 1):
        ep_t0 = time.time()
        model.train()
        tl, n = 0.0, 0
        for b in tr_dl:
            opt.zero_grad()
            loss = _loss(canonical_name, model, b, dev)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            batch_len = len(b["sequence"])
            tl += loss.item() * batch_len
            n += batch_len

        model.eval()
        vl, m = 0.0, 0
        with torch.no_grad():
            for b in va_dl:
                loss_v = _loss(canonical_name, model, b, dev)
                batch_len = len(b["sequence"])
                vl += loss_v.item() * batch_len
                m += batch_len

        tl = tl / max(n, 1)
        vl = vl / max(m, 1)
        ep_elapsed = time.time() - ep_t0
        if ep == 1:
            log(f"  epoch 1 time: {ep_elapsed:.2f}s")

        hist.append({
            "epoch": ep,
            "train_loss": tl,
            "val_loss": vl,
            "lr": lr,
            "time_sec": round(ep_elapsed, 4),
        })

        if vl < best - 1e-7:
            best, bad = vl, 0
            tmp_path = path + ".tmp"
            torch.save(model.state_dict(), tmp_path)
            os.replace(tmp_path, path)
        else:
            bad += 1
            if bad >= patience:
                log(f"  early stop @ epoch {ep}")
                break

    return best, pd.DataFrame(hist), path


def run_smoke_mode(cfg, device: torch.device):
    """Execute rapid 1-epoch smoke verification on lightweight synthetic data writing strictly to tempfile."""
    print("[SMOKE] Executing rapid smoke test training pipeline on synthetic data...")
    set_seed(42)
    seq_len = getattr(cfg, "SEQ_LEN", 10)
    n_features = getattr(cfg, "NUM_INPUT_FEATURES", 5)
    n_nodes = getattr(cfg, "NUM_NODES", 4)

    X_tr = torch.rand(128, seq_len, n_features)
    y_tr = X_tr[:, -1, :n_nodes] + torch.randn(128, n_nodes) * 0.01
    X_va = torch.rand(32, seq_len, n_features)
    y_va = X_va[:, -1, :n_nodes] + torch.randn(32, n_nodes) * 0.01
    tr = SupplyChainDataset(X_tr, y_tr)
    va = SupplyChainDataset(X_va, y_va)

    import tempfile
    tmp_dir = tempfile.mkdtemp()
    cfg_smoke = copy.copy(cfg)
    cfg_smoke.CKPT_DIR = tmp_dir
    cfg_smoke.MAX_EPOCHS = 1
    cfg_smoke.BATCH_SIZE = 32
    cfg_smoke.PATIENCE = 2

    best, hist, path = train_one(cfg_smoke, "lstm", 42, tr, va, device=device)
    smoke_loss_csv = os.path.join(tmp_dir, "lstm_seed42_loss.csv")
    hist.to_csv(smoke_loss_csv, index=False)
    smoke_summary_csv = os.path.join(tmp_dir, "training_summary.csv")
    pd.DataFrame([{
        "model": "lstm",
        "seed": 42,
        "best_val_loss": best,
        "wall_clock_s": 0.25,
    }]).to_csv(smoke_summary_csv, index=False)

    print(f"[SMOKE] Smoke run completed successfully. Best val loss: {best:.6f}, checkpoint saved to: {path}")
    return best, hist, path


def main(cfg=None, models=None, seeds=None, cli_args=None):
    """Main training entry point for CLI and programmatic execution."""
    args = parse_args(cli_args)
    cfg = cfg or Config()

    # CLI hyperparameter overrides
    if args.epochs is not None:
        cfg.MAX_EPOCHS = args.epochs
    if args.batch_size is not None:
        cfg.BATCH_SIZE = args.batch_size
    if args.lr is not None:
        cfg.LEARNING_RATE = args.lr

    device = resolve_device(args.device)
    raw_models = models if models is not None else args.models
    resolved_models = [MODEL_ALIASES.get(m, m) for m in raw_models]
    resolved_seeds = seeds if seeds is not None else args.seeds

    os.makedirs(cfg.CKPT_DIR, exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)

    print(f"SupplyGuard Training Pipeline | Device: {device} | Models: {resolved_models} | Seeds: {resolved_seeds}")

    # Handle smoke test mode
    if args.smoke:
        return run_smoke_mode(cfg, device)

    # Safety Guardrail: Prevent full CPU grid execution on local machine
    if device.type == "cpu" and (len(resolved_models) * len(resolved_seeds) > 2) and not args.force_cpu:
        raise RuntimeError(
            "SAFETY GUARDRAIL ENFORCED: Full multi-seed grid training on CPU is prohibited per Plan A constraints. "
            "Please run this on Google Colab with GPU using 'python -m training.train --device cuda'. "
            "To test pipeline functionality locally, use '--smoke' or specify '--force-cpu'."
        )

    should_save_scaler = args.force or not os.path.exists(cfg.SCALER_PATH)
    tr, va, te, scaler, info = build_datasets(cfg, save_scaler=should_save_scaler)
    print(f"rows used={info['n_rows_used']}  span={info['start']} -> {info['end']}  "
          f"train/val/test windows={len(tr)}/{len(va)}/{len(te)}")

    try:
        import mlflow
        mlflow.set_experiment("SupplyGuard")
    except Exception:
        mlflow = None

    summary_path = "outputs/results/training_summary.csv"
    if os.path.exists(summary_path):
        try:
            summary_df = pd.read_csv(summary_path)
            rows = summary_df.to_dict("records")
        except Exception:
            rows = []
    else:
        rows = []

    for name in resolved_models:
        for seed in resolved_seeds:
            path = ckpt_path(cfg, name, seed)
            if os.path.exists(path) and not args.force:
                print(f"[{name} seed={seed}] [Skip] Checkpoint exists: {path}")
                continue

            t0 = time.time()
            print(f"[{name} seed={seed}]")
            if mlflow:
                with mlflow.start_run(run_name=f"{name}_s{seed}"):
                    mlflow.log_params({
                        "model": name,
                        "seed": seed,
                        "graph_mode": getattr(cfg, "GRAPH_MODE", "directed"),
                        "lr": getattr(cfg, "LEARNING_RATE", 1e-3),
                        "horizon": getattr(cfg, "HORIZON", 5),
                        "seq_len": getattr(cfg, "SEQ_LEN", 10),
                        "residual": getattr(cfg, "RESIDUAL", True),
                    })
                    best, hist, path = train_one(cfg, name, seed, tr, va, device=device)
                    for r in hist.itertuples():
                        mlflow.log_metric("train_loss", r.train_loss, step=r.epoch)
                        mlflow.log_metric("val_loss", r.val_loss, step=r.epoch)
                    mlflow.log_artifact(path)
            else:
                best, hist, path = train_one(cfg, name, seed, tr, va, device=device)

            elapsed = time.time() - t0
            loss_file = f"outputs/results/{name}_seed{seed}_loss.csv"
            hist.to_csv(loss_file, index=False)
            print(f"  best val loss {best:.6f}  ({elapsed:.0f}s, {len(hist)} epochs)")

            # Deduplicate and append summary row progressively
            rows = [r for r in rows if not (r.get("model") == name and r.get("seed") == seed)]
            rows.append({
                "model": name,
                "seed": seed,
                "best_val_loss": best,
                "wall_clock_s": round(elapsed, 2),
            })
            pd.DataFrame(rows).to_csv(summary_path, index=False)

    completion_marker = os.path.join("outputs", "results", ".training_completed")
    all_complete = all(os.path.exists(ckpt_path(cfg, m, s)) for m in MODEL_NAMES for s in [42, 43, 44, 45, 46])
    if all_complete:
        with open(completion_marker, "w", encoding="utf-8") as f:
            f.write("COMPLETE\n")



if __name__ == "__main__":
    main()
