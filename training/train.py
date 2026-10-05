"""Train every model for every seed.  Run from project root:  python -m training.train

* Same loss (node-wise MSE; TRI MSE for the paper-style model) for train AND val -> comparable curves
* Early stopping on val loss; best weights saved per (model, seed)
* Seeds set for python / numpy / torch
* MLflow is optional: used only if installed
"""
import os, random, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from src.config import Config
from src.dataset import build_datasets
from src.models.st_gcn_lstm import build_model

MODEL_NAMES = ["lstm", "paper_overall", "st_gcn_lstm"]


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _loss(model_name, model, batch):
    pred = model(batch["sequence"])
    tgt = batch["tri_target"] if model_name == "paper_overall" else batch["node_target"]
    return F.mse_loss(pred, tgt)


def ckpt_path(cfg, name, seed):
    tag = f"{name}_{cfg.GRAPH_MODE}" if name == "st_gcn_lstm" else name
    return os.path.join(cfg.CKPT_DIR, f"{tag}_seed{seed}.pt")


def train_one(cfg, name, seed, tr, va, log=print):
    set_seed(seed)
    model = build_model(name, cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    g = torch.Generator().manual_seed(seed)
    tr_dl = DataLoader(tr, batch_size=cfg.BATCH_SIZE, shuffle=True, generator=g)  # shuffles batches only, never the split
    va_dl = DataLoader(va, batch_size=256, shuffle=False)
    best, bad, hist = float("inf"), 0, []
    path = ckpt_path(cfg, name, seed)
    for ep in range(1, cfg.MAX_EPOCHS + 1):
        model.train(); tl, n = 0.0, 0
        for b in tr_dl:
            opt.zero_grad(); loss = _loss(name, model, b); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP); opt.step()
            tl += loss.item() * len(b["sequence"]); n += len(b["sequence"])
        model.eval(); vl, m = 0.0, 0
        with torch.no_grad():
            for b in va_dl:
                vl += _loss(name, model, b).item() * len(b["sequence"]); m += len(b["sequence"])
        tl, vl = tl / n, vl / m
        hist.append({"epoch": ep, "train_loss": tl, "val_loss": vl})
        if vl < best - 1e-7:
            best, bad = vl, 0; torch.save(model.state_dict(), path)
        else:
            bad += 1
            if bad >= cfg.PATIENCE:
                log(f"  early stop @ {ep}"); break
    return best, pd.DataFrame(hist), path


def main(cfg=None, models=MODEL_NAMES, seeds=None):
    cfg = cfg or Config()
    seeds = seeds or cfg.SEEDS
    os.makedirs(cfg.CKPT_DIR, exist_ok=True); os.makedirs("outputs/results", exist_ok=True)
    tr, va, te, scaler, info = build_datasets(cfg)
    print(f"rows used={info['n_rows_used']}  span={info['start']} -> {info['end']}  "
          f"train/val/test windows={len(tr)}/{len(va)}/{len(te)}")
    try:
        import mlflow; mlflow.set_experiment("SupplyGuard")
    except Exception:
        mlflow = None
    rows = []
    for name in models:
        for seed in seeds:
            t0 = time.time(); print(f"[{name} seed={seed}]")
            if mlflow:
                with mlflow.start_run(run_name=f"{name}_s{seed}"):
                    mlflow.log_params({"model": name, "seed": seed, "graph_mode": cfg.GRAPH_MODE, "lr": cfg.LEARNING_RATE,
                                       "horizon": cfg.HORIZON, "seq_len": cfg.SEQ_LEN, "residual": cfg.RESIDUAL})
                    best, hist, path = train_one(cfg, name, seed, tr, va)
                    for r in hist.itertuples():
                        mlflow.log_metric("train_loss", r.train_loss, step=r.epoch); mlflow.log_metric("val_loss", r.val_loss, step=r.epoch)
                    mlflow.log_artifact(path)
            else:
                best, hist, path = train_one(cfg, name, seed, tr, va)
            hist.to_csv(f"outputs/results/{name}_{cfg.GRAPH_MODE}_seed{seed}_loss.csv", index=False)
            print(f"  best val loss {best:.6f}  ({time.time()-t0:.0f}s, {len(hist)} epochs)")
            rows.append({"model": name, "seed": seed, "best_val_loss": best})
    pd.DataFrame(rows).to_csv("outputs/results/training_summary.csv", index=False)


if __name__ == "__main__":
    main()
