"""Reproducible evidence behind every number quoted in the research paper.

Run from the project root:  python -m scripts.paper_evidence

Reads the production checkpoints in outputs/models and the cleaned dataset, and writes
(never into outputs/models or outputs/results):
  docs/paper/evidence/paper_facts.json        dataset split, Table II statistics, shock-window skill, latency
  docs/paper/evidence/ig_audit_weighted.csv   Integrated-Gradients audit (5 seeds x 100 windows x 4 targets x 2 modes)
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd
import torch

from src.config import Config
from src.dataset import build_datasets, load_clean_frame
from src.explainability import NODE_NAMES, RiskExplainer
from src.models.st_gcn_lstm import build_model

OUT_DIR = os.path.join("docs", "paper", "evidence")
N_IG_WINDOWS = 100


def load_dir_model(cfg, seed):
    m = build_model("st_gcn_lstm_dir", cfg)
    m.load_state_dict(torch.load(f"outputs/models/st_gcn_lstm_dir_seed{seed}.pt",
                                 map_location="cpu", weights_only=True))
    return m.eval()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = Config()
    torch.set_num_threads(4)
    tr, va, te, scaler, info = build_datasets(cfg, save_scaler=False)
    out = {"windows_train_val_test": [len(tr), len(va), len(te)],
           "rows_used": info["n_rows_used"], "row_loss": info["row_loss"],
           "data_start": info["start"], "data_end": info["end"],
           "train_time_range": [str(tr.timestamps.min()), str(tr.timestamps.max())],
           "test_time_range": [str(te.timestamps.min()), str(te.timestamps.max())]}

    # Table II: statistics of the cleaned TRAIN partition (raw units and min-max scaled)
    df = load_clean_frame(cfg)
    n_tr = int(len(df) * cfg.TRAIN_RATIO)
    out["partition_rows_train_val_test"] = [n_tr, int(len(df) * cfg.VAL_RATIO),
                                            len(df) - n_tr - int(len(df) * cfg.VAL_RATIO)]
    t = df.iloc[:n_tr][cfg.FEATURE_COLS]
    out["table2_raw_train_mean_std"] = {c: [float(t[c].mean()), float(t[c].std())] for c in cfg.FEATURE_COLS}
    ts = pd.DataFrame(scaler.transform(t.values), columns=cfg.FEATURE_COLS)
    out["table2_scaled_train_mean_std"] = {c: [float(ts[c].mean()), float(ts[c].std())] for c in cfg.FEATURE_COLS}

    # Severity definition (train terciles per node) and test class balance
    lo, hi = np.quantile(tr.node_targets.numpy(), [1 / 3, 2 / 3], axis=0)
    out["tercile_thresholds_scaled"] = {"low_to_med": lo.tolist(), "med_to_high": hi.tolist()}
    Y, X = te.node_targets.numpy(), te.sequences.numpy()
    cls = np.concatenate([np.digitize(Y[:, i], [lo[i], hi[i]]) for i in range(4)])
    out["test_class_counts_low_med_high"] = np.bincount(cls, minlength=3).tolist()
    out["corr_xt_vs_ytH_test"] = [float(np.corrcoef(X[:, -1, i], Y[:, i])[0, 1]) for i in range(4)]

    # Shock-window skill: top 5% of mean |y(t+H) - y(t)|
    d = np.abs(Y - X[:, -1, :4]).mean(1)
    thr = np.quantile(d, 0.95)
    m = d >= thr
    pers = X[:, -1, :4]
    preds = {}
    for s in cfg.SEEDS:
        with torch.no_grad():
            preds[s] = load_dir_model(cfg, s)(te.sequences).numpy()
    out["shock_top5pct"] = {
        "n_windows": int(m.sum()), "threshold": float(thr),
        "mse_persistence_shock": float(((pers - Y)[m] ** 2).mean()),
        "mse_directed_shock_by_seed": [float(((preds[s] - Y)[m] ** 2).mean()) for s in cfg.SEEDS],
        "mse_persistence_calm": float(((pers - Y)[~m] ** 2).mean()),
        "mse_directed_calm_by_seed": [float(((preds[s] - Y)[~m] ** 2).mean()) for s in cfg.SEEDS]}

    # Integrated Gradients audit
    idx = np.linspace(0, len(te) - 1, N_IG_WINDOWS).astype(int)
    rows = []
    for s in cfg.SEEDS:
        ex = RiskExplainer(load_dir_model(cfg, s), info["train_mean"], steps=64)
        for i in idx:
            for k in range(4):
                for delta in (False, True):
                    r = ex.explain(te.sequences[i], k, residual_delta=delta)
                    a = np.abs(r["attribution"]).sum(0)
                    rows.append(dict(seed=s, window=int(i), target=NODE_NAMES[k], delta=delta,
                                     gap=r["completeness_gap"], total=r["total_abs_attribution"],
                                     S=a[0], M=a[1], D=a[2], R=a[3], Cost=a[4]))
    ig = pd.DataFrame(rows)
    ig.to_csv(os.path.join(OUT_DIR, "ig_audit_weighted.csv"), index=False)
    raw_gap = ig[~ig.delta].gap.abs()
    out["ig_completeness_abs_gap_raw_mode"] = {
        "n": int(len(raw_gap)), "median": float(raw_gap.median()), "mean": float(raw_gap.mean()),
        "p95": float(raw_gap.quantile(0.95)), "p99": float(raw_gap.quantile(0.99)), "max": float(raw_gap.max())}

    # Latency (CPU, batch size 1)
    mdl = load_dir_model(cfg, 42)
    x1 = te.sequences[:1]
    with torch.no_grad():
        for _ in range(20):
            mdl(x1)
        t0 = time.perf_counter()
        for _ in range(300):
            mdl(x1)
        out["inference_ms_cpu_batch1"] = (time.perf_counter() - t0) / 300 * 1000
    ex = RiskExplainer(mdl, info["train_mean"], steps=64)
    for _ in range(3):
        ex.explain(te.sequences[0], 3)
    t0 = time.perf_counter()
    for j in range(30):
        ex.explain(te.sequences[j], 3)
    out["ig_ms_cpu_per_target_64steps"] = (time.perf_counter() - t0) / 30 * 1000
    out["torch"], out["threads"] = torch.__version__, torch.get_num_threads()

    with open(os.path.join(OUT_DIR, "paper_facts.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=str)
    print(json.dumps({k: out[k] for k in ("windows_train_val_test", "partition_rows_train_val_test",
                                          "inference_ms_cpu_batch1", "ig_ms_cpu_per_target_64steps")}, indent=1))


if __name__ == "__main__":
    main()
