"""Write four Sandbox demo CSVs taken from the real held-out test partition (raw dataset units),
plus one invalid file, and print the expected Sandbox output for each.

Run from the project root:  python -m scripts.make_sandbox_demo
Windows are chosen by a fixed rule, not by hand:
  calm         = the test window with the median absolute change over the 10-minute horizon
  <echelon>    = the test window with the largest RISE of that echelon's risk index over the horizon
Expected values = directed ST-GCN-LSTM, seed 42 (the app's default), per-echelon training terciles,
Integrated Gradients with the training-mean baseline (64 steps), as the Sandbox computes them.
"""
import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
import torch

from src.config import Config
from src.dataset import build_datasets
from src.explainability import RiskExplainer
from src.models.st_gcn_lstm import build_model

cfg = Config()
tr, _, te, sc, info = build_datasets(cfg, save_scaler=False)
X, Y = te.sequences.numpy(), te.node_targets.numpy()
model = build_model("st_gcn_lstm_dir", cfg)
model.load_state_dict(torch.load("outputs/models/st_gcn_lstm_dir_seed42.pt", map_location="cpu", weights_only=True))
model.eval()
with torch.no_grad():
    P = model(te.sequences).numpy()
explainer = RiskExplainer(model, torch.tensor(info["train_mean"], dtype=torch.float32), steps=64)

lo, hi = np.quantile(tr.node_targets.numpy(), [1 / 3, 2 / 3], axis=0)
tier = lambda v, k: "Low" if v < lo[k] else ("Medium" if v <= hi[k] else "High")
now = X[:, -1, :4]
rise = Y - now
names = cfg.NODE_NAMES

no_zero_dist = (X[:, :, 2] > 1e-6).all(axis=1)          # 88 test windows hold a raw Distributor reading of 0.00
dist_rise = np.where(no_zero_dist, rise[:, 2], -np.inf)
picks = [
    ("1_calm_normal_operations.csv", int(np.argsort(np.abs(rise).mean(axis=1))[len(rise) // 2]), "median-change (typical calm) window"),
    ("2_retailer_demand_shock.csv", int(np.argmax(rise[:, 3])), "largest Retailer risk rise in the test period"),
    ("3_distributor_disruption.csv", int(np.argmax(dist_rise)), "largest Distributor risk rise without a 0.00 (missing-like) reading"),
    ("4_manufacturer_sudden_shock.csv", int(np.argmax(rise[:, 1])), "largest Manufacturer risk rise in the test period"),
]

os.makedirs("sandbox_demo", exist_ok=True)
for fname, i, rule in picks:
    raw = sc.inverse_transform(X[i])
    df = pd.DataFrame(raw, columns=cfg.FEATURE_COLS)
    if te.seq_timestamps is not None:
        df.insert(0, cfg.DATE_COL, [str(t) for t in te.seq_timestamps[i]])
    df.round(4).to_csv(os.path.join("sandbox_demo", fname), index=False)
    print(f"\n### {fname}  ({rule}; test window {i}; last step {te.seq_timestamps[i][-1] if te.seq_timestamps is not None else ''})")
    print(f"  raw now: " + ", ".join(f"{n} {r:.2f}" for n, r in zip(names + ['Cost'], raw[-1])))
    for k, n in enumerate(names):
        print(f"  {n:13s} now {now[i, k]:.3f} ({tier(now[i, k], k)}) -> forecast {P[i, k]:.3f} ({tier(P[i, k], k)}) | actual t+5 {Y[i, k]:.3f} ({tier(Y[i, k], k)}) | err model {abs(P[i, k] - Y[i, k]):.3f} vs persistence {abs(now[i, k] - Y[i, k]):.3f}")
    seq = torch.tensor(X[i])
    edges = []
    for (u, v), tgt, feat in [(("Supplier", "Manufacturer"), 1, 0), (("Manufacturer", "Distributor"), 2, 1), (("Distributor", "Retailer"), 3, 2)]:
        r = explainer.explain(seq, tgt)
        share = r["feature_importance"][feat]
        edges.append(f"{u[0]}->{v[0]} {share:.0%} ({'FLOW' if share > 0.05 else 'LINK'})")
    print("  network edges: " + ", ".join(edges))
    highs = [k for k in range(4) if tier(P[i, k], k) == "High"]
    if highs:
        k = max(highs, key=lambda k: P[i, k] - hi[k])
        print(f"  HIGHEST RISK marker: {names[k]}")

bad = pd.read_csv(os.path.join("sandbox_demo", "1_calm_normal_operations.csv")).drop(columns=["Total_Cost"])
bad.to_csv(os.path.join("sandbox_demo", "5_invalid_missing_total_cost.csv"), index=False)
print("\n### 5_invalid_missing_total_cost.csv: file 1 without the Total_Cost column (validation demo)")
