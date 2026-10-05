"""Evaluate every trained model + baselines on the untouched TEST partition.
Run from project root:  python -m training.evaluate

Outputs (outputs/results/):
  overall_metrics.csv   model comparison on derived TRI (mean +/- std over seeds)
  node_metrics.csv      per-echelon metrics for node-level models and baselines
  severity_metrics.csv  3-tier severity accuracy / macro-F1 (thresholds from TRAIN quantiles)
Metrics are reported in scaled units AND raw risk-index units.
"""
import os, numpy as np, pandas as pd, torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, accuracy_score
from src.config import Config
from src.dataset import build_datasets
from src.models.st_gcn_lstm import build_model
from training.train import ckpt_path


def reg(y, p):
    mse = mean_squared_error(y, p)
    return {"MSE": mse, "MAE": mean_absolute_error(y, p), "RMSE": float(np.sqrt(mse)), "R2": r2_score(y, p)}


def predict(model, ds):
    model.eval()
    with torch.no_grad():
        return model(ds.sequences).numpy()


def severity(x, lo, hi):
    return np.digitize(x, [lo, hi])          # 0 Low, 1 Medium, 2 High


def agg(df, keys):
    g = df.groupby(keys)
    out = g.mean(numeric_only=True).add_suffix("_mean").join(g.std(numeric_only=True).add_suffix("_std"))
    return out.drop(columns=[c for c in out.columns if c.startswith("seed")], errors="ignore").reset_index()


def main(cfg=None, models=("lstm", "paper_overall", "st_gcn_lstm"), seeds=None):
    cfg = cfg or Config(); seeds = seeds or cfg.SEEDS
    tr, va, te, scaler, info = build_datasets(cfg, save_scaler=False)
    y = te.node_targets.numpy(); tri = y.mean(1)
    rng = scaler.data_range_[:4]                          # raw = scaled * range + min
    lo, hi = np.quantile(tr.node_targets.numpy(), [1 / 3, 2 / 3])   # tiers from TRAIN distribution only

    preds = {}                                            # name -> list of [N,4] (or [N] for paper) arrays
    preds["persistence"] = [te.sequences[:, -1, :4].numpy()]
    Xtr, Xte = tr.sequences.numpy().reshape(len(tr), -1), te.sequences.numpy().reshape(len(te), -1)
    preds["ar10_ridge"] = [Ridge(alpha=1e-3).fit(Xtr, tr.node_targets.numpy()).predict(Xte)]
    for name in models:
        key = f"{name}_{cfg.GRAPH_MODE}" if name == "st_gcn_lstm" else name
        preds[key] = []
        for s in seeds:
            p = ckpt_path(cfg, name, s)
            if not os.path.exists(p):
                print("missing", p); continue
            m = build_model(name, cfg); m.load_state_dict(torch.load(p, map_location="cpu")); preds[key].append(predict(m, te))

    overall, node, sev = [], [], []
    for name, plist in preds.items():
        for s, p in enumerate(plist):
            is_node = p.ndim == 2
            tri_p = p.mean(1) if is_node else p
            r = reg(tri, tri_p); r_raw = reg(tri * rng.mean(), tri_p * rng.mean())   # approx raw TRI scale (mean range)
            overall.append({"model": name, "seed": s, **r, "MSE_raw_approx": r_raw["MSE"]})
            if is_node:
                for i, nn_ in enumerate(cfg.NODE_NAMES):
                    rr = reg(y[:, i], p[:, i])
                    node.append({"model": name, "node": nn_, "seed": s, **rr, "MSE_raw": rr["MSE"] * rng[i] ** 2,
                                 "RMSE_raw": rr["RMSE"] * rng[i]})
                st, sp = severity(y.ravel(), lo, hi), severity(p.ravel(), lo, hi)
                sev.append({"model": name, "seed": s, "accuracy": accuracy_score(st, sp), "macro_F1": f1_score(st, sp, average="macro")})

    os.makedirs("outputs/results", exist_ok=True)
    o, n_, s_ = agg(pd.DataFrame(overall), ["model"]), agg(pd.DataFrame(node), ["model", "node"]), agg(pd.DataFrame(sev), ["model"])
    o.to_csv("outputs/results/overall_metrics.csv", index=False)
    n_.to_csv("outputs/results/node_metrics.csv", index=False)
    s_.to_csv("outputs/results/severity_metrics.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n== OVERALL (derived TRI) ==\n", o[["model", "MSE_mean", "MSE_std", "MAE_mean", "R2_mean", "R2_std"]].to_string(index=False))
    print("\n== SEVERITY (tiers from train terciles: %.3f / %.3f) ==\n" % (lo, hi), s_.to_string(index=False))
    return o, n_, s_


if __name__ == "__main__":
    main()
