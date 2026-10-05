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


def main(cfg=None, models=("lstm", "paper_overall", "st_gcn_lstm_sym", "st_gcn_lstm_dir"), seeds=None):
    cfg = cfg or Config(); seeds = seeds or cfg.SEEDS
    tr, va, te, scaler, info = build_datasets(cfg, save_scaler=False)
    y = te.node_targets.numpy(); tri = y.mean(1)
    rng = scaler.data_range_[:4]                          # raw = scaled * range + min
    dmin = scaler.data_min_[:4]
    
    # Exact raw TRI for targets
    raw_y = y * rng + dmin
    raw_tri = raw_y.mean(1)

    # Terciles per node (from train distribution)
    lo, hi = np.quantile(tr.node_targets.numpy(), [1 / 3, 2 / 3], axis=0)

    preds = {}                                            # name -> list of [N,4] (or [N] for paper) arrays
    preds["persistence"] = [te.sequences[:, -1, :4].numpy()]
    Xtr, Xte = tr.sequences.numpy().reshape(len(tr), -1), te.sequences.numpy().reshape(len(te), -1)
    preds["ar10_ridge"] = [Ridge(alpha=1e-3).fit(Xtr, tr.node_targets.numpy()).predict(Xte)]
    for name in models:
        preds[name] = []
        for s in seeds:
            p = ckpt_path(cfg, name, s)
            if not os.path.exists(p):
                print("missing", p); continue
            m = build_model(name, cfg); m.load_state_dict(torch.load(p, map_location="cpu")); preds[name].append(predict(m, te))

    overall, node, sev = [], [], []
    for name, plist in preds.items():
        for s, p in enumerate(plist):
            is_node = p.ndim == 2
            tri_p = p.mean(1) if is_node else p
            r = reg(tri, tri_p)
            
            # Exact raw TRI prediction
            if is_node:
                raw_p = p * rng + dmin
                raw_tri_p = raw_p.mean(1)
            else:
                # For paper_overall, TRI is directly predicted in scaled space
                # We approximate back by using mean range and min, though paper model doesn't output per-node
                raw_tri_p = tri_p * rng.mean() + dmin.mean()
                
            r_raw = reg(raw_tri, raw_tri_p)
            overall.append({"model": name, "seed": s, **r, "MSE_raw": r_raw["MSE"]})
            if is_node:
                for i, nn_ in enumerate(cfg.NODE_NAMES):
                    rr = reg(y[:, i], p[:, i])
                    node.append({"model": name, "node": nn_, "seed": s, **rr, "MSE_raw": rr["MSE"] * rng[i] ** 2,
                                 "RMSE_raw": rr["RMSE"] * rng[i]})
                
                # Severity metrics per node
                st_list, sp_list = [], []
                for i in range(4):
                    st_list.append(severity(y[:, i], lo[i], hi[i]))
                    sp_list.append(severity(p[:, i], lo[i], hi[i]))
                st = np.concatenate(st_list)
                sp = np.concatenate(sp_list)
                sev.append({"model": name, "seed": s, "accuracy": accuracy_score(st, sp), "macro_F1": f1_score(st, sp, average="macro")})

    os.makedirs("outputs/results", exist_ok=True)
    o, n_, s_ = agg(pd.DataFrame(overall), ["model"]), agg(pd.DataFrame(node), ["model", "node"]), agg(pd.DataFrame(sev), ["model"])
    o.to_csv("outputs/results/overall_metrics.csv", index=False)
    n_.to_csv("outputs/results/node_metrics.csv", index=False)
    s_.to_csv("outputs/results/severity_metrics.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n== OVERALL (derived TRI) ==\n", o[["model", "MSE_mean", "MSE_std", "MAE_mean", "R2_mean", "R2_std"]].to_string(index=False))
    print("\n== SEVERITY (tiers per node) ==\n", s_.to_string(index=False))
    return o, n_, s_


if __name__ == "__main__":
    main()
