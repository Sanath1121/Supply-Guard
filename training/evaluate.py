"""Evaluate every trained model + baselines on the untouched TEST partition.
Run from project root:  python -m training.evaluate

Outputs (outputs/results/):
  overall_metrics.csv   model comparison on derived TRI (mean +/- std over seeds)
  node_metrics.csv      per-echelon metrics, raw units, relative skill scores, and delta R2
  severity_metrics.csv  3-tier severity accuracy / macro-F1 (thresholds from TRAIN quantiles)
"""
import os
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, accuracy_score

from src.config import Config
from src.dataset import build_datasets
from src.models.st_gcn_lstm import build_model
from training.train import MODEL_ALIASES, ckpt_path


def reg(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MSE, MAE, RMSE, and R2 regression metrics."""
    mse = float(mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return {"MSE": mse, "MAE": mae, "RMSE": rmse, "R2": r2}


def predict(model: torch.nn.Module, ds, batch_size: int = 2048, device: str = "cpu") -> np.ndarray:
    """Run batched inference to prevent OOM on large test partitions (M3)."""
    model.eval()
    dev = torch.device(device)
    model.to(dev)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for batch in loader:
            seq = batch["sequence"].to(dev)
            p = model(seq).cpu().numpy()
            preds.append(p)
    return np.concatenate(preds, axis=0)


def severity(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Assign continuous values to 3 ordinal severity tiers (0: Low, 1: Medium, 2: High)."""
    return np.digitize(x, [lo, hi])


def tune_ridge_alpha(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray) -> float:
    """Tune Ridge regularization parameter alpha on validation split (m3)."""
    alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    best_alpha, best_mse = alphas[0], float("inf")
    for a in alphas:
        r = Ridge(alpha=a).fit(X_tr, y_tr)
        val_pred = r.predict(X_va)
        val_mse = mean_squared_error(y_va, val_pred)
        if val_mse < best_mse:
            best_mse = val_mse
            best_alpha = a
    return best_alpha


def agg(df: pd.DataFrame, keys: list) -> pd.DataFrame:
    """Compute mean and std across runs/seeds for numeric columns."""
    g = df.groupby(keys)
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in keys and c != "seed"]
    mean_df = g[numeric_cols].mean().add_suffix("_mean")
    std_df = g[numeric_cols].std().add_suffix("_std")
    out = mean_df.join(std_df).reset_index()
    return out


def main(cfg=None, models=("lstm", "paper_overall", "st_gcn_lstm_sym", "st_gcn_lstm_dir"), seeds=None):
    """Main evaluation entry point evaluating models and baselines on the test partition."""
    cfg = cfg or Config()
    seeds = seeds or cfg.SEEDS

    print("[EVAL] Loading chronological partitions...")
    tr, va, te, scaler, info = build_datasets(cfg, save_scaler=False)
    if os.path.exists(cfg.SCALER_PATH):
        try:
            loaded_obj = joblib.load(cfg.SCALER_PATH)
            prod_scaler = loaded_obj["scaler"] if isinstance(loaded_obj, dict) and "scaler" in loaded_obj else loaded_obj
            if np.allclose(prod_scaler.data_max_, scaler.data_max_, atol=1e-5):
                scaler = prod_scaler
                print(f"[EVAL] Successfully loaded production scaler from {cfg.SCALER_PATH}")
        except Exception as e:
            print(f"[WARNING] Could not load saved scaler: {e}")

    y_te = te.node_targets.numpy()               # [N, 4]
    tri_te = y_te.mean(axis=1)                   # [N] derived true TRI
    last_step_te = te.sequences[:, -1, :4].numpy() # [N, 4] persistence forecast y_t
    delta_true = y_te - last_step_te             # [N, 4] true change \Delta y

    rng = scaler.data_range_[:4]                 # raw = scaled * range + min
    dmin = scaler.data_min_[:4]
    raw_y_te = y_te * rng + dmin
    raw_tri_te = raw_y_te.mean(axis=1)

    # Compute tercile boundaries per node from train split
    lo, hi = np.quantile(tr.node_targets.numpy(), [1 / 3, 2 / 3], axis=0)

    # 1. Baseline: Persistence
    pers_pred = last_step_te
    pers_mse_per_node = [mean_squared_error(y_te[:, i], pers_pred[:, i]) for i in range(4)]

    # 2. Baseline: Ridge-AR(10) with validation tuning
    X_tr = tr.sequences.numpy().reshape(len(tr), -1)
    X_va = va.sequences.numpy().reshape(len(va), -1)
    X_te = te.sequences.numpy().reshape(len(te), -1)
    best_alpha = tune_ridge_alpha(X_tr, tr.node_targets.numpy(), X_va, va.node_targets.numpy())
    print(f"[EVAL] Tuned Ridge-AR(10) optimal alpha on validation set: {best_alpha}")
    ridge_model = Ridge(alpha=best_alpha).fit(X_tr, tr.node_targets.numpy())
    ridge_pred = ridge_model.predict(X_te)

    # Store predictions mapping (model_name, seed) -> pred_array
    preds = {
        ("persistence", "deterministic"): pers_pred,
        ("ar10_ridge", "deterministic"): ridge_pred,
    }

    # Evaluate deep learning checkpoints
    for name in models:
        canonical_name = MODEL_ALIASES.get(name, name)
        for s in seeds:
            p = ckpt_path(cfg, canonical_name, s)
            if not os.path.exists(p):
                print(f"[MISSING] {p} not found; skipping.")
                continue
            m = build_model(canonical_name, cfg)
            m.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
            preds[(canonical_name, s)] = predict(m, te, batch_size=2048, device="cpu")

    overall_rows, node_rows, sev_rows = [], [], []

    for (name, s), p in preds.items():
        is_node = (p.ndim == 2 and p.shape[1] == 4)

        if is_node:
            tri_p = p.mean(axis=1)
            raw_p = p * rng + dmin
            raw_tri_p = raw_p.mean(axis=1)
            r = reg(tri_te, tri_p)
            r_raw = reg(raw_tri_te, raw_tri_p)
            overall_rows.append({
                "model": name,
                "seed": s,
                "target_type": "derived_4node_mean",
                **r,
                "MSE_raw": r_raw["MSE"],
                "MAE_raw": r_raw["MAE"],
            })

            # Per-node evaluation
            for i, node_name in enumerate(cfg.NODE_NAMES):
                rr = reg(y_te[:, i], p[:, i])
                raw_mse_i = rr["MSE"] * (rng[i] ** 2)
                raw_rmse_i = rr["RMSE"] * rng[i]
                raw_mae_i = rr["MAE"] * rng[i]

                # Relative skill score over persistence: 1 - MSE / MSE_pers (M5)
                pers_mse_i = pers_mse_per_node[i]
                skill = 1.0 - (rr["MSE"] / pers_mse_i) if pers_mse_i > 0 else 0.0

                # Change R^2 on \Delta y
                pred_delta_i = p[:, i] - last_step_te[:, i]
                true_delta_i = delta_true[:, i]
                r2_delta = float(r2_score(true_delta_i, pred_delta_i))

                node_rows.append({
                    "model": name,
                    "seed": s,
                    "node": node_name,
                    **rr,
                    "MSE_raw": raw_mse_i,
                    "RMSE_raw": raw_rmse_i,
                    "MAE_raw": raw_mae_i,
                    "skill_score": skill,
                    "R2_delta": r2_delta,
                })

            # Severity tier evaluation
            st_list, sp_list = [], []
            for i in range(4):
                st_list.append(severity(y_te[:, i], lo[i], hi[i]))
                sp_list.append(severity(p[:, i], lo[i], hi[i]))
            st = np.concatenate(st_list)
            sp = np.concatenate(sp_list)
            sev_rows.append({
                "model": name,
                "seed": s,
                "accuracy": float(accuracy_score(st, sp)),
                "macro_F1": float(f1_score(st, sp, average="macro")),
            })
        else:
            # Paper hybrid overall directly predicts scalar TRI (M4)
            # Reported strictly in scaled TRI space; no raw-unit approximation
            r = reg(tri_te, p)
            overall_rows.append({
                "model": name,
                "seed": s,
                "target_type": "direct_scalar_tri",
                **r,
                "MSE_raw": np.nan,
                "MAE_raw": np.nan,
            })

    os.makedirs("outputs/results", exist_ok=True)
    overall_df = pd.DataFrame(overall_rows)
    node_df = pd.DataFrame(node_rows)
    sev_df = pd.DataFrame(sev_rows)

    overall_agg = agg(overall_df, ["model", "target_type"])
    node_agg = agg(node_df, ["model", "node"])
    sev_agg = agg(sev_df, ["model"])

    overall_agg.to_csv("outputs/results/overall_metrics.csv", index=False)
    node_agg.to_csv("outputs/results/node_metrics.csv", index=False)
    sev_agg.to_csv("outputs/results/severity_metrics.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 80)
    print("== OVERALL METRICS (DERIVED TRI & SCALAR TRI) ==")
    print("=" * 80)
    print(overall_agg[["model", "target_type", "MSE_mean", "MSE_std", "MAE_mean", "R2_mean", "R2_std"]].to_string(index=False))

    print("\n" + "=" * 80)
    print("== SEVERITY CLASSIFICATION METRICS (3-TIER) ==")
    print("=" * 80)
    print(sev_agg[["model", "accuracy_mean", "accuracy_std", "macro_F1_mean", "macro_F1_std"]].to_string(index=False))

    return overall_agg, node_agg, sev_agg


if __name__ == "__main__":
    main()
