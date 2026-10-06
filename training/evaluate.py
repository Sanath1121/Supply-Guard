"""Evaluate every trained model + baselines on the untouched TEST partition.
Run from project root:  python -m training.evaluate

Outputs:
  outputs/results/overall_metrics.csv       model comparison on derived TRI (mean +/- std over seeds)
  outputs/results/node_metrics.csv          per-echelon metrics, raw units, relative skill scores, and delta R2
  outputs/results/severity_metrics.csv      3-tier severity accuracy / macro-F1 (thresholds from TRAIN quantiles)
  outputs/results/confusion_matrix.csv      detailed 3x3 severity tier confusion matrix
  outputs/figures/prediction_vs_truth.png   200-step time series forecast vs ground truth per echelon
  outputs/figures/error_by_node.png         per-echelon MSE comparison bar chart with skill scores
  outputs/figures/severity_confusion_matrix.png 3-tier severity confusion matrix heatmap
"""
import os
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, accuracy_score, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def plot_prediction_vs_truth(y_true: np.ndarray, preds_dict: dict, node_names: list, out_path: str, slice_len: int = 200):
    """Plot multi-echelon time-series slice comparing ground truth against model forecasts."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    slice_idx = slice(100, 100 + slice_len)
    time_steps = np.arange(slice_len)

    colors = {
        "Ground Truth": "#1f77b4",
        "Persistence": "#7f7f7f",
        "Ridge-AR(10)": "#ff7f0e",
        "LSTM": "#2ca02c",
        "ST-GCN-LSTM (Directed)": "#d62728",
    }

    # Aggregate predictions across seeds for models
    mean_preds = {
        "Persistence": preds_dict[("persistence", "deterministic")],
        "Ridge-AR(10)": preds_dict[("ar10_ridge", "deterministic")],
    }
    for m_key, label in [("lstm", "LSTM"), ("st_gcn_lstm_dir", "ST-GCN-LSTM (Directed)")]:
        m_seed_preds = [preds_dict[(m_key, s)] for s in [42, 43, 44, 45, 46] if (m_key, s) in preds_dict]
        if m_seed_preds:
            mean_preds[label] = np.mean(m_seed_preds, axis=0)

    for i, (ax, node) in enumerate(zip(axes, node_names)):
        ax.plot(time_steps, y_true[slice_idx, i], label="Ground Truth", color=colors["Ground Truth"], linewidth=2.0)
        ax.plot(time_steps, mean_preds["Persistence"][slice_idx, i], label="Persistence", color=colors["Persistence"], linestyle="--", alpha=0.7)
        if "Ridge-AR(10)" in mean_preds:
            ax.plot(time_steps, mean_preds["Ridge-AR(10)"][slice_idx, i], label="Ridge-AR(10)", color=colors["Ridge-AR(10)"], linestyle=":", alpha=0.8)
        if "LSTM" in mean_preds:
            ax.plot(time_steps, mean_preds["LSTM"][slice_idx, i], label="LSTM (5-seed mean)", color=colors["LSTM"], alpha=0.85)
        if "ST-GCN-LSTM (Directed)" in mean_preds:
            ax.plot(time_steps, mean_preds["ST-GCN-LSTM (Directed)"][slice_idx, i], label="ST-GCN-LSTM Dir (5-seed mean)", color=colors["ST-GCN-LSTM (Directed)"], linewidth=1.8)

        ax.set_ylabel(f"{node} Risk Index", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        if i == 0:
            ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
            ax.set_title(f"SupplyGuard Test Slice Forecast vs. Truth (H=10 min, Steps t+5)", fontsize=13, fontweight="bold")

    axes[-1].set_xlabel("Test Window Step Index (Nominal 2-min Cadence)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[FIGURE] Saved prediction vs truth plot: {out_path}")


def plot_error_by_node(node_df: pd.DataFrame, out_path: str):
    """Plot bar chart of test MSE by node across models with skill score annotations."""
    agg_df = node_df.groupby(["model", "node"])[["MSE", "skill_score"]].mean().reset_index()
    models = ["persistence", "ar10_ridge", "lstm", "st_gcn_lstm_sym", "st_gcn_lstm_dir"]
    model_labels = ["Persistence", "Ridge-AR(10)", "LSTM", "ST-GCN Sym", "ST-GCN Dir"]
    nodes = ["Supplier", "Manufacturer", "Distributor", "Retailer"]

    x = np.arange(len(nodes))
    width = 0.16
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#7f7f7f", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728"]

    for idx, (m, lbl, col) in enumerate(zip(models, model_labels, colors)):
        m_data = agg_df[agg_df["model"] == m].set_index("node").reindex(nodes)
        mses = m_data["MSE"].values
        rects = ax.bar(x + idx * width - (len(models) / 2) * width + width / 2, mses, width, label=lbl, color=col, alpha=0.9)

        if m == "st_gcn_lstm_dir":
            skills = m_data["skill_score"].values
            for rect, skill in zip(rects, skills):
                height = rect.get_height()
                ax.annotate(f"{skill*100:.1f}%",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8, fontweight="bold", color="#d62728")

    ax.set_ylabel("Test MSE (Lower is Better)", fontsize=11, fontweight="bold")
    ax.set_title("Test Prediction MSE & Relative Skill Score vs. Persistence Across Echelons", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(nodes, fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[FIGURE] Saved per-node MSE bar chart: {out_path}")


def plot_severity_confusion_matrix(cm: np.ndarray, out_path: str):
    """Plot 3-tier severity classification confusion matrix heatmap."""
    tiers = ["Low", "Medium", "High"]
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap="Blues", alpha=0.85)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = val / cm.sum() * 100
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, f"{val:,}\n({pct:.1f}%)", ha="center", va="center", color=color, fontweight="bold")

    fig.colorbar(cax)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(tiers, fontsize=10, fontweight="bold")
    ax.set_yticklabels(tiers, fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted Severity Tier", fontsize=11, fontweight="bold")
    ax.set_ylabel("Actual Severity Tier", fontsize=11, fontweight="bold")
    ax.set_title("ST-GCN-LSTM (Directed) Severity Tier Matrix (Test Set)", fontsize=12, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[FIGURE] Saved severity confusion matrix: {out_path}")


def evaluate_headline_claim(overall_agg: pd.DataFrame, node_agg: pd.DataFrame) -> dict:
    """Evaluate pre-registered Claims Table (§4) against empirical test set metrics."""
    derived_df = overall_agg[overall_agg["target_type"] == "derived_4node_mean"].set_index("model")

    mse_pers = float(derived_df.loc["persistence", "MSE_mean"])
    mse_lstm = float(derived_df.loc["lstm", "MSE_mean"])
    std_lstm = float(derived_df.loc["lstm", "MSE_std"])
    mse_dir = float(derived_df.loc["st_gcn_lstm_dir", "MSE_mean"])
    std_dir = float(derived_df.loc["st_gcn_lstm_dir", "MSE_std"])
    mse_sym = float(derived_df.loc["st_gcn_lstm_sym", "MSE_mean"])

    diff_lstm_dir = mse_lstm - mse_dir
    threshold = std_lstm + std_dir

    # Node level check for Supplier
    sup_df = node_agg[node_agg["node"] == "Supplier"].set_index("model")
    sup_mse_lstm = float(sup_df.loc["lstm", "MSE_mean"])
    sup_mse_dir = float(sup_df.loc["st_gcn_lstm_dir", "MSE_mean"])
    sup_std_lstm = float(sup_df.loc["lstm", "MSE_std"])
    sup_std_dir = float(sup_df.loc["st_gcn_lstm_dir", "MSE_std"])

    beats_beyond_1std = (diff_lstm_dir > threshold)
    beats_supplier = (sup_mse_lstm - sup_mse_dir > sup_std_lstm + sup_std_dir)
    directed_beats_sym = (mse_dir < mse_sym)

    if beats_beyond_1std and beats_supplier:
        headline_claim = "Graph structure improves echelon-level forecasts on this dataset"
    elif mse_dir < mse_pers:
        headline_claim = "Temporal modelling helps; the assumed graph adds no measurable accuracy but enables per-node attribution"
    else:
        headline_claim = "Dataset is dominated by short-term persistence"

    return {
        "headline_claim": headline_claim,
        "mse_persistence": mse_pers,
        "mse_lstm": mse_lstm,
        "std_lstm": std_lstm,
        "mse_st_gcn_dir": mse_dir,
        "std_st_gcn_dir": std_dir,
        "diff_lstm_dir": diff_lstm_dir,
        "threshold_1std": threshold,
        "beats_beyond_1std": beats_beyond_1std,
        "directed_beats_sym": directed_beats_sym,
    }


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
    pers_tri_mse = mean_squared_error(tri_te, pers_pred.mean(axis=1))

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
    best_cm = None

    for (name, s), p in preds.items():
        is_node = (p.ndim == 2 and p.shape[1] == 4)

        if is_node:
            tri_p = p.mean(axis=1)
            raw_p = p * rng + dmin
            raw_tri_p = raw_p.mean(axis=1)
            r = reg(tri_te, tri_p)
            r_raw = reg(raw_tri_te, raw_tri_p)

            # Overall MSE improvement vs persistence
            pct_improvement_tri = ((pers_tri_mse - r["MSE"]) / pers_tri_mse) * 100.0

            overall_rows.append({
                "model": name,
                "seed": s,
                "target_type": "derived_4node_mean",
                **r,
                "MSE_raw": r_raw["MSE"],
                "MAE_raw": r_raw["MAE"],
                "pct_improvement_pers": pct_improvement_tri,
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
                pct_improvement_node = skill * 100.0

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
                    "pct_improvement_pers": pct_improvement_node,
                    "R2_delta": r2_delta,
                })

            # Severity tier evaluation
            st_list, sp_list = [], []
            for i in range(4):
                st_list.append(severity(y_te[:, i], lo[i], hi[i]))
                sp_list.append(severity(p[:, i], lo[i], hi[i]))
            st = np.concatenate(st_list)
            sp = np.concatenate(sp_list)
            cm = confusion_matrix(st, sp, labels=[0, 1, 2])
            if name == "st_gcn_lstm_dir" and (s == 42 or best_cm is None):
                best_cm = cm

            sev_rows.append({
                "model": name,
                "seed": s,
                "accuracy": float(accuracy_score(st, sp)),
                "macro_F1": float(f1_score(st, sp, average="macro")),
            })
        else:
            # Paper hybrid overall directly predicts scalar TRI (M4)
            r = reg(tri_te, p)
            pct_improvement_tri = ((pers_tri_mse - r["MSE"]) / pers_tri_mse) * 100.0
            overall_rows.append({
                "model": name,
                "seed": s,
                "target_type": "direct_scalar_tri",
                **r,
                "MSE_raw": np.nan,
                "MAE_raw": np.nan,
                "pct_improvement_pers": pct_improvement_tri,
            })

    os.makedirs("outputs/results", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    overall_df = pd.DataFrame(overall_rows)
    node_df = pd.DataFrame(node_rows)
    sev_df = pd.DataFrame(sev_rows)

    overall_agg = agg(overall_df, ["model", "target_type"])
    node_agg = agg(node_df, ["model", "node"])
    sev_agg = agg(sev_df, ["model"])

    overall_agg.to_csv("outputs/results/overall_metrics.csv", index=False)
    node_agg.to_csv("outputs/results/node_metrics.csv", index=False)
    sev_agg.to_csv("outputs/results/severity_metrics.csv", index=False)

    if best_cm is not None:
        pd.DataFrame(best_cm, index=["Actual_Low", "Actual_Med", "Actual_High"],
                     columns=["Pred_Low", "Pred_Med", "Pred_High"]).to_csv("outputs/results/confusion_matrix.csv")

    # Generate Figures
    plot_prediction_vs_truth(y_te, preds, cfg.NODE_NAMES, "outputs/figures/prediction_vs_truth.png")
    plot_error_by_node(node_df, "outputs/figures/error_by_node.png")
    if best_cm is not None:
        plot_severity_confusion_matrix(best_cm, "outputs/figures/severity_confusion_matrix.png")

    # Evaluate Claims Table (§4)
    claims_info = evaluate_headline_claim(overall_agg, node_agg)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 80)
    print("== OVERALL METRICS (DERIVED TRI & SCALAR TRI) ==")
    print("=" * 80)
    print(overall_agg[["model", "target_type", "MSE_mean", "MSE_std", "R2_mean", "R2_std", "pct_improvement_pers_mean", "pct_improvement_pers_std"]].to_string(index=False))

    print("\n" + "=" * 80)
    print("== SEVERITY CLASSIFICATION METRICS (3-TIER) ==")
    print("=" * 80)
    print(sev_agg[["model", "accuracy_mean", "accuracy_std", "macro_F1_mean", "macro_F1_std"]].to_string(index=False))

    print("\n" + "=" * 80)
    print("== CLAIMS TABLE (§4) EVALUATION VERDICT ==")
    print("=" * 80)
    print(f"Persistence Test MSE : {claims_info['mse_persistence']:.6f}")
    print(f"LSTM Test MSE        : {claims_info['mse_lstm']:.6f} +/- {claims_info['std_lstm']:.6f}")
    print(f"ST-GCN Dir Test MSE  : {claims_info['mse_st_gcn_dir']:.6f} +/- {claims_info['std_st_gcn_dir']:.6f}")
    print(f"Difference (LSTM - Dir) : {claims_info['diff_lstm_dir']:.6f} (Threshold 1-std: {claims_info['threshold_1std']:.6f})")
    print(f"ST-GCN beats LSTM beyond 1 std : {claims_info['beats_beyond_1std']}")
    print(f"ST-GCN Dir beats ST-GCN Sym   : {claims_info['directed_beats_sym']}")
    print(f"PRE-REGISTERED HEADLINE CLAIM : \"{claims_info['headline_claim']}\"")
    print("=" * 80)

    return overall_agg, node_agg, sev_agg, claims_info


if __name__ == "__main__":
    main()
