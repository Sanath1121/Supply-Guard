"""Results Loader & RQ Verdict Evaluator.

Loads empirical evaluation results strictly from outputs/results/ CSV files.
If files do not exist, returns None or Pending states with reproduction instructions.
Never fabricates hardcoded numbers.
"""
from typing import Dict, Any, Optional, Tuple
import os
import pandas as pd
import numpy as np


RESULTS_DIR = os.path.join("outputs", "results")


def load_overall_metrics() -> Optional[pd.DataFrame]:
    """Load outputs/results/overall_metrics.csv or return None."""
    path = os.path.join(RESULTS_DIR, "overall_metrics.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        required = ["model", "MSE_mean", "MSE_std", "MAE_mean", "R2_mean"]
        if all(col in df.columns for col in required):
            return df
    except Exception:
        pass
    return None


def load_severity_metrics() -> Optional[pd.DataFrame]:
    """Load outputs/results/severity_metrics.csv or return None."""
    path = os.path.join(RESULTS_DIR, "severity_metrics.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        required = ["model", "accuracy_mean", "macro_F1_mean"]
        if all(col in df.columns for col in required):
            return df
    except Exception:
        pass
    return None


def load_node_metrics() -> Optional[pd.DataFrame]:
    """Load outputs/results/node_metrics.csv or return None."""
    path = os.path.join(RESULTS_DIR, "node_metrics.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if "model" in df.columns and "node" in df.columns:
            return df
    except Exception:
        pass
    return None


def evaluate_rq_verdicts(overall_df: Optional[pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    """Compute empirical verdicts for RQ1-RQ4 strictly from results dataframe.
    
    Allowed states: 'Supported', 'Not supported', 'Inconclusive', 'Pending – run Phase 5'.
    Defaults strictly to 'Pending', never 'Supported'.
    """
    verdicts = {
        "RQ1": {
            "title": "RQ1: Beating Persistence",
            "hypothesis": "ST-GCN-LSTM MSE < Persistence MSE by > 1 std dev.",
            "status": "Pending – run Phase 5 evaluation",
            "state": "pending",
            "details": "Evaluation results not yet generated in outputs/results/overall_metrics.csv."
        },
        "RQ2": {
            "title": "RQ2: Graph vs. Graph-Free LSTM",
            "hypothesis": "ST-GCN-LSTM MSE < Standalone LSTM MSE by > 1 std dev.",
            "status": "Pending – run Phase 5 evaluation",
            "state": "pending",
            "details": "Evaluation results not yet generated in outputs/results/overall_metrics.csv."
        },
        "RQ3": {
            "title": "RQ3: Directed vs. Symmetric Graph",
            "hypothesis": "ST-GCN-LSTM (Directed) MSE < ST-GCN-LSTM (Symmetric) MSE by > 1 std dev.",
            "status": "Pending – run Phase 5 evaluation",
            "state": "pending",
            "details": "Evaluation results not yet generated in outputs/results/overall_metrics.csv."
        },
        "RQ4": {
            "title": "RQ4: Attribution Validity (Deletion Test)",
            "hypothesis": "Removing top IG-attributed features causes greater error degradation than random removal.",
            "status": "Pending – run Phase 6 validation",
            "state": "pending",
            "details": "Phase 6 deletion test results file not yet generated."
        },
    }

    if overall_df is None or overall_df.empty:
        return verdicts

    # Index by model name
    try:
        df_models = overall_df.set_index("model")
    except Exception:
        return verdicts

    # Helper to get MSE and std
    def get_mse_stats(name):
        if name in df_models.index:
            row = df_models.loc[name]
            mse = float(row.get("MSE_mean", float("nan")))
            std = float(row.get("MSE_std", 0.0))
            return mse, std
        return None, None

    # Evaluate RQ1: ST-GCN-LSTM vs persistence
    mse_pers, std_pers = get_mse_stats("persistence")
    mse_prop_dir, std_prop_dir = get_mse_stats("st_gcn_lstm_directed")
    mse_prop = mse_prop_dir if mse_prop_dir is not None else get_mse_stats("st_gcn_lstm")[0]
    std_prop = std_prop_dir if std_prop_dir is not None else get_mse_stats("st_gcn_lstm")[1]

    if mse_pers is not None and mse_prop is not None:
        diff = mse_pers - mse_prop
        threshold = std_prop if (std_prop is not None and not np.isnan(std_prop)) else 0.0
        pct_imp = (diff / mse_pers) * 100.0 if mse_pers > 0 else 0.0
        if diff > threshold:
            verdicts["RQ1"]["status"] = "Supported"
            verdicts["RQ1"]["state"] = "supported"
            verdicts["RQ1"]["details"] = f"ST-GCN-LSTM MSE ({mse_prop:.4f}±{threshold:.4f}) beats Persistence ({mse_pers:.4f}) by {pct_imp:.1f}% (> 1 std)."
        elif diff > 0:
            verdicts["RQ1"]["status"] = "Inconclusive"
            verdicts["RQ1"]["state"] = "inconclusive"
            verdicts["RQ1"]["details"] = f"Improvement ({pct_imp:.1f}%) is within 1 std dev ({threshold:.4f})."
        else:
            verdicts["RQ1"]["status"] = "Not supported"
            verdicts["RQ1"]["state"] = "not_supported"
            verdicts["RQ1"]["details"] = f"Persistence MSE ({mse_pers:.4f}) is lower than or equal to ST-GCN-LSTM ({mse_prop:.4f})."

    # Evaluate RQ2: ST-GCN-LSTM vs LSTM
    mse_lstm, std_lstm = get_mse_stats("lstm")
    if mse_lstm is not None and mse_prop is not None:
        diff_lstm = mse_lstm - mse_prop
        thresh_lstm = std_prop if (std_prop is not None and not np.isnan(std_prop)) else 0.0
        if diff_lstm > thresh_lstm:
            verdicts["RQ2"]["status"] = "Supported"
            verdicts["RQ2"]["state"] = "supported"
            verdicts["RQ2"]["details"] = f"Graph model ({mse_prop:.4f}) beats Graph-free LSTM ({mse_lstm:.4f}) beyond 1 std dev."
        elif diff_lstm > 0:
            verdicts["RQ2"]["status"] = "Inconclusive"
            verdicts["RQ2"]["state"] = "inconclusive"
            verdicts["RQ2"]["details"] = f"Difference ({diff_lstm:.4f}) is within 1 std dev ({thresh_lstm:.4f})."
        else:
            verdicts["RQ2"]["status"] = "Not supported"
            verdicts["RQ2"]["state"] = "not_supported"
            verdicts["RQ2"]["details"] = f"Graph layers did not improve accuracy over standalone LSTM."

    # Evaluate RQ3: Directed vs Symmetric
    mse_sym, std_sym = get_mse_stats("st_gcn_lstm_symmetric")
    if mse_prop_dir is not None and mse_sym is not None:
        diff_dir = mse_sym - mse_prop_dir
        thresh_dir = std_prop_dir if (std_prop_dir is not None and not np.isnan(std_prop_dir)) else 0.0
        if diff_dir > thresh_dir:
            verdicts["RQ3"]["status"] = "Supported"
            verdicts["RQ3"]["state"] = "supported"
            verdicts["RQ3"]["details"] = f"Directed mode ({mse_prop_dir:.4f}) outperforms Symmetric ({mse_sym:.4f}) beyond 1 std dev."
        elif diff_dir > 0:
            verdicts["RQ3"]["status"] = "Inconclusive"
            verdicts["RQ3"]["state"] = "inconclusive"
            verdicts["RQ3"]["details"] = f"Difference ({diff_dir:.4f}) is within 1 std dev."
        else:
            verdicts["RQ3"]["status"] = "Not supported"
            verdicts["RQ3"]["state"] = "not_supported"
            verdicts["RQ3"]["details"] = f"Directed mode did not outperform symmetric mode."

    # Evaluate RQ4: deletion test file
    del_path = os.path.join(RESULTS_DIR, "deletion_test.csv")
    if os.path.exists(del_path):
        try:
            del_df = pd.read_csv(del_path)
            if "top_feature_diff" in del_df.columns and "random_feature_diff" in del_df.columns:
                t_diff = del_df["top_feature_diff"].mean()
                r_diff = del_df["random_feature_diff"].mean()
                if t_diff > r_diff:
                    verdicts["RQ4"]["status"] = "Supported"
                    verdicts["RQ4"]["state"] = "supported"
                    verdicts["RQ4"]["details"] = f"Top-attributed deletion error ({t_diff:.4f}) > random deletion error ({r_diff:.4f})."
                else:
                    verdicts["RQ4"]["status"] = "Not supported"
                    verdicts["RQ4"]["state"] = "not_supported"
                    verdicts["RQ4"]["details"] = "Attributions failed the deletion degradation test."
        except Exception:
            pass

    return verdicts
