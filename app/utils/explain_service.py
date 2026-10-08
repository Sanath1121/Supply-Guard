"""Explainability Service & Cached Inference.

Provides lazy, cached Integrated Gradients attributions and sensitivity testing.
Uses real model passes and RiskExplainer; never hardcodes attribution numbers
or fake counterfactual adjustments.
"""
from typing import Dict, Any, Optional, Tuple, List
import hashlib
import torch
import numpy as np
import streamlit as st

from src.explainability import RiskExplainer, FEATURE_NAMES, NODE_NAMES, narrate
from src.config import Config


@st.cache_resource(show_spinner=False)
def get_train_mean_baseline() -> Optional[Tuple[float, ...]]:
    """Training-partition mean feature vector (scaled), the IG baseline used in the paper
    and scripts/generate_attributions.py. None if the raw dataset is unavailable."""
    import os
    if not os.path.exists(Config.RAW_DATA_PATH):
        return None
    try:
        from src.dataset import build_datasets
        _, _, _, _, info = build_datasets(Config(), save_scaler=False)
        return tuple(float(v) for v in info["train_mean"])
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_explainer(model_key: str, _model: torch.nn.Module,
                  baseline_arr: Optional[Tuple[float, ...]] = None) -> RiskExplainer:
    """Build and cache a RiskExplainer instance for the given model.

    Cached by model_key and baseline (`_model` itself is not hashed by Streamlit, so
    model_key must identify it). Uses the given baseline, otherwise zeros.
    """
    if baseline_arr is not None:
        baseline = torch.tensor(baseline_arr, dtype=torch.float32)
    else:
        # Fallback baseline: zero feature vector
        baseline = torch.zeros(Config.NUM_INPUT_FEATURES, dtype=torch.float32)
    return RiskExplainer(_model, baseline, steps=64)


def _hash_seq(seq: np.ndarray) -> str:
    """Fast hash of sequence array for caching."""
    return hashlib.md5(seq.tobytes()).hexdigest()[:12]


@st.cache_data(show_spinner=False)
def compute_explanation(
    model_key: str,
    seq: np.ndarray,
    target_node: int,
    residual_delta: bool = False,
    baseline_tuple: Optional[Tuple[float, ...]] = None
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Compute attribution for a single target node.

    Cached by model key, sequence hash, target node, residual mode and baseline
    (model_key must NOT be underscore-prefixed, or Streamlit drops it from the cache key).
    baseline_tuple=None uses the training-mean baseline (zeros if the dataset is missing).
    Returns: (result_dict, error_message)
    """
    try:
        from app.utils.artifacts import get_model
        # Parse model_key: e.g. "st_gcn_lstm:directed:42"
        parts = model_key.split(":")
        m_name = parts[0]
        g_mode = parts[1] if len(parts) > 1 else "directed"
        seed = int(parts[2]) if len(parts) > 2 else 42

        model, _, _, _ = get_model(m_name, g_mode, seed)
        if baseline_tuple is None:
            baseline_tuple = get_train_mean_baseline()
        explainer = get_explainer(model_key, model, baseline_tuple)
        
        seq_tensor = torch.tensor(seq, dtype=torch.float32)
        res = explainer.explain(seq_tensor, target_node, residual_delta=residual_delta)
        return res, None
    except Exception as e:
        return None, str(e)


@st.cache_data(show_spinner=False)
def compute_all_edge_shares(
    model_key: str,
    seq: np.ndarray,
    baseline_tuple: Optional[Tuple[float, ...]] = None
) -> Tuple[Optional[Dict[Tuple[str, str], float]], Optional[str]]:
    """Compute real edge shares from per-target explanations.
    
    Edge (Supplier -> Manufacturer): target=Manufacturer (1), feature 0 (Supplier RI)
    Edge (Manufacturer -> Distributor): target=Distributor (2), feature 1 (Manufacturer RI)
    Edge (Distributor -> Retailer): target=Retailer (3), feature 2 (Distributor RI)
    
    Returns: (shares_dict, error_message)
    """
    shares = {}
    edges_to_query = [
        (("Supplier", "Manufacturer"), 1, 0),
        (("Manufacturer", "Distributor"), 2, 1),
        (("Distributor", "Retailer"), 3, 2),
    ]
    
    for (u, v), tgt_idx, feat_idx in edges_to_query:
        res, err = compute_explanation(model_key, seq, tgt_idx, residual_delta=False, baseline_tuple=baseline_tuple)
        if err or res is None:
            return None, err or f"Failed to explain target {v}"
        
        # feature_importance[feat_idx] is the share of attribution on target v from echelon u
        fi = res["feature_importance"]
        share = float(fi[feat_idx]) if feat_idx < len(fi) else 0.0
        shares[(u, v)] = share

    return shares, None


def run_deletion_test(
    model: torch.nn.Module,
    seq: np.ndarray,
    target_node: int,
    neutralize_feature_idx: int,
    baseline_val: float = 0.0
) -> Dict[str, Any]:
    """Perform real upstream deletion / sensitivity test.
    
    Replaces the neutralized feature column across the entire window with baseline_val,
    re-runs model inference, and reports real before -> after risk.
    Never uses invented multipliers.
    """
    seq_orig = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
    seq_mod = seq.copy()
    seq_mod[:, neutralize_feature_idx] = baseline_val
    seq_mod_tensor = torch.tensor(seq_mod, dtype=torch.float32).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        out_orig = model(seq_orig).squeeze(0)
        out_mod = model(seq_mod_tensor).squeeze(0)

        if out_orig.dim() > 0 and out_orig.shape[-1] > target_node:
            orig_val = float(out_orig[target_node])
            mod_val = float(out_mod[target_node])
        else:
            orig_val = float(out_orig.squeeze())
            mod_val = float(out_mod.squeeze())

    return {
        "original_risk": orig_val,
        "modified_risk": mod_val,
        "delta": mod_val - orig_val,
        "feature_name": FEATURE_NAMES[neutralize_feature_idx] if neutralize_feature_idx < len(FEATURE_NAMES) else f"Feature {neutralize_feature_idx}",
        "target_node": NODE_NAMES[target_node] if target_node < len(NODE_NAMES) else f"Node {target_node}",
    }
