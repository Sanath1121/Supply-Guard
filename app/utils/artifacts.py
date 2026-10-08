"""Artifact Discovery, Loading, and Caching Layer.

Provides honest discovery and caching of models, scalers, test-partition windows,
and evaluation tiers. Reports honest statuses and human-readable warnings instead
of fabricating fallbacks.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import os
import json
import numpy as np
import torch
import joblib
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

from src.config import Config
from src.models.st_gcn_lstm import build_model


@dataclass(frozen=True)
class ArtifactStatus:
    model_loaded: bool            # True only if a real checkpoint was loaded with strict=True
    checkpoint_path: Optional[str]
    model_name: str
    graph_mode: str
    seed: int
    scaler_fitted: bool           # True only if scaler.joblib loaded
    tiers_source: str             # "train_terciles" | "provisional_default"
    data_source: str              # "test_partition" | "sandbox" | "unavailable"
    warnings: Tuple[str, ...]     # human-readable, shown in Status Strip


@st.cache_resource(show_spinner=False)
def get_scaler() -> Tuple[Optional[MinMaxScaler], bool]:
    """Load fitted scaler from outputs/models/scaler.joblib.
    
    Returns (scaler, True) if found, else (None, False).
    Never fabricates invented ranges.
    """
    path = Config.SCALER_PATH
    if os.path.exists(path):
        try:
            data = joblib.load(path)
            if isinstance(data, dict) and "scaler" in data:
                return data["scaler"], True
            elif isinstance(data, MinMaxScaler):
                return data, True
        except Exception:
            return None, False
    return None, False


def get_checkpoint_filename(model_name: str, graph_mode: str, seed: int) -> str:
    """Return the exact expected checkpoint filename matching training/train.py."""
    if model_name == "st_gcn_lstm":
        # Support both canonical short forms (dir, sym) and descriptive names (directed, symmetric)
        short_mode = "dir" if "dir" in graph_mode else ("sym" if "sym" in graph_mode else graph_mode)
        short_file = f"st_gcn_lstm_{short_mode}_seed{seed}.pt"
        if os.path.exists(os.path.join(Config.CKPT_DIR, short_file)):
            return short_file
        tag = f"st_gcn_lstm_{graph_mode}"
    else:
        tag = model_name
    return f"{tag}_seed{seed}.pt"


@st.cache_resource(show_spinner=False)
def get_model(model_name: str, graph_mode: str, seed: int) -> Tuple[torch.nn.Module, bool, Optional[str], Optional[str]]:
    """Load a model with strict checkpoint weights or return an eval-mode untrained model.
    
    Returns: (model, is_loaded, checkpoint_path, warning_message)
    """
    cfg = Config()
    cfg.GRAPH_MODE = graph_mode
    model = build_model(model_name, cfg)
    model.eval()

    expected_file = get_checkpoint_filename(model_name, graph_mode, seed)
    ckpt_path = os.path.join(Config.CKPT_DIR, expected_file)

    if not os.path.exists(ckpt_path):
        warning = f"No trained checkpoint found at '{ckpt_path}' – predictions come from UNTRAINED weights."
        return model, False, None, warning

    try:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        model.load_state_dict(sd, strict=True)
        return model, True, ckpt_path, None
    except Exception as e:
        warning = f"Failed to load checkpoint '{ckpt_path}' with strict=True: {e}. Model is UNTRAINED."
        return model, False, ckpt_path, warning


@st.cache_data(show_spinner=False)
def get_tiers() -> Dict[str, Any]:
    """Retrieve tercile thresholds from outputs/models/tiers.json or train set.
    
    Returns dict with keys 'p33', 'p66', and 'source'.
    """
    tiers_path = os.path.join(Config.CKPT_DIR, "tiers.json")
    if os.path.exists(tiers_path):
        try:
            with open(tiers_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "p33" in data and "p66" in data:
                    return {"p33": float(data["p33"]), "p66": float(data["p66"]), "source": "train_terciles"}
        except Exception:
            pass

    # Check results directory
    res_tiers = os.path.join("outputs", "results", "tiers.json")
    if os.path.exists(res_tiers):
        try:
            with open(res_tiers, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "p33" in data and "p66" in data:
                    return {"p33": float(data["p33"]), "p66": float(data["p66"]), "source": "train_terciles"}
        except Exception:
            pass

    # Same definition as training/evaluate.py: per-node terciles of the TRAIN targets
    # (p33/p66 for the mean risk index are the terciles of the train node-mean).
    if os.path.exists(Config.RAW_DATA_PATH):
        try:
            from src.dataset import build_datasets
            tr, _, _, _, _ = build_datasets(Config(), save_scaler=False)
            node_targets = tr.node_targets.numpy()
            lo, hi = np.quantile(node_targets, [1 / 3, 2 / 3], axis=0)
            tri_lo, tri_hi = np.quantile(node_targets.mean(axis=1), [1 / 3, 2 / 3])
            return {
                "p33": float(tri_lo), "p66": float(tri_hi),
                "node_p33": [float(v) for v in lo], "node_p66": [float(v) for v in hi],
                "source": "train_terciles",
            }
        except Exception:
            pass

    return {"p33": 0.35, "p66": 0.65, "source": "provisional_default"}


@st.cache_data(show_spinner=False)
def get_test_windows(max_windows: Optional[int] = 500) -> Tuple[List[Dict[str, Any]], str, Tuple[str, ...]]:
    """Extract sliding windows strictly from the untouched test partition.
    
    Returns: (windows, data_source, warnings)
    Where each window dict has: {"window_id", "timestamp", "sequence" [10, 5], "ground_truth" [4]}
    """
    if not os.path.exists(Config.RAW_DATA_PATH):
        warning = f"Raw dataset missing at '{Config.RAW_DATA_PATH}'. Run 'python setup_and_download.py' to acquire."
        return [], "unavailable", (warning,)

    try:
        from src.dataset import build_datasets
        cfg = Config()
        # Build leak-free datasets; te is test partition SupplyChainDataset
        _, _, te, scaler, _ = build_datasets(cfg, save_scaler=False)
        total_te = len(te)
        if total_te == 0:
            return [], "unavailable", ("Test partition has 0 valid segment windows.",)

        # Evenly subsample if more than max_windows
        if max_windows and total_te > max_windows:
            indices = np.linspace(0, total_te - 1, max_windows, dtype=int)
        else:
            indices = np.arange(total_te)

        windows = []
        for i, idx in enumerate(indices):
            seq = te.sequences[idx].numpy() # [10, 5]
            tgt = te.node_targets[idx].numpy() # [4]
            # Retrieve timestamp from sequence or target
            ts_str = f"Window #{idx + 1}"
            if te.timestamps is not None and idx < len(te.timestamps):
                ts_str = str(te.timestamps[idx])
            elif te.seq_timestamps is not None and idx < len(te.seq_timestamps):
                ts_str = str(te.seq_timestamps[idx][-1])

            windows.append({
                "window_id": int(idx),
                "timestamp": ts_str,
                "sequence": seq,
                "ground_truth": tgt
            })

        return windows, "test_partition", ()
    except Exception as e:
        return [], "unavailable", (f"Error extracting test partition windows: {e}",)


def predict_window(model: torch.nn.Module, seq: np.ndarray) -> np.ndarray:
    """Run forward prediction on a single [10, 5] sequence.
    
    Returns 1D array: [4] for node models, or [1] for scalar models.
    """
    model.eval()
    with torch.no_grad():
        inp = torch.tensor(seq, dtype=torch.float32).unsqueeze(0) # [1, 10, 5]
        out = model(inp).squeeze(0).cpu().numpy()
        return np.atleast_1d(out)
