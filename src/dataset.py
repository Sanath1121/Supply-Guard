"""Leak-free forecasting pipeline.

Guarantees
----------
* rows are parsed, sorted by Timestamp and de-duplicated before anything else
* chronological 80:10:10 split (no shuffling of the split itself)
* MinMaxScaler is fitted on the TRAIN partition only, and saved for the dashboard
* val/test windows borrow their first L+H-1 context rows from the preceding partition
  (inputs only; every TARGET lies strictly inside its own partition)
* each sample: X = rows [i, i+L)  ->  Y = node risks at row i+L+H-1   (true forward forecast)
* the graph input is NOT stored separately: the models derive it from X[:, -1, :4],
  so there is exactly one input tensor (this makes attribution unambiguous)
"""
import numpy as np
import pandas as pd
import joblib
import torch
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


class SupplyChainDataset(Dataset):
    def __init__(self, seqs, node_tgts, timestamps=None):
        self.sequences = torch.tensor(seqs, dtype=torch.float32)          # [N, L, F]
        self.node_targets = torch.tensor(node_tgts, dtype=torch.float32)  # [N, 4]
        self.tri_targets = self.node_targets.mean(dim=1)                  # [N]
        self.timestamps = timestamps                                      # target time (or None)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {"sequence": self.sequences[idx],
                "node_target": self.node_targets[idx],
                "tri_target": self.tri_targets[idx]}


def load_clean_frame(cfg) -> pd.DataFrame:
    df = pd.read_csv(cfg.RAW_DATA_PATH)
    missing = [c for c in cfg.FEATURE_COLS + [cfg.DATE_COL] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns {missing}; found {list(df.columns)}")
    df[cfg.DATE_COL] = pd.to_datetime(df[cfg.DATE_COL], errors="coerce")
    df = df.dropna(subset=[cfg.DATE_COL])
    df = (df.sort_values(cfg.DATE_COL, kind="stable")
            .drop_duplicates(subset=cfg.DATE_COL, keep="first")
            .reset_index(drop=True))
    null_frac = df[cfg.FEATURE_COLS].isna().mean()
    if (null_frac > 0.01).any():
        raise ValueError(f"More than 1% nulls:\n{null_frac[null_frac > 0.01]}")
    df[cfg.FEATURE_COLS] = df[cfg.FEATURE_COLS].ffill()        # past-only fill
    return df.dropna(subset=cfg.FEATURE_COLS).reset_index(drop=True)


def subsample(df: pd.DataFrame, cfg) -> pd.DataFrame:
    if cfg.MAX_SAMPLES is None or len(df) <= cfg.MAX_SAMPLES:
        return df
    if cfg.SAMPLING == "head":     # contiguous but covers only a short time span
        return df.iloc[: cfg.MAX_SAMPLES].reset_index(drop=True)
    stride = int(np.ceil(len(df) / cfg.MAX_SAMPLES))   # covers the full span; step = stride x raw step
    return df.iloc[::stride].reset_index(drop=True)


def _windows(ext: np.ndarray, ts: np.ndarray, n_ctx: int, L: int, H: int):
    """ext = [context rows ; partition rows]. Only windows whose TARGET is in the partition."""
    n_win = len(ext) - L - H + 1
    X = sliding_window_view(ext, (L, ext.shape[1]))[:n_win, 0]       # [n_win, L, F]
    tgt_idx = np.arange(n_win) + L + H - 1
    keep = tgt_idx >= n_ctx
    return (np.ascontiguousarray(X[keep]),
            ext[tgt_idx[keep], :4],
            ts[tgt_idx[keep]])


def build_datasets(cfg, save_scaler: bool = True):
    df = subsample(load_clean_frame(cfg), cfg)
    n = len(df)
    n_tr, n_va = int(n * cfg.TRAIN_RATIO), int(n * cfg.VAL_RATIO)
    parts = {"train": df.iloc[:n_tr], "val": df.iloc[n_tr:n_tr + n_va], "test": df.iloc[n_tr + n_va:]}

    scaler = MinMaxScaler().fit(parts["train"][cfg.FEATURE_COLS].values)   # TRAIN ONLY
    if save_scaler:
        joblib.dump({"scaler": scaler, "cols": cfg.FEATURE_COLS}, cfg.SCALER_PATH)

    scaled = {k: scaler.transform(v[cfg.FEATURE_COLS].values) for k, v in parts.items()}
    stamps = {k: v[cfg.DATE_COL].values for k, v in parts.items()}

    L, H = cfg.SEQ_LEN, cfg.HORIZON
    n_ctx = L + H - 1
    out, prev = {}, None
    for k in ["train", "val", "test"]:
        if prev is None:
            ext, ets, nc = scaled[k], stamps[k], 0
        else:
            ext = np.vstack([scaled[prev][-n_ctx:], scaled[k]])
            ets = np.concatenate([stamps[prev][-n_ctx:], stamps[k]])
            nc = n_ctx
        X, Y, T = _windows(ext, ets, nc, L, H)
        out[k] = SupplyChainDataset(X, Y, T)
        prev = k

    info = {"n_rows_used": n, "start": str(df[cfg.DATE_COL].iloc[0]), "end": str(df[cfg.DATE_COL].iloc[-1]),
            "train_mean": torch.tensor(scaled["train"].mean(axis=0), dtype=torch.float32)}
    return out["train"], out["val"], out["test"], scaler, info


def persistence_metrics(ds: SupplyChainDataset) -> dict:
    """Naive forecast y_hat(t+H) = y(t). Every model must beat this to claim anything."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    pred = ds.sequences[:, -1, :4].numpy()
    true = ds.node_targets.numpy()
    res = {"overall": {"MSE": mean_squared_error(true.mean(1), pred.mean(1)),
                       "MAE": mean_absolute_error(true.mean(1), pred.mean(1)),
                       "R2": r2_score(true.mean(1), pred.mean(1))}}
    for i in range(4):
        res[i] = {"MSE": mean_squared_error(true[:, i], pred[:, i]),
                  "MAE": mean_absolute_error(true[:, i], pred[:, i]),
                  "R2": r2_score(true[:, i], pred[:, i])}
    return res
