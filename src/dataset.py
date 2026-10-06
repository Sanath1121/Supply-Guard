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
    def __init__(self, seqs, node_tgts, timestamps=None, seq_timestamps=None):
        self.sequences = torch.tensor(seqs, dtype=torch.float32) if not isinstance(seqs, torch.Tensor) else seqs
        self.node_targets = torch.tensor(node_tgts, dtype=torch.float32) if not isinstance(node_tgts, torch.Tensor) else node_tgts
        self.tri_targets = self.node_targets.mean(dim=1)                  # [N]
        self.timestamps = timestamps                                      # target time (or None)
        self.seq_timestamps = seq_timestamps                              # sequence timestamps [N, L]

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

    n_raw = len(df)
    ts_fmt = getattr(cfg, "TIMESTAMP_FORMAT", "%m/%d/%Y %I:%M:%S %p")
    df[cfg.DATE_COL] = pd.to_datetime(df[cfg.DATE_COL], format=ts_fmt, errors="coerce")
    n_bad_ts = int(df[cfg.DATE_COL].isna().sum())
    df = df.dropna(subset=[cfg.DATE_COL])

    df = (df.sort_values(cfg.DATE_COL, kind="stable")
            .drop_duplicates(subset=[cfg.DATE_COL], keep="first"))
            
    df = df.set_index(cfg.DATE_COL).resample("2min").asfreq()
    
    n_dup_dropped = (n_raw - n_bad_ts) - len(df[~df[cfg.FEATURE_COLS].isna().all(axis=1)])

    ffill_limit = getattr(cfg, "FFILL_LIMIT", 5)
    df[cfg.FEATURE_COLS] = df[cfg.FEATURE_COLS].ffill(limit=ffill_limit)

    has_nan = df[cfg.FEATURE_COLS].isna().any(axis=1)
    n_nan_dropped = int(has_nan.sum())
    clean_df = df[~has_nan].copy()

    min_seg_len = getattr(cfg, "SEQ_LEN", 10) + getattr(cfg, "HORIZON", 5)
    gap_max_min = getattr(cfg, "GAP_MAX_MIN", 6.0)

    if len(clean_df) > 0:
        time_diffs_clean = clean_df.index.to_series().diff()
        is_new_seg = time_diffs_clean > pd.Timedelta(minutes=gap_max_min)
        is_new_seg.iloc[0] = False
        clean_df["seg_id"] = is_new_seg.cumsum()

        seg_counts = clean_df.groupby("seg_id")["seg_id"].transform("count")
        short_mask = seg_counts < min_seg_len
        n_short_seg_rows = int(short_mask.sum())
        valid_df = clean_df[~short_mask].copy().reset_index()
        if len(valid_df) > 0:
            valid_df["seg_id"] = pd.factorize(valid_df["seg_id"])[0]
    else:
        n_short_seg_rows = 0
        valid_df = clean_df.copy().reset_index()

    n_clean_retained = len(valid_df)
    row_loss_stats = {
        "n_raw": n_raw,
        "n_bad_timestamps": n_bad_ts,
        "n_duplicates_dropped": n_dup_dropped,
        "n_nans_dropped": n_nan_dropped,
        "n_short_seg_rows_dropped": n_short_seg_rows,
        "n_clean_rows_retained": n_clean_retained,
        "pct_retained": (n_clean_retained / n_raw * 100.0) if n_raw > 0 else 0.0,
    }
    valid_df.attrs["row_loss"] = row_loss_stats
    print(
        f"[DATA INGESTION] Row Loss Accounting:\n"
        f"  - Raw records: {n_raw}\n"
        f"  - Unparseable timestamps dropped: {n_bad_ts}\n"
        f"  - Duplicate timestamps dropped: {n_dup_dropped}\n"
        f"  - Rows with unfilled NaNs dropped: {n_nan_dropped}\n"
        f"  - Rows in short segments (< {min_seg_len} steps) dropped: {n_short_seg_rows}\n"
        f"  - Clean records retained: {n_clean_retained} ({row_loss_stats['pct_retained']:.2f}%)\n"
        f"  - Clean contiguous segments: {valid_df['seg_id'].nunique() if len(valid_df) > 0 else 0}"
    )
    return valid_df



def build_datasets(cfg, save_scaler: bool = True):
    clean_df = load_clean_frame(cfg)
    n_total = len(clean_df)
    n_tr = int(n_total * cfg.TRAIN_RATIO)
    n_va = int(n_total * cfg.VAL_RATIO)

    # MinMaxScaler is fitted strictly on TRAIN partition rows only
    train_raw = clean_df.iloc[:n_tr][cfg.FEATURE_COLS].values
    scaler = MinMaxScaler().fit(train_raw)
    if save_scaler:
        import os
        os.makedirs(os.path.dirname(cfg.SCALER_PATH), exist_ok=True)
        joblib.dump({"scaler": scaler, "cols": cfg.FEATURE_COLS}, cfg.SCALER_PATH)

    scaled = scaler.transform(clean_df[cfg.FEATURE_COLS].values)
    stamps = clean_df[cfg.DATE_COL].values
    L, H = cfg.SEQ_LEN, cfg.HORIZON

    tr_X, tr_Y, tr_T, tr_seq_T = [], [], [], []
    va_X, va_Y, va_T, va_seq_T = [], [], [], []
    te_X, te_Y, te_T, te_seq_T = [], [], [], []

    if n_total > 0 and "seg_id" in clean_df.columns:
        seg_ids = clean_df["seg_id"].values
        seg_change = np.where(seg_ids[:-1] != seg_ids[1:])[0] + 1
        seg_starts = np.concatenate([[0], seg_change])
        seg_ends = np.concatenate([seg_change, [n_total]])

        for r_start, r_end in zip(seg_starts, seg_ends):
            M = r_end - r_start
            if M < L + H:
                continue
            n_win = M - L - H + 1
            seg_scaled = scaled[r_start:r_end]
            seg_stamps = stamps[r_start:r_end]

            win_X = sliding_window_view(seg_scaled, (L, 5))[:n_win, 0]
            win_seq_T = sliding_window_view(seg_stamps, L)[:n_win]
            tgt_local = np.arange(n_win) + L + H - 1
            win_Y = seg_scaled[tgt_local, :4]
            win_T = seg_stamps[tgt_local]
            global_tgt = r_start + tgt_local

            mask_tr = global_tgt < n_tr
            mask_va = (global_tgt >= n_tr) & (global_tgt < n_tr + n_va)
            mask_te = global_tgt >= n_tr + n_va

            if mask_tr.any():
                tr_X.append(win_X[mask_tr])
                tr_Y.append(win_Y[mask_tr])
                tr_T.append(win_T[mask_tr])
                tr_seq_T.append(win_seq_T[mask_tr])
            if mask_va.any():
                va_X.append(win_X[mask_va])
                va_Y.append(win_Y[mask_va])
                va_T.append(win_T[mask_va])
                va_seq_T.append(win_seq_T[mask_va])
            if mask_te.any():
                te_X.append(win_X[mask_te])
                te_Y.append(win_Y[mask_te])
                te_T.append(win_T[mask_te])
                te_seq_T.append(win_seq_T[mask_te])

    def _concat(lst, shape_tail, dtype):
        if lst:
            return np.ascontiguousarray(np.concatenate(lst, axis=0))
        return np.empty((0, *shape_tail), dtype=dtype)

    X_tr = _concat(tr_X, (L, 5), np.float32)
    Y_tr = _concat(tr_Y, (4,), np.float32)
    T_tr = _concat(tr_T, (), stamps.dtype)
    seq_T_tr = _concat(tr_seq_T, (L,), stamps.dtype)

    X_va = _concat(va_X, (L, 5), np.float32)
    Y_va = _concat(va_Y, (4,), np.float32)
    T_va = _concat(va_T, (), stamps.dtype)
    seq_T_va = _concat(va_seq_T, (L,), stamps.dtype)

    X_te = _concat(te_X, (L, 5), np.float32)
    Y_te = _concat(te_Y, (4,), np.float32)
    T_te = _concat(te_T, (), stamps.dtype)
    seq_T_te = _concat(te_seq_T, (L,), stamps.dtype)

    if getattr(cfg, "MAX_SAMPLES", None) is not None:
        max_s = cfg.MAX_SAMPLES
        total_wins = len(X_tr) + len(X_va) + len(X_te)
        if total_wins > max_s:
            stride = int(np.ceil(total_wins / max_s))
            X_tr, Y_tr, T_tr, seq_T_tr = X_tr[::stride], Y_tr[::stride], T_tr[::stride], seq_T_tr[::stride]
            X_va, Y_va, T_va, seq_T_va = X_va[::stride], Y_va[::stride], T_va[::stride], seq_T_va[::stride]
            X_te, Y_te, T_te, seq_T_te = X_te[::stride], Y_te[::stride], T_te[::stride], seq_T_te[::stride]

    out = {
        "train": SupplyChainDataset(X_tr, Y_tr, T_tr, seq_T_tr),
        "val": SupplyChainDataset(X_va, Y_va, T_va, seq_T_va),
        "test": SupplyChainDataset(X_te, Y_te, T_te, seq_T_te),
    }

    row_loss = clean_df.attrs.get("row_loss", {})
    train_mean = (torch.tensor(scaled[:n_tr].mean(axis=0), dtype=torch.float32)
                  if n_tr > 0 else torch.zeros(5, dtype=torch.float32))
    info = {
        "n_rows_used": n_total,
        "start": str(clean_df[cfg.DATE_COL].iloc[0]) if n_total > 0 else "",
        "end": str(clean_df[cfg.DATE_COL].iloc[-1]) if n_total > 0 else "",
        "train_mean": train_mean,
        "row_loss": row_loss,
        "n_rows_lost_total": (row_loss.get("n_duplicates_dropped", 0) +
                              row_loss.get("n_nans_dropped", 0) +
                              row_loss.get("n_short_seg_rows_dropped", 0)),
    }
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
