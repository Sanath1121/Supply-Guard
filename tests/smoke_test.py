"""Run from project root:  python -m tests.smoke_test
Synthetic data with a KNOWN lagged S->M->D->R dependency, so we can check the code end to end."""
import os, numpy as np, pandas as pd, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from src.config import Config
from src.dataset import build_datasets, persistence_metrics
from src.graph_builder import build_graphs, normalised_laplacian
from src.models.st_gcn_lstm import build_model
from src.explainability import RiskExplainer, narrate


def make_csv(path, n=6000, seed=0):
    rng = np.random.default_rng(seed)
    s = np.zeros((n, 4))
    for t in range(1, n):
        s[t, 0] = 0.9 * s[t-1, 0] + rng.normal(0, 0.3)
        for k in range(1, 4):                       # each echelon follows its upstream parent with lag 1
            s[t, k] = 0.5 * s[t-1, k] + 0.5 * s[t-1, k-1] + rng.normal(0, 0.1)

    # 2-minute cadence with injected 24-hour temporal gap (> 6 min) at row 2000
    gap_idx = 2000
    t1 = pd.date_range("2018-01-01 00:00:00", periods=gap_idx, freq="2min")
    t2_start = t1[-1] + pd.Timedelta(hours=24)
    t2 = pd.date_range(t2_start, periods=n - gap_idx, freq="2min")
    ts = list(t1) + list(t2)

    df = pd.DataFrame({
        "Timestamp": [t.strftime("%m/%d/%Y %I:%M:%S %p") for t in ts],
        "RI_Supplier1": s[:, 0],
        "RI_Distributor1": s[:, 2],
        "RI_Manufacturer1": s[:, 1],
        "RI_Retailer1": s[:, 3],
        "Total_Cost": s.sum(1) + rng.normal(0, .1, n),
        "SCMstability_category": 1
    })

    # Inject duplicate timestamp
    df.loc[15, "Timestamp"] = df.loc[14, "Timestamp"]

    # Inject bounded null run (<= 5 rows, forward-fillable)
    df.loc[500:502, "RI_Distributor1"] = np.nan

    # Inject extreme null run (> 5 rows, e.g. 8 rows, splits segment)
    df.loc[4000:4007, "RI_Distributor1"] = np.nan

    df = df.sample(frac=1.0, random_state=1)        # shuffled on disk: loader must sort by time
    df.to_csv(path, index=False)


def main():
    import tempfile, shutil
    tmp_dir = tempfile.mkdtemp()
    cfg = Config()
    cfg.RAW_DATA_PATH = os.path.join(tmp_dir, "_synthetic.csv")
    cfg.CKPT_DIR = os.path.join(tmp_dir, "models")
    os.makedirs(cfg.CKPT_DIR, exist_ok=True)
    cfg.MAX_SAMPLES = None
    make_csv(cfg.RAW_DATA_PATH)

    # --- graph sanity -------------------------------------------------------
    g = build_graphs()
    assert torch.equal(g["A_hat"], g["A_hat"].T)
    ev = torch.linalg.eigvalsh(normalised_laplacian()); assert ev.min() > -1e-6 and ev.max() < 2 + 1e-6
    assert g["A_down"][1, 0] == 1 and g["A_down"][0].sum() == 0      # M receives from S; S has no parent
    assert g["A_up"][0, 1] == 1 and g["A_up"][3].sum() == 0          # S sends to M; R has no child
    print("graph OK")

    # --- dataset: chronology, leakage, shapes, gap integrity ----------------
    tr, va, te, sc, info = build_datasets(cfg, save_scaler=False)

    # Assert shapes: sequences [N, 10, 5], node_targets [N, 4]
    for split_name, ds in [("train", tr), ("val", va), ("test", te)]:
        assert ds.sequences.shape[1:] == (cfg.SEQ_LEN, 5), f"{split_name} seq shape mismatch"
        assert ds.node_targets.shape[1] == 4, f"{split_name} target shape mismatch"

    # Assert zero NaNs in sequences and targets
    for split_name, ds in [("train", tr), ("val", va), ("test", te)]:
        assert not torch.isnan(ds.sequences).any(), f"NaNs in {split_name} sequences"
        assert not torch.isnan(ds.node_targets).any(), f"NaNs in {split_name} targets"

    # Assert chronological order of partitions
    assert tr.timestamps.max() < va.timestamps.min() < va.timestamps.max() < te.timestamps.min()
    assert np.all(np.diff(tr.timestamps.astype("int64")) > 0)          # sorted despite shuffled file
    assert np.all(np.diff(va.timestamps.astype("int64")) > 0)
    assert np.all(np.diff(te.timestamps.astype("int64")) > 0)

    # Assert row loss was logged
    assert "row_loss" in info, "Row loss accounting missing from info dict"
    assert info["n_rows_lost_total"] > 0, "Row loss total must be greater than zero"

    # Assert no window spans a gap
    max_gap_ns = int(cfg.GAP_MAX_MIN * 60 * 1e9)
    for split_name, ds in [("train", tr), ("val", va), ("test", te)]:
        if ds.seq_timestamps is not None and len(ds) > 0:
            seq_diffs = np.diff(ds.seq_timestamps.astype("int64"), axis=1)
            assert np.all(seq_diffs <= max_gap_ns), f"Sequence window in {split_name} spans a gap > GAP_MAX_MIN"
            tgt_diff = ds.timestamps.astype("int64") - ds.seq_timestamps[:, -1].astype("int64")
            assert np.all(tgt_diff <= int(cfg.HORIZON * max_gap_ns)), f"Target in {split_name} spans a gap"

    # Target is genuinely t+5 (index 14): first-sample target equals row 14 in clean frame
    clean_full = pd.read_csv(cfg.RAW_DATA_PATH)
    clean_full["dt"] = pd.to_datetime(clean_full[cfg.DATE_COL], format=cfg.TIMESTAMP_FORMAT)
    clean_full = clean_full.sort_values("dt", kind="stable").drop_duplicates("dt", keep="first").reset_index(drop=True)
    x0 = sc.transform(clean_full[cfg.FEATURE_COLS].values)
    assert np.allclose(tr.sequences[0].numpy(), x0[:10], atol=1e-5), "First sequence mismatch"
    assert np.allclose(tr.node_targets[0].numpy(), x0[14, :4], atol=1e-5), "Target at step t+5 (index 14) mismatch"
    print(f"dataset OK  train={len(tr)} val={len(va)} test={len(te)}  span={info['start']} -> {info['end']}")

    pers = persistence_metrics(te)
    print(f"persistence (test): overall MSE={pers['overall']['MSE']:.5f} R2={pers['overall']['R2']:.4f}")

    # --- models: shapes + 2-hop reach + quick training -----------------------
    torch.manual_seed(0)
    for name in ["st_gcn_lstm", "lstm", "paper_overall"]:
        m = build_model(name, cfg); o = m(torch.randn(8, 10, 5))
        assert o.shape == ((8, 4) if name != "paper_overall" else (8,)), (name, o.shape)

    cfg.GRAPH_MODE = "directed"; m = build_model("st_gcn_lstm", cfg)
    x = torch.rand(2, 10, 5, requires_grad=True)
    cfg.RESIDUAL = False; m2 = build_model("st_gcn_lstm", cfg); cfg.RESIDUAL = True
    for p in m2.head[-1].parameters(): torch.nn.init.normal_(p, std=0.5)   # non-zero head so grads flow
    m2.eval(); m2(x)[:, 2].sum().backward()
    assert x.grad[:, :, 0].abs().sum() > 0, "Supplier input must influence Distributor (2 hops)"
    print("models OK (2-hop gradient S->D non-zero)")

    def fit(model, epochs=6):
        opt = torch.optim.Adam(model.parameters(), 1e-3)
        dl = DataLoader(tr, batch_size=cfg.BATCH_SIZE, shuffle=True)
        for _ in range(epochs):
            model.train()
            for b in dl:
                opt.zero_grad(); F.mse_loss(model(b["sequence"]), b["node_target"]).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP); opt.step()
        model.eval()
        with torch.no_grad():
            p = model(te.sequences)
        return F.mse_loss(p, te.node_targets).item()

    base_mse = float(np.mean([pers[i]["MSE"] for i in range(4)]))
    res = {n: fit(build_model(n, cfg)) for n in ["lstm", "st_gcn_lstm"]}
    print(f"test node-MSE  persistence={base_mse:.5f}  " + "  ".join(f"{k}={v:.5f}" for k, v in res.items()))

    # --- explainability: completeness ----------------------------------------
    model = build_model("st_gcn_lstm", cfg); fit(model, 3)
    ex = RiskExplainer(model, info["train_mean"], steps=128)
    r = ex.explain(te.sequences[0], target_node=3)
    assert abs(sum(r["feature_importance"]) - 1) < 1e-6 and abs(sum(r["time_importance"]) - 1) < 1e-6
    print(f"IG completeness gap = {r['completeness_gap']:.5f} (pred-baseline = {r['predicted_risk']-r['baseline_risk']:.4f})")
    print(narrate(r))
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
