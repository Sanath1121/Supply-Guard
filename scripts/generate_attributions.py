import os
import random
import tempfile
import torch
import numpy as np
import pandas as pd

from src.config import Config
from src.dataset import build_datasets
from src.models.st_gcn_lstm import build_model
from src.explainability import RiskExplainer, upstream_share, narrate
from training.train import ckpt_path

def main():
    # Set explicit random seeds for reproducible attribution analysis
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    cfg = Config()
    # Defensively redirect SCALER_PATH to prevent touching production scaler
    cfg.SCALER_PATH = os.path.join(tempfile.gettempdir(), "supplyguard_temp_scaler.joblib")

    print("Loading datasets...")
    tr, va, te, scaler, info = build_datasets(cfg, save_scaler=False)
    
    # Target node: Retailer (3) so we can measure upstream share (S, M, D)
    target_node = 3
    
    print("Loading models (seed 42)...")
    model_dir = build_model("st_gcn_lstm_dir", cfg).eval()
    model_sym = build_model("st_gcn_lstm_sym", cfg).eval()
    
    p_dir = ckpt_path(cfg, "st_gcn_lstm_dir", 42)
    p_sym = ckpt_path(cfg, "st_gcn_lstm_sym", 42)
    
    model_dir.load_state_dict(torch.load(p_dir, map_location="cpu", weights_only=True))
    model_sym.load_state_dict(torch.load(p_sym, map_location="cpu", weights_only=True))
    
    # Find windows where the model actually predicts a significant change from persistence
    # Only search the first 2000 to avoid CPU OOM
    search_seqs = te.sequences[:2000]
    with torch.no_grad():
        all_preds = model_dir(search_seqs)
        deltas = torch.abs(all_preds[:, target_node] - search_seqs[:, -1, target_node])
        
    # Pick top 10 indices with largest delta
    top_indices = torch.argsort(deltas, descending=True)[:10].tolist()
    
    baseline = info["train_mean"]
    
    explainer_dir = RiskExplainer(model_dir, baseline, steps=64)
    explainer_sym = RiskExplainer(model_sym, baseline, steps=64)
    
    results = []
    
    for idx, i in enumerate(top_indices):
        seq = te.sequences[i]
        
        # 1. Normal IG
        res_dir = explainer_dir.explain(seq, target_node, residual_delta=False)
        res_sym = explainer_sym.explain(seq, target_node, residual_delta=False)
        
        # 2. Delta IG
        res_dir_delta = explainer_dir.explain(seq, target_node, residual_delta=True)
        
        # 3. Upstream share (on Normal IG, or Delta IG? Plan implies on Delta IG, let's use delta)
        up_dir = upstream_share(res_dir_delta)
        up_sym = upstream_share(explainer_sym.explain(seq, target_node, residual_delta=True))
        
        # 4. Deletion test vs Random on Delta IG
        feat_imp = np.array(res_dir_delta["feature_importance"])
        top_feat = int(np.argmax(feat_imp))
        
        rand_feat = random.choice([f for f in range(5) if f != top_feat])
        
        delta_pred_orig = res_dir_delta["predicted_risk"] - res_dir_delta["baseline_risk"]
        
        # Mask top feature
        seq_del_top = seq.clone()
        seq_del_top[:, top_feat] = baseline[top_feat]
        res_del_top = explainer_dir.explain(seq_del_top, target_node, residual_delta=True)
        delta_pred_del_top = res_del_top["predicted_risk"] - res_del_top["baseline_risk"]
        drop_top = abs(delta_pred_orig - delta_pred_del_top)
        
        # Mask random feature
        seq_del_rand = seq.clone()
        seq_del_rand[:, rand_feat] = baseline[rand_feat]
        res_del_rand = explainer_dir.explain(seq_del_rand, target_node, residual_delta=True)
        delta_pred_del_rand = res_del_rand["predicted_risk"] - res_del_rand["baseline_risk"]
        drop_rand = abs(delta_pred_orig - delta_pred_del_rand)

        # Multi-feature benchmark: compute mean drop across all non-top features
        other_drops = []
        for f in range(5):
            if f != top_feat:
                seq_del_f = seq.clone()
                seq_del_f[:, f] = baseline[f]
                res_del_f = explainer_dir.explain(seq_del_f, target_node, residual_delta=True)
                delta_pred_del_f = res_del_f["predicted_risk"] - res_del_f["baseline_risk"]
                other_drops.append(abs(delta_pred_orig - delta_pred_del_f))
        drop_mean_other = float(np.mean(other_drops))

        # 5. Stability (compare attribution to next window i+1)
        if i + 1 < len(te.sequences):
            seq_next = te.sequences[i+1]
            res_next = explainer_dir.explain(seq_next, target_node, residual_delta=True)
            stability = np.linalg.norm(res_dir_delta["attribution"] - res_next["attribution"])
        else:
            stability = np.nan

        results.append({
            "window_index": i,
            "target_node": target_node,
            "completeness_gap": res_dir_delta["completeness_gap"],
            "upstream_share_dir": up_dir,
            "upstream_share_sym": up_sym,
            "deletion_drop_top": drop_top,
            "deletion_drop_rand": drop_rand,
            "deletion_drop_mean_other": drop_mean_other,
            "deletion_test_passed": drop_top > drop_rand,
            "stability_l2": stability,
            "narrative": narrate(res_dir_delta, seq_len=cfg.SEQ_LEN)
        })
        
    df = pd.DataFrame(results)
    os.makedirs("outputs/results", exist_ok=True)
    df.to_csv("outputs/results/attribution_examples.csv", index=False)
    print("Done! Results saved to outputs/results/attribution_examples.csv")
    
    pass_rate = df["deletion_test_passed"].mean()
    print(f"Deletion Test Pass Rate: {pass_rate:.1%}")
    print(f"Average upstream share (Directed): {df['upstream_share_dir'].mean():.1%}")
    print(f"Average upstream share (Symmetric): {df['upstream_share_sym'].mean():.1%}")

if __name__ == "__main__":
    main()
