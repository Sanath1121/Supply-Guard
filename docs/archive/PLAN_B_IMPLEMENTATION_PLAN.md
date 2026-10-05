# SupplyGuard — PLAN B Implementation Plan (Optional extensions)

**Do not start until Plan A Gate 8 has passed** (`PLAN_A_IMPLEMENTATION_PLAN.md`). Every item is independent. Pick by remaining time. Colab is available for the heavy training items (B1, B2, B5).

**Prerequisites for all items:** Plan A repo, saved scaler, 5-seed baseline results, `requirements.lock`, passing smoke test. Re-use Plan A's evaluation protocol (persistence first, 5 seeds, mean ± std, pre-agreed claims) — an extension must beat the Plan A result under the same protocol to be reported as an improvement.

| Order | Item | Effort | Compute | Value |
|---|---|---|---|---|
| 1 | B1 Ablations | 3–4 h | Colab | High, cheap |
| 2 | B6 VAR + Granger heat-map | 2–3 h | Local | Strengthens the graph argument |
| 3 | B3 Streaming-replay mode | 3–4 h | Local | Good viva demo |
| 4 | B2 Graph attention (GAT) | 4–5 h | Colab | Research depth |
| 5 | B5 PatchTST vs LSTM | 5–6 h | Colab | Replaces the old unproven "Transformers need big data" claim with evidence |
| 6 | B4 LAN / Streamlit Cloud deploy | 2–3 h | Local | Polish |

---

## B1 — Ablations (3–4 h, Colab)
- **Goal:** show sensitivity of results to key choices.
- **Tasks:** vary `SEQ_LEN ∈ {5,10,20}`, `GCN_HIDDEN_DIM ∈ {16,32,64}`, `HORIZON_MIN ∈ {2,10,20,60}`; reuse the Plan A training/eval scripts with config overrides; 3 seeds per cell is acceptable (state it).
- **Deliverables:** `outputs/results/ablation.csv`, heatmap, one paragraph in the report.
- **Gate:** every cell reports persistence-relative improvement ± std.

## B6 — VAR baseline and Granger heat-map (2–3 h, local)
- **Goal:** classical multivariate baseline and direct evidence about the assumed S→M→D→R chain.
- **Tasks:** `statsmodels` VAR on scaled series within segments; Granger p-value matrix (all ordered pairs) as a heat-map; add VAR row to the comparison table.
- **Deliverables:** heat-map figure, VAR metrics row, interpretation text (feeds RQ2 discussion).

## B3 — Streaming-replay mode (3–4 h, local)
- **Goal:** step through real test windows in time order in Streamlit.
- **Tasks:** play/pause/step controls over chronological test windows; live-updating node risk cards and tier changes; threshold-crossing log.
- **Rules:** still "offline replay of historical data" — not real-time; no synthetic shocks presented as real events (a clearly labelled "what-if" mode is acceptable if added).
- **Gate:** replay runs through ≥200 consecutive windows without error.

## B2 — Graph attention layer (4–5 h, Colab)
- **Goal:** test learned edge weights against fixed adjacency.
- **Tasks:** manual single-head GAT layer in `graph_layers.py` as a third `GraphConv`-compatible mode (`attention`); train/evaluate as another `st_gcn_lstm` variant under the 5-seed protocol; log attention weights per edge.
- **Rules:** the 4-node chain gives each node ≤2 neighbours, so expect little gain; report honestly. Attention weights are not explanations by themselves — compare with IG upstream share before claiming anything.
- **Gate:** variant included in `overall_metrics.csv` with std.

## B5 — PatchTST vs LSTM temporal encoder (5–6 h, Colab)
- **Goal:** empirically test whether a Transformer encoder beats the LSTM on this data.
- **Tasks:** lightweight patch-based Transformer encoder as the temporal branch (same input/output shapes as the LSTM branch); match parameter count to the LSTM variant; 5 seeds; same early stopping.
- **Rules:** make no prior claim about which wins; report parameter counts and training time.
- **Gate:** comparison table with std and a one-paragraph conclusion that matches it.

## B4 — LAN / cloud deployment (2–3 h, local)
- **Goal:** let examiners open the demo from another device.
- **Tasks:** `streamlit run app/streamlit_app.py --server.address 0.0.0.0`; optionally Streamlit Community Cloud (requires the repo to be on GitHub; data and checkpoints must be included or downloaded at start-up — check dataset licence terms and file-size limits first).
- **Gate:** app opens from a second device and all tabs render.

---

## Risks specific to Plan B

| Risk | Mitigation |
|---|---|
| Extensions crowd out viva preparation | Hard stop: finish Plan A Phase 8 first; reserve the last day before submission for rehearsal only |
| Extension results contradict Plan A claims | Report both; update the claims table, never silently swap |
| Colab runtime limits | Same checkpoint-resume pattern as Plan A Phase 4 |
