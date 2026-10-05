# SupplyGuard — PLAN A Implementation Plan (Core, mandatory)

**Scope:** everything needed to submit a complete mini project. **Plan B extensions are NOT in this file** (see `PLAN_B_IMPLEMENTATION_PLAN.md`; start them only after Gate 8).
**Companion docs:** `MASTER_PLAN_AND_TECHSTACK.md` (tech stack table, facts from the real-data probe, decision log). Where they disagree on phases or locked choices, **this file wins**.
**Base paper:** *Hybrid GNN-LSTM Model for Real-Time Supply Chain Risk Prediction*, IEEE ICCMC 2025 (DOI 10.1109/ICCMC65190.2025.11140739)
**Dataset:** Banerjee et al. (2019), Mendeley Data V2 (DOI 10.17632/gystn6d3r4.2, CC BY 4.0)

---

## 0. Locked choices (confirmed by you)

| # | Decision | Value |
|---|---|---|
| L1 | Dataset | **`SCRM_timeSeries_2018_train.csv` only** (649,999 rows, Banerjee dataset). Test CSV not used. Reason: the train file alone is ample, avoids cross-year drift, and keeps the pipeline simple |
| L2 | Core spatial model | **2-layer GCN** (`directed` and `symmetric` modes). GAT is Plan B only |
| L3 | Horizon | **10 minutes ahead** (config in minutes, converted to steps). Final value confirmed at Gate 1 using EDA persistence results; fall back to 20 min if persistence still dominates |
| L4 | Seeds | **5** (`42–46`) |
| L5 | Compute | **Google Colab for training/evaluation**, local machine for EDA, Streamlit and the viva demo. Checkpoints and CSVs are copied back to the repo |
| L6 | Framing | Re-implementation of the paper's model as a baseline, plus a node-level extension. Every research question may be answered "no" |

**Research questions:** RQ1 beat persistence? RQ2 graph vs graph-free LSTM? RQ3 directed vs symmetric? RQ4 are IG explanations valid (deletion test)?

---

## 1. Environment strategy (Colab + local)

- **Colab (train/evaluate):** use Colab's default Python. This removes the Python 3.14 wheel risk for training. `pip install -r requirements.txt` in the first cell; mount Google Drive or clone from git.
- **Local (EDA, dashboard):** try Python 3.14 (15-min timebox) → else a 3.12 venv.
- **Version consistency:** pin `requirements.lock` from Colab; checkpoints are plain PyTorch `state_dict`, portable between environments.
- **Colab hygiene:** Colab sessions are disconnectable. `train.py` must save a checkpoint and a loss CSV per (model, mode, seed) as it goes, and skip runs whose checkpoint already exists (resume by re-running).
- **Repo location:** keep it **outside OneDrive** (e.g. `C:\dev\supplyguard`) under git, with GitHub as the bridge to Colab.

---

## 2. Structure

```
supplyguard/
├── docs/                    this plan, Plan B, master doc, archive/ (old docs)
├── data/raw/                SCRM_timeSeries_2018_train.csv (git-ignored)
├── src/{config,graph_builder,dataset,explainability}.py
├── src/models/{graph_layers,st_gcn_lstm}.py
├── training/{train,evaluate}.py
├── tests/smoke_test.py
├── notebooks/{01_EDA,02_Results,colab_train}.ipynb
├── app/streamlit_app.py
├── outputs/{figures,models,results}/
├── AGENTS.md  requirements.txt  requirements.lock  setup_and_download.py  README.md
```

---

## 3. Phases and gates

Starting code exists in `Review/` (Response 2: config, graph, models, dataset, explainability, smoke test; Response 3: train, evaluate, setup). It is a **draft**: it was only run on synthetic data and **will crash on the real file** (see Phase 0/2 fixes).

### Phase 0 — Repository, data acquisition (3–4 h)
1. Create repo, `git init`, `.gitignore` (`data/`, `outputs/models/`, `mlruns/`, `.venv/`); copy the Review code into the structure; add `__init__.py` to `src/`, `src/models/`, `training/`, `tests/`; archive old docs.
2. **Fix `setup_and_download.py`:** drop the `nulls < 1%` assert (real nulls: `RI_Distributor1` 5.1%, `Total_Cost` 5.5%); pin the mirror commit hash; record sha256; parse timestamps with `format="%m/%d/%Y %I:%M:%S %p"`; print row count, span (2015-01-28 → 2018-12-19), median step (2 min), the largest gaps (max 428 days), null-run lengths (max 5,860 rows), duplicate stamps (2,363).
3. Write `AGENTS.md`: source of truth = `docs/` Plan A + tech stack; never change leakage-critical logic in `dataset.py` without asking; models take one tensor `seq [B,L,5]`; no PyG; run `python -m tests.smoke_test` after changes; never claim a model wins unless `evaluate.py` shows it; explanations = sensitivity, not "root cause".
4. Set up local env and a Colab environment; `pip freeze > requirements.lock`.
- **Gate 0:** setup runs cleanly; you can state rows, span, step, gap structure in one sentence.

### Phase 1 — EDA and go/no-go (4–6 h, local) · `notebooks/01_EDA.ipynb`
1. Distributions, missingness map, null-run and segment-length histograms.
2. ACF (lags 1–50); rolling mean/variance.
3. Cross-correlation of adjacent echelons (lags 0–20, both directions); **Granger** S→M, M→D, D→R and reverse, computed **within segments**.
4. Persistence R² per node at horizons 2, 10, 20, 60 minutes on the series that will actually be used.
5. `SCMstability_category` vs the four RIs (association only).
6. **Check the paper's quoted metrics in the PDF** before any critique appears in the report.
- **Decision table:** persistence R² ≥ 0.99 at the chosen horizon → lengthen it. No cross-node lead/lag → proceed, but frame RQ2/RQ3 as "tests whether the assumed graph helps" (the probe already suggests this). i.i.d. noise → stop and change dataset/framing.
- **Gate 1:** one-page EDA summary with the table filled in; **`HORIZON_MIN` locked** (default 10).

### Phase 2 — Data pipeline hardening (4–5 h) · `src/dataset.py`, `src/config.py`
1. Fixed-format timestamp parse → stable sort → drop duplicate stamps.
2. **Segmentation:** new segment wherever the gap > `GAP_MAX` (default 6 min = 3× median). Windows never cross segments.
3. **Null policy:** `ffill(limit=5)` within segments; remaining NaNs split the segment; drop segments shorter than `L+H`; report rows lost.
4. **Stride sampling within segments** if `MAX_SAMPLES` is set; effective step = stride × 2 min. Horizon is stored in minutes and converted to steps.
5. Keep from Response 2: chronological 80:10:10 by row position, train-only scaler saved to `outputs/models/scaler.joblib`, context rows borrowed from the previous partition (only when contiguous in the same segment), targets inside their own partition, single input tensor.
6. Extend the smoke test with injected gaps and null runs; assert no window spans a gap.
- **Gate 2:** smoke test passes **and** a manual real-file check confirms target = row H steps after the window end, same segment.

### Phase 3 — Models and baselines (3–4 h) · `src/models/`, `src/graph_builder.py`
1. Adopt `GraphConv` (symmetric | directed), `STGCNLSTM`, `LSTMBaseline`, `PaperHybridOverall`.
2. **Remove the `Sigmoid` from `PaperHybridOverall`** (scaled val/test values can exceed [0,1]); document the deviation.
3. Parameter-count table → `outputs/results/param_counts.csv`.
4. Terminology: "renormalised adjacency" (not Laplacian); echelon embedding is *added*; "re-implementation", not "reproduction".
- **Gate 3:** shapes `[B,4]` / `[B]`; non-zero 2-hop gradient Supplier→Distributor; parameter table exists.

### Phase 4 — Training on Colab (2 h hands-on + wall clock) · `training/train.py`
1. **Time one epoch per model first**, then fix `MAX_SAMPLES` and `MAX_EPOCHS`. Colab makes the full 5-seed grid realistic.
2. Grid: `lstm`, `paper_overall`, `st_gcn_lstm[directed]`, `st_gcn_lstm[symmetric]` × 5 seeds. Only `st_gcn_lstm` is re-trained per graph mode.
3. Same loss for train and val; early stopping; `GRAD_CLIP`; per-run loss CSV; skip-if-checkpoint-exists (Colab resume); optional MLflow (auto-skipped if unavailable).
4. `notebooks/colab_train.ipynb`: clone repo → install → run training → zip `outputs/` → download.
- **Gate 4:** no NaN or epoch-1 stops; train/val curves comparable; `training_summary.csv` has wall-clock times.

### Phase 5 — Evaluation (3–4 h) · `training/evaluate.py`
1. Evaluate **both graph modes in one run** (the draft overwrites CSVs per mode).
2. Order: persistence → Ridge-AR(10) → LSTM → paper-style → STGCN symmetric/directed; report % MSE improvement vs persistence ± std.
3. Metrics per node and overall (MSE/MAE/RMSE/R²), scaled and raw, mean ± std over 5 seeds.
4. Severity tiers: **per-node terciles from train**; confusion matrix + macro-F1.
5. Plots: prediction vs truth on a test slice; error by node.
6. Headline claims only from the pre-agreed table (§4).
- **Gate 5:** the report's headline sentence matches the table exactly.

### Phase 6 — Explainability with validation (4–5 h) · `src/explainability.py`
1. Integrated Gradients (64 steps, training-mean baseline, signed, completeness gap).
2. **Δ-attribution:** also explain `f(x) − y_t` (what the network adds beyond persistence); raw IG on the residual output mostly rediscovers "own RI at t".
3. 8–10 hand-picked windows: completeness gap ≈ 0; **deletion test** vs random removal; neighbouring-window stability; upstream share directed vs symmetric.
4. Save `attribution_examples.csv`. Wording: "most sensitive to…", never "root cause".
- **Gate 6:** deletion test beats random. If it fails, the dashboard labels explanations "unvalidated".

### Phase 7 — SupplyGuard dashboard (6–8 h, local) · `app/streamlit_app.py`
Tabs: (1) echelon risk cards on a real test window with tier, derived TRI, raw units via the saved scaler; (2) directed topology coloured by tier, edge width = computed upstream share; (3) "Why this forecast?" signed feature/time bars, Δ toggle, `narrate()` sentence, completeness caption; (4) benchmarks with ± std. Rules: label "offline replay of historical windows"; no sliders, invented confidence or hard-coded explanation text; `st.cache_resource` for models/scaler.
- **Gate 7:** every tab renders for ≥3 windows from a fresh `streamlit run`.

### Phase 8 — Docs and viva (5–7 h)
README (setup order, sha256, commit, seeds, hardware, Colab runtimes), `02_Results.ipynb`, report with a limitations section (assumed topology, simulated data, offline replay, sensitivity ≠ causation, gap/null handling), viva rehearsal using the R2 viva table, final clean run from a fresh clone.
- **Gate 8:** a fresh clone reproduces results following the README. **Plan A is complete here.**

---

## 4. Claims table (decide before seeing results)

| Result | Allowed claim |
|---|---|
| Models ≫ persistence and `st_gcn_lstm` beats `lstm` beyond 1 std | "Graph structure improves echelon-level forecasts on this dataset" |
| Models > persistence; `st_gcn_lstm` ≈ `lstm` | "Temporal modelling helps; the assumed graph adds no measurable accuracy but enables per-node attribution" |
| Persistence ≥ models | Lengthen horizon; if still true, "dataset is dominated by short-term persistence" |
| `directed` > `symmetric` | Supports direction-aware propagation; check upstream share |
| Attributions fail deletion test | Report explanations as unreliable |

---

## 5. Verification checklist

| # | Check | Pass if |
|---|---|---|
| 1 | Dataset integrity | sha256 pinned; span, step, gaps, null runs printed |
| 2 | Sorted + deduplicated | smoke test on shuffled input |
| 3 | Chronological split | `max(train) < min(val) < max(val) < min(test)` |
| 4 | No scaler leakage | scaler fit on train only, saved |
| 5 | Forward target, no gap crossing | smoke test + real-file spot check |
| 6 | Graph correctness | Â symmetric; Laplacian eigenvalues in [0,2]; `A_down[M,S]=1`, `A_up[S,M]=1` |
| 7 | Node identity preserved | no pooling in `STGCNLSTM`; embedding `[B,4,64]` |
| 8 | 2-hop reach | non-zero Supplier→Distributor gradient |
| 9 | Persistence | reported with seed std (answer may be "no") |
| 10 | Graph ablation | `lstm` vs `st_gcn_lstm` (sym, dir) table over 5 seeds |
| 11 | Loss curves | train/val comparable, no divergence |
| 12 | Metrics files | node, overall, severity CSVs with mean ± std |
| 13 | IG completeness | gap ≈ 0; shares sum to 1 |
| 14 | Explanation validity | deletion test beats random |
| 15 | Dashboard | all tabs render on real windows |

---

## 6. Schedule (≈ 42–56 h, 8–10 working days)

| Day | Work |
|---|---|
| 1 | Phase 0 |
| 2 | Phase 1 (Gate 1) |
| 3 | Phase 2 |
| 4 | Phase 3; epoch timing; launch Phase 4 on Colab |
| 5 | Phase 4 completes; Phase 5 |
| 6 | Phase 6 |
| 7–8 | Phase 7 |
| 9 | Phase 8 |
| 10 | Buffer / clean-run |

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| Reviewer scripts crash on real nulls (5%) | Phase 0/2 fixes; Gate 2 real-file check |
| Windows cross 428-day gaps / long ffills | Segmentation + `ffill(limit=5)` |
| Persistence unbeatable | Horizon chosen at Gate 1; claims table |
| Graph adds nothing (probe: weak, lag-free cross-corr) | RQ2 reframed; ablation is the result |
| Explanations dominated by persistence path | Δ-attribution + deletion test |
| Colab disconnects mid-training | Per-run checkpoints; skip-if-exists; re-run to resume |
| Python/wheel differences Colab vs local | `requirements.lock`; portable `state_dict` checkpoints |
| Training slow | Epoch timing gate; `MAX_SAMPLES`/epochs knobs; Colab |
| Paper critique wrong | Verify numbers in the PDF in Phase 1 |

---

## Status

✅ **All decisions resolved.** Repository location is managed directly by the user (no active OneDrive sync in effect). Plan A is 100% frozen and ready for execution starting with Phase 0.
