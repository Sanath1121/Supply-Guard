# SupplyGuard — Master Tech Stack

**Status:** Consolidated master tech stack. Supersedes `../PROJECT_TECHSTACK.md` (v1) and `Review/response_2-PROJECT_TECHSTACK.md` (v2).
**Date:** 2026-10-05
**Inputs read:** root (`NEW_AIM.md`, `final_implementation_plan.md`), `Review/` (Response 1 review; Response 2 text + 7 code artifacts + tech stack; Response 3 text + plan + `train.py`/`evaluate.py`/`setup_and_download.py`; final response). `old/` treated as superseded history.

> [!WARNING]
> **Input gap:** there is no tech-stack file in the root folder. I used `mini project/PROJECT_TECHSTACK.md` (the file Response 1 reviewed) as the "current tech stack".

---

## 0. Facts checked against the real dataset (2026-10-05)

The reviews disagree on several facts and nobody ran their checks on the real file, so I ran a one-off probe (`scratch/data_probe2.py`) against both mirror CSVs. These results override any claim in the documents.

| Fact | Claimed by | Actual | Consequence |
|---|---|---|---|
| `_test.csv` `RI_Distributor1` is 100% null | Current plan Decision 4, Response 3 | **False: 7.3% null.** Span 2016-12-08 → 2017-08-28 | The "train only" rule can stay for simplicity, but the stated reason has to change |
| `_train.csv` is "650k clean 2018 records" | Current plan | **649,999 rows, span 2015-01-28 → 2018-12-19, not sorted as stored**, 2,363 duplicate stamps | Sorting and deduplication are mandatory (Response 2 already does this) |
| Nulls < 1% | Response 2 `dataset.py` (raises error), Response 3 setup (assert) | **`RI_Distributor1` 5.13%, `Total_Cost` 5.46%**; Distributor null runs up to **5,860 consecutive rows** | 🔴 **Both reviewer scripts will crash on real data.** And a blind `ffill` would create flat segments days long, which inflates persistence |
| Sampling interval | unknown | **Median 2 min**; maximum gap **428 days** | Windows must not cross gaps, so segment-aware windowing is needed (no review covers this) |
| Timestamp format | unstated | `MM/DD/YYYY hh:mm:ss AM/PM`: parses with 0 failures in month-first form | Lock `format="%m/%d/%Y %I:%M:%S %p"` and never let pandas infer it |
| Persistence strength (raw step, H=1) | Response 1 predicted R² ≈ 0.99 | **R² 0.946 (S), 0.991 (M), 0.983 (D), 0.986 (R)**; H=10: 0.77–0.94 | At H=1 the task is close to trivial. The horizon must be chosen in EDA (minutes, not rows) |
| Cross-echelon structure | Assumed by all plans | **corr(S,M)=0.03, corr(M,D)=0.29, corr(D,R)=0.08**, flat across lags 0–3 (no lead/lag) | The graph premise is **weak**. Expect RQ2 ("graph beats LSTM") to come out negative, so the plan must be built to survive that |
| `SCMstability_category` | Plan v1: 3 tiers | 5 classes, imbalanced (86k/176k/251k/110k/27k) | It is a single global label, not per node, so it cannot directly define per-node tiers |

*These probe numbers are indicative: they were computed on raw, sorted, forward-filled data that still crosses gaps. Phase 1 EDA recomputes them properly.*

---

## 1. Project aim (for confirmation)

SupplyGuard re-implements the IEEE ICCMC 2025 hybrid GCN+LSTM as a paper-style baseline. It then extends it into a node-level spatio-temporal model that forecasts next-step risk separately for Supplier, Manufacturer, Distributor and Retailer, and derives overall TRI as their mean. Each forecast is explained with validated gradient attributions, and everything is shown in a Streamlit dashboard. The contribution is honest evaluation: beating persistence, measuring whether the assumed graph helps, and explanations that pass validity tests. It is not about chasing the paper's numbers.

---

## 2. Analysis of the current plan (v1) and tech stack (v1)

**Strengths (kept):** chronological 80:10:10 split with a train-only scaler; a true t+1 target; columns selected by name, which catches the S-D-M-R order trap; a clear Plan A / Plan B split with explicit exclusions; a hand-written GCN; regression metrics instead of the "88%"; a real-test-window UI with no fake sliders; dual models (paper-style plus node-level).

**Gaps and risks (fixed in this master):**
1. The graph branch adds no information. `graph_x` equals the last row of `seq`, so GCN ≈ re-mixed LSTM input, and attribution is double-counted. *(Response 1, correct)*
2. There is no persistence baseline, even though the probe shows persistence R² of 0.95–0.99.
3. EDA comes last, yet the probe shows the cross-node structure is weak. EDA has to gate the work.
4. There is no timestamp parsing or sorting, and `MAX_SAMPLES` takes the first 50k rows (≈ 69 days of a 4-year series).
5. A symmetric-only adjacency contradicts the "upstream propagation" explanations.
6. Input×Gradient with `.abs()` and a zero baseline in scaled space; "root cause" wording; hard-coded explanation text.
7. Arbitrary 0.35/0.65 tiers; train loss ≠ val loss (extra TRI term); no seeds; scaler not saved; `GRAD_CLIP`/`edge_index` unused; only 2 models evaluated; "<2 min" unmeasured.
8. Wrong facts: "100% null test file", "normalized Laplacian" (it is the renormalised adjacency), "PyG can't install" (PyG ≥ 2.3 core is pure Python).
9. Tech stack v1 is stale: it describes TRI-same-step, quantile "88%", sliders, and "~10k rows".
10. **Not raised by any review:** time gaps (up to 428 days), multi-day null runs, a timestamp-format ambiguity, and the diagrams/PPT titled *"Attention-Based … Classification … Real-Time"*, which does not match this plan.

---

## MASTER TECH STACK

| Layer / component | Chosen technology | Source | Reason |
|---|---|---|---|
| Runtime | Python **3.14 first** (15-min install timebox) → fall back to **3.12 venv** if `statsmodels`/`mlflow`/`scipy` wheels fail | merged (current + R2/R3) | 3.14 is already installed; the fallback removes the wheel risk without dictating it |
| Env reproducibility | `venv` + `requirements.txt` + `pip freeze > requirements.lock` | R2/R3 | Exact versions for the viva re-run |
| Version control | `git` (commit before every agent session) | R1/R3/final | Rollback; audit trail of agent edits |
| DL framework | PyTorch 2.x (CPU) | current | Installed; autograd needed for IG |
| Graph layer | Hand-written `GraphConv` with **`symmetric`** (Kipf–Welling renormalised adjacency) and **`directed`** (separate `A_down`/`A_up` weights) modes | R2 | Directed mode is needed to talk about upstream propagation. Justified on transparency and the 4-node size, **not** on "PyG can't install" |
| Proposed model | `STGCNLSTM`: GCN at every time step → shared 2-layer LSTM per node → shared head, `RESIDUAL=True` (`ŷ = y_t + Δ`) | R2 | Gives each node its own temporal embedding, so the graph genuinely participates. Residual starts training at persistence |
| Paper-style baseline | `PaperHybridOverall` (1 GCN → mean pool ‖ LSTM → scalar TRI) | current Model A, R2 code | Paper fidelity. Renamed "re-implementation", not "reproduction" |
| Other baselines | Persistence; **Ridge-AR(10)** (multivariate linear AR on flattened windows); `LSTMBaseline` (graph-free) | R2/R3 | Persistence is mandatory given R² 0.95+. Ridge-AR replaces the ARIMA/AR10 fallback with no extra dependency. LSTM is the RQ2 ablation |
| Data pipeline | pandas, NumPy (`sliding_window_view`), scikit-learn `MinMaxScaler` (train-only), `joblib` scaler save | R2 + master fixes | Leak-free; scaler saved for raw-unit display. **Plus segment-aware windows and the null policy** |
| Metrics | `sklearn.metrics`: MSE/MAE/RMSE/R² per node + derived TRI, scaled **and** raw, mean ± std over seeds; per-node tercile severity tiers + confusion matrix + macro-F1 | R3 (modified) | Defined, defensible. Tiers are per node because `SCMstability_category` is global |
| Explainability | Integrated Gradients (64 steps, training-mean baseline, signed, completeness gap) on **both** the full forecast and the **residual Δ**; deletion test + stability check | R2 + R3 + master | IG fixes the baseline and sign issues. Δ-attribution is a master addition |
| Statistics / EDA | `statsmodels` (Granger, ADF, ACF); VAR optional | R1/R2 | Evidence for or against the assumed graph |
| Experiment tracking | MLflow (**optional**, auto-skipped if import fails) | current + R3 | Viva value, but must never block training |
| Plots | Matplotlib, Seaborn, NetworkX, Plotly | current | Already installed / standard |
| UI | Streamlit + Plotly; `st.cache_resource`; real test windows only | current + R3 rules | Pure Python; no fabricated numbers |
| Tests | `tests/smoke_test.py` (synthetic data with a known S→M→D→R lag) **plus a real-data spot check** | R2 + master | Acceptance check after each change |
| Data source | Mendeley Data V2 (CC BY 4.0) as the citation; GitHub mirror **pinned to a commit hash + sha256** for download | merged | Reproducible; correct attribution |
| Agent guard rails | `AGENTS.md` at the repo root (supported by Antigravity, alongside `GEMINI.md` / `.agents/rules/*.md`) | final response | Stops agents re-implementing archived designs |
| Excluded | PyG, Docker, Kafka/Spark, IoT, news, imagery, React/FastAPI, Transformers in Plan A | current | Scope control |

`requirements.txt`: `torch, numpy, pandas, scikit-learn, joblib, statsmodels, networkx, matplotlib, seaborn, plotly, streamlit, requests, mlflow` (mlflow optional).

---

## Resolved decisions (2026-10-05)

| # | Question | Your answer | Effect |
|---|---|---|---|
| 1 | Which data files | Banerjee dataset is enough → **train CSV only** | Test CSV unused; Plan A L1 |
| 2 | Core spatial model | **2-layer GCN**; GAT moved to Plan B (B2) | Plan A L2 |
| 3 | Horizon | **10 minutes** (fallback 20, locked at Gate 1) | Plan A L3 |
| 4 | Seeds / compute | **5 seeds**; **Google Colab** for training | Plan A L4/L5; checkpoint-resume added |
| 5 | Plan split | Plan A and Plan B in **separate files** | `PLAN_A_IMPLEMENTATION_PLAN.md`, `PLAN_B_IMPLEMENTATION_PLAN.md` |

## Status

✅ **All decisions resolved.** Repository location is managed directly by the user (no active OneDrive sync in effect). The master tech stack is 100% frozen and ready for execution.
