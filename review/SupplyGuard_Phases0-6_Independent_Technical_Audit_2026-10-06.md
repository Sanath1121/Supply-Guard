# SupplyGuard — Independent Senior Technical Audit (Phases 0–6)
**Audit date:** 2026-10-06  
**Repository:** `Sanath1121/Supply-Guard`  
**Audited commit:** `68390f18e1a87417ca5044a558bccc7a282dbb92` (`68390f1`, *feat(phase-6): complete and harden Phase 6 explainability engine, test suite, and report*)  
**Scope:** Plan A Phases 0–6. Phase 7 used only for forward handoff/readiness checks; Phase 7 is not audited as completed work.  
**Method:** GitHub connector, because a direct `git clone` in this environment failed with DNS resolution (`Could not resolve host: github.com`). No remote mutations were performed.

## 1. Input and execution status

Two source-input limitations were identified before the audit: there is no standalone `PHASE_7.md` upload in the supplied files, and no project ZIP is attached. The repository itself also contains no ZIP archive at the audited commit, so exact ZIP-vs-repo parity is **not verifiable**. I therefore use the Phase 7 forward-looking handoff document in the repository, but do not treat Phase 7 as implemented.

Dependency installation was attempted from the current `requirements.txt`; it could not complete because this environment cannot reach PyPI/DNS, failing on `streamlit>=1.25.0`. Existing packages were inspected, but the exact repository environment was not installed. Consequently, I did **not** claim a fresh full-suite execution; historical test outputs in the project's own reports are treated as evidence to reconcile, not as an independent rerun.

The audited commit is materially later than the Colab training execution commit recorded in the repository (`3d90be6`). The checkpoints and result artifacts are therefore evaluated as artifacts produced by the training run and subsequently carried into commit `68390f1`; current code compatibility is checked statically and through the repository's documented verification results.

## 2. Step 1 — Understanding (7 lines)

1. SupplyGuard aims to forecast supply-chain risk at Supplier/Manufacturer/Distributor/Retailer echelons using a spatio-temporal GCN+LSTM model and to provide interpretable, offline historical replay rather than a proven live-streaming system.
2. Phase 0 established the repository, data acquisition, guardrails, dependency definitions, and the pinned Banerjee/Mendeley dataset workflow.
3. Phase 1 performed EDA, persistence/horizon analysis, cross-echelon correlation/Granger analysis, and a paper-method audit, locking a nominal 10-minute (5-step) horizon.
4. Phase 2 hardened parsing, deduplication, gap segmentation, bounded forward fill, chronological target-safe splitting, and train-only MinMax scaling.
5. Phase 3 implemented native PyTorch graph convolutions, symmetric/directed ST-GCN-LSTM variants, an LSTM baseline, a paper-style scalar TRI hybrid, and parameter counts.
6. Phase 4 trained 4 architectures × 5 seeds on an A100, while Phase 5 evaluated persistence/Ridge and the learned models on the test set; Phase 6 added Integrated Gradients, Δ-attribution, deletion checks, stability checks, upstream-share summaries, and narratives.
7. The latest supplied progress report claims Phases 0–6 are complete and verified, with 37/37 gate checks, all 20 checkpoints verified, and a graph-structure headline claim; this audit finds that several numerical artifacts are real and reproducible-looking, but some scientific conclusions, report statements, and the Phase 7 handoff remain overstated or inconsistent.

## 3. Overall verdict

**READY WITH FIXES — not ready for a results-backed Phase 7 handoff yet.** The core data pipeline is substantially sound: chronological splitting, train-only scaling, gap-aware segmentation, checkpoint loading, and the main regression artifacts are internally coherent, and many earlier engineering defects were genuinely fixed. The main blocking issue is scientific rather than infrastructural: the current `lstm` baseline is not a parameter-matched no-graph version of `STGCNLSTM`, so the Phase 5 statement that the graph itself improves accuracy is not causally identified. Phase 6 also contains semantic errors in the explanation narrative and a weakly validated notion of “upstream contribution,” while the Phase 7 handoff contains incorrect model/checkpoint/API examples; these should be corrected before implementation proceeds.

## 4. Phase-by-phase scorecard

| Phase | Plan compliance | Code quality | Testing | Report accuracy | Audit verdict |
|---|---:|---:|---:|---:|---|
| 0 | 4/5 | 4/5 | 3/5 | 3/5 | Mostly complete; documentation/version drift remains |
| 1 | 3/5 | 4/5 | 2/5 | 3/5 | Analysis exists, but gate tests are not strong enough and some reported EDA claims are weakly evidenced |
| 2 | 4/5 | 4/5 | 3/5 | 3/5 | Pipeline implementation is good; horizon semantics and report details need correction |
| 3 | 3/5 | 4/5 | 4/5 | 2/5 | Core models work; several report claims are false/stale |
| 4 | 4/5 | 4/5 | 4/5 | 3.5/5 | Training artifacts are present and consistent, but reproducibility/provenance is incomplete |
| 5 | 4/5 | 4/5 | 3.5/5 | 2.5/5 | Metrics are useful, but graph-specific inference and some report artifacts are not trustworthy as stated |
| 6 | 3.5/5 | 4/5 | 3/5 | 2.5/5 | Explainability engine exists, but validation and narrative semantics need correction |

## 5. Plan compliance audit

### Phase 0 — Setup / provenance / Gate 0
**Status: PARTIAL.** The repository structure, `AGENTS.md`, `requirements.txt`, `requirements.lock`, dataset download script and Guardrails exist. The current `setup_and_download.py` now contains an actual checksum comparison/halt path, resolving the earlier defect. However, the phase documentation still contains historical claims such as repository initialization on `master` while the repository uses `main`, and the current `.gitignore` no longer matches the older Phase 0 description because important output artifacts and checkpoints are now unignored/tracked. The lock file is clean UTF-8 but is not the exact Colab training environment because training used unpinned `requirements.txt` and the Colab PyTorch version was recorded as unknown (`requirements.lock`, header; `MASTER_TECHSTACK.md`).

### Phase 1 — EDA / go-no-go
**Status: PARTIAL.** Required ACF, cross-correlation, persistence and Granger analyses exist in `notebooks/01_EDA.ipynb` and the Phase 1 documentation. The plan also requires `SCMstability_category` association and verification of the paper's quoted metrics before critique; the category association is not clearly evidenced in the Phase 1 report, and some paper-audit assertions were historically stronger than the available evidence. Current Phase 1 tests are not a faithful executable gate over the real EDA pipeline: several tests calculate metrics using helper functions defined inside the test itself and use synthetic data rather than asserting notebook-derived values (`tests/test_phase1_eda.py`).

### Phase 2 — Data pipeline hardening
**Status: MOSTLY DONE.** Production code implements stable sorting/deduplication, gap-aware segmentation at `GAP_MAX_MIN=6`, `ffill(limit=5)`, dropping remaining NaNs, chronological partitioning, train-only MinMaxScaler, target-safe split masks, and context borrowing (`src/dataset.py`). Adversarial suites now call production ingestion APIs, fixing the earlier import/dead-code problem. Remaining issues: the configured `HORIZON=5` is a step count while actual elapsed target times are not fixed at 10 minutes for every window, and the long-gap null policy partially forward-fills runs before dropping later rows.

### Phase 3 — Models and baselines
**Status: PARTIAL / CORE DONE.** Native `GraphConv` supports symmetric and directed modes; `STGCNLSTM`, `LSTMBaseline`, and `PaperHybridOverall` are implemented; the 2-hop graph-gradient behavior and parameter counts are present. The current parameter artifact is 53,668 / 57,793 / 63,937 / 61,761 parameters for LSTM / paper hybrid / directed / symmetric (`outputs/results/param_counts.csv`). The Phase 3 report's claims about `hidden_dim` aliasing and 32-dim LSTM configuration are false for the audited code; the actual LSTM hidden size is 64 (`src/config.py:39–43`, `src/models/st_gcn_lstm.py:60,74`).

### Phase 4 — Multi-seed training
**Status: DONE WITH REPRODUCIBILITY DEFECTS.** All 20 checkpoints and 20 loss histories are present in the audited repository, together with `training_summary.csv`. Training uses seeded Python/NumPy/PyTorch, Adam, early stopping and gradient clipping. The checkpoint format is the raw `state_dict`, not a wrapper dictionary. The training run provenance is only partially reproducible because the actual Colab PyTorch version was unknown and training used an unpinned requirements file; the current `requirements.lock` is a later environment snapshot, not the exact training lock. The plan's explicit “time one epoch per model first” prerequisite is not documented as an executed decision gate; only final run timings are preserved.

### Phase 5 — Test evaluation and benchmark
**Status: DONE NUMERICALLY, PARTIAL SCIENTIFICALLY.** The evaluator performs batched inference, persistence and Ridge-AR(10) comparison, per-node regression metrics, severity tiers from training quantiles, confusion matrix generation, and 5-seed aggregation (`training/evaluate.py`). The outputs exist and are numerically coherent. However, the RQ2 comparison is confounded because `LSTMBaseline` and `STGCNLSTM` differ in more than the graph: the former is one LSTM over all 5 features with a joint output head, while the proposed model is a two-GCN + shared-per-node-LSTM architecture with distinct node embeddings and a shared head. Therefore, lower STGCN error cannot be attributed to graph message passing alone.

### Phase 6 — Explainability
**Status: PARTIAL.** Integrated Gradients with 64 steps, a training-mean baseline, signed attribution, Δ-attribution, deletion checks, stability checks, upstream-share calculation, and narrative generation exist. The artifact has 10 examples and 7/10 deletion-vs-random passes. The implementation does not fully satisfy the spirit of the plan's explanatory validity criterion: the stored examples are selected from the most extreme prediction adjustments in the first 2,000 test windows rather than a predeclared representative sample; `upstream_share()` is a feature-column share, not a graph-path attribution; the unit test does not enforce the plan's random-deletion criterion; and the narrative mislabels a training-mean baseline as a “window-average baseline” in Δ mode.

## 6. Test execution and coverage

### 6.1 What I could execute in this environment

I inspected the current environment:

- Python 3.13.5
- torch 2.10.0+cpu
- NumPy 2.3.5
- pandas 2.2.3
- scikit-learn 1.8.0
- joblib 1.5.3
- statsmodels 0.14.6
- networkx 3.6.1
- matplotlib 3.10.8
- seaborn 0.13.2
- plotly 6.5.2
- requests 2.32.5
- **streamlit missing**
- **mlflow missing**

A fresh `pip install -r requirements.txt` was attempted. It failed because DNS/network access to PyPI is unavailable, specifically while resolving `streamlit>=1.25.0`. No repository mutation was performed.

Because the repository was not cloned locally and the current environment could not install the complete dependency set, I did **not** execute `python tests/run_phase_tests.py --all` against the live repository. The test results quoted in the project's reports are therefore historical/recorded evidence rather than an independent fresh run.

### 6.2 Coverage quality

| Area | Covered | Main limitation |
|---|---|---|
| Phase 0 setup | Yes | Some tests are simple environment/structure checks |
| Phase 1 EDA | Partial | Several tests validate local helper logic rather than the executed EDA notebook |
| Phase 2 data pipeline | Yes, including adversarial production-path tests | Manual full real-file target spot-check remains limited in the archived evidence |
| Phase 3 models | Yes | Parameter/shape/gradient checks are useful; scientific performance is not tested here |
| Phase 4 training | Yes | Some tests validate training behavior on synthetic datasets rather than the complete production grid |
| Phase 5 evaluation | Yes | Important claims logic is tested, but artifact identity/provenance checks are weak |
| Phase 6 explainability | Partial | The core IG math is tested, but selection bias, random deletion, real-window coverage, and upstream semantics are under-tested |
| Phase 7 | Not implemented | Test module is intentionally skipped and no `app/` implementation exists |

The current Gate runner is correctly hardened against the old false-positive problem: when all tests are skipped it emits `[SKIPPED / UNIMPLEMENTED]` and exits nonzero (`tests/run_phase_tests.py`). That earlier defect is genuinely fixed.

## 7. Training and evaluation artifacts

### 7.1 Stored training results

`outputs/results/training_summary.csv` contains all 20 architecture/seed combinations. The key validation means are:

- LSTM: about `0.000941 ± 0.000014`
- Paper hybrid: about `0.000249 ± 0.000004` (scalar TRI target)
- ST-GCN symmetric: `0.001145 ± 0.000252`, with seeds 43 and 44 at about `0.001421`, i.e. near persistence
- ST-GCN directed: about `0.000904 ± 0.000002`

The two symmetric plateaus are real in the stored loss histories and were also analyzed in `scripts/diagnose_sym_collapse.py`. Treating them as ordinary random-seed variation is too generous; they are clearly optimization failures/plateaus under this initialization and patience configuration.

### 7.2 Test regression metrics

Current `outputs/results/overall_metrics.csv` reports:

| Model | Test MSE | MSE std | R² | Improvement vs persistence |
|---|---:|---:|---:|---:|
| Persistence | 0.000396623 | — | 0.96131 | 0.00% |
| Ridge-AR(10) | 0.000353731 | — | 0.96550 | 10.81% |
| LSTM | 0.000283693 | 0.000003233 | 0.97233 | 28.47% |
| Paper hybrid | 0.000287795 | 0.000008263 | 0.97193 | 27.44% |
| ST-GCN directed | 0.000273027 | 0.000002715 | 0.97337 | 31.16% |
| ST-GCN symmetric | 0.000330122 | 0.000060856 | 0.96780 | 16.77% |

These are internally sensible as regression numbers. The directed model is numerically best on derived TRI, but the gap to LSTM is small (`~1.1e-5` MSE).

### 7.3 Severity classification tells a different story

`outputs/results/severity_metrics.csv` shows persistence is **best** on the 3-tier classification task:

- Persistence accuracy `0.8362`, macro-F1 `0.8431`
- LSTM accuracy `0.8047`, macro-F1 `0.8114`
- Directed ST-GCN accuracy `0.8130`, macro-F1 `0.8188`
- Symmetric ST-GCN accuracy `0.8193`, macro-F1 `0.8250`

Therefore the project should not present the learned forecaster as clearly superior for severity-alert classification. It is better on continuous regression metrics, while persistence is better on the current tiered alert metric.

### 7.4 Confusion-matrix artifact mismatch

The tracked `outputs/results/confusion_matrix.csv` is:

```text
           Pred_Low  Pred_Med  Pred_High
Actual_Low     48359      5155        551
Actual_Med      4692     71284      10314
Actual_High      528     15460      75157
```

But the supplied Phase 5 report's seed-42 matrix is a different matrix (same total 231,500, different cell counts). The current evaluator is written to choose the directed-model seed 42 confusion matrix (`training/evaluate.py:371–374`). The artifact therefore does not match the report's stated seed-42 contents. The right fix is to regenerate the artifact and record which exact run produced it, rather than trying to infer which historical version is correct.

### 7.5 Checkpoint integrity

`outputs/models/checkpoints.sha256` contains the 20 checkpoints, scaler and tracked result CSVs. `outputs/results/attribution_examples.csv` is **not** in the manifest, nor are the PNG figures. The project's checkpoint verifier also exists (`scripts/verify_checkpoints.py`) and the remediation report records 20/20 validation-loss matches. That makes the learned weights plausibly intact, but the audit trail should cover the complete Phase 6 artifact set as well.

## 8. Major correctness and scientific issues

### MAJOR-1 — RQ2 graph advantage is confounded
**Location:** `src/models/st_gcn_lstm.py:79–106`, `PHASE_5.md` lines 20–26, `training/evaluate.py:200–243`  
**Verified.** `LSTMBaseline` is a single LSTM over all 5 input features with a joint 4-output head; `STGCNLSTM` first applies two graph-convolution layers at every time step, adds node embeddings, then reshapes to a per-node sequence and runs a shared LSTM/head. The models therefore differ in architecture, parameterization, and representation, not just graph message passing.  
**Why it matters:** The statement “graph structure improves…” is not isolated by the current ablation.  
**Fix:** Add a graph-free control with the *same STGCNLSTM architecture* but identity adjacency / no message passing, train the same seeds, and compare it directly against the graph-enabled version.

### MAJOR-2 — Phase 6 narrative uses the wrong baseline and wrong quantity
**Location:** `src/explainability.py:78–88`, `outputs/results/attribution_examples.csv`  
**Verified.** The explanation baseline is `self.baseline`, described as the training-mean feature vector. The narrative says “window-average baseline.” Worse, `scripts/generate_attributions.py` calls `narrate()` on a **Δ-attribution result**, where `predicted_risk = f(x) - x[-1,target]`, so the text “Forecast Retailer risk = 0.22” is actually describing a residual adjustment, not the raw forecast.  
**Fix:** Rename the baseline to “training-mean reference”; in Δ mode say “Predicted adjustment from persistence = …” or otherwise keep raw forecast and residual adjustment as separate fields.

### MAJOR-3 — `upstream_share()` is not graph/path attribution
**Location:** `src/explainability.py:72–75`, `PHASE_6.md` sections 4.2–4.3  
**Verified.** For target node index `t`, `upstream_share` is simply `sum(feature_importance[:t])`. It does not inspect graph edges, graph weights, path contributions, or message-passing internals. It is therefore a share of attribution assigned to earlier echelon feature columns.  
**Fix:** Rename it to something explicit such as `upstream_feature_share`, or implement a graph/path attribution method before claiming “upstream diffusion” or “propagation.”

### MAJOR-4 — Phase 6 windows are selected using the model output itself
**Location:** `scripts/generate_attributions.py`  
**Verified.** The script searches the first 2,000 test windows and selects the 10 largest absolute prediction adjustments. This is useful for illustrative case studies, but it is a selected/extreme sample, not a neutral validation sample.  
**Fix:** Keep the extreme examples as a “case study” artifact, but add a separately sampled or stratified attribution-validation set for the deletion/stability pass rate.

### MAJOR-5 — Phase 6 unit test does not validate the planned random-deletion criterion
**Location:** `tests/test_phase6_explainability.py:59–83`  
**Verified.** The unit test compares deletion of the top feature against deletion of the *least*-important feature. The Plan A Gate 6 criterion is top-attributed feature vs a random feature. The production artifact script does perform top-vs-random, but the unit test is not testing that contractual rule.  
**Fix:** Make the test perform the same top-vs-random protocol as the production validation and assert the required pass rate on a fixed/derived sample.

### MAJOR-6 — Phase 6 test can silently fall back to an untrained random model
**Location:** `tests/test_phase6_explainability.py:29–37`  
**Verified.** The checkpoint is loaded only if the file exists. If absent, the test keeps the freshly instantiated model and continues. This converts a missing production checkpoint from a hard failure into a different experiment.  
**Fix:** Require the checkpoint to exist and fail explicitly if it does not; derive the baseline from the production dataset instead of hardcoding its five values.

### MAJOR-7 — Evaluation scaler mismatch does not fail closed
**Location:** `training/evaluate.py:252–261`  
**Verified.** Evaluation builds a new train-only scaler and then attempts to replace it with the saved production scaler only if `data_max_` matches. If the saved scaler differs, evaluation silently continues using the newly fitted scaler.  
**Fix:** Require the saved scaler to exist, compare all scaler parameters (`min_`, `scale_`, `data_min_`, `data_max_`, feature range) and raise on mismatch.

### MAJOR-8 — Phase 5 “significance” is an informal one-standard-deviation heuristic
**Location:** `training/evaluate.py:211–224`, `PHASE_5.md:20–23`  
**Verified.** The headline logic is `difference > std_lstm + std_dir`; there is no paired statistical test, bootstrap interval, or confidence interval in the implementation.  
**Fix:** Report paired per-seed differences and a bootstrap CI over test-window errors (or another pre-registered inferential method). Call the current test exactly what it is: a >1σ heuristic.

### MAJOR-9 — Phase 5 confusion matrix report does not match tracked artifact
**Location:** `outputs/results/confusion_matrix.csv`, `PHASE_5.md` lines 80–87  
**Verified.** Same total count, different cell values.  
**Fix:** Regenerate and commit one canonical matrix, then cite its producing model/seed and code commit.

### MAJOR-10 — Reproducibility lock does not reproduce the Colab training environment
**Location:** `requirements.txt`, `requirements.lock`, `notebooks/colab_train.ipynb`  
**Verified.** Training used `pip install -r requirements.txt` where `torch>=2.0.0` and other dependencies are unpinned. The lock lists `torch==2.12.0`, but its header explicitly says the Colab PyTorch version is unknown; the lock is a later environment snapshot, not the exact run environment.  
**Fix:** Record Python version, exact torch/CUDA build, GPU, package lock, training script SHA, dataset SHA, config snapshot and run metadata in a single machine-readable manifest.

## 9. Additional minor issues

1. **Horizon semantics are still ambiguous.** `src/config.py:25–27` calls H=5 a nominal 10-minute horizon, but previous real-data probing recorded a wider elapsed-time range (about 4–22 minutes, median 10). Either resample to a regular 2-minute grid, select targets by elapsed time, or label the result “5 steps, median 10.0 min.”
2. **`HORIZON=5` is manually duplicated.** It should be derived from `HORIZON_MIN / CADENCE_MIN` or validated with an assertion to avoid configuration drift.
3. **Phase 1 test quality.** `tests/test_phase1_eda.py` validates local helper functions rather than the actual executed notebook outputs, so it cannot catch a notebook/report divergence.
4. **Phase 1 Granger reporting is too qualitative.** “Minimal Granger predictive causality” is not accompanied by the F-statistics/p-values needed to support that wording.
5. **Phase 1 cross-correlation numbers differ across documents.** Older/current project documents mix lag-0 and peak-lag values; use one executed source of truth and label whether each number is lag-0 or max-|r|.
6. **Phase 3 report stale terminology.** The plan calls Phase 3 models/baselines; one phase report calls it graph-topology implementation.
7. **`.gitignore` vs tech-stack policy is inconsistent.** `AGENTS.md` / `MASTER_TECHSTACK.md` say `.pt` weights remain git-ignored, but the audited commit explicitly unignores and tracks `!outputs/models/*.pt`.
8. **`attribution_examples.csv` is tracked by accident rather than by explicit whitelist.** It is not listed in the current `.gitignore` whitelist, making future local modifications vulnerable to being ignored.
9. **`checkpoints.sha256.bak` is tracked but not represented in the manifest itself.** Not dangerous, but it complicates the notion of a complete cryptographic artifact set.
10. **No formal deterministic CUDA setting.** Seeds are set, but `torch.use_deterministic_algorithms()` / backend settings are not explicitly controlled, so bit-for-bit reproducibility on GPU is not guaranteed.
11. **No epoch-level resume.** `train.py` skips a run when the checkpoint/loss history exists; it does not resume optimizer state/epoch state from a partial run. The project should describe this as “complete-run resume/skip,” not mid-epoch fault tolerance.
12. **Phase 4 speed claim should be framed carefully.** The 2.33h cumulative A100 wall time is real in the recorded summary; the older T4 “under 20 min” type claim was not measured in the codebase and should not be revived.

## 10. Report discrepancies — claim vs reality

| Source | Claim | Reality at audited commit |
|---|---|---|
| `PHASE_0.md` | Repository initialized on `master` | Repo is `main`; historical statement |
| `PHASE_0.md` | Outputs are excluded by `.gitignore` | Current `.gitignore` selectively tracks checkpoints, scaler, result CSVs and figures |
| `PHASE_0.md` | Phase 0 foundation is “100% stable” | Strong foundation, but environment is not fully pinned and docs have drift |
| `PHASE_1.md` | ACF CI = ±0.0276 supports the displayed full-data analysis | Requires notebook provenance; value looks like a subsample-size-based band and should be explicitly justified |
| `PHASE_1.md` | Adjacent correlations are 0.0272 / 0.2134 / -0.0160 while other documents use ~0.04 / 0.32 / 0.27 | Documents are mixing different correlation definitions and/or data subsets; one canonical executed table is needed |
| `PHASE_1.md` | “Minimal Granger predictive causality” | No p-value/F-statistic table is included in the report to support the claim |
| `PHASE_2.md` | Gate 2 is 5/5 | Current Gate 2 test module is a different 4-test structure; report is stale on count |
| `PHASE_2.md` | 57,400 rows were dropped due to “catastrophic gaps…” | Earlier independent accounting separated duplicate, NaN and short-segment losses; gap detection itself does not directly drop rows |
| `PHASE_3.md` | Hidden-dimension aliasing exists | No `hidden_dim_lstm` / `hidden_dim_gcn` aliases are present; actual config uses 64-dim LSTM |
| `PHASE_3.md` | Directed model extra-parameter formula includes biases | Extra 2,176 count is correct, but the written decomposition is not: directed down/up layers are bias-free |
| `PHASE_3.md` | All 9 phase gates passed at the time | Future gates later became explicitly skipped/unimplemented; historical wording should be scoped to gate stubs, not treated as evidence of completed phases |
| `PHASE_5.md` | Directed ST-GCN “proves” graph message passing adds value | Current baseline is not parameter/architecture matched; this does not isolate the graph effect |
| `PHASE_5.md` | “Significantly” outperforms symmetric | No formal significance test; only mean/std comparison plus a simple MSE ordering rule |
| `PHASE_5.md` | Seed-42 confusion matrix shown in report | Tracked `confusion_matrix.csv` contains different cell values |
| `PHASE_5.md` | MAE standard deviations as printed in the report | They do not match current `overall_metrics.csv`; several are substantially different |
| `PHASE_6.md` | “window-average baseline” | Actual baseline is the training-mean feature vector |
| `PHASE_6.md` | 70% deletion pass validates explainability | The artifact supports a 7/10 top-vs-random pass rate, but windows were model-selected extremes and the unit test uses top-vs-min instead of top-vs-random |
| `PHASE_6.md` | “true propagated disruptions” / directional diffusion conclusions | Current `upstream_share` is simply earlier-column attribution share, not path-level graph attribution |
| `PHASE_6.md` | Narrative example is a risk forecast | In Δ mode the reported quantity is a residual adjustment from persistence |
| `PHASE_6.md` | Some mechanistic explanations (“because the network couples…”, etc.) are established | They are interpretations not demonstrated by a causal/mechanistic experiment in the audited code |
| `PROJECT_PROGRESS_AND_EVALUATION_REPORT.md` | Mean±std provides “publication-grade confidence intervals” | Mean±std over five seeds are not confidence intervals |
| Progress report | Severity model performance supports alert superiority | Persistence has the highest severity accuracy and macro-F1 in the tracked results |
| Progress report | 37/37 gates and full verification establish Phase 0–6 readiness | Several tests are synthetic/self-referential, the current environment could not rerun the suite, and Phase 6/API handoff defects remain |

## 11. Cross-phase issues

### A. The scientific unit of comparison changes across phases
Phase 3’s proposed model is node-level; Phase 5’s headline TRI is a mean of node predictions. The paper-style model directly predicts scalar TRI. Those are legitimate artifacts, but they are not the same target definition. The reports must keep the target spaces explicit and never present the scalar paper model's smaller MSE as a direct “better model” comparison against node MSE.

### B. Persistence is both the baseline and a structural component of the proposed model
The residual head explicitly starts from `y_t` (`src/models/st_gcn_lstm.py:75–76`). This is a sensible engineering choice for a highly persistent dataset, but it also means explainability must carefully distinguish “what moves the forecast away from persistence” from “what explains the forecast.” Phase 6's Δ-attribution is the right conceptual direction, but the current narrative semantics do not consistently preserve that distinction.

### C. The project's main claim depends on a comparison that was never isolated
Phase 5 concludes that graph structure improves forecasting, but the baseline comparison is not a graph-only ablation. This is the single most important end-to-end validity issue because it affects the interpretation of RQ2, the progress report, and the dashboard headline.

### D. Artifact provenance is weaker than artifact integrity
The checkpoints are hashed and validation-loss verification exists, which is good. But a hash says “this byte stream is unchanged,” not “this is the exact result generated by the exact code/environment/config described by the paper/report.” A model manifest should include dataset SHA, code commit, config, Python, PyTorch/CUDA and training command.

### E. Phase 7 is not yet technically wired to Phase 6
The handoff document contains examples that cannot run against the audited code. This is a real interface contract defect, not a Phase 7 implementation defect.

## 12. Earlier-review fixes — what actually stuck

The current commit genuinely resolves several previously reported issues:

- **Checksum enforcement:** fixed in `setup_and_download.py`.
- **Smoke-test scaler/checkpoint safety:** fixed; current smoke uses temporary paths and `save_scaler=False`.
- **All-skipped gate false positives:** fixed in `tests/run_phase_tests.py`.
- **Broken Phase 2 adversarial imports:** fixed; suites now use production ingestion APIs.
- **Model mode selection:** `build_model()` now maps `st_gcn_lstm_sym` and `st_gcn_lstm_dir` correctly.
- **Sigmoid/non-residual issue:** removed from the audited `STGCNLSTM` implementation.
- **Checkpoint integrity verification:** new `scripts/verify_checkpoints.py` exists and the remediation report records 20/20 validation matches.
- **Cryptographic manifest:** present and tracks 20 checkpoints + scaler + result/loss artifacts.

Issues that remain despite earlier “100% remediated” language:

- graph-vs-LSTM ablation confounding;
- irregular 5-step wall-clock horizon;
- evaluation fail-open on scaler mismatch;
- Phase 6 selection bias and weak upstream-share semantics;
- Phase 6 unit-test mismatch with the actual random-deletion gate;
- missing real statistical confidence intervals despite later prose implying them;
- documentation/report inconsistencies including the confusion matrix and MAE standard deviations.

Therefore `review/fix_report_pre_phase5.md` and `review/POST_PHASE4_AUDIT_REMEDIATION_REPORT.md` should not be treated as proof that every prior issue is resolved; their “100% complete” language is itself stronger than the current evidence warrants.

## 13. Phase 7 readiness checklist (forward-looking only)

| Requirement for Phase 7 | Current state | Verdict |
|---|---|---|
| `app/streamlit_app.py` exists | **Missing** | BLOCKING for implementation, expected at this stage |
| Correct model name | Handoff uses `build_model("st_gcn_lstm", cfg)`; audited code uses `st_gcn_lstm_dir` / `st_gcn_lstm_sym` | **Missing / wrong example** |
| Correct checkpoint path | Handoff uses `st_gcn_lstm_directed_seed42.pt`; actual file is `st_gcn_lstm_dir_seed42.pt` | **Missing / wrong example** |
| Correct checkpoint format | Handoff expects `checkpoint["model_state_dict"]`; actual checkpoint is raw `state_dict()` | **Missing / wrong example** |
| Correct explainer argument | Handoff uses `residual_delta=...`; audited API uses `delta_mode=...` | **Missing / wrong example** |
| Explanation result schema | Handoff claims `upstream_share` inside result; audited `explain()` does not return it | **Missing / wrong example** |
| Saved production scaler | Present as tracked artifact | **Present** |
| Fail-closed scaler validation | Current evaluator can silently use a newly fitted scaler if saved scaler mismatches | **Missing** |
| Real historical replay windows | Data pipeline can produce them; no app implementation | **Backend ready, UI missing** |
| Multi-seed benchmark panel | Phase 5 result CSVs exist | **Present** |
| Severity confusion matrix | Present, but report/artifact provenance needs normalization | **Present with fix** |
| Explainability “sensitivity, not cause” wording | Engine avoids “root cause” but Phase 6 report overinterprets upstream shares | **Needs fix** |
| Gate 7 real three-window rendering | Test module intentionally skips; no app | **Missing** |
| Offline replay framing | Architecture document explicitly calls for single-process offline replay | **Present** |

**Phase 7 readiness conclusion:** the data/model artifacts are close to usable by a dashboard, but the interface contract in the current handoff document must be corrected before a frontend engineer copies the examples. Do not build against the handoff code as written.

## 14. Top 5 priority fixes before Phase 7

### 1. Fix the RQ2 experiment
Implement a parameter/architecture-matched no-message-passing STGCNLSTM control (identity graph) and retrain the same five seeds. Re-run Phase 5. Without this, the phrase “graph structure improves forecasting” should be removed or clearly labeled as an architectural comparison rather than a graph-effect finding.

### 2. Canonicalize evaluation artifacts and make evaluation fail closed
Regenerate `confusion_matrix.csv`, verify it against the report, and create a run manifest linking code commit, dataset SHA, model seeds, scaler hash and result files. Change `training/evaluate.py` to raise on scaler mismatch rather than silently recomputing.

### 3. Repair Phase 6 semantics and validation
Rename the baseline to training-mean, separate raw forecast vs persistence adjustment in `narrate()`, require real checkpoints, remove the hardcoded test baseline vector, align unit tests with top-vs-random deletion, and label `upstream_share` as feature-column attribution unless a path-aware graph attribution method is added. Then regenerate the Phase 6 artifact.

### 4. Close the reproducibility gap
Record the exact Colab Python/PyTorch/CUDA versions and a project-specific dependency lock, plus the exact training commit (`3d90be6`) and configuration snapshot. Do not claim the current `requirements.lock` reproduces the Colab run when its own header says the Colab torch version is unknown.

### 5. Fix the Phase 7 handoff and documentation source of truth
Correct the model/checkpoint/explainer APIs in `docs/PHASE_7_FRONTEND_CONTEXT_AND_STREAMLIT_DECISION.md`; keep the dashboard explicitly “offline historical replay”; align root and `docs/` progress reports; remove claims such as “publication-grade confidence intervals” and “proves graph diffusion.”

## 15. Does anything require re-running training or evaluation?

**Yes.**

- **Training rerun required:** the identity/no-message-passing control needed to answer RQ2. This is a new scientific control, so the current 20 runs cannot replace it.
- **Evaluation rerun required:** after the new ablation, and after canonicalizing the confusion matrix/scaler verification. Use the same untouched test set; do not retune on test.
- **Phase 6 attribution rerun required:** after correcting the narrative and validation protocol. Existing checkpoints can be reused for the same architectures if their hashes remain unchanged.
- **No retraining strictly required for:** merely correcting report text, checkpoint loading examples, gate logic, and scaler fail-closed behavior—provided the existing weights remain intact.

### Trust assessment of existing results

- **Data integrity:** strong. Earlier recorded runs reproduced 592,599 retained rows and 1,198 segments; the pinned dataset SHA was verified in prior audit evidence.
- **Checkpoint byte integrity:** strong. The repository contains 20 checkpoint hashes and a checkpoint-verification script; the remediation report records 20/20 validation-loss matches.
- **Current regression metrics as calculations of those checkpoints:** reasonably trustworthy.
- **Graph-specific causal interpretation:** **not trustworthy yet** because of the ablation confound.
- **Severity-alert superiority:** **not supported**; persistence is strongest on the current tiered classification metrics.
- **Phase 6 general explainability validity:** **not established** by the current 10 selected examples and tests.

## 16. Suggestions to strengthen the final results and write-up

1. Make the primary scientific table a paired, per-seed comparison: persistence vs LSTM vs matched-no-graph STGCNLSTM vs directed STGCNLSTM vs symmetric STGCNLSTM, with mean difference and confidence interval.
2. Report both regression skill and alert classification performance, and explicitly discuss the fact that persistence wins macro-F1/accuracy while learned models win continuous MSE/R².
3. Add a “target definition” box in the paper/dashboard stating that H=5 is a 5-step forecast with nominal 10-minute cadence, or fix the time-grid issue and make it truly 10 minutes.
4. Separate **case-study explanations** (extreme windows) from **explainability validation** (representative/stratified windows).
5. Replace “upstream propagation” wording with “upstream-feature attribution share” unless graph-path attribution is implemented.
6. Put the exact model/checkpoint/config/environment provenance into a machine-readable JSON manifest so any examiner can reproduce the artifact lineage.
7. Keep the project title honest: the current implementation is an offline historical replay/forecasting system, not a demonstrated real-time ingestion service.
8. For the paper comparison, explicitly label the project as an empirical implementation/extension of the cited paper rather than an exact reproduction, because split strategy, feature usage, graph implementation and evaluation protocol differ.

## 17. Open questions for the project owner

1. Do you want the final scientific claim to be **“graph structure improves accuracy”**, or are you willing to downgrade this to an architectural comparison until the identity-graph ablation is trained?
2. Should the official forecast horizon be a strict elapsed-time target (recommended), or is “5 steps / median 10 minutes” acceptable for the project scope?
3. Do you want the Phase 6 explanation to remain a lightweight sensitivity explainer, or do you want a true graph/path contribution analysis for the “upstream propagation” narrative?
4. Which artifact is the intended canonical seed-42 severity confusion matrix: the tracked CSV in commit `68390f1`, or the matrix printed in the supplied Phase 5 report?
5. Should the final project be described as a **real-time alert system** or explicitly as an **offline historical replay / risk forecasting prototype**?

## 18. Key evidence index

- **Audited commit:** `68390f18e1a87417ca5044a558bccc7a282dbb92`
- **Plan:** `PLAN_A_IMPLEMENTATION_PLAN.md`, especially Phase 0–6 sections
- **Tech stack:** `MASTER_TECHSTACK.md`
- **Guardrails:** `AGENTS.md:5–68`
- **Data pipeline:** `src/dataset.py`
- **Config:** `src/config.py:19–54`
- **Graph/model implementation:** `src/models/graph_layers.py`, `src/models/st_gcn_lstm.py:35–151`
- **Training:** `training/train.py`
- **Evaluation:** `training/evaluate.py:41–243, 252–418`
- **Explainability:** `src/explainability.py:20–92`
- **Attribution generation:** `scripts/generate_attributions.py`
- **Phase 6 tests:** `tests/test_phase6_explainability.py`
- **Gate runner:** `tests/run_phase_tests.py`
- **Stored metrics:** `outputs/results/overall_metrics.csv`, `node_metrics.csv`, `severity_metrics.csv`, `confusion_matrix.csv`, `training_summary.csv`, `attribution_examples.csv`
- **Model integrity:** `outputs/models/checkpoints.sha256`, `scripts/verify_checkpoints.py`
- **Forward Phase 7 handoff:** `docs/PHASE_7_FRONTEND_CONTEXT_AND_STREAMLIT_DECISION.md`
- **Earlier remediation claims:** `review/fix_report_pre_phase5.md`, `review/POST_PHASE4_AUDIT_REMEDIATION_REPORT.md`

**Bottom line:** proceed to Phase 7 only as an implementation step after correcting the handoff/API contract and documentation; do not treat the current graph-specific superiority statement or Phase 6 “upstream diffusion” narrative as scientifically established until the ablation and explainability fixes are completed.
