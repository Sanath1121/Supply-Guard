# SupplyGuard — Master Implementation Plan

**Status:** Consolidated master implementation plan. Supersedes `final_implementation_plan.md` (v1) and `Review/response-3_final_implementation_plan.md` (v2).
**Date:** 2026-10-05
**Inputs read:** root (`NEW_AIM.md`, `final_implementation_plan.md`), `Review/` (Response 1 review; Response 2 text + 7 code artifacts + tech stack; Response 3 text + plan + `train.py`/`evaluate.py`/`setup_and_download.py`; final response). `old/` treated as superseded history.

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

## MASTER IMPLEMENTATION PLAN

**Starting point:** about 70% of the core code already exists in `Review/` (R2: config, graph builder, graph layers, models, dataset, explainability, smoke test; R3: train, evaluate, setup). Treat it as a **draft to be fixed and understood**, not finished work. Each phase ends with a **gate**: do not start the next phase until the gate passes.

### Phase 0 — Environment, repository, data acquisition (3–4 h)
- **Goal:** a reproducible environment and a validated raw file.
- **Tasks:**
  1. Create the repo at the agreed location; `git init`; `.gitignore` for `data/`, `outputs/models/`, `mlruns/`, `.venv/`.
  2. Lay out the structure (R3 §3); copy the R2/R3 code; add `__init__.py` in `src/`, `src/models/`, `training/`, `tests/`; move v1 docs to `docs/archive/`; put this file + tech stack in `docs/`.
  3. Install on 3.14 (timebox 15 min) → else a 3.12 venv; `pip freeze > requirements.lock`.
  4. **Fix `setup_and_download.py`:** replace `assert nulls < 1%` with a report-only check; pin the commit hash; record sha256; explicit timestamp format; print gap statistics (count of gaps > 10 min, largest 10 gaps) and null-run lengths.
  5. Check whether the `_test.csv` span (2016-12 → 2017-08) sits inside a gap of `_train.csv`.
  6. Write `AGENTS.md` (final-response rules + "never assert nulls < 1%" + "parse timestamps with the fixed format").
- **Deliverables:** repo skeleton, `requirements.lock`, setup output (rows, sha256, span, median step, gaps, null shares), `AGENTS.md`.
- **Gate 0:** the setup script exits cleanly, and you can state in one sentence the row count, time span, sampling interval and gap structure.

### Phase 1 — EDA and go/no-go (4–6 h) · `notebooks/01_EDA.ipynb`
- **Goal:** decide whether the graph premise and the horizon are defensible *before* training anything.
- **Tasks** (on the cleaned, segment-aware frame from Phase 2's `load_clean_frame`, written first as a minimal version):
  1. Distributions, missingness map, null-run histogram, segment lengths.
  2. ACF (lags 1–50) per index; rolling mean/variance.
  3. Cross-correlation of adjacent echelons at lags 0–20, both directions; **Granger** S→M, M→D, D→R and reverse (run *within* segments).
  4. Persistence R² per node at horizons of **2, 10, 20, 60 minutes** on the sampled series that will actually be used.
  5. Relationship of `SCMstability_category` to the four RIs (association only).
  6. **Verify the paper's numbers in the PDF** (MSE 0.12, MAE 0.08, R² 0.92, 88%) before any critique goes into the report.
- **Decision table** (R3, adopted): persistence R² ≥ 0.99 → lengthen the horizon. No cross-node lead/lag → keep the project, reframe RQ2/RQ3 as "tests whether the assumed graph helps", expect a negative result. i.i.d. noise → stop.
- **Deliverables:** notebook + a one-page EDA summary with the decision table filled in; a locked `HORIZON_MIN` and `SAMPLING` choice.
- **Gate 1:** summary exists; horizon locked; framing chosen (probe suggests the "reframe RQ2" branch).

### Phase 2 — Data pipeline hardening (4–5 h) · `src/dataset.py`, `src/config.py`
- **Goal:** a leak-free, gap-safe windowing pipeline on the real file.
- **Tasks** (modify R2 `dataset.py`):
  1. Fixed-format timestamp parse → sort (stable) → drop duplicate stamps (0.36%).
  2. **Segmentation:** start a new segment wherever the gap exceeds `GAP_MAX` (default 3× median = 6 min). Windows never cross segments.
  3. **Null policy (replaces the >1% raise):** `ffill(limit=FFILL_LIMIT)` *within segments* (default 5 rows ≈ 10 min); remaining NaNs split the segment; drop segments shorter than `L+H`. Report rows lost.
  4. **Sampling:** stride *within segments*, so the effective step = stride × 2 min. Express `HORIZON` in minutes in config and convert to steps. Keep `MAX_SAMPLES` as a knob, tuned after timing in Phase 4.
  5. Keep R2's guarantees: chronological 80:10:10 by row position over the concatenated segments, train-only scaler saved, val/test context borrowed from the previous partition (only when contiguous in the same segment), targets inside their own partition.
  6. Extend the smoke test: synthetic data with injected gaps and null runs; assert that no window spans a gap.
- **Deliverables:** updated `dataset.py`/`config.py`; a printed `info` (rows used, segments, span, effective step, windows per split); a real-data spot check (first window vs raw rows).
- **Dependencies:** Phase 1 (horizon, sampling).
- **Gate 2:** smoke test passes **and** the real-file spot check confirms target = the row H steps after the window end, in the same segment.

### Phase 3 — Models and baselines (3–4 h) · `src/models/`, `src/graph_builder.py`
- **Goal:** every model runs on one input tensor `seq [B,L,5]`, and capacity is documented.
- **Tasks:** adopt R2's `GraphConv`, `STGCNLSTM`, `LSTMBaseline`, `PaperHybridOverall`; **remove the `Sigmoid` from `PaperHybridOverall`** (scaled val/test values can exceed [0,1], and the residual models are unbounded anyway; document it as a deviation). Produce a parameter-count table. Fix the doc wording "echelon embedding is *added*, not concatenated".
- **Deliverables:** a model forward-pass check; `outputs/results/param_counts.csv`.
- **Gate 3:** shapes `[B,4]`/`[B]`; non-zero 2-hop Supplier→Distributor gradient (smoke test); parameter table exists.

### Phase 4 — Training (2 h hands-on + 3–6 h wall clock) · `training/train.py`
- **Goal:** comparable, seeded runs within the time budget.
- **Tasks** (modify R3 `train.py`):
  1. **Time one epoch per model first**, then set `MAX_SAMPLES`, `MAX_EPOCHS` and seeds.
  2. Seeds: **3 by default** (42–44), 5 if the timing allows (user approved 5 seeds with Colab).
  3. Train the grid `lstm`, `paper_overall`, `st_gcn_lstm[directed]`, `st_gcn_lstm[symmetric]`. Only re-train `st_gcn_lstm` when switching `GRAPH_MODE` (R3 re-trains everything). Fix the loss-CSV naming so non-graph models don't get a graph-mode tag.
  4. Same loss for train and val; early stopping; `GRAD_CLIP`; optional MLflow with `log_artifact`.
- **Deliverables:** checkpoints per (model, mode, seed); loss CSVs; `training_summary.csv` with wall-clock times.
- **Gate 4:** no NaN or epoch-1 stops; train and val curves comparable.

### Phase 5 — Evaluation (3–4 h) · `training/evaluate.py`
- **Goal:** answer RQ1–RQ3 with one table each.
- **Tasks** (modify R3 `evaluate.py`):
  1. Evaluate **both graph modes in one run** (R3 overwrites the CSVs per mode).
  2. Order: persistence → Ridge-AR → LSTM → paper-style → STGCN sym/dir; % MSE improvement over persistence ± std.
  3. Severity: **per-node terciles from train** (not pooled), confusion matrix + macro-F1.
  4. Plots: prediction vs truth (test slice), error by node, error vs horizon if more than one is run.
  5. Write headline claims only from R3 §7's pre-agreed claims table.
- **Deliverables:** `overall_metrics.csv`, `node_metrics.csv`, `severity_metrics.csv`, figures.
- **Gate 5:** the headline sentence in the report matches the table exactly.

### Phase 6 — Explainability with validation (4–5 h) · `src/explainability.py`
- **Goal:** explanations that are *tested*, not just displayed.
- **Tasks:**
  1. Adopt R2's IG `RiskExplainer`.
  2. **Add Δ-attribution:** explain `f(x) − y_t` (what the network adds beyond persistence) as well as `f(x)`. With `RESIDUAL=True`, raw IG is dominated by the identity path, so "own RI at time t" would trivially win every time.
  3. For 8–10 hand-picked windows: completeness gap ≈ 0; **deletion test** vs random removal; neighbour-window stability; upstream share directed vs symmetric.
  4. Save `attribution_examples.csv`. Wording: "most sensitive to", never "root cause".
- **Deliverables:** deletion-test plot, examples CSV, a short validity note.
- **Gate 6:** the deletion test beats random. If it fails, the dashboard labels the explanations "unvalidated" (R3 §7).

### Phase 7 — SupplyGuard dashboard (6–8 h) · `app/streamlit_app.py`
- **Goal:** working demo on real test windows.
- **Tabs** (the v1 layout, kept, with R3 rules):
  1. Echelon risk cards: predicted node risks, tier, derived TRI, raw units via the saved scaler.
  2. Directed topology coloured by tier; edge width = computed upstream share.
  3. "Why this forecast?": signed feature/time bars, Δ-attribution toggle, `narrate()` sentence, completeness caption.
  4. Benchmarks with ± std error bars.
- **Rules:** label it "offline replay of historical windows"; no sliders, invented confidence or hard-coded text.
- **Gate 7:** every tab renders for 3 or more windows from a fresh `streamlit run`.

### Phase 8 — Documentation, artefact alignment, viva (5–7 h)
- **Tasks:**
  1. README (setup order, sha256, commit, seeds, hardware, wall-clock times).
  2. `02_Results.ipynb`.
  3. Report with a limitations section: assumed topology, simulated data, offline replay, sensitivity ≠ causation, gaps/nulls handling.
  4. Rewrite `NEW_AIM.md` claims ("learns topological dependencies" becomes "tests whether …"; "reproduction" becomes "re-implementation").
  5. Align the UML diagrams and PPT with this plan.
  6. Viva rehearsal using R2 §7 table + updated answers; final clean run from a fresh clone.
- **Gate 8:** a fresh clone runs end to end following the README.

**Schedule:** ≈ **42–56 h over 8–10 working days** (R3's 40–55 h is closer to reality than v1's 32–42 h; there is extra work for gaps/nulls and Δ-attribution, and saved time from reusing code).

| Day | Work |
|---|---|
| 1 | Phase 0 |
| 2 | Phase 1 (Gate 1) |
| 3 | Phase 2 |
| 4 | Phase 3 + epoch timing; launch Phase 4 |
| 5 | Phase 4 finishes; Phase 5 |
| 6 | Phase 6 |
| 7–8 | Phase 7 |
| 9 | Phase 8 |
| 10 | Buffer / clean-run |

### Key risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Reviewer scripts crash on real nulls (5%) | **certain if unfixed** | Phase 0/2 null-policy fixes; real-data spot check in Gate 2 |
| Windows crossing 428-day gaps or long ffills | high | Segmentation + `FFILL_LIMIT` (Phase 2) |
| Persistence is unbeatable at a short horizon | high (probe R² 0.95–0.99) | Horizon chosen in Phase 1; claims table |
| Graph adds nothing (probe: weak, lag-free cross-corr) | high | Reframed RQ2; ablation *is* the result; per-node attribution still delivered |
| Explanations dominated by the persistence path | high | Δ-attribution (D14) + deletion test |
| Training too slow (per-node LSTM = 4× sequences, 4 configs × 3–5 seeds) | medium | Epoch timing gate; `MAX_SAMPLES`/seeds knobs; early stopping |
| Wheels fail on 3.14 | medium | 3.12 venv fallback |
| Paper critique wrong | low–medium | Verify against the PDF in Phase 1 |
| OneDrive sync corrupts `mlruns/`/checkpoints | medium | Managed by user; at minimum git + keep `mlruns/` git-ignored |
| Presentation artefacts contradict the implementation | high (already true) | Phase 8 alignment |

### Plan B
Moved to its own file: `PLAN_B_IMPLEMENTATION_PLAN.md` (start only after Plan A Gate 8).

---

## DECISION LOG

**Adopted**
| # | Item | From | Why |
|---|---|---|---|
| D1 | Single input tensor `seq`; graph derived inside the model | R1/R2 | Removes duplicated information and attribution double-counting |
| D2 | `STGCNLSTM` (GCN per step → per-node LSTM) replaces `NodeRiskGNNLSTM` | R1/R2 | The graph participates across the window; per-node temporal embeddings |
| D3 | Directed + symmetric graph modes, compared | R1/R2 | Propagation claims need direction; symmetric is kept for Kipf fidelity |
| D4 | Persistence + Ridge-AR + graph-free LSTM baselines | R1/R2/R3 | Probe confirms persistence is strong |
| D5 | EDA moved to Phase 1 with a go/no-go gate | R1/R3 | Probe shows weak cross-node structure |
| D6 | Timestamp sort/dedup, saved scaler, borrowed context rows | R2 | Correct, already coded |
| D7 | Same train/val loss; seeds; mean ± std | R1/R3 | Comparable curves; small expected differences |
| D8 | IG with training-mean baseline, signed, completeness gap; deletion/stability tests | R1/R2/R3 | Defensible attribution |
| D9 | Terminology: "renormalised adjacency", "re-implementation", "sensitivity not cause" | R1 | Correct; avoids viva traps |
| D10 | Pre-agreed claims table; RQs allowed to be "no" | R3 | Prevents over-claiming |
| D11 | `AGENTS.md`, git, archive stale docs | final response | Verified: Antigravity reads `AGENTS.md`/`GEMINI.md`/`.agents/rules/` |
| D12 | Realistic schedule (~42–56 h) | R3 (adjusted) | v1 "<2 min" and 32–42 h unmeasured |

**Master additions (not in any review)**
| # | Item | Why |
|---|---|---|
| D13 | Segment-aware windowing + bounded `ffill` + no "<1% nulls" assertion | Real data: 5% nulls, 5,860-row null runs, 428-day gaps; both reviewer scripts would crash |
| D14 | Δ-attribution for residual models | With `ŷ = y_t + Δ`, IG on `ŷ` mostly rediscovers persistence |
| D15 | Horizon defined in minutes; stride applied within segments | Stride changes the effective step; rows ≠ time |
| D16 | Fixed timestamp format string | Avoids day/month inference errors |
| D17 | Evaluate both graph modes in one run; per-node tercile tiers | Fixes R3 CSV overwrite; `SCMstability_category` is global |
| D18 | Drop `Sigmoid` from paper baseline (documented deviation) | Scaled test values can exceed [0,1] |
| D19 | Align diagrams/PPT with the implementation | They currently say "Attention-Based … Classification … Real-Time" |

**Rejected / modified**
| # | Item | From | Decision | Why |
|---|---|---|---|---|
| R1 | "Test CSV `RI_Distributor1` 100% null" | current, R3 | **Rejected as fact** | Measured 7.3%. Train-only kept for simplicity |
| R2 | Fail on >1% nulls | R2, R3 | **Rejected** | Real data is 5%; replaced by D13 |
| R3 | 0.35/0.65 "industrial" thresholds | current | Rejected | No source |
| R4 | Tiers from `SCMstability_category` | R1/R2 option | Rejected as primary | Global 5-class label, not per node; used only in EDA |
| R5 | Node-loss + 0.25·TRI-loss | current | Rejected | Train/val mismatch; TRI is derived anyway |
| R6 | StandaloneGCN baseline | current | Dropped | Last-step-only GCN adds no question that persistence/Ridge-AR/LSTM don't already answer |
| R7 | ARIMA / statsmodels-AR10 fallback | current | Replaced by Ridge-AR | Multivariate, dependency-free; VAR moves to Plan B6 |
| R8 | "PyG can't install" / "Transformers need 100k rows" | current stack v1 | Rejected | Unverified or wrong; replaced with transparency/fidelity arguments + B5 |
| R9 | 5 seeds mandatory | R2/R3 | Modified to 3 default, 5 if timing allows | Feasibility on CPU (per-node LSTM ×4 configs) |
| R10 | VAR as a core baseline | R1 | Deferred to B6 | Ridge-AR covers multivariate linear AR at zero cost |
| R11 | `.antigravityignore` | final response | Not adopted as a requirement | Not confirmed in the Antigravity docs available; `.gitignore` + git suffice |
| R12 | Copy R2/R3 code as "done" | R3/final | Modified | Untested on real data; must pass Gate 2 fixes |
| R13 | Mandatory move outside OneDrive | R1/R2/R3/final | Not decided | Managed directly by user (no active OneDrive sync) |

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

✅ **All decisions resolved.** Repository location is managed directly by the user (no active OneDrive sync in effect). The master implementation plan is 100% frozen and ready for execution.
