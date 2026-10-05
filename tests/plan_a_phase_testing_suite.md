# Plan A: Phase-by-Phase Test Suite and Gate Verification Architecture

## Goal Description
This plan establishes an end-to-end, phase-gated testing harness for **SupplyGuard (Plan A)**. For each of the 9 development phases (Phase 0 through Phase 8), a dedicated test file and verification checklist will validate all phase deliverables against its defined **Gate** before advancing to the next phase. A unified test runner (`run_phase_tests.py`) will allow running individual phase tests or progressive regression tests up to any given phase.

```
+---------------------------------------------------------------------------------------------------------+
|                                    SUPPLYGUARD PHASE GATING PIPELINE                                    |
+---------------------------------------------------------------------------------------------------------+
|  Phase 0: Setup & Data   --> Gate 0: test_phase0_setup.py          [Checksums, Schema, Gaps, Nulls]     |
|         |                                                                                               |
|  Phase 1: EDA & Horizon  --> Gate 1: test_phase1_eda.py            [ACF, Granger, Persistence, Horizon] |
|         |                                                                                               |
|  Phase 2: Dataset Safe   --> Gate 2: test_phase2_dataset.py        [Segments, Bounded Ffill, No-Leak]   |
|         |                                                                                               |
|  Phase 3: Models & GCN   --> Gate 3: test_phase3_models.py         [Shapes, 2-Hop Gradients, Params]    |
|         |                                                                                               |
|  Phase 4: Training Loop  --> Gate 4: test_phase4_training.py       [Loss, Checkpoints, Colab Resume]    |
|         |                                                                                               |
|  Phase 5: Evaluation     --> Gate 5: test_phase5_evaluation.py     [Baselines, Metrics, Terciles, F1]   |
|         |                                                                                               |
|  Phase 6: Explainability --> Gate 6: test_phase6_explainability.py [IG Axiom, Delta-Attr, Deletion Test]|
|         |                                                                                               |
|  Phase 7: Streamlit App  --> Gate 7: test_phase7_app.py            [4 Tabs Replay, Scaler Unscale]      |
|         |                                                                                               |
|  Phase 8: Final Viva     --> Gate 8: test_phase8_e2e.py            [Clean Repro, Claims Audit, Packaged]|
+---------------------------------------------------------------------------------------------------------+
```

---

## User Review Required

> [!IMPORTANT]
> **Test Data Strategy (Synthetic vs Real File):**
> 1. Each phase test script will include a **self-contained synthetic fixture** with known ground-truth properties (injected gaps, known lags, controlled null runs) so tests can run in seconds in any environment without depending on a 650,000-row CSV download.
> 2. When the real dataset (`data/raw/SCRM_timeSeries_2018_train.csv`) is present, the test suite will automatically run the **real-data spot check** to verify production data integrity.
> 
> Please review and confirm that you want this hybrid approach (synthetic mock for instant unit testing + real data spot check when downloaded).

> [!NOTE]
> **Execution Tooling:**
> Tests will be written using standard Python `unittest` compatible with `pytest`. You can run them via:
> - `python run_phase_tests.py --phase <0-8>` (Our custom colorful CLI runner)
> - `pytest tests/test_phase<N>_*.py` (Standard pytest runner)

---

## Proposed Changes

The test suite will reside in the `tests/` directory alongside the unified test harness `run_phase_tests.py`.

```
supplyguard/
├── run_phase_tests.py                 # [NEW] Master CLI runner for gating and review
└── tests/
    ├── __init__.py                    # [NEW]
    ├── test_phase0_setup.py           # [NEW] Gate 0: Setup, env, data acquisition, schema
    ├── test_phase1_eda.py             # [NEW] Gate 1: EDA metrics, persistence baseline, horizon
    ├── test_phase2_dataset.py         # [NEW] Gate 2: Segmentation, bounded ffill, leakage check
    ├── test_phase3_models.py          # [NEW] Gate 3: GCN/LSTM shapes, 2-hop gradients, parameters
    ├── test_phase4_training.py        # [NEW] Gate 4: Train loop, convergence, Colab checkpoint resume
    ├── test_phase5_evaluation.py      # [NEW] Gate 5: Persistence, Ridge-AR, metrics CSVs, terciles
    ├── test_phase6_explainability.py  # [NEW] Gate 6: IG completeness, delta-attribution, deletion test
    ├── test_phase7_app.py             # [NEW] Gate 7: Streamlit components headless smoke test
    └── test_phase8_e2e.py             # [NEW] Gate 8: Clean run reproduction, claims audit
```

---

### Component 1: Master Phase Runner (`run_phase_tests.py`)

#### [NEW] `run_phase_tests.py`
A CLI utility that executes the test suite for a given phase and prints a structured scorecard:
- Flags:
  - `python run_phase_tests.py --phase 0` (Runs Gate 0)
  - `python run_phase_tests.py --phase 2` (Runs Gate 2)
  - `python run_phase_tests.py --up-to 4` (Runs Gates 0 through 4)
  - `python run_phase_tests.py --all` (Runs complete test suite)
- Displays:
  - Gate status (`[PASSED]` / `[BLOCKED]`)
  - Summary of pass/fail assertions
  - Clear diagnosis and exact next steps if any check fails.

---

### Component 2: Phase 0 Verification (`tests/test_phase0_setup.py`)

#### [NEW] `tests/test_phase0_setup.py`
Validates environment reproducibility, directory skeleton, and dataset acquisition integrity.
- **Test Cases:**
  1. `test_directory_structure()`: Asserts existence of `data/raw`, `src/models`, `training`, `tests`, `notebooks`, `app`, `outputs/models`, `outputs/results`, `outputs/figures`, `docs`.
  2. `test_environment_imports()`: Asserts that core packages (`torch`, `pandas`, `numpy`, `sklearn`, `joblib`, `networkx`, `streamlit`) import without errors.
  3. `test_timestamp_format_parsing()`: Tests that dates in format `%m/%d/%Y %I:%M:%S %p` parse without month/day swapping or errors.
  4. `test_null_policy_non_crashing()`: Verifies setup scripts do NOT enforce a fragile `assert nulls < 1%`, but rather report the actual ~5% null distribution safely.
  5. `test_raw_data_spot_check()`: *(Conditional on file presence)* Verifies SHA-256 hash, 649,999 row count, and presence of all 6 required columns.

---

### Component 3: Phase 1 Verification (`tests/test_phase1_eda.py`)

#### [NEW] `tests/test_phase1_eda.py`
Validates EDA functions, horizon determination, and research question framing.
- **Test Cases:**
  1. `test_acf_calculation()`: Verifies autocorrelation computation across lags 1–50 on synthetic series.
  2. `test_within_segment_cross_correlation()`: Verifies cross-correlation and Granger causality functions only operate *within* contiguous segments and reject bridging time gaps.
  3. `test_persistence_evaluation_across_horizons()`: Tests persistence $R^2$ at $H \in \{2, 10, 20, 60\}$ minutes; verifies calculation logic.
  4. `test_eda_summary_artifact_integrity()`: Validates that the EDA summary outputs and decision table are properly structured.

---

### Component 4: Phase 2 Verification (`tests/test_phase2_dataset.py`)

#### [NEW] `tests/test_phase2_dataset.py`
Validates data pipeline hardening, segmentation, bounded forward-fill, and leakage prevention.
- **Test Cases:**
  1. `test_sorting_and_deduplication()`: Shuffled input with duplicate timestamps is correctly ordered and deduplicated.
  2. `test_segmentation_on_gaps()`: Injects a 20-minute gap into synthetic data; verifies pipeline splits data into two distinct segments without spanning.
  3. `test_no_window_spans_gap()`: Loops through all created windows $(X, y)$; strictly asserts that no window has timestamps crossing a segment boundary.
  4. `test_bounded_ffill_policy()`: Injects a 3-row null run (filled) and an 8-row null run (splits segment); asserts no null values reach the window tensors.
  5. `test_scaler_no_leakage()`: Verifies `MinMaxScaler` is fit strictly on train split, and that test set min/max values outside $[0, 1]$ do not raise runtime exceptions.
  6. `test_chronological_split_strictness()`: Asserts `train.timestamps.max() < val.timestamps.min() < val.timestamps.max() < test.timestamps.min()`.
  7. `test_target_offset_alignment()`: Verifies sample 0 target matches row exactly $H$ steps ahead.

---

### Component 5: Phase 3 Verification (`tests/test_phase3_models.py`)

#### [NEW] `tests/test_phase3_models.py`
Validates model architectures, graph convolutions, and mathematical correctness.
- **Test Cases:**
  1. `test_model_output_shapes()`:
     - `st_gcn_lstm` (directed & symmetric) -> `[B, 4]`
     - `lstm` baseline -> `[B, 4]`
     - `paper_overall` -> `[B]`
  2. `test_graph_adjacency_and_laplacian()`:
     - Symmetric $\hat{A} = \hat{A}^T$.
     - Normalised Laplacian eigenvalues $\in [-10^{-6}, 2 + 10^{-6}]$.
     - Directed $A_{down}[M, S] = 1, A_{up}[S, M] = 1$.
  3. `test_two_hop_gradient_reachability()`: In a 2-layer GCN, asserts $\frac{\partial \hat{y}_{Distributor}}{\partial x_{Supplier}} \ne 0$.
  4. `test_residual_path_identity()`: With head weights zeroed, asserts $\hat{y} = y_t$ (starts at persistence baseline).
  5. `test_paper_hybrid_sigmoid_removal()`: Asserts `PaperHybridOverall` outputs unbounded risk values rather than squashing through `Sigmoid`.

---

### Component 6: Phase 4 Verification (`tests/test_phase4_training.py`)

#### [NEW] `tests/test_phase4_training.py`
Validates training execution, convergence, and Colab checkpoint-resume idempotency.
- **Test Cases:**
  1. `test_mini_training_convergence()`: Runs 3 training epochs on synthetic data; verifies loss monotonically decreases or stays stable without NaNs.
  2. `test_gradient_clipping_enforced()`: Injects large loss gradients; verifies `clip_grad_norm_` limits parameter gradients to $\le 1.0$.
  3. `test_checkpoint_save_and_resume()`:
     - Trains 1 epoch and saves checkpoint.
     - Re-launches training; asserts checkpoint is discovered and resumed without re-running or overwriting.
  4. `test_loss_csv_logging()`: Verifies loss CSV schema (`epoch,train_loss,val_loss,lr,time`).

---

### Component 7: Phase 5 Verification (`tests/test_phase5_evaluation.py`)

#### [NEW] `tests/test_phase5_evaluation.py`
Validates evaluation metrics, baseline benchmarking, and severity classification.
- **Test Cases:**
  1. `test_persistence_and_ridge_ar_metrics()`: Verifies calculation of persistence baseline and Ridge-AR(10) baseline.
  2. `test_per_node_and_tri_metrics()`: Computes MSE, MAE, RMSE, $R^2$ per node and verifies derived Total Risk Index (TRI) as the mean of the 4 nodes.
  3. `test_raw_unit_metric_restoration()`: Verifies that metrics computed in raw units match manual inverse-transform math.
  4. `test_tercile_severity_and_macro_f1()`: Validates train-derived tercile binning, confusion matrix calculation, and macro-F1 scoring.
  5. `test_metrics_csv_isolation()`: Asserts evaluation runs for symmetric and directed modes do NOT overwrite each other's CSV files.

---

### Component 8: Phase 6 Verification (`tests/test_phase6_explainability.py`)

#### [NEW] `tests/test_phase6_explainability.py`
Validates explainability methods, Integrated Gradients axioms, and deletion test logic.
- **Test Cases:**
  1. `test_integrated_gradients_completeness()`: Asserts that $\left| \sum \text{Attributions} - (F(x) - F(x_0)) \right| < 10^{-3}$ (completeness axiom).
  2. `test_delta_attribution_computation()`: Verifies that $\Delta$-attribution explains $f(x) - y_t$, isolating network contribution from the persistence shortcut.
  3. `test_deletion_test_validation()`: Verifies deletion test algorithm: removing top-ranked features causes a larger prediction drop than removing random features.
  4. `test_upstream_share_directionality()`: Verifies upstream attribution attribution calculation under directed vs symmetric modes.

---

### Component 9: Phase 7 Verification (`tests/test_phase7_app.py`)

#### [NEW] `tests/test_phase7_app.py`
Validates the Streamlit dashboard components in headless execution.
- **Test Cases:**
  1. `test_app_helper_functions()`: Tests scaler unscaling, tier badge color mapping, and timestamp formatting functions.
  2. `test_topology_graph_rendering()`: Generates NetworkX/Plotly supply chain graph; asserts 4 nodes and directed edges render without exceptions.
  3. `test_replay_window_retrieval()`: Retrieves test windows from test dataset and verifies valid input tensors for all 4 dashboard tabs.
  4. `test_no_synthetic_sliders_present()`: Inspects app source to verify adherence to Plan A rule: no fake sliders, no fabricated metrics.

---

### Component 10: Phase 8 Verification (`tests/test_phase8_e2e.py`)

#### [NEW] `tests/test_phase8_e2e.py`
Validates end-to-end integration, reproducibility, and claim alignment.
- **Test Cases:**
  1. `test_all_output_artifacts_present()`: Checks existence of scaler, checkpoints, evaluation CSVs, figures, and `training_summary.csv`.
  2. `test_claims_table_consistency()`: Parses `overall_metrics.csv` and verifies that report claims adhere strictly to Plan A Section 4 claims table (e.g. no claiming graph wins if $p$-value / std does not support it).
  3. `test_seed_reproducibility()`: Re-runs 1 seed with fixed seed; asserts predictions match saved checkpoint outputs within numerical precision.

---

## Verification Plan

### Automated Tests
Run individual phase tests:
```powershell
# Phase 0 setup & acquisition
python run_phase_tests.py --phase 0

# Phase 2 dataset & segmentation
python run_phase_tests.py --phase 2

# Phase 3 model architectures & gradients
python run_phase_tests.py --phase 3

# Run all phase tests up to Phase 3
python run_phase_tests.py --up-to 3

# Or standard pytest invocation:
python -m pytest tests/test_phase*.py -v
```

### Manual Verification
1. Review the printed test output at each phase gate.
2. Confirm that each gate output gives a clear green signal before advancing to the implementation of the next phase.
