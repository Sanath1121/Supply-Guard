# Phase 0-3 Issue Fix Report

## Overview
This report documents the resolution of the issues identified in `merged_review_phase0-3.md`. All fixes have been implemented, tested, and verified against the criteria without starting Phase 4.

## Fix Status Table

| ID  | Severity | Status | Files Changed | Description / Verification |
| --- | -------- | ------ | ------------- | -------------------------- |
| **R-3** | Critical | Fixed | `tests/smoke_test.py` | Fixed `smoke_test.py` by passing `save_scaler=False`. Regenerated real scaler using `build_datasets`. Un-ignored `param_counts.csv` in `.gitignore`. |
| **R-4** | Major | Fixed | `training/train.py`, `training/evaluate.py` | Made scripts Phase-4 ready: skip existing runs, output both `sym/dir` models, exact raw TRI calculation, and per-node severity terciles, appended results to `training_summary.csv` with wall-clock time. |
| **R-5** | Major | Fixed | `tests/test_phase[4-8]*.py` | Replaced false `pass` stubs with `raise unittest.SkipTest("Phase not implemented yet")`. Test runners now correctly mark these as skipped instead of falsely claiming completion. |
| **R-8, R-10, R-11** | Major | Fixed | `PHASE_0.md`, `PHASE_1.md`, `PHASE_2.md`, `PHASE_3.md`, `setup_and_download.py` | Added exact SHA-256 halt logic in `setup_and_download.py` to stop execution on mismatch. Corrected documentation regarding SHA halt, Farzhana et al. base paper citation, explicit L+H drop reasoning, and GCN_HIDDEN_DIM aliasing. |
| **R-12** | Major | Fixed | `src/models/st_gcn_lstm.py` | Removed the `Sigmoid` activation from non-residual paths. Added string-parsing in `build_model` to instantiate `symmetric` or `directed` modes depending on name (`st_gcn_lstm_sym`, `st_gcn_lstm_dir`). |
| **R-1, R-6** | Minor | Fixed | `tests/test_phase[0-2]*.py` | Removed hardcoded tautologies. Replaced tests with actual calls to `build_datasets` / `load_clean_frame` to dynamically compute metrics and shapes on real pipeline behavior. |
| **R-2** | Minor | Fixed | `src/dataset.py` | Removed dead/duplicate code (`subsample`, `_windows`, and the duplicate `segment_and_window`). |
| **N-1, N-2** | Minor | Fixed | `tests/audit_executed_notebook.py`, `tests/test_phase1_challenge_oracle.py` | Updated default notebook path and wrapped oracle inside a `unittest.TestCase` class so it reliably runs in test harness. |

## Test Results
**Before Fixes:** 
*   `smoke_test.py` was corrupting the scaler.
*   Gates 4-8 gave false positive passes.
*   Gates 0-2 checked synthetic mock logic instead of production code.

**After Fixes:**
*   Gate tests 0-3 pass dynamically by loading and validating the actual dataset logic.
*   Gate tests 4-8 correctly show as `[SKIPPED]` in unittest reporting.
*   All Phase reports are perfectly aligned with the actual code and data behavior.

## Deviations
*   The review feedback from `deep-research-report.md` was intentionally discarded based on user confirmation that it was for an unrelated manuscript.

## Readiness
The Phase 0-3 codebase is now strictly aligned with the implementation plan. There are no false test assurances, the data ingestion pipeline prevents leakage, the scaler is safe from overwrite, and the project is fully clean and ready to begin Phase 4 (Model Training).
