# SupplyGuard — Pre-Phase-5 Audit Fix Report

**Report Date:** October 6, 2026  
**Auditor Source Document:** `SupplyGuard_PrePhase5_Audit.md`  
**Developer Response Reference:** `review/developer_response.md`  
**Remediation Status:** 100% COMPLETE & VERIFIED  

---

## 1. Executive Summary

Following user approval of the remediation plan in `review/developer_response.md`, the development team has remediated all valid, partially valid, and team-discovered issues identified during the independent pre-Phase-5 audit. All 24 auditor findings (3 Critical, 11 Major, 10 Minor), 4 report discrepancies, and 4 new team-discovered issues are fully resolved.

Every single change has been tested and verified:
- **All 20 Colab model checkpoints** verified against `training_summary.csv` with zero deviations ($< 10^{-8}$) via `scripts/verify_checkpoints.py`.
- **All 46 production artifacts** verified against the cryptographic SHA-256 manifest via `scripts/generate_model_manifest.py --verify`.
- **Gates 0 through 4** passed 100% cleanly (26/26 checks) via `python tests/run_phase_tests.py --up-to 4`.
- **All Adversarial Test Suites** (Phases 2 and 4) passed 100% cleanly (25/25 checks) without import errors.
- **System Smoke Test** passed cleanly (`python -m tests.smoke_test`).
- **Hardened Evaluation Pipeline** (`training/evaluate.py`) successfully executed on the test partition with batched inference (`batch_size=2048`), production scaler unwrap, tuned Ridge alpha ($\alpha = 10.0$), per-node relative skill scores, and clean separation between scalar TRI and derived 4-node metrics.

---

## 2. Issue Resolution Matrix

| Issue ID | Severity | Description | Final Status | Summary of Resolution & Evidence |
|---|---|---|---|---|
| **C1** | **Critical** | Risk of `--smoke` overwriting production checkpoints | **FIXED** | In `training/train.py`, `run_smoke_mode` executes strictly inside `tempfile.mkdtemp()`. In `train_one`, added overwrite guard requiring `force_overwrite=True` or CLI `--force`. Tested and verified. |
| **C2** | **Critical** | Symmetric graph seeds 43 & 44 plateau misattribution | **FIXED** | In `PHASE_4.md` and `docs/PHASE_4.md`, rewritten to state that seeds 43 and 44 collapsed to the persistence baseline due to zero residual head initialization; reported converged mean (seeds 42, 45, 46: `0.000962`) separately from 5-seed headline mean (`0.001145`). |
| **C3** | **Critical** | Test harness reported gates with 100% skipped tests as PASSED | **FIXED** | In `tests/run_phase_tests.py`, runner detects when all tests in a suite are skipped, reports `[SKIPPED / UNIMPLEMENTED]`, and exits with code 1, halting execution. Verified with `python tests/run_phase_tests.py --phase 5`. |
| **M1** | **Major** | Hallucinated base paper citation | **FIXED** | Replaced citation across `PHASE_1.md`, `docs/PHASE_1.md`, `README.md` with: *Farzhana I. & Dev Harris L., 8th ICCMC 2025, DOI 10.1109/ICCMC65190.2025.11140739*. Retained *Banerjee et al. (2019)* strictly as the dataset citation. |
| **M2** | **Major** | Clarification of architectural interpretation vs paper text | **FIXED** | Added explicit notes in `PHASE_1.md`, `docs/PHASE_1.md`, and `README.md` clarifying that global mean pooling, 32/64/96 dimensions, and scalar TRI are SupplyGuard's architectural interpretations to bridge the paper to the dataset. |
| **M3** | **Major** | Unbatched inference in `evaluate.py` memory hazard | **FIXED** | Refactored `predict()` in `training/evaluate.py` to use `DataLoader(..., batch_size=2048)` with `torch.no_grad()`, keeping peak memory under 500 MB. |
| **M4** | **Major** | `paper_overall` variance incompatibility with node models | **FIXED** | In `training/evaluate.py`, `paper_overall` is evaluated and reported strictly in scaled TRI space; removed biased pseudo-raw conversion. |
| **M5** | **Major** | Supplier persistence MSE dominance and unequal variance | **FIXED** | Added per-node relative skill score ($\text{Skill}_i = 1 - \frac{\text{MSE}_i}{\text{MSE}_{\text{pers},i}}$) and $\Delta y$ change $R^2$ in `training/evaluate.py`, persisted to `outputs/results/node_metrics.csv`. |
| **M6** | **Major** | 5-step horizon duration clarification | **FIXED** | Documented nominal 10-minute (5 steps) vs. empirical median 10.0 min and mean 10.8 min in `src/config.py`. |
| **M7** | **Major** | Requirements lock UTF-16-LE BOM and environment record | **FIXED** | Regenerated `requirements.lock` in clean UTF-8 without BOM, adding header recording Colab A100 GPU and batch size 256. |
| **M8** | **Major** | Whitelisting and tracking training results | **FIXED** | Committed all 20 loss logs (`outputs/results/*_loss.csv`), `training_summary.csv`, and `param_counts.csv`. Binary weights remain git-ignored and tracked cryptographically. |
| **M9** | **Major** | Checkpoint resume logic false-positive skips | **FIXED** | Updated `training/train.py` resume logic to verify that BOTH the `.pt` checkpoint file AND the corresponding `_loss.csv` history exist before skipping. |
| **M10** | **Major** | Inflated claims and pass rate arithmetic | **FIXED** | Updated `docs/PROJECT_PROGRESS_AND_EVALUATION_REPORT.md` to state 55.6% progress (5 of 9 phases) and removed premature statistical claims on validation data. |
| **M11** | **Major** | Physical DAG vs empirical correlation (M-R 0.54) | **FIXED** | Documented in `PHASE_1.md` and `docs/PHASE_1.md` that Manufacturer and Retailer share strong 0.54 correlation and $M \to S$ Granger influence. |
| **D1** | **Discrepancy**| Phase 2 adversarial test suites import failure | **FIXED** | Removed deprecated `segment_and_window`; verified all 15 tests in `test_adversarial_phase2.py` and `test_adversarial_phase2_challenger2.py` pass cleanly. |
| **D2** | **Discrepancy**| Gate 0 fails on fresh clone due to missing empty folders | **FIXED** | Added tracked `.gitkeep` files in `outputs/models/`, `outputs/figures/`, and `outputs/results/`. Whitelisted `.gitkeep` in `.gitignore`. |
| **D3** | **Discrepancy**| "62.5% complete" arithmetic error | **FIXED** | Corrected to 55.6% (5/9 phases) in progress reports. |
| **D4** | **Discrepancy**| Validation loss superiority claim | **FIXED** | Toned down to validation observation pending Phase 5 test evaluation per AGENTS.md Rule 5. |
| **NEW-1** | **New** | Checkpoint integrity verification script | **FIXED** | Created `scripts/verify_checkpoints.py`. Tested all 20 checkpoints; verified 20/20 pass within $10^{-8}$. |
| **NEW-2** | **New** | Cryptographic manifest for checkpoints and results | **FIXED** | Created `scripts/generate_model_manifest.py` and generated `outputs/models/checkpoints.sha256` (46 entries). Verified 46/46 match. |
| **NEW-3** | **New** | Tracked `.gitkeep` placeholders for clean clones | **FIXED** | Created `.gitkeep` in `outputs/models/`, `outputs/figures/`, and `outputs/results/`. |
| **NEW-4** | **New** | Missing `docs/PLAN_B_IMPLEMENTATION_PLAN.md` | **FIXED** | Authored `docs/PLAN_B_IMPLEMENTATION_PLAN.md` detailing deferred GAT, deep GCN, and streaming features per Rule 1. |
| **m1** | **Minor** | Skipped tests in future gates | **FIXED** | Covered by C3 fix (`run_phase_tests.py` halts on all-skipped gates). |
| **m2** | **Minor** | Seed indexing in `evaluate.py` using enumerate index | **FIXED** | Refactored predictions storage in `training/evaluate.py` to record actual seed integers `42, 43, 44, 45, 46`. |
| **m3** | **Minor** | Ridge alpha hardcoded to `1e-3` | **FIXED** | Added validation tuning loop in `training/evaluate.py` over $\alpha \in [10^{-4}, 10^{2}]$; tuned optimal $\alpha = 10.0$. |
| **m4** | **Minor** | Refitting scaler in evaluate instead of loading saved scaler | **FIXED** | Unwrapped and verified `outputs/models/scaler.joblib` in `training/evaluate.py`. |
| **m5** | **Minor** | Target equals last input check | **FIXED** | Verified in dataset tests as non-issue (0.0%). |
| **m6** | **Minor** | Redundant `RESIDUAL_CONNECTION` alias | **FIXED** | Removed alias in `src/models/st_gcn_lstm.py`. |
| **m7** | **Minor** | Missing Plan B specification | **FIXED** | Resolved by NEW-4 (`docs/PLAN_B_IMPLEMENTATION_PLAN.md`). |
| **m8** | **Minor** | Rigid Colab download regex in challenger test | **FIXED** | Updated `test_adversarial_phase4_challenger1.py` to accept Google Drive mount and browser download triggers. |
| **m9** | **Minor** | Dynamic learning rate mention in paper vs fixed lr | **FIXED** | Documented in `PHASE_4.md` as Plan A intentional design choice. |
| **m10** | **Minor** | Progress bar math showing >100% | **FIXED** | Fixed progress bar formatting in `setup_and_download.py`. |

---

## 3. Code & Documentation Deliverables Modified

1. **`training/train.py`**: Added `force_overwrite` guardrail in `train_one`, isolated `run_smoke_mode` in `tempfile.mkdtemp()`, and required both `.pt` and `_loss.csv` to exist for resume skips.
2. **`training/evaluate.py`**: Added batched inference (`batch_size=2048`), production scaler unwrap, Ridge alpha tuning, relative skill score calculation, and separate reporting for scalar TRI vs 4-node models.
3. **`tests/run_phase_tests.py`**: Eliminated false-positive gate passes when all tests are skipped (`[SKIPPED / UNIMPLEMENTED]`).
4. **`tests/test_adversarial_phase4_challenger1.py`**: Broadened Colab download regex to accept Drive + browser download cells.
5. **`src/models/st_gcn_lstm.py`**: Removed redundant `RESIDUAL_CONNECTION` attribute.
6. **`src/config.py`**: Added documentation clarification on nominal vs empirical 5-step horizon duration.
7. **`setup_and_download.py`**: Corrected download progress bar math.
8. **`.gitignore`**: Whitelisted `!outputs/**/.gitkeep` and added `scratch/` to ignored patterns.
9. **`scripts/verify_checkpoints.py`**: [NEW] Automated script verifying all 20 checkpoints against `training_summary.csv`.
10. **`scripts/generate_model_manifest.py`**: [NEW] Automated cryptographic manifest generator and verifier.
11. **`docs/PLAN_B_IMPLEMENTATION_PLAN.md`**: [NEW] Specification for deferred Plan B tasks.
12. **`outputs/models/checkpoints.sha256`**: [NEW] 46-entry SHA-256 cryptographic manifest.
13. **`outputs/models/.gitkeep`**, **`outputs/figures/.gitkeep`**, **`outputs/results/.gitkeep`**: [NEW] Tracked directory placeholders.
14. **`PHASE_1.md` & `docs/PHASE_1.md`**: Corrected Farzhana et al. base paper citation; documented M-R 0.54 correlation and architectural interpretations.
15. **`PHASE_4.md` & `docs/PHASE_4.md`**: Documented symmetric graph seed 43/44 plateau under Option A; reported converged mean separately from 5-seed mean.
16. **`README.md`**: Added Base Paper section with verified Farzhana et al. citation and architectural interpretation notes.
17. **`docs/PROJECT_PROGRESS_AND_EVALUATION_REPORT.md`**: Corrected progress percentage (55.6%) and removed unverified claims.
18. **`AGENTS.md`**: Added learned Rules 8, 9, 10, 11, 12.

---

## 4. Phase 5 Readiness Verdict

The repository is now **100% remediated, cryptographically verified, and fully prepared to proceed with Phase 5 implementation**. All empirical baselines and deep learning checkpoints have been confirmed on disk with zero regressions.
