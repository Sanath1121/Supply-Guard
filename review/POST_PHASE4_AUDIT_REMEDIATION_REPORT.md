# SupplyGuard — Post-Phase 4 Audit Remediation & Hardening Report

**Project Title:** SupplyGuard — Real-Time Multi-Echelon Supply Chain Risk Alert System  
**Document Type:** Formal Audit Remediation, Test Isolation & Model Hardening Report  
**Target Milestone:** Pre-Phase 5 Quality Assurance & Handoff Sign-Off  
**Audit Reference:** `review/SupplyGuard_PrePhase5_Audit.md` (Pre-Phase 5 Independent Audit)  
**Primary Remediation Commit:** [`ba2ef7c`](https://github.com/Sanath1121/Supply-Guard/commit/ba2ef7c)  
**Current HEAD Commit:** [`75775a3`](https://github.com/Sanath1121/Supply-Guard/commit/75775a3)  
**Date:** October 6, 2026  
**Auditor Traceability:** Tech Lead, Core Architecture Team, Empirical Challengers  

---

## 1. Executive Summary

Following the completion of Phase 4 (Multi-Architecture Multi-Seed Training on Google Colab GPU), an independent pre-Phase 5 audit was conducted (`SupplyGuard_PrePhase5_Audit.md`), identifying 3 Critical (C1–C3), 11 Major (M1–M11), and 8 Minor (m1–m8) findings. The primary risks centered around:
1. Potential overwrite of production checkpoints during test/smoke runs (`C1`).
2. Two symmetric graph seeds (43 & 44) collapsing to the analytical persistence baseline without explicit diagnostic documentation (`C2`).
3. False-positive test runner reporting when test suites were skipped (`C3`).
4. Broken imports in legacy Phase 2 adversarial tests (`D1`).
5. Overwriting the production `scaler.joblib` during adversarial test executions.
6. Documentation inaccuracies regarding citations, model pooling interpretations, and percentage reporting.

This report documents the exhaustive, end-to-end remediation of all audit findings. Every remediation step was executed without modifying `src/dataset.py`, `src/models/*`, or the binary contents of any `.pt` checkpoint file. All 20 trained models have been re-verified against recorded validation metrics and analytically confirmed against the persistence baseline.

---

## 2. Order of Work: Verification & Hardening Results

```
+---------------------------------------------------------------------------------------------------------+
|                                    AUDIT REMEDIATION WORKFLOW ROADMAP                                   |
+---------------------------------------------------------------------------------------------------------+
|  [STEP 1] Checkpoint Hash Manifest    | Status: VERIFIED | 43 files hashed from raw Colab outputs.zip   |
|  [STEP 2] Sandbox Isolation (C1)      | Status: VERIFIED | tempfile.mkdtemp(), --force guard, invariance |
|  [STEP 3] Checkpoint Verification     | Status: VERIFIED | 20/20 models pass within 3.9e-5 rel tol      |
|  [STEP 4] Test Runner Skips (C3)      | Status: VERIFIED | [SKIPPED / UNIMPLEMENTED] exit code 1 gating |
|  [STEP 5] Adversarial Suite Fix (D1)  | Status: VERIFIED | Production APIs, zero scaler mutation       |
|  [STEP 6] Diagnostic & Option A (C2)  | Status: VERIFIED | Read-only parameter & delta activation audit |
+---------------------------------------------------------------------------------------------------------+
```

---

### Step 1: Cryptographic Checkpoint Hash Manifest
To establish an indisputable, immutable baseline, [`outputs/models/checkpoints.sha256`](file:///c:/Users/srila/OneDrive/Desktop/BTECH/4th%20year/mini%20project/Supply_chain_alret_system/outputs/models/checkpoints.sha256) was constructed directly from the raw byte stream of the original Google Colab output archive (`outputs.zip`), not from local disk.

- **Scope:** 43 production files:
  - All 20 PyTorch model checkpoints (`.pt`) across 4 architectures and 5 seeds.
  - Production fitted `scaler.joblib`.
  - Master `training_summary.csv`.
  - Model parameter report `param_counts.csv`.
  - All 20 per-run loss history files (`*_loss.csv`).
- **Backup & Verification:** A read-only backup [`outputs/models/checkpoints.sha256.bak`](file:///c:/Users/srila/OneDrive/Desktop/BTECH/4th%20year/mini%20project/Supply_chain_alret_system/outputs/models/checkpoints.sha256.bak) was archived. The local disk was cross-checked, confirming a 100% bit-for-bit match with zero byte discrepancies.

---

### Step 2: C1 Sandbox Isolation & Overwrite Guards
The production training and testing pipelines were hardened against accidental overwrites:

1. **Test & Smoke Sandboxing:**
   - Modified `run_smoke_mode` in [`training/train.py`](file:///c:/Users/srila/OneDrive/Desktop/BTECH/4th%20year/mini%20project/Supply_chain_alret_system/training/train.py) and [`tests/smoke_test.py`](file:///c:/Users/srila/OneDrive/Desktop/BTECH/4th%20year/mini%20project/Supply_chain_alret_system/tests/smoke_test.py) to write strictly to `tempfile.mkdtemp()`. Temporary directories are deleted upon exit.
2. **`--force` Overwrite Guard:**
   - Added a strict overwrite guard to `train_one` and `build_datasets`. The script prohibits overwriting existing `.pt` model files or `scaler.joblib` unless `--force` is explicitly specified via the CLI.
3. **Atomic Writes & Completion Marker (M9):**
   - Checkpoints write to temporary filenames (`.tmp`) before atomic promotion via `os.replace`.
   - Grid completion creates [`outputs/results/.training_completed`](file:///c:/Users/srila/OneDrive/Desktop/BTECH/4th%20year/mini%20project/Supply_chain_alret_system/outputs/results/.training_completed).
4. **Idempotent Resume Verification:**
   - Verified that running `python training/train.py --device cpu --force-cpu` detects all 20 existing checkpoints and exits cleanly with `[Skip]` messages without retraining.
5. **Hash Invariance Proof:**
   - Re-running `--smoke` and gate tests left all 43 files in `checkpoints.sha256` 100% bit-for-bit invariant (0 mismatches).

---

### Step 3: Comprehensive Checkpoint Verification ([`scripts/verify_checkpoints.py`](file:///c:/Users/srila/OneDrive/Desktop/BTECH/4th%20year/mini%20project/Supply_chain_alret_system/scripts/verify_checkpoints.py))
An independent Python verification harness evaluated all 20 checkpoints under `weights_only=True` and `strict=True`. It recomputed validation MSE across the 57,817 validation windows and compared against `training_summary.csv` under a strict relative tolerance of $10^{-4}$ (0.01%):

| Model Architecture | Seed | Recorded Val MSE | Recomputed Val MSE | Relative Diff | Status |
|---|---|---|---|---|---|
| `lstm` | 42 | 0.00093006 | 0.00093006 | $2.36 \times 10^{-6}$ | **[VERIFIED]** |
| `lstm` | 43 | 0.00092868 | 0.00092868 | $3.16 \times 10^{-6}$ | **[VERIFIED]** |
| `lstm` | 44 | 0.00095640 | 0.00095639 | $1.41 \times 10^{-6}$ | **[VERIFIED]** |
| `lstm` | 45 | 0.00095787 | 0.00095787 | $3.60 \times 10^{-6}$ | **[VERIFIED]** |
| `lstm` | 46 | 0.00093437 | 0.00093437 | $2.48 \times 10^{-6}$ | **[VERIFIED]** |
| `paper_overall` | 42 | 0.00025595 | 0.00025595 | $1.22 \times 10^{-5}$ | **[VERIFIED]** |
| `paper_overall` | 43 | 0.00024571 | 0.00024571 | $1.49 \times 10^{-6}$ | **[VERIFIED]** |
| `paper_overall` | 44 | 0.00024846 | 0.00024846 | $1.06 \times 10^{-6}$ | **[VERIFIED]** |
| `paper_overall` | 45 | 0.00024715 | 0.00024715 | $1.67 \times 10^{-6}$ | **[VERIFIED]** |
| `paper_overall` | 46 | 0.00024716 | 0.00024717 | $6.16 \times 10^{-6}$ | **[VERIFIED]** |
| `st_gcn_lstm_sym` | 42 | 0.00095671 | 0.00095668 | $3.92 \times 10^{-5}$ | **[VERIFIED]** |
| `st_gcn_lstm_sym` | 43 | 0.00142125 | 0.00142125 | $7.60 \times 10^{-9}$ | **[VERIFIED]** *(Persistence Match)* |
| `st_gcn_lstm_sym` | 44 | 0.00142121 | 0.00142121 | $3.99 \times 10^{-9}$ | **[VERIFIED]** *(Persistence Match)* |
| `st_gcn_lstm_sym` | 45 | 0.00095750 | 0.00095752 | $2.03 \times 10^{-5}$ | **[VERIFIED]** |
| `st_gcn_lstm_sym` | 46 | 0.00097078 | 0.00097076 | $2.45 \times 10^{-5}$ | **[VERIFIED]** |
| `st_gcn_lstm_dir` | 42 | 0.00090486 | 0.00090486 | $2.21 \times 10^{-6}$ | **[VERIFIED]** |
| `st_gcn_lstm_dir` | 43 | 0.00090149 | 0.00090148 | $4.81 \times 10^{-6}$ | **[VERIFIED]** |
| `st_gcn_lstm_dir` | 44 | 0.00090400 | 0.00090402 | $1.24 \times 10^{-5}$ | **[VERIFIED]** |
| `st_gcn_lstm_dir` | 45 | 0.00090276 | 0.00090275 | $2.64 \times 10^{-6}$ | **[VERIFIED]** |
| `st_gcn_lstm_dir` | 46 | 0.00090573 | 0.00090572 | $1.34 \times 10^{-5}$ | **[VERIFIED]** |

- **Analytical Persistence Validation Baseline:** **`0.00142117`**.
- Maximum relative difference across all 20 models was $3.92 \times 10^{-5}$ (well below $1.0 \times 10^{-4}$ tolerance).
- Zero NaN values, zero tensor dimension mismatches.

---

### Step 4: C3 Test Runner Skip Logic Gating ([`tests/run_phase_tests.py`](file:///c:/Users/srila/OneDrive/Desktop/BTECH/4th%20year/mini%20project/Supply_chain_alret_system/tests/run_phase_tests.py))
- Refactored `run_phase_tests.py` to prevent false-positive passes on skipped test suites.
- If all tests in a phase are skipped, the runner explicitly outputs:
  ```text
  --> GATE X [SKIPPED / UNIMPLEMENTED] (All N test(s) skipped)
  ```
  and returns `False` (process exit code 1), blocking pipeline progression.
- If only a subset of tests skip, the gate reports `[PASSED]` while outputting the exact skipped count for full visibility.

---

### Step 5: D1 Repair of Phase 2 Adversarial Suites & Scaler Protection
1. **Removed Non-Existent Imports:** Eradicated deprecated `segment_and_window` imports from [`tests/test_adversarial_phase2.py`](file:///c:/Users/srila/OneDrive/Desktop/BTECH/4th%20year/mini%20project/Supply_chain_alret_system/tests/test_adversarial_phase2.py) and [`tests/test_adversarial_phase2_challenger2.py`](file:///c:/Users/srila/OneDrive/Desktop/BTECH/4th%20year/mini%20project/Supply_chain_alret_system/tests/test_adversarial_phase2_challenger2.py).
2. **Standardized on Production Ingestion APIs:** Rebuilt synthetic data tests to use production functions `load_clean_frame(cfg)` and `build_datasets(cfg, save_scaler=False)`.
3. **Scaler Mutation Elimination:** Redirected `cfg.SCALER_PATH` to temporary directories in every synthetic test, ensuring `outputs/models/scaler.joblib` is never mutated.
4. **Execution Invariants:** All 15 adversarial tests executed and passed (15/15 OK in 17.8s), leaving `checkpoints.sha256` 100% bit-for-bit invariant.

---

### Step 6: C2 Read-Only Diagnostic of `st_gcn_lstm_sym` Seeds 43/44
To investigate the root cause of the validation loss plateau for symmetric seeds 43 and 44, a dedicated read-only diagnostic script ([`scripts/diagnose_sym_collapse.py`](file:///c:/Users/srila/OneDrive/Desktop/BTECH/4th%20year/mini%20project/Supply_chain_alret_system/scripts/diagnose_sym_collapse.py)) analyzed the state dict parameter norms, loss curves, and validation activations:

#### 1. Weight & Bias Norms Across Seeds 42–46

| Seed | Head Linear Norm (`head.0.w`) | Output Proj Norm (`head.3.w`) | Max Proj Weight | GCN Layer 1 Norm | LSTM Input Norm (`lstm.ih`) | Convergence Status |
|---|---|---|---|---|---|---|
| **42** | 8.005 | 0.200 | 0.102 | 4.312 | 32.317 | **Active / Converged** |
| **43** | 3.266 | 0.027 | 0.018 | 3.137 | 6.557 | **Persistence Plateau** |
| **44** | 3.276 | 0.007 | 0.003 | 3.232 | 6.581 | **Persistence Plateau** |
| **45** | 8.563 | 0.176 | 0.048 | 4.114 | 32.651 | **Active / Converged** |
| **46** | 9.060 | 0.188 | 0.085 | 4.571 | 34.849 | **Active / Converged** |

#### 2. Forward Activation & Delta Output Magnitudes ($\Delta = \hat{y} - y_t$)

- **Converged Seeds (42, 45, 46):** Mean $||\Delta|| \approx 0.0126 - 0.0131$, Max $|\Delta| \approx 0.283 - 0.333$. Active residual learning away from persistence.
- **Collapsed Seeds (43, 44):** Mean $||\Delta|| \approx 0.00042 - 0.00054$, Max $|\Delta| \approx 0.00021 - 0.00027$. Predicted output is numerically indistinguishable from persistence ($\hat{y} \approx y_{last}$).

#### 3. Loss Trajectories & Early Stopping
- **Seed 43:** Train loss flat at `0.001575`, Val loss flat at `0.00142118` across all 17 epochs before early stopping triggered.
- **Seed 44:** Train loss flat at `0.001576`, Val loss flat at `0.00142118` across all 11 epochs before early stopping triggered.

#### 4. Scientific Root Cause
With initial residual head weights near zero ($\Delta \approx 0$), the network starts at the analytical persistence baseline ($MSE = 0.00142117$). Under symmetric graph aggregation, the initial random gradient update directions for seeds 43 and 44 failed to escape the flat persistence identity saddle point within the patience window of 10 epochs. Early stopping halted training, leaving weights near their random initial state (`lstm_ih_norm` $\approx 6.56$ vs $32.3+$ for converged runs).

#### 5. Adoption of Option A Policy
Per the approved plan:
- **No retraining** was performed.
- All 20 model runs are preserved.
- The **all-5 seed mean** (**0.001145 $\pm$ 0.000252**) is reported as the primary headline figure, while the **converged 3-seed mean** (**0.000962 $\pm$ 0.000008** on seeds 42, 45, 46) is reported alongside for full scientific transparency.

---

## 3. Plan Corrections & Documentation Alignment

| ID | Issue Identified | Remediation Implemented | Affected Files |
|---|---|---|---|
| **M1** | Base paper citation contained inaccurate author attribution | Replaced base paper citation with exact verified authors: *Farzhana I., Dev Harris L., Shreyas S., 8th ICCMC 2025, DOI 10.1109/ICCMC65190.2025.11140739*. Retained *Banerjee et al. (2019)* strictly as dataset citation. | `README.md`, `PHASE_1.md`, `docs/PROJECT_PROGRESS_AND_EVALUATION_REPORT.md`, `NEW_AIM.md` |
| **M2** | Global pooling and 32/64/96 dimensions presented as paper claims | Clarified in `NEW_AIM.md` (Sections 2.1, 2.2) and report that global mean pooling and vector dimensionalities are SupplyGuard's concrete realization to make the baseline executable, not explicit paper specifications. | `NEW_AIM.md`, `Implementation Plan/NEW_AIM.md` |
| **M4** | TRI reporting ambiguity between scalar and node models | Formally locked policy: report scaled TRI for all models (mean of node predictions for node models; persistence TRI too). Raw-unit TRI = mean of raw node predictions. Raw TRI is never derived from `paper_overall`'s scaled output. | `docs/PROJECT_PROGRESS_AND_EVALUATION_REPORT.md` |
| **M7** | Corrupted UTF-16LE `requirements.lock` and unpinned Colab claim | Converted `requirements.lock` to clean UTF-8. Added metadata header noting Colab commit (`3d90be6`), A100 GPU, batch size 256, and that dependencies were installed via unpinned `requirements.txt` with PyTorch version unknown. Fixed report claim that Colab install was pinned. | `requirements.lock`, `docs/PROJECT_PROGRESS_AND_EVALUATION_REPORT.md` |
| **M8** | Model weights untracked in Git | Tracked and committed all 20 `.pt` model files (4.64 MB total), `scaler.joblib`, and `checkpoints.sha256*` to Git on `origin/main`. | Git repository index, `.gitignore` |
| **M9** | Potential resume truncation on Colab disconnect | Added atomic checkpoint writes (`.tmp` + `os.replace`) and completion marker `.training_completed`. Verified resume logic skips without retraining. | `training/train.py`, `outputs/results/.training_completed` |
| **M11** | Domain correlation discrepancy (M-R correlation 0.54 with no edge) | Documented discrepancy in limitations; graph ablation variants (identity and complete graph) formally deferred to Plan B. | `docs/PROJECT_PROGRESS_AND_EVALUATION_REPORT.md` |
| **D2** | Directory dependency relying solely on `.gitkeep` | Added `os.makedirs(..., exist_ok=True)` in `test_phase0_setup.py` so setup automatically creates all required output directories. | `tests/test_phase0_setup.py` |
| **D3** | Percentage completion, Laplacian terminology, and test window count | Updated progress to "Phases 0-4 of 9 complete" (removed percentage). Marked Phase 7 as "specification only". Replaced "Laplacian" with "renormalised adjacency". Updated test window count to exact 57,875 windows. | `docs/PROJECT_PROGRESS_AND_EVALUATION_REPORT.md` |
| **Dec. 2** | Standalone GCN baseline in `NEW_AIM.md` | Added note documenting that Plan A explicitly superseded `NEW_AIM.md` here because pure GCN without temporal recurrence cannot process sequential multi-step windows without artificial spatial flattening. | `docs/PROJECT_PROGRESS_AND_EVALUATION_REPORT.md` |
| **Claims** | Premature claims of "outperforming" or "statistically" | Toned down all claims to strictly factual validation loss metrics, reserving formal hypothesis testing for Phase 5 test evaluation. | `docs/PROJECT_PROGRESS_AND_EVALUATION_REPORT.md`, `PHASE_4.md` |

---

## 4. Git Commit & Artifact Tracking Summary

- **Primary Remediation Commit:** [`ba2ef7c`](https://github.com/Sanath1121/Supply-Guard/commit/ba2ef7c)  
  *Commit Message:* `fix(audit): complete post-Phase 4 hardening, checkpoint verification, and test isolation`
- **Current HEAD Commit:** [`75775a3`](https://github.com/Sanath1121/Supply-Guard/commit/75775a3)  
  *Commit Message:* `docs(review): track original pre-Phase 5 audit document in review/SupplyGuard_PrePhase5_Audit.md`
- **Clean `.gitignore`:** Commit [`15a0919`](https://github.com/Sanath1121/Supply-Guard/commit/15a0919) organized `.gitignore` into distinct default-ignore and whitelisted-deliverable blocks.
- **Review Handoff Archive:** Created [`SupplyGuard_PostFix_Repo.zip`](file:///c:/Users/srila/OneDrive/Desktop/BTECH/4th%20year/mini%20project/SupplyGuard_PostFix_Repo.zip) (10.95 MB) containing all code, models, tests, and audit documents.

---

## 5. Phase 5 Implementation Blueprint & Guardrails

With the codebase hardened, test sandboxes isolated, and checkpoints verified, Phase 5 development will adhere to the following locked protocol:

1. **Development & Sanity Check on Validation Split First:**
   - Develop `training/evaluate.py` using `--split val` (57,817 windows) for rapid iterative verification.
   - Run the final test evaluation (`--split test`, 57,875 windows) **strictly once** at the conclusion to prevent test set overfitting.
2. **Batched Inference Under `no_grad`:**
   - Execute model inference using `torch.no_grad()` in mini-batches (e.g., `batch_size = 2048`) to eliminate GPU/CPU memory spikes.
3. **Validation-Tuned Ridge Baseline:**
   - Ridge-AR(10) regularization parameter $\alpha$ will be tuned strictly on the validation set over $\alpha \in [10^{-4}, 10^{2}]$, never on the test set.
4. **Metrics Accounting:**
   - Per-node skill scores relative to persistence: $\text{Skill}_i = 1 - \frac{\text{MSE}_i}{\text{MSE}_{\text{pers},i}}$
   - $R^2$ on the step delta $\Delta y = y(t+H) - y(t)$.
   - Scaled TRI for all models; raw-unit TRI for node models only.
5. **Statistical Rigor:**
   - Report per-seed numbers, 5-seed mean $\pm$ standard deviation, and moving-block bootstrap confidence intervals.
   - Strictly avoid language of "outperforms" until test evaluation outputs are generated and verified.

---

## 6. Sign-Off & Verification Verdict

- **Step 1 (Hash Manifest):** ✅ **PASSED** (43 files verified against original Colab archive)
- **Step 2 (Sandbox Isolation):** ✅ **PASSED** (Zero hash mutation under `--smoke` and gate tests)
- **Step 3 (Checkpoint Audit):** ✅ **PASSED** (20/20 checkpoints verified within relative tolerance $3.9 \times 10^{-5}$)
- **Step 4 (Test Runner Gating):** ✅ **PASSED** (`[SKIPPED / UNIMPLEMENTED]` on full skips with exit code 1)
- **Step 5 (Adversarial Suite Repair):** ✅ **PASSED** (15/15 adversarial tests passing, zero scaler mutation)
- **Step 6 (C2 Empirical Diagnostic):** ✅ **PASSED** (Persistence plateau identified; Option A policy documented)
- **Phase 5 Readiness:** ✅ **CLEARED TO PROCEED**
