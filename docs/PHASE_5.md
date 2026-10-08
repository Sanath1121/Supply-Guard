# Phase 5 Completion Report: Model Evaluation, Baseline Benchmarking & Statistical Verification

**Project:** SupplyGuard — Real-Time Multi-Echelon Supply Chain Risk Alert System  
**Phase:** Phase 5 (Model Evaluation, Baseline Benchmarking & Statistical Verification)  
**Status:** ✅ **COMPLETED & VERIFIED (Gate 5 Passed — VICTORY CONFIRMED)**  
**Date of Completion:** October 6, 2026  
**Primary References:** `docs/PLAN_A_IMPLEMENTATION_PLAN.md`, `docs/MASTER_TECHSTACK.md`, `AGENTS.md`, `review/fix_report_pre_phase5.md`

---

## 1. Executive Summary

Phase 5 executed the comprehensive, multi-seed statistical evaluation of all trained deep learning architectures against linear and persistence baselines on the untouched, chronological **Test Partition (57,875 windows / 231,500 echelon evaluation points)**.

### Headline Finding & Pre-Registered Claim
Per **Plan A Implementation Plan §4 (Claims Table)**, the pre-registered decision boundary was evaluated:
- **Naive Persistence Baseline Test MSE:** `0.000397` ($R^2 = 0.9613$)
- **Temporal Baseline (`lstm`) Test MSE:** `0.000284 ± 0.000003` ($R^2 = 0.9723$)
- **Proposed Model (`st_gcn_lstm_dir`) Test MSE:** `0.000273 ± 0.000003` ($R^2 = 0.9734$)
- **Difference (`lstm` − `st_gcn_lstm_dir`):** `0.000011`, which exceeds the combined 1-standard-deviation threshold (`0.000006`) by a factor of $1.83\times$.
- **Per-Node Supplier Benefit:** On the Supplier echelon, `st_gcn_lstm_dir` achieves a relative skill score of **23.4%** ($MSE = 6.64\times 10^{-5}$) vs. only **3.4%** for `lstm` ($MSE = 8.38\times 10^{-5}$). The Supplier has no upstream neighbour, and the two models differ in more than the graph, so this is not evidence that upstream message passing causes the gain (see §3.2).
- **Directed vs. Symmetric:** Directed mode has the lower mean MSE (`0.000273 ± 0.000003` vs. `0.000330 ± 0.000061`), but the gap (`0.000057`) is within the summed seed std (`0.000064`) because the symmetric model is unstable across seeds (two of five seeds stopped at the persistence solution). By the 1-std rule this is not a significant difference.

**Pre-Registered Headline Claim (output of `training/evaluate.py`):**  
> **"Graph structure improves echelon-level forecasts on this dataset."**

*Caveat:* the rule that produced this claim compares `st_gcn_lstm_dir` with a graph-free `lstm` that also differs in architecture (joint LSTM vs. shared per-node LSTM with echelon embeddings). The paper therefore does not attribute the gain to graph structure alone, and Naive Persistence remains the most accurate model on 3-tier severity (§3.3).

---

## 2. Key Deliverables & Implementation Summary

### 2.1 Production Evaluation Harness (`training/evaluate.py`)
- **Batched Memory-Safe Inference:** Implemented mini-batch evaluation using PyTorch `DataLoader(..., batch_size=2048)` with `torch.no_grad()`, ensuring peak memory remains $< 500$ MB across all 57,875 test windows.
- **Production Scaler Isolation:** Explicitly verified and loaded `outputs/models/scaler.joblib` fitted strictly on the train partition, verifying exact parameter parity.
- **Tuned Multivariate Ridge-AR(10) Baseline:** Optimized Ridge regularization parameter $\alpha \in [10^{-4}, 10^2]$ on the validation split, selecting optimal $\alpha = 10.0$ before test evaluation.
- **Fair Target Space Separation:** Separated direct scalar TRI evaluation (`paper_overall`) from 4-echelon node-level models (`lstm`, `st_gcn_lstm_sym`, `st_gcn_lstm_dir`, `persistence`, `ar10_ridge`).
- **Relative Skill Scores & Change $R^2$:** Implemented relative skill scores over persistence ($\text{Skill}_i = 1 - \frac{\text{MSE}_i}{\text{MSE}_{\text{pers},i}}$) and evaluated $R^2$ on the change $\Delta y = y(t+H) - y(t)$.
- **Automated Publication Graphics:** Generated high-resolution visualization figures saved to `outputs/figures/`.

### 2.2 Gate 5 Anti-Tautological Unit Suite (`tests/test_phase5_evaluation.py`)
- Completely eliminated `SkipTest` and mock definitions.
- Directly imports and verifies production functions (`reg`, `predict`, `severity`, `tune_ridge_alpha`, `evaluate_headline_claim`).
- Confirms mathematical equivalence between batched and unbatched inference, validates quantile digitizing, and asserts presence and validity of all result artifacts.

### 2.3 Adversarial Stress Test Suite (`tests/test_adversarial_phase5.py`)
- Tests skill score boundary conditions, conservation of the 231,500-sample confusion matrix, decision-boundary transitions in the Claims Table, and binary PNG magic headers.

---

## 3. Empirical Results & Benchmark Tables

### 3.1 Overall Multi-Seed Benchmark (Derived TRI & Direct Scalar TRI)

| Model Architecture | Target Space | Test MSE ($\pm$ Std) | Test MAE ($\pm$ Std) | Test $R^2$ ($\pm$ Std) | % MSE Improvement vs. Persistence |
|---|---|---|---|---|---|
| **`st_gcn_lstm_dir`** | Derived 4-Node Mean | **0.000273 $\pm$ 0.000003** | **0.005238 $\pm$ 0.000024** | **0.9734 $\pm$ 0.0003** | **+31.16% $\pm$ 0.68%** |
| **`lstm`** | Derived 4-Node Mean | 0.000284 $\pm$ 0.000003 | 0.005118 $\pm$ 0.000031 | 0.9723 $\pm$ 0.0003 | +28.47% $\pm$ 0.82% |
| **`paper_overall`** | Direct Scalar TRI | 0.000288 $\pm$ 0.000008 | 0.006088 $\pm$ 0.000095 | 0.9719 $\pm$ 0.0008 | +27.44% $\pm$ 2.08% |
| **`st_gcn_lstm_sym`** | Derived 4-Node Mean | 0.000330 $\pm$ 0.000061 | 0.005385 $\pm$ 0.000185 | 0.9678 $\pm$ 0.0059 | +16.77% $\pm$ 15.34% |
| **`ar10_ridge`** ($\alpha=10$) | Derived 4-Node Mean | 0.000354 (deterministic) | 0.006351 | 0.9655 | +10.81% |
| **`persistence`** | Derived 4-Node Mean | 0.000397 (deterministic) | 0.005493 | 0.9613 | 0.00% (Baseline) |

### 3.2 Per-Echelon Performance Breakdown & Relative Skill

| Echelon Node | Baseline Pers. MSE | `lstm` Test MSE | `st_gcn_lstm_dir` Test MSE | `lstm` Skill Score | `st_gcn_lstm_dir` Skill Score | Graph Skill Advantage |
|---|---|---|---|---|---|---|
| **Supplier** | $8.67 \times 10^{-5}$ | $8.38 \times 10^{-5}$ | **$6.64 \times 10^{-5}$** | +3.4% | **+23.4%** | **+20.0%** |
| **Manufacturer** | $3.74 \times 10^{-3}$ | $2.81 \times 10^{-3}$ | **$2.75 \times 10^{-3}$** | +24.8% | **+26.5%** | +1.7% |
| **Distributor** | $1.28 \times 10^{-3}$ | $7.14 \times 10^{-4}$ | **$6.63 \times 10^{-4}$** | +44.1% | **+48.0%** | +3.9% |
| **Retailer** | $1.10 \times 10^{-3}$ | $3.96 \times 10^{-4}$ | **$3.37 \times 10^{-4}$** | +64.0% | **+69.3%** | +5.3% |

*Insight:* On downstream tiers (Retailer, Distributor), both models perform well. At the **Supplier** tier, `lstm` captures almost no change beyond persistence (3.4% skill), whereas `st_gcn_lstm_dir` reaches **23.4% skill**. This gain cannot come from upstream messages: the Supplier is the first node of S→M→D→R and has no upstream neighbour (it only receives downstream messages from the Manufacturer). The two models also differ in more than the graph (`st_gcn_lstm_dir` uses a shared per-node LSTM with echelon embeddings, `lstm` one joint LSTM), so the gain is not attributed to graph structure alone; a same-architecture identity-adjacency control was not run (see the paper's limitation 4).

### 3.3 3-Tier Severity Tier Classification (Low / Medium / High)

Tercile boundaries calculated per node from the train partition (from `outputs/results/severity_metrics.csv`):
- **`persistence`:** Accuracy = **83.62%**, Macro-F1 = **0.8431** (deterministic) — the most accurate on this task
- **`st_gcn_lstm_sym`:** Accuracy = 81.93% ± 1.77%, Macro-F1 = 0.8250 ± 0.0195
- **`st_gcn_lstm_dir`:** Accuracy = 81.30% ± 2.38%, Macro-F1 = 0.8188 ± 0.0254
- **`lstm`:** Accuracy = 80.47% ± 1.59%, Macro-F1 = 0.8114 ± 0.0158
- **`ar10_ridge`:** Accuracy = 79.42%, Macro-F1 = 0.7988

No learned model exceeds Naive Persistence on 3-tier severity, although the directed model has the lowest MSE (§3.1).

#### Test Set Confusion Matrix (`st_gcn_lstm_dir`, Seed 42, 231,500 total points; `outputs/results/confusion_matrix.csv`):
| Actual \ Predicted | Predicted Low | Predicted Medium | Predicted High | Total Actual |
|---|---|---|---|---|
| **Actual Low** | **48,359** (20.9%) | 5,155 (2.2%) | 551 (0.2%) | 54,065 |
| **Actual Medium** | 4,692 (2.0%) | **71,284** (30.8%) | 10,314 (4.5%) | 86,290 |
| **Actual High** | 528 (0.2%) | 15,460 (6.7%) | **75,157** (32.5%) | 91,145 |

Severe misclassifications (Low predicted as High, or High predicted as Low) occur in **0.47%** of cases (1,079 out of 231,500 instances). Seed-42 accuracy is 84.15%.

---

## 4. Generated Artifacts & Visualizations

1. **`outputs/results/overall_metrics.csv`**: Comprehensive derived TRI comparison.
2. **`outputs/results/node_metrics.csv`**: Per-echelon metrics, raw units, skill scores, and $\Delta y$ change $R^2$.
3. **`outputs/results/severity_metrics.csv`**: Accuracy and macro-F1 metrics.
4. **`outputs/results/confusion_matrix.csv`**: Detailed 3x3 confusion matrix counts.
5. **`outputs/figures/prediction_vs_truth.png`**: 4-panel time-series slice comparing models against ground truth.
6. **`outputs/figures/error_by_node.png`**: Bar chart of MSE across echelons with skill annotations.
7. **`outputs/figures/severity_confusion_matrix.png`**: Confusion matrix heatmap.
8. **`outputs/models/checkpoints.sha256`**: Updated cryptographic manifest covering 47 artifacts.

---

## 5. Verification Gate Scorecard

| Verification Target | Command | Result | Duration | Checks Passed |
|---|---|---|---|---|
| **Gate 5 Test Suite** | `python tests/run_phase_tests.py --phase 5` | **PASSED** | 0.08s | 5 / 5 |
| **Full Regression (Gates 0–5)** | `python tests/run_phase_tests.py --up-to 5` | **PASSED** | 36.05s | 31 / 31 |
| **Phase 5 Adversarial Suite** | `python -m unittest tests.test_adversarial_phase5` | **PASSED** | 0.03s | 4 / 4 |
| **End-to-End Smoke Test** | `python -m tests.smoke_test` | **PASSED** | 48.2s | All |
| **Cryptographic Manifest** | `python scripts/generate_model_manifest.py --verify` | **PASSED** | 0.65s | 47 / 47 |

---

## 6. How to Reproduce & Verify Locally

From the repository root:
```bash
# 1. Run the evaluation harness (takes ~18s on CPU)
python -m training.evaluate

# 2. Run Gate 5 verification
python tests/run_phase_tests.py --phase 5

# 3. Run full regression across Gates 0 through 5
python tests/run_phase_tests.py --up-to 5

# 4. Verify cryptographic manifest
python scripts/generate_model_manifest.py --verify
```

---

## 7. Readiness for Phase 6

Phase 5 deliverables are **100% complete, tested, and validated against pre-registered claims**.  
The system is ready to proceed to **Phase 6: Explainability with Validation (`src/explainability.py`, Integrated Gradients, Δ-attribution, and deletion testing)**.
