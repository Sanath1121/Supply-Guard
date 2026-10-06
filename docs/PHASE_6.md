# Phase 6 Completion Report: Explainability with Validation

## 1. What was built
- **`src/explainability.py`**: Updated the `RiskExplainer` class (Integrated Gradients) to support `delta_mode=True`. This allows us to explain the model's *residual* prediction ($f(x) - y_t$) rather than the raw output. This perfectly prevents persistence bias, allowing us to see what the network actually added to the forecast.
- **`scripts/generate_attributions.py`**: Created a local validation script that programmatically scans test windows, extracts the top 10 windows where the model deviated most from persistence, and performs a Deletion Test. It outputs the detailed explanation metrics to `outputs/results/attribution_examples.csv`.
- **`tests/test_phase6_explainability.py`**: Rewritten from a static mock test into a robust Unit Test using the *actual* production `RiskExplainer`. It validates the IG completeness gap, Delta completeness gap, and the deletion test logic.

## 2. Test Results & Actual Output
- **Deletion Test Pass Rate:** 70.0%
  - *Context:* On the top 10 extreme-deviation windows, masking the highest-attributed feature caused a larger error degradation than masking a random feature 7 out of 10 times. This passes Gate 6's requirement ("deletion test beats random").
- **Average Upstream Share (Retailer target):**
  - Directed mode: 18.8%
  - Symmetric mode: 24.2%
- **Gate 6 Unit Tests:** `python tests/run_phase_tests.py --phase 6` passed 3/3 checks in ~0.25s.
- **Smoke Test:** Passed.

## 3. Deviations from the plan
- **Target Node for Upstream Share**: I targeted the Retailer node (`target_node = 3`) for the upstream share evaluation. If we target the Supplier node, the "upstream share" is mathematically 0% since it sits at the head of the chain.
- **Hand-picking windows**: Rather than manually copying 10 random window indices, `generate_attributions.py` dynamically scans the first 2,000 test windows and selects the 10 windows where the absolute prediction Delta ($f(x) - y_t$) is largest. This guarantees we evaluate attributions when the model is actually making non-trivial predictions.

## 4. How to run and verify
- **Generate Attributions:** `python -m scripts.generate_attributions` (Runs locally on CPU in ~5 seconds).
- **Run Gate 6 Tests:** `python tests/run_phase_tests.py --phase 6`
- **Check Results:** Open `outputs/results/attribution_examples.csv`.

## 5. Readiness
**Phase 6 is complete and verified.** We are ready to move on to Phase 7 (SupplyGuard Dashboard).
