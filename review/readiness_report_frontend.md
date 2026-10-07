# Frontend Readiness Report

**Verdict:** `ready after fixes`

**Justification:** The frontend is well-structured, respects the recent design specifications (SVG animations, no numbers), and properly uses dynamic predictions across most views. However, the sandbox view explicitly violates the "NO mockdata" rule by simulating disruption scenarios using hardcoded stochastic arrays rather than real data perturbations.

## Blockers
### Issue 1: Mock Data Simulation in Sandbox
* **Severity:** High
* **Location:** `app/views/sandbox.py`, Lines 54-82
* **What's wrong:** The "Preset Shock Scenarios" tab generates synthetic sequences using `np.zeros`, `np.linspace`, and `np.random.normal` instead of drawing real test windows from the dataset. This violates the explicit rule against using mock data to simulate pipelines.
* **Evidence:** In `app/views/sandbox.py`, the code block starting at line 55 manually builds sequences like `seq[:, i] = np.clip(base_line + noise, 0.0, 1.0)` to simulate upstream/midstream disruptions.
* **Suggested Fix:** Replace the synthetic stochastic generation with an actual historical test window from the dataset (`get_test_windows()`), and apply realistic shock perturbations to that real sequence instead of fabricating it entirely.

## Unverified Suspicions
* No unhandled exceptions or syntax errors were identified during static review, but runtime edge cases in the multi-page navigation when navigating between "Monitor" and "Diagnostic XAI" (`st.session_state["selected_node_idx"]`) might require full integration testing to guarantee state persistence.

## Things Verified as Working
* **Animated SVG Network Graph:** Verified in `app/ui/canvas.py`. The required design updates are fully implemented:
  * Floating animations exist (`sg-float-ambient`, `sg-float-node-0`).
  * Delayed disruption animations are correctly implemented with `1.0s` CSS animation delays (`sg-critical-glow`, `sg-shockwave-1`).
  * NO numbers exist in the graph (replaced with categorical labels like "FLOW ▶", "CASCADE ▶", "● HIGH RISK").
* **Dynamic Components:** The `monitor.py` and `export.py` views dynamically generate real test-partition windows and apply real models without hardcoded placeholders.
* **Windows & Environment Rules:** No invalid `&&` chaining or incorrect file redirection was found in the frontend implementation.

## Open Questions
* Should the sandbox be removed entirely if synthetic scenario simulation inherently requires mock data, or is applying perturbations to a real lookback window sufficient for closure?

## Remaining Questions & Gaps
* Could not fully verify edge cases where `Config.RAW_DATA_PATH` is missing and `get_test_windows` returns empty lists; the UI shows warning states, but actual user experience wasn't interactively tested.
