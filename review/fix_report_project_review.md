# Fix Report — Whole-Project Review (2026-10-09)

Scope: issues raised in the whole-project review of 2026-10-09 (Claude Code session "Project review").
Fix plan approved by the user with these decisions: use the training-mean IG baseline in the app; skip
`tiers.json` caching; regenerate `attribution_examples.csv`; leave the gate runner's skip reporting unchanged;
delete only files that must go.

A second agent (the front-end / presentation session) was editing `app/` at the same time. Its uncommitted
edits fixed review issues 2 and 3; this session did not edit those files and only verified them.

## Issue status

| ID | Review issue | Severity | Status | Change |
|---|---|---|---|---|
| F1 | Explanations ignored the selected model (Streamlit does not hash `_`-prefixed args) | Critical | **Fixed** | `app/utils/explain_service.py`: `_model_key` → `model_key`; explainer cache keyed by `model_key`; default IG baseline is now the training-partition mean (`get_train_mean_baseline()`), matching the paper and `scripts/generate_attributions.py`. Regression test `test_09_explanation_cache_keyed_by_model` added to `tests/test_phase7_app.py` (fails on the old code, passes now). |
| F2 | Benchmarks page showed all research-question verdicts as "Pending" | Major | **Fixed (by front-end agent), verified** | `app/utils/results_loader.py` (other agent). Added `test_10_rq_verdicts_from_real_results` to `tests/test_phase7_app.py`. On the real results: RQ1 Supported, RQ2 Supported, RQ3 Inconclusive (directed vs. symmetric gap is within the summed seed std), RQ4 Partly supported (7/10). |
| F3 | Dashboard severity used provisional 0.35/0.65 thresholds | Major | **Fixed (by front-end agent), verified** | `app/utils/artifacts.py` (other agent). Verified that the per-node thresholds equal `paper_facts.json` → `tercile_thresholds_scaled`. Caching them in `tiers.json` was skipped by user decision (saves ~15 s once at startup; would have meant editing a file another agent was changing). |
| F4 | `PHASE_5.md`: stale confusion matrix, persistence omitted from severity results, impossible "upstream message passing" claim for the Supplier | Major | **Fixed** | `PHASE_5.md` and `docs/PHASE_5.md` (identical). Confusion matrix and severity figures now taken from `confusion_matrix.csv` / `severity_metrics.csv` (persistence 83.62% is the best); Supplier statement corrected (the Supplier has no upstream neighbour, and the ablation is confounded); "directed significantly outperforms symmetric" corrected (gap 0.000057 < summed std 0.000064); caveat added under the pre-registered headline claim. |
| F5 | `PROJECT_PROGRESS_AND_EVALUATION_REPORT.md`: over-claims, garbled characters, root and `docs/` copies differed | Major | **Fixed** | Both copies now identical. Garbled UTF-8 repaired. Base = root copy (Phase 5 sections regenerated from CSVs on 2026-10-09); Phase 4 / §4.1 training numbers taken from the `docs/` copy because they match `training_summary.csv` (the root copy still had stale subset-run values: ~0.0013 val loss, "~200 seconds"). Removed "All Pre-Registered Claims Empirically Confirmed" and other over-claims; rewrote the corrupted Q1 answer; Q3 and Q4 no longer state unsupported explanations. |
| F6 | `narrate()` labelled the Δ-mode change as "Forecast risk" and the baseline as "window-average" | Major | **Fixed** | `src/explainability.py`: `explain()` records `residual_delta`; `narrate()` says "risk change vs. its current value" in Δ mode and "baseline-input forecast" otherwise. `outputs/results/attribution_examples.csv` regenerated: all numeric columns identical to before (max abs diff 0.0); only the `narrative` text changed. |
| F7 | Gate 8 tested a re-implemented copy of the claims rule | Major | **Fixed** | `tests/test_phase8_e2e.py` now calls `training.evaluate.evaluate_headline_claim` with tables in the evaluator's schema (4 cases, including "overall gap passes but Supplier gap does not"). |
| F8 | `patch_tests.py` rewrote test assertions to make them pass | Major | **Fixed** | File deleted (it had not been applied). |
| F9 | Sandbox enabled by default, contradicting the README and front-end docs | Minor | **Resolved the other way (user decision)** | The user wants the sandbox available in every default run, so `src/config.py` is unchanged (on unless `SG_ENABLE_SANDBOX=0`) and `README.md` now documents it as on by default. |
| F10a | `training/train.py` crashed the grid when a checkpoint existed without its loss CSV | Minor | **Fixed (changed approach)** | Now skips that run with a message to rerun with `--force`, instead of crashing. The plan said "retrain"; skipping was chosen so that existing checkpoints are never overwritten without `--force`. |
| F10b | Monitor ◀/▶ tooltip says "t-1/t+1" but steps ~116 windows | Minor | **Handed to front-end agent** | `app/views/monitor.py` belongs to the other agent, which is changing this text. |
| F10c | Sandbox presets use unseeded random numbers | Minor | **Skipped** | Not necessary; `app/views/sandbox.py` is front-end territory. |
| — | Gate runner reports "PASSED" when some tests are skipped | Minor | **Skipped (user decision)** | Unchanged. |
| — | Other stray files (`scratch_loss.py`, `checkpoints.sha256.bak`, untracked `test_*.py`, `test.html`, `scratch_test.py`) | Minor | **Skipped (user decision)** | Not deleted. |
| — | Graph-free ablation is confounded (no identity-adjacency control) | Research | **Not changed** | Acknowledged as a limitation (paper limitation 4); out of scope for a mini project. |

## Open item for the user

- **Base-paper author list is inconsistent.** `PROJECT_PROGRESS_AND_EVALUATION_REPORT.md` and
  `review/POST_PHASE4_AUDIT_REMEDIATION_REPORT.md` cite *Farzhana I., Dev Harris L., Shreyas S.*, while `README.md` and
  the paper's bibliography cite *Farzhana I. and Dev Harris L.* This was not changed; please confirm the author list
  against the IEEE Xplore record (DOI 10.1109/ICCMC65190.2025.11140739).
- `training/evaluate.py` still prints the pre-registered claim "Graph structure improves echelon-level forecasts on this
  dataset" for the real results; the reports now carry the confound caveat next to it. The rule itself was not changed.

## Files changed by this session

- `app/utils/explain_service.py`
- `src/explainability.py`
- `README.md` (sandbox section)
- `training/train.py`
- `tests/test_phase7_app.py`, `tests/test_phase8_e2e.py`
- `PHASE_5.md`, `docs/PHASE_5.md`
- `PROJECT_PROGRESS_AND_EVALUATION_REPORT.md`, `docs/PROJECT_PROGRESS_AND_EVALUATION_REPORT.md`
- `outputs/results/attribution_examples.csv` (regenerated)
- `patch_tests.py` (deleted)
- `review/fix_report_project_review.md` (this file)

## Verification (run 2026-10-09, with the raw dataset present, after all changes above)

| Check | Result |
|---|---|
| Gates 0–8 (`python tests/run_phase_tests.py --phase N`) | All PASSED, no skips: 5, 4, 4, 6, 7, 5, 6, 13 (was 11; +2 new tests), 2 checks |
| Smoke test (`python -m tests.smoke_test`) | ALL CHECKS PASSED |
| Adversarial suites (phase 2, 2-ch2, 3-ch1, 4-ch1, 4-ch2, 5, phase 1 oracle, phase 7 AppTest smoke) | Ran 52 tests, OK |
| New F1 test against the old `explain_service.py` | Fails as expected (0.367 ≠ 0.084 for symmetric seed 43) |
| Checkpoint SHA-256 vs. `checkpoints.sha256` | 21/21 match (checked before the fixes; no checkpoint was modified) |
