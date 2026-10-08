# Fix report — research paper audit and project restore (2026-10-09)

Scope: findings in `Review.txt` plus issues found while checking the paper against the code and outputs.
Branch `checkpoint-2` (GitHub) preserves the state before any change (`d6ca45e`).

## Root cause behind most numeric findings

`src/dataset.py` (commit `f48fa62`) resampled to a 2-minute grid with `asfreq()`. The raw timestamps carry
arbitrary seconds (only 10,940 of 649,999 rows are on `:00`), so 98.2% of rows were discarded: 11,934 rows,
1,194 test windows, 17 days of 2015. The checkpoints, results, figures and four tests (which asserted the bug,
e.g. `n_clean_rows_retained == 11934`) all followed it. The paper's Table IV mixed the full-data Phase 5
persistence/Ridge rows with copies of the persistence row for the LSTM and SupplyGuard.

## Project changes (all on `main`)

| File / artifact | Change |
|---|---|
| `src/dataset.py`, `src/models/st_gcn_lstm.py` | Restored from `74ed751` (full-data pipeline; model definitions that match the 20 hash-verified checkpoints). |
| `tests/test_adversarial_phase2.py`, `…phase2_challenger2.py`, `…phase5.py`, `test_phase5_evaluation.py` | Restored from `74ed751`; they had been rewritten to assert the subset-run values. |
| `outputs/models/*.pt`, `outputs/results/*_loss.csv`, `training_summary.csv` | From `outputs-old.zip` (20/20 hashes match the `74ed751` manifest). |
| `outputs/results/{overall,node,severity}_metrics.csv`, `confusion_matrix.csv`, `param_counts.csv`, figures | Regenerated with `training.evaluate` and `scripts.count_parameters`; identical to the committed Phase 5 numbers. |
| `outputs/results/attribution_examples.csv`, `checkpoints.sha256` | Regenerated from the restored weights. |
| `scripts/paper_evidence.py`, `scripts/paper_figures.py`, `docs/paper/**` | New: reproducible evidence, paper-sized figures, corrected paper. |

Verification: `verify_checkpoints` 20/20 PASS; manifest verify OK; `run_phase_tests.py --all` Gates 0–8 PASSED;
48 adversarial tests OK; `tests.smoke_test` ALL CHECKS PASSED.

## Paper findings

| ID | Finding | Status |
|---|---|---|
| 1 | Table IV identical values for three models | **Fixed.** Rebuilt from `overall_metrics.csv` / `severity_metrics.csv` (5 seeds, ± std). Not Δy ≈ 0: the rows had been copied. |
| 2 | "Outperforms all baselines" vs Ridge | **Fixed.** Ridge 3.54e-4 vs directed 2.73e-4; severity accuracy now stated honestly (persistence best). |
| 3 | Fig. 6 3-class counts vs 4-class text | **Fixed.** Fig. 6 and text both use the code's 3-tier scheme (terciles). 4-class numbers removed (no code produced them). |
| 4 | Split sizes / date range | **Fixed.** 592,599 rows; 474,079/59,259/59,261 rows; 460,135/57,817/57,875 windows; no 2017 records stated. |
| 5 | GCN direction | **Fixed.** Eq. (1) rewritten to the implemented relational layer (`A_down = rownorm(Aᵀ)`, `A_up = rownorm(A)`, separate weights). |
| 6 | IG scalar target; Algorithm 1 dimensions | **Fixed.** Scalar target node k; Algorithm 1 uses the code's shapes (2 node features, 32 channels, per-node LSTM of hidden 64). The `Review.txt` "clean Algorithm 1" (129-dim LSTM input, h ∈ ℝ³²) did **not** match the code and was not used. |
| 7 | Hardware | **Fixed.** A100 40 GB per `PHASE_4.md`; timings labelled as CPU. Please confirm this matches your actual run. |
| 8 | Overclaims ("proves", "35%", "cannot") | **Fixed.** Supplier-share claim removed: the audit shows the Supplier contributes ≈0.9% to the Distributor and ≈0% to the Retailer forecast. |
| 9 | References | **Partly fixed.** [1] replaced (Ivanov–Dolgui–Sokolov 2018, DOI found online); [2] title corrected; [3] and [10] corrected to the README's citations; [7] title corrected (verified online); [11] replaced; all 20 now cited. Still to confirm by you: [10] (not found in web search; DOI and authors taken from `README.md`), and bibliographic details of the classic references, which were not re-checked online. |
| – | Hyphenation artefacts, hype phrases | **Fixed** (text rewritten). |
| – | Numbered objectives | **Added** (O1–O5, Section I-B). |
| – | Figs 1–3 showed GAT, React, SQLite, T = 14, a fifth node | **Redrawn** from the code structure. |
| – | Table II statistics did not match the data | **Fixed** (computed from the cleaned training partition). |
| – | Latency (12.4 / 14.2 ms), IG gap (0.000238), 98.x% accuracy | **Replaced** with measured values or removed. |

## Not done / for you

- Plagiarism check (Turnitin or similar) — must be run by you.
- Author list, title and affiliation were left unchanged.
- No identity-adjacency control with the same architecture was run (listed as a limitation).
- The deletion test covers 10 windows chosen for large corrections; 7 of 10 pass.
- Training was not re-run; results are the Phase 5 run. The Colab environment is documented in `PHASE_4.md`, not re-verified.
- Layout checked by opening in Word (7 pages, 7 figures, 8 tables); please skim it once in Word before submitting.

## Documentation reconciled after the restore

`PHASE_5.md`, `docs/PHASE_5.md` and `PROJECT_PROGRESS_AND_EVALUATION_REPORT.md` still carried the subset-run numbers
(e.g. persistence MSE 0.000995, 4,776 test points). `PHASE_5.md` and `docs/PHASE_5.md` were restored from `74ed751`;
the Phase 5 sections of the progress report (dashboard line, key achievements 1-3, Tables 4.2-4.4) were regenerated
from `outputs/results/*.csv`. The old "proves node attribution benefit" wording was removed, and the report now notes
that the graph-free ablation is architecture-confounded. Grep for the old values (`0.000749`, `4,776`, `1,194 windows`)
now returns no hits outside historical review files and the evidence note.
