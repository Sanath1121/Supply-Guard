# SupplyGuard: Merged Review, Phases 0–3

Scope: review-merge only. No code was modified. The only file created in the repo is this document.
Method: every issue below was checked against code, tests, plan, or a real run. All runs were in a scratch copy of the repo (`brain/…/scratch/repo_copy`). SHA-256 hashes of all 56 non-git files in the original repo were identical before and after (verified).

---

## 0. Input status (read this first)

| Input | Status |
|---|---|
| My own review `review/progress_review_phase0-3.md` | **Did not exist.** I stopped at a plan (`review_plan.md`) and never wrote the review. "My review" in this merge means the 13 preliminary findings in that plan. I have now re-verified each by running code. |
| Second review `C:\Users\srila\Downloads\deep-research-report.md` (note: `.md` extension) | Readable, 89 lines. **It is not a review of this project** (see §1). |
| Plan A, tech stack, `AGENTS.md`, `PHASE_0–3.md`, progress report, all code | Read in full. |
| Missing: `PLAN_B_IMPLEMENTATION_PLAN.md`, `MASTER_PLAN_AND_TECHSTACK.md`, the IEEE paper PDF | Not in repo. Plan B is out of scope; the PDF blocks verification of paper claims. |

---

## 1. The second review

**What it is:** a generic summary of reviewer comments on a *written manuscript* ("introduction is scattered", "Figure 3 caption unclear", "add related work by Smith et al., 2024", "sample selection missing"). It contains 0 mentions of SupplyGuard, phases, datasets, GCN, LSTM, persistence, scaler or any file (searched: 0 matches). It looks like the wrong file was downloaded, or it was generated from an unrelated conversation.

| Item from second review | Verdict | Evidence |
|---|---|---|
| Introduction scattered / concise | **Rejected (not applicable)** | No introduction exists in the repo to review. The nearest text is the README "Project Aim", a single paragraph. |
| Methodology missing "sample selection" | **Rejected (not applicable)** | Sample selection is explicit and documented: single CSV (Plan A L1), 80:10:10 split, segmentation, `ffill(limit=5)`. The real gaps in methodology documentation are different (see §3). |
| "Add citations, e.g. Smith et al. 2024" | **Rejected** | No such reference exists in the repo or plan. The real citation problem is the inconsistent base-paper citation (issue R-10). |
| "Conclusion lacks limitations" | **Unverifiable / not applicable** | No conclusion document exists yet. Plan Phase 8 requires a limitations section. Premature at Phase 3. |
| "Figure 2 / Figure 3 captions" | **Rejected (not applicable)** | No numbered figures in the repo. The EDA notebook has plots, not these figures. |
| "Use `collections.deque` instead of `list.pop(0)`" | **Rejected** | Searched all `.py`/`.ipynb`: **0** uses of `.pop(0)`, 0 of `deque`. The only `insert(0, …)` calls are `sys.path.insert(0, ROOT)`, which is not a queue. The data pipeline uses `sliding_window_view` and pandas, not list queues. |
| "Reviewers agree on needing more detail" | **Not applicable** | Refers to the reviewers of some other document. |

**Agreed (found by both reviews):** none.
**New (valid issues found only by the second review):** none.
**Yours/mine only:** all 13 findings in §2.
**Conflicts in severity or conclusion:** none exist, because the second review makes no claims about this project.

**Conclusion:** the second review adds nothing to the SupplyGuard audit. Please confirm you meant to attach a different file. If another review exists, send it and I will redo this section.

---

## 2. Re-verification of my own findings

Status: **Confirmed** (checked and real), **Corrected** (my original claim was wrong or overstated; corrected here), **Partly** (partly confirmed). Severity is my re-assessment.

| # | Finding | Status | Evidence | Severity |
|---|---|---|---|---|
| R-1 | Gate 2 test does not test production code | **Corrected** | `tests/test_phase2_dataset.py` L63–125 defines its own `segment_and_window` and tests that (L155–164 only checks shapes; L177–203 never builds real partitions). **But** the two adversarial suites do import `load_clean_frame`, `build_datasets` from `src.dataset` (`test_adversarial_phase2.py` L20, `…challenger2.py` L31) and passed in my run (10/10 and 5/5). So production code *is* tested, just not by the gate test. The gate is weak; the overall coverage is better than I first said. | Major → **Minor** (gate is weak; coverage exists elsewhere) |
| R-2 | `src/dataset.py` has dead duplicate code | **Corrected** | `subsample` (L111–119) and `_windows` (L122–130) are unused by any caller (only definitions found). `segment_and_window` (L133–192) is **not dead**: the adversarial tests import it. It is a test-only reference implementation living in `src/`, with hardcoded column names, and its null policy differs from `load_clean_frame`. | Minor |
| R-3 | `smoke_test` overwrites the real scaler | **Confirmed by experiment** | In the scratch copy, clean sequential run: scaler `data_min` before = `[0, 1.25, 0, 1, −23.94]` (real); after `python -m tests.smoke_test` = `[−2.48, −2.37, −2.23, −2.15, −7.81]` (synthetic). The **real repo's `outputs/models/scaler.joblib` already holds the synthetic values** (`data_min` −2.478…, −7.81) while the real CSV has min `[0, 1.251, 0, 1, −23.94]`. So the file in the repo is not the real train-partition scaler, contradicting `PHASE_2.md` and the progress report ("fit on the 80% train partition and saved"). Phase 4 retraining would overwrite it, but it is a latent trap: any later smoke run silently corrupts raw-unit conversion (Phase 5/7). Cause: `smoke_test.py` L66 calls `build_datasets(cfg)` with default `save_scaler=True`. | **Critical** for data integrity of reports; easy fix |
| R-4 | `train.py` / `evaluate.py` are unmodified drafts | **Confirmed by dry run** | 1-epoch run on a 3k-window subset (copy): works end-to-end (15.9 s for 3 models). But: (a) a second run of `lstm` seed 42 **retrained (8.2 s) instead of skipping** an existing checkpoint, so no Colab resume (plan Phase 4 §3); (b) `training_summary.csv` is rewritten per `main()` call and held only the last run (`lstm`) after the second call, so partial/resumed runs lose rows; (c) no wall-clock column in the summary (Gate 4 requires it); (d) `ckpt_path` L28–30 and `evaluate.py` L52 key checkpoints by `cfg.GRAPH_MODE`, so one run trains/evaluates one mode: evaluating `symmetric` printed `missing …st_gcn_lstm_symmetric_seed42.pt` and omitted the model; (e) `evaluate.py` L45 uses pooled terciles across all nodes, not per-node as the plan requires; (f) `MSE_raw_approx` uses the mean range for TRI. Note these are Phase 4/5 deliverables, so they are "not started", not "wrong for Phase 3". They matter because Phase 4 scope is under-specified in the repo. | Major (for Phase 4 start) |
| R-5 | Gates 4–8 pass with nothing implemented | **Confirmed by run** | `run_phase_tests.py --all` in my run: Gates 0–8 all `[PASSED]`, "READY TO PROCEED". `test_phase4_training.py` imports models from `tests.test_phase3_models`, tests its own loop, and never imports `training/train.py`. Greps: `test_phase6/7/8` import nothing from `src.explainability`, the app, or `training` (0 matches). Gate 5 defines its own `compute_metrics`/`compute_tercile_tiers`. So `PHASE_3.md` L91–92 ("all 9 gates passed") is true but meaningless. | **Major** (false assurance) |
| R-6 | Gate 0/1 tests are tautological | **Confirmed** (code read) | `test_phase0_setup.py` L75–96 builds a mock frame and asserts the mock's own null share is ≤10%. `test_phase1_eda.py` L91–101 asserts `10 if 0.985 >= 0.95 else 2` on constants, and 0.29 < 0.40 on a constant. Neither reads real results. (Gate 0 test 05 is a genuine spot check and passes.) | Minor/Major |
| R-7 | `PHASE_3.md` claims `hidden_dim` aliasing | **Confirmed** | Greps: `hidden_dim_lstm` and `hidden_dim_gcn` have 0 hits in `src/`, `scripts/`, `tests/`. `STGCNLSTM.__init__` (`st_gcn_lstm.py` L35–60) only reads `GCN_HIDDEN_DIM`/`LSTM_HIDDEN_DIM`. `count_parameters.py` uses `Config` defaults (LSTM hidden = 64, not the "32" stated in `PHASE_3.md` L34). | Minor (report error) |
| R-8 | `PHASE_0.md` claims a SHA-256 halt | **Confirmed** | `setup_and_download.py` L59–90 only computes and prints the hash; no comparison against `d2e71ae7…`. 0 hits for any expected-hash constant. Also `PHASE_0.md` lists `tests/phase_test_config.py` (does not exist) and says branch `master` (repo is `main`). | **Major** (a "pinned" integrity check that doesn't check) |
| R-9 | Cross-correlation numbers disagree | **Corrected (partly)** | They are different quantities, unlabeled. Oracle (run): peak |r| S–M 0.0576, M–D 0.3209 (lag 0), D–R 0.2739 (at lag +20, the edge of the tested range); lag-0 values 0.0397 / 0.3209 / 0.1598. Progress report (0.04 / 0.32 / 0.27) matches *peak* values. `PHASE_1.md` L72–76 (0.0272 / 0.2134 / −0.0160) and notebook cell 14 (0.0478 / 0.1976 / −0.0122) are lag-0 values that **disagree with each other and with the oracle's lag-0 values** (0.0397 / 0.3209 / 0.1598). The cause is unknown (different row subsets?). Also, the claim "lead-lag flat across −20…+20" is contradicted by the D–R peak at +20. | Minor/Major (credibility of RQ2 framing) |
| R-10 | Base-paper citation inconsistent | **Confirmed**, claims **Unverifiable** | Plan: *"Hybrid GNN-LSTM Model for Real-Time Supply Chain Risk Prediction"*, DOI 10.1109/ICCMC65190.2025.11140739. `PHASE_1.md` L117 and notebook cell 14: *"Graph-Based Risk Propagation and ML for Supply Chain Resilience"* by "Banerjee, Sharma, Kumar" (Banerjee is the dataset author). Quoted paper metrics and "Sections IV.D and V say synthetic" cannot be checked: no PDF in repo, and Plan Phase 1 §6 required that check before any critique. | **Major** (possibly fabricated critique) |
| R-11 | Progress report overstated | **Confirmed** | "50% complete": 4 of 9 phases = 44%; by plan hours (14–19 of 42–56) ≈ 25–45%. "Ahead of schedule": no evidence. Locked decisions L1–L5 in the report differ from plan L1–L6 (L5/L6 meaning changed). "~20 min on T4" and "5+ h on CPU": unmeasured (my 3k-window, 1-epoch run took ~2–5 s per model; no full-epoch timing exists). `+2,176` delta is right (layer 1: 2×(2×32)=128; layer 2: 2×(32×32)=2,048), but the report's formula ("2 weights + 2 biases") is wrong, since `down_lin`/`up_lin` have no bias (`graph_layers.py` L23–24). "Notebook Audit PASSED 1/1": see R-14. | Minor/Major (credibility) |
| R-12 | Sigmoid on non-residual paths; `build_model` can't pick mode | **Confirmed** | `st_gcn_lstm.py` L76 and L106 return `torch.sigmoid(out)` when `residual=False`. Only the Phase 3 test for 2-hop gradients uses that path. `build_model` (L141–145) takes only `name, cfg`; the Phase 4 plan names `st_gcn_lstm_sym/dir` but nothing maps them. | Minor / Major (Phase 4) |
| R-13 | Repo layout and hygiene | **Confirmed** | Repo path is under OneDrive (Plan §1 says outside). `colab_train.ipynb`: only a pip line, a commented clone URL (placeholder `your-username`) and `setup_and_download.py`; no train step. `.gitignore` ignores `outputs/results/`, `data/` and `*.csv`; `git ls-files` shows `param_counts.csv` is untracked, yet Gate 3 test 06 requires it. A fresh clone would fail Gate 3 (inferred, I did not clone). `PHASE_0–3.md`, `MASTER_TECHSTACK.md`, `PLAN_A…` are byte-identical at root and in `docs/` (hash check). | Minor |

### Additional findings from this verification pass (not in my original plan)

| # | Finding | Evidence | Severity |
|---|---|---|---|
| N-1 | `tests/audit_executed_notebook.py` defaults to `notebooks/test_executed.ipynb`, which doesn't exist, so it **fails when run as documented** ("FAILED: … does not exist"). It passes only when given `notebooks/01_EDA.ipynb` as an argument. The progress report lists it as "PASSED 1/1". | Run in copy; script L: `sys.argv[1] if len(sys.argv)>1 else "notebooks/test_executed.ipynb"` | Minor |
| N-2 | `test_phase1_challenge_oracle.py` is a script, not a unittest module: `python -m unittest` reports **"Ran 0 tests / NO TESTS RAN"**. It works when run directly. The progress report counts it as "3/3". | Run in copy | Minor |
| N-3 | **Real persistence numbers on the actual partitions** (production `build_datasets`): test R² per node 0.874 / 0.961 / 0.868 / 0.877 (mean 0.895); derived-TRI test R² **0.961**; val per node 0.909 / 0.965 / 0.888 / 0.885. In the dry run, Ridge on TRI scored 0.968 vs persistence 0.961 (3k-window subset, 1 seed). TRI R² is inside the 0.95–0.99 band that `AGENTS.md` §5 says must be discussed honestly. Gate 1 evaluated per-node R², while evaluate.py reports TRI. | Probe in copy | Informational/Major (for claims) |
| N-4 | Real partition dates: train 2015-01-28 → 2018-06-19, val → 2018-09-17, test → 2018-09-17 → 2018-12-19. Splits are chronological and non-overlapping (confirmed), but train spans ~3.4 years while val/test are all late 2018. Plan L1 justified train-only to "avoid cross-year drift", yet drift inside the file is unmeasured. | Probe | Minor |
| N-5 | Row accounting reconciles: 649,999 raw − 2,363 duplicates − 53,960 unfilled NaN − 1,077 short-segment = **592,599 (91.17%)**, 1,198 segments (all confirmed by run). `PHASE_2.md` L52 attributes the 57,400 dropped rows to "catastrophic gaps, extreme NaN runs, or short segments"; gaps themselves drop **no** rows. NaN rows (53,960) are 8.3% of raw, more than the 5.1/5.5% null shares, because ffilled rows are only the first 5 of each run. | Run | Minor |
| N-6 | Real-file target alignment checked on 3 test windows (first/middle/last): target − last-input time = **10.0 min** each; window spans 18–20 min (so a window can contain a 4-min step; allowed up to `GAP_MAX_MIN` 6). I did **not** verify all windows (my full-array check used a wrong unit and is discarded). The plan's Gate 2 manual real-file check is thus only spot-verified. | Probe | Minor |
| N-7 | Phase 1 persistence (notebook cell 13) uses `group[col].dropna().values` then offsets by H, so over null holes the effective horizon exceeds H steps for those columns. Magnitude not quantified. | Notebook source | Unverified/Minor |
| N-8 | Positive confirmations: gates 0–3 pass 20/20 in my run (6.38 s + 0.02 + 0.20 + 0.39); adversarial suites 2 (10/10), 2-challenger (5/5), 3-challenger (8/8) pass; smoke test passes (synthetic: persistence MSE 0.01416 vs lstm 0.00819 vs st_gcn_lstm 0.00790; IG completeness gap −0.00006); param counts match the CSV; 2-hop gradient test is meaningful (supplier RI reaches the distributor only through the graph). | Runs | n/a |

---

## 3. Verdict

**Ready with fixes.** The data pipeline and models are sound: leakage-safe chronological split, gap-aware segmentation (592,599 rows, 1,198 segments, reproduced), correct 10-minute targets on spot checks, 20/20 gate checks and all adversarial suites passing. But three issues should be fixed before launching Colab training: the stored `scaler.joblib` is synthetic and a normal smoke run overwrites it, `train.py` has no resume and cannot cover both graph modes in one run, and Gates 4–8 pass without any implementation, so "all gates pass" carries no information. Several phase reports also contain claims the code doesn't support (SHA halt, `hidden_dim` aliasing, paper citation, 50% completion), which should be corrected before they are shown to an examiner.

## 4. Scorecard

| Phase | Plan compliance | Code quality | Testing | Notes |
|---|---|---|---|---|
| 0 | Mostly done. Repo, `.gitignore`, `AGENTS.md`, lock file present. Partial: Colab env is a stub, repo is in OneDrive, SHA not enforced. | Good | Weak (tautological) | R-6, R-8, R-13 |
| 1 | Done: ACF, Granger, CCF, persistence by horizon, `HORIZON_MIN` locked. Partial: paper check not verifiable (no PDF), citation inconsistent. | n/a (notebook) | Weak gate; oracle is a script | R-9, R-10, N-2, N-7 |
| 2 | Done: sort/dedup, segmentation, null policy, train-only scaler, no gap crossing (verified). Partial: saved scaler is synthetic (R-3); stride sampling is over windows, not rows (a deviation from plan §Phase 2.4, not documented as such). | Good, some unused/duplicated helpers | Gate test weak; adversarial suites good | R-1, R-2, R-3, N-5 |
| 3 | Done: `GraphConv` (sym/dir), `STGCNLSTM`, `LSTMBaseline`, `PaperHybridOverall` without sigmoid, param table. Partial: report claims not in code. | Good | Good (2-hop test meaningful) | R-7, R-12 |

## 5. Deviations from the plan

| Deviation | Justified? |
|---|---|
| Repo inside OneDrive (plan §1: outside) | Plan "Status" says user-managed, no active sync. Acceptable, but undocumented in the plan body. |
| Gate 1 persistence rule "any node R² ≥ 0.99" (plan: "≥ 0.99 at chosen horizon") | Justified, stricter. |
| `MAX_SAMPLES` strides windows after windowing, not rows within segments | Not documented. Minor. |
| Sigmoid retained on non-residual paths | Not justified; harmless today. |
| Gate tests for phases 4–8 created early and passing | Not justified: gives false assurance. |
| Progress report "L1–L5" differs from plan "L1–L6" | Not justified. |

## 6. Report discrepancies (claim vs reality)

| Claim | Reality |
|---|---|
| `PHASE_0.md`: SHA-256 check halts on mismatch | Hash printed only; no comparison. |
| `PHASE_0.md`: `tests/phase_test_config.py`; branch `master` | File absent; branch `main`. |
| `PHASE_2`/progress: scaler fit on train partition and saved | File in repo is synthetic (smoke-test output). Real fit is correct in code. |
| `PHASE_2.md` L52: 57,400 rows dropped due to "catastrophic gaps…" | Dups 2,363 + NaN 53,960 + short 1,077; gaps drop no rows. |
| `PHASE_3.md`: `hidden_dim` / `hidden_dim_lstm` / `hidden_dim_gcn` aliasing; script uses `hidden_dim=32` | Not in code; LSTM hidden is 64. |
| `PHASE_3.md`/progress: `--all` passes all 9 gates | True, but Gates 4–8 test nothing in the repo. |
| Progress: "Notebook Audit PASSED 1/1", "Oracle 3/3" | Audit fails with default args; oracle is 0 tests under unittest. |
| Progress/`PHASE_1.md`/notebook: three different sets of cross-correlations | Mixed lag-0 and peak metrics; lag-0 numbers differ between sources. |
| `PHASE_1.md`: paper title/authors | Differ from Plan A citation; unverifiable. |
| Progress: "50% complete, ahead of schedule" | 44% by phases; no schedule evidence. |
| Progress: "+2,176 = 2×weights + 2×biases" | Number right; derivation wrong (no bias in `down`/`up`). |
| Progress: "~20 min on T4" | Unmeasured. |

## 7. Top 5 priority fixes before Phase 4

1. **Scaler safety.** Make `smoke_test` use `save_scaler=False` (or a temp `SCALER_PATH`), and regenerate `outputs/models/scaler.joblib` from the real file. Add a check that `scaler.data_min_` equals the real train-partition minimum.
2. **Make `train.py` Phase-4 ready.** Skip runs whose checkpoint exists, train both graph modes in one grid (e.g. names `st_gcn_lstm_sym/dir`), append to `training_summary.csv` with wall-clock per run, and time one epoch before choosing `MAX_SAMPLES`/epochs (plan Phase 4 §1).
3. **Remove false assurance from Gates 4–8.** Make those gate tests fail or skip until the code exists, and make them import `training.train`, `training.evaluate`, `src.explainability` and the app.
4. **Fix Gate 2/0/1 tests.** Gate 2 should call `build_datasets` on injected-gap data and on the real file (assert target = row +H in the same segment); replace the tautological Gate 0/1 assertions with checks on real outputs.
5. **Correct the reports.** Fix the SHA claim (and actually enforce the hash), the `hidden_dim` claim, the base-paper citation (check against the PDF), the 57,400 explanation, the correlation numbers (state lag-0 vs peak), the completion percentage and the unmeasured timing claims. Un-ignore (or regenerate in the gate) `param_counts.csv`.

## 8. Open questions for you

1. **Is `deep-research-report.md` the file you meant?** It reviews a manuscript, not this project. If another review exists, send it and I'll redo §1.
2. **Can you provide the base paper PDF** (or confirm the correct title and authors) so the Phase 1 paper claims can be verified?
3. **Which graph-mode naming do you want for Phase 4** (`st_gcn_lstm_sym/dir` per the progress report, or `st_gcn_lstm` + mode)? Fixes 2 and 3 depend on it.
4. **Should the Gate 4–8 tests be disabled (skipped) now**, or kept as specifications that fail until implemented?
5. **Should this document be committed?** `AGENTS.md` rule 7 covers phase completion only, so I left it uncommitted.
