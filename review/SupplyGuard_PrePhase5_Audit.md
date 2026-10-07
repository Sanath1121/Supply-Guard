# SupplyGuard — Independent Pre-Phase-5 Audit

**Repo reviewed:** `Sanath1121/Supply-Guard`, branch `main`, commit **`0b3030c60e34578876b057c3e7ac7fd1d51a6a27`** ("docs: update master progress and evaluation report through Phase 4")
**Access method:** full `git clone` succeeded (public repo). Nothing was pushed, committed or changed in the remote repo. All runs happened in a throwaway local clone.
**Audit date:** 2026-10-06
**Scope:** Plan A Phases 0–4 as claimed in `PROJECT_PROGRESS_AND_EVALUATION_REPORT.md`, readiness for Phase 5.

---

## 0. What I could and could not access (read this first)

| Item | Status |
|---|---|
| Project files (NEW_AIM, MASTER_TECHSTACK, PLAN_A, Base_paper.pdf) and the progress report | Read in full. Base paper: text extracted with `pdftotext`; **Fig. 3.1 (architecture) is an image I did not inspect** |
| Source code, tests, notebooks, docs at the commit above | Read in full |
| Raw dataset | Downloaded with the repo's own `setup_and_download.py`; **SHA-256 matched the pinned value** (`d2e71ae7…6ff3`, 649,999 rows) |
| **Phase 4 outputs: 20 checkpoints, 20 loss CSVs, `training_summary.csv`, `scaler.joblib`** | **NOT in the repo** (git-ignored: `.gitignore` has `*.csv`, `outputs/models/`, `outputs/results/*`). **I could not inspect any of them.** Everything I say about Phase 4 results is either from the report's numbers or inferred by comparing those numbers with values I computed myself from the real data |
| GPU training | Not runnable here (CPU only). I did not re-train anything at scale. I trained throwaway 1-epoch models on a 30k-window subset only to test the checkpoint → evaluate integration path; those numbers are meaningless as results and are not used below |
| Environment | Python 3.12.3, torch 2.14.1 (CPU, installed `--no-deps` because the CUDA wheels filled the sandbox disk), scikit-learn 1.8.0, pandas 3.0.2. The repo's `requirements.lock` pins torch 2.12.0 and pandas 3.0.0; I could not install from the lock (see M7), so minor version drift exists |

Convention below: **[V]** = verified by running code / reading the file; **[I]** = inferred, with the reasoning shown.

---

## 1. Understanding check (6 lines)

SupplyGuard re-implements the IEEE ICCMC 2025 hybrid GCN+LSTM as a paper-style baseline (`paper_overall`) and extends it to a node-level spatio-temporal model (`STGCNLSTM`, directed/symmetric graph modes, residual-on-persistence head) forecasting next-step risk per echelon S/M/D/R, with overall TRI = mean of nodes. Data: Banerjee et al. 2019, one train CSV (649,999 rows), cleaned to 592,599 rows / 1,198 gap-free segments, 80:10:10 chronological split, horizon H = 5 steps ("10 min"), L = 10, 5 features. Plan A has Phases 0–8 with Gates 0–8; the report claims Phases 0–4 complete (Gates 0–4 pass 26/26), a 20-run Colab A100 training grid (4 models × 5 seeds, 685 epochs, 2.33 h), 110+ passing tests, and "62.5% complete". Headline training claim: `st_gcn_lstm_dir` has the lowest mean val MSE (0.000904 ± 0.000002) and "statistically outperforms" `lstm` (0.000941) and `st_gcn_lstm_sym` (0.001145). Phase 5 (evaluation vs. persistence/Ridge-AR(10), terciles, multi-seed aggregation) is next; Phase 7 has only a spec document.

---

## 2. Verdict

### **READY WITH FIXES**

The data pipeline and model code are sound: I re-ran the real-data pipeline and reproduced the report's row counts, segment counts and persistence figures, found no leakage across the chronological split, and confirmed the checkpoint → `evaluate.py` path works. But Phase 4's *results* cannot be verified from the repo, **two of the five `st_gcn_lstm_sym` runs look like failed runs that the report presents as "seed instability"**, and **running the project's own `--smoke` command (or its test suite) silently overwrites a real checkpoint filename (`outputs/models/lstm_seed42.pt`)** — you must confirm your Colab checkpoints are intact before evaluating anything. The gate harness also reports Gates 5–7 as `[PASSED]` while every test in them is skipped, so Gate 5 as built would be meaningless. None of this needs retraining except possibly the two sym seeds, but items C1–C3 must be resolved before Phase 5 numbers can be trusted.

---

## 3. Phase-by-phase scorecard

| Phase | Plan deliverables | In repo @ `0b3030c` | Status | Notes |
|---|---|---|---|---|
| **0** Setup / data | repo, `.gitignore`, `setup_and_download.py` (sha256, gap/null printout), `AGENTS.md`, env + `requirements.lock` | all present; script ran clean and matched pinned sha [V] | **Partial** | Gate 0 **fails on a fresh clone** (needs gitignored `outputs/models`, `outputs/figures`) [V]; `requirements.lock` is UTF-16 and a whole-workstation freeze [V] (M7) |
| **1** EDA / go-no-go | executed `01_EDA.ipynb`, decision table, horizon locked, check paper's metrics | notebook has executed outputs [V]; PHASE_1.md decision table present | **Done with defects** | Paper-citation wrong (M1); several quoted correlation/ACF numbers don't reproduce (M10); EDA ignores the strongest dependency M↔R (M11) |
| **2** Pipeline hardening | sort/dedup, segmentation, null policy, leak-free split, saved scaler, smoke test with injected gaps | `src/dataset.py` implements all [V] | **Done (code); tests partially broken** | Both adversarial Phase-2 suites fail at import (C3); horizon is only nominally 10 min (M6) |
| **3** Models / baselines | `GraphConv`, `STGCNLSTM`, `LSTMBaseline`, `PaperHybridOverall` (no Sigmoid), param table | all present; `param_counts.csv` tracked and matches report [V] | **Done** | Standalone-GCN baseline from NEW_AIM §6.1 is absent from Plan A / code (see §6) |
| **4** Colab training | `train.py`, `colab_train.ipynb`, 20 checkpoints + loss CSVs + summary | code and notebook present; **outputs absent** | **Partial / unverifiable** | Notebook committed with no outputs [V]; results untracked (M8); 2 sym runs suspicious (C2); smoke-overwrite hazard (C1) |
| **5** Evaluation | `evaluate.py`, metrics CSVs, plots, claims table | `training/evaluate.py` exists as a **draft**; runs on baselines [V] | **Draft only** | See §9 checklist |
| **6** Explainability | IG, Δ-attribution, deletion test | `src/explainability.py` has IG + narrate; **no Δ-attribution, no deletion test** | **Missing (not yet due)** | Report's file inventory says "Integrated Gradients & Delta-attribution" — Δ part does not exist |
| **7** Dashboard | `app/streamlit_app.py` | spec doc only (`docs/PHASE_7_…md`); no `app/` | **Missing (not yet due)** | Report calls Phase 7 both "SPEC READY" and "COMPLETED & SPECIFIED" |
| **8** Docs / viva | README, results notebook, clean-clone run | README exists | **Missing (not yet due)** | Gate 8 clean-clone check currently fails (Gate 0) |

---

## 4. Issues list

Severity: **Critical** = invalidates results or can destroy work; **Major** = misleads or blocks Phase 5 credibility; **Minor** = hygiene.

### CRITICAL

**C1 — `--smoke` and the test suite overwrite production checkpoint filenames.**
- *Where:* `training/train.py:198-220` (`run_smoke_mode` calls `train_one(cfg_smoke, "lstm", 42, …)` with the default `cfg.CKPT_DIR = outputs/models`); `train.py:186-188` (`train_one` saves unconditionally on first epoch; the skip-if-exists guard lives only in `main()` at line 281). `tests/test_adversarial_phase4_challenger1.py::test_02_cli_smoke…` runs the CLI smoke as a subprocess.
- *Verified [V]:* I placed a sentinel file at `outputs/models/lstm_seed42.pt`, ran `python -m training.train --smoke`; the file was replaced by a model trained on synthetic data (md5 changed). After running the test suite my clone also contained `outputs/results/training_summary.csv` with a stray row `paper_overall,43,0.0178,0.02` and `paper_overall_seed43_loss.csv` written by tests (AGENTS.md Rule 8 forbids this).
- *Impact:* If you ran the suite or `--smoke` in the folder holding your extracted Colab results, **`lstm_seed42.pt` (and possibly `training_summary.csv` / a loss CSV) may already be a synthetic-data artifact.**
- *Fix:* (1) before Phase 5, recompute val MSE from each of the 20 checkpoints and compare to `best_val_loss` in the summary (should match to ~1e-6); also compare sha256 with the original `outputs.zip` on Drive. (2) Make smoke/tests use `tempfile` dirs for `CKPT_DIR` and results; (3) make `train_one` refuse to overwrite an existing checkpoint unless `--force`; (4) commit a `checkpoints.sha256` manifest.

**C2 — Two of five `st_gcn_lstm_sym` runs look like failed (collapsed-to-persistence) runs, presented as "seed instability".**
- *Evidence:* Report §4.1 lists seeds 43 and 44 at **0.001421** and 0.001421. I computed the **persistence** validation MSE (mean over the 4 nodes, 57,817 val windows, via `persistence_metrics`) = **0.001421** [V]. Identical to 4 significant figures for two different seeds is not "instability"; it is the zero-initialised residual head (`_head`, `st_gcn_lstm.py:105-110`) never leaving its starting point (output = persistence). My own 1-epoch throwaway runs show the same plateau (lstm/sym ≈ 0.00144 vs persistence-level; dir already slightly below) [V, weak]. Conclusion that these were failed runs is **[I]**, because I do not have the loss CSVs.
- *Consequences:* the sym mean ± std (0.001145 ± 0.000252) is contaminated. The three healthy sym seeds average ≈ **0.000962**, i.e. *slightly worse than plain `lstm` (0.000941)*, not "blurred by relational mixing". Report §4.2 item 2's causal story is unsupported. RQ3 (directed vs symmetric) must not be answered from these numbers.
- *Fix:* open `st_gcn_lstm_sym_seed43/44_loss.csv` — if train loss is flat from epoch 1 and early stop fired at epoch ≈ 11, treat as failed. Decide a failure protocol now (e.g. re-run with a different init seed, or a small non-zero head init), report per-seed values rather than only mean ± std, and rewrite §4.2.

**C3 — Gate harness reports vacuous passes; Gate 5 as written cannot gate Phase 5.**
- *Where:* `tests/run_phase_tests.py` (`run_phase_test`: success = `result.wasSuccessful()`, message prints `result.testsRun`); `tests/test_phase5_evaluation.py` (`setUp` raises `SkipTest`).
- *Verified [V]:* `python tests/run_phase_tests.py --phase 5` prints `GATE 5 [PASSED] … (4 checks passed)` with `OK (skipped=4)`; Gates 6 and 7 identically (3 and 3). `--all` would therefore show every gate green. The Gate-5 test file also defines its own `compute_metrics` / `compute_tercile_tiers` and never imports `training.evaluate` (violates AGENTS.md Rule 8) and checks no artifacts, although the report §9 says Gate 5 "confirms all metrics artifacts exist".
- *Fix:* treat skips as not-passed in the runner; rewrite the Gate-5 test to import `training.evaluate`, run it on a small synthetic/tmp setup and on the real output CSVs, and assert the claims table rules (Plan §4).

### MAJOR

**M1 — Wrong citation of the base paper in `PHASE_1.md` §6.** It cites *"S. Banerjee, D. Sharma, R. Kumar, 'Graph-Based Risk Propagation and Machine Learning for Supply Chain Resilience', 9th ICCMC"*. The actual paper (checked against the PDF) is **Farzhana I. & Dev Harris L., "Hybrid GNN-LSTM Model for Real-Time Supply Chain Risk Prediction", 8th ICCMC 2025, DOI 10.1109/ICCMC65190.2025.11140739.** This is exactly the hallucinated-citation case AGENTS.md Rule 9 bans. Fix everywhere before submission (grep all `docs/`, README, notebook markdown).

**M2 — Statements about the paper's internals are not in the paper text.** NEW_AIM §2.1 and the report assert the paper uses global mean pooling, 32/64/96-dim vectors, a single TRI scalar and "current-step reconstruction". My keyword search of the extracted paper text found **no** mention of pooling, hidden sizes, sliding window, horizon or "next/t+1" [V]. It may be in Fig. 3.1 (an image I did not open) or in released code — **[unknown]**. What the paper text does say: MinMax on all numeric attributes **including the stability category as an input**, chronological **80:20** split, Adam + MSE, **50 epochs**, **dynamic LR scheduling**, gradient clipping, early stopping, 2-layer stacked LSTM with dropout, baselines **LSTM-only, GNN-only, ARIMA**, metrics MSE/MAE/R²/"accuracy 88%". Also **[I]:** the paper's MSE 0.12 with R² 0.92 implies target variance = 0.12/0.08 = 1.5, impossible for a MinMax-[0,1] target (max variance 0.25) — the reported numbers cannot all hold on scaled data; useful viva ammunition, but say "appears inconsistent", not "wrong". Either cite where the pooling claim comes from or word it as "our interpretation of the fusion module".

**M3 — `evaluate.py` runs inference on the whole test set in one batch.** `training/evaluate.py:24-27` calls `model(ds.sequences)` on 57,875 windows. [V] On the real test set the directed STGCN **was OOM-killed in my sandbox** (memory limit unknown). It may pass on a 16 GB laptop; it is fragile and will also hit the explainer. *Fix:* batch at ~4096 with `torch.no_grad()`.

**M4 — Cross-model comparison is not apples-to-apples.** `paper_overall` is trained and early-stopped on the **TRI** target and has no residual anchor (`train.py:122`), so its val loss (0.000249) is a different quantity from the node-MSE of the other three (report §4.1 puts them in one column; §4.2 item 3 only half-acknowledges it). In `evaluate.py:80` raw-unit TRI for `paper_overall` is computed as `tri_p * mean(range) + mean(min)`, which is **not** equal to the mean of per-node raw values (ranges differ 1.2–6.4 raw units [V]), so its raw metrics are biased. *Fix:* compare in scaled-TRI space only for `paper_overall`; do not report raw TRI for it; compare against persistence/Ridge **TRI** computed identically.

**M5 — Scaled MSE hides the Supplier node.** [V] Test persistence MSE per node (scaled): Manufacturer 3.7e-3, Distributor 1.3e-3, Retailer 1.1e-3, **Supplier 8.7e-5** (≈43× smaller). The equal-weight MSE loss on MinMax targets is dominated by Manufacturer; mean-node-MSE and TRI are too. Per-node claims (the project's whole point) need per-node *relative* metrics: skill = 1 − MSE_model / MSE_persistence, and R² of the *change* y(t+5) − y(t).

**M6 — "10-minute horizon" is a 5-row horizon.** [V, all windows] Actual target-minus-last-input time: **median 10.0, mean 10.8, min 4.0, max 22.0 min**; ~30% of windows fall outside 9–11 min; the 10-row input window spans up to 34 min (nominal 18). Cause: `GAP_MAX_MIN = 6` allows steps up to 3× cadence. Say "5 steps (≈10 min median)" in all documents, and in Phase 5 report results stratified by true Δt.

**M7 — Reproducibility artifacts are unreliable.**
- `requirements.lock` is **UTF-16-LE with BOM and CRLF** [V]; `pip install -r` will choke on it. It is also a full workstation freeze (187 lines: `google-cloud-aiplatform`, `fastapi`, `capstone`, …), not a project environment.
- `colab_train.ipynb` installs from **unpinned** `requirements.txt` (`>=`), while its markdown says "pinned". `mlflow` is in the tech stack but absent from `requirements.txt`.
- Colab was run with `--batch-size 256`; `Config.BATCH_SIZE` is 64. No artifact records batch size, Colab torch/CUDA versions or the git commit: `training_summary.csv` only stores model, seed, best_val_loss, wall_clock.
- *Fix:* regenerate the lock as UTF-8 from a clean venv (`pip freeze` in a project venv, saved with `> file` under bash/`Out-File -Encoding utf8` on Windows); write `run_config.json` (cfg snapshot, batch size, `torch.__version__`, git SHA, data SHA) next to each checkpoint going forward.

**M8 — Phase 4 evidence is not in the repo.** Checkpoints, loss CSVs, `training_summary.csv`, `scaler.joblib` are git-ignored; `colab_train.ipynb` was committed with **no outputs and no execution counts** [V]. Nobody (examiner included) can verify "20 runs, A100, 685 epochs" from the repository. Each checkpoint is ~0.2 MB (my throwaway `lstm` ckpt was 219 KB) so all 20 + CSVs fit easily in git or a GitHub Release.

**M9 — Skip-if-exists resume can accept a half-trained model.** `train.py:281` skips any run whose `.pt` exists, but `train_one` writes the `.pt` at the *first* improving epoch (line 188), long before the run finishes. A Colab disconnect mid-run leaves a "best-so-far" checkpoint that a re-run treats as complete, with no loss CSV or summary row. *Check:* for all 20 runs confirm a loss CSV and summary row exist, and that each run either ran 50 epochs or stopped with `last_epoch − best_epoch == 10`. 685 epochs / 20 runs = 34 average, so plausible but unverified.

**M10 — Report claims that are overstated or do not reproduce** (full list in §5). Most important: "statistically outperforming" with no test (AGENTS Rule 5 requires `evaluate.py` evidence beyond 1 std across seeds), "100% pass rate / 110+ tests" (not reproducible, see §5), 62.5% completion arithmetic, adjacent-tier correlations, "ACF > 0.99 for all echelons", and the L1 rationale about cross-year drift.

**M11 — The assumed S→M→D→R graph conflicts with the data's own dependency structure, and the EDA doesn't say so.** [V, recomputed within segments] Pearson levels: S–M 0.047, M–D 0.196, **D–R −0.014**, but **M–R 0.54** (the strongest pair, not an edge), S–D 0.18, S–R 0.17. The executed notebook's Granger table shows **M→S significant at 2–5 steps (F up to 34)** while S→M is significant only at 5 steps — the direction S→M is not supported at short lags. PHASE_1.md concludes "weak coupling" but omits M–R. *Implications:* RQ2/RQ3 are well-posed but the "upstream share" explanations in Phase 6 need heavy caveats; add cheap topology controls in Phase 5/6 (graph-free LSTM already exists; add *complete graph* and *random/permuted graph* variants with the hand-written GCN, optionally an M–R edge). Also note the notebook computes the correlation matrix over the full frame (cell 11), crossing gaps, and ACF/Granger on the single largest segment (5,045 rows, `bfill(limit=5)` used in the notebook), whereas PHASE_1.md says "strictly within segments".

### MINOR

- **m1** `tests/test_phase5/6/7_*.py` are `SkipTest` skeletons; fine as placeholders but counted as passes (C3).
- **m2** `evaluate.py:68` stores the *enumeration index* as `seed`; wrong if a checkpoint is missing. Persist real seed values and per-seed rows (`overall_by_seed.csv`).
- **m3** `evaluate.py:57` `Ridge(alpha=1e-3)` is untuned; select alpha on **val**, and also fit Ridge on the residual target `y(t+5) − y(t)`.
- **m4** `evaluate.py:42` refits the scaler (`save_scaler=False`) instead of loading `outputs/models/scaler.joblib`; assert they match. Scaler `Total_Cost` min is −23.9 (negative cost), range 223.9 [V] — check for outliers/units before the dashboard shows raw cost.
- **m5** `ffill(limit=5)` fills the first 5 rows of a long null run before the segment breaks; those flat rows can end up as targets. Effect looks negligible ([V] 0.0% of windows have target == last input on all four nodes) but is unquantified per node.
- **m6** `RESIDUAL_CONNECTION` is a shadow attribute of `RESIDUAL` (`st_gcn_lstm.py:49-52, 92-95`); harmless but confusing. `MAX_SAMPLES` subsampling is a global stride, not the segment-wise stride described in Plan Phase 2.
- **m7** Seven docs are duplicated (root and `docs/`, byte-identical [V]); `review/` is tracked; `PLAN_B_IMPLEMENTATION_PLAN.md` is referenced by AGENTS.md and the report but **not in the repo**.
- **m8** `test_adversarial_phase4_challenger1.py::test_09` hard-codes the string `from google.colab import files` but the notebook now uses `from google.colab import drive, files` (brittle test; the notebook is fine).
- **m9** No LR scheduler although the paper mentions dynamic LR scheduling (`train.py` logs a constant `lr`). Justify or note as a deviation.
- **m10** `setup_and_download.py` progress shows >100% (content-length is the gzip size); cosmetic.

---

## 5. Report discrepancies (claim vs. reality)

| # | Report claim | Reality | Basis |
|---|---|---|---|
| 1 | "100% pass rate across 15 suites, 110+ tests, 0 failures" (§1, §5, §10) | At `0b3030c`: **2 suites don't import** (`test_adversarial_phase2.py`, `…_challenger2.py` → `ImportError: segment_and_window`; the function was added in `a96cfa6` and **removed in `7de888a`**, before Phase 4). pytest on the rest: **57 passed, 10 skipped, 1 failed**. Gate 0 fails on a fresh clone. So "15 Phase-2 adversarial tests passed" was true only before `7de888a` | [V] |
| 2 | "26/26 gate checks pass" | 25/26 on a fresh clone (Gate 0 test_01 fails; Streamlit also had to be installed for test_02). Gates 5–7 print PASSED with all tests skipped | [V] |
| 3 | "62.5% complete, 5 of 8 core phases" | Plan A has **Phases 0–8 = 9 phases**; 5/9 = **55.6%**. Phase 7 is listed both "SPEC READY" and "✅ COMPLETED" | [V] |
| 4 | "`st_gcn_lstm_dir` statistically outperforming `lstm`" | No statistical test exists; validation only (one 3-month slice); dir vs lstm = 0.000904 vs 0.000941 (−3.9%) with +10.3k parameters. AGENTS Rule 5 requires `evaluate.py` on test | [V] code / report numbers |
| 5 | "sym suffered seed instability… relational blurring" | Seeds 43/44 = persistence-level val loss (C2); healthy sym seeds ≈ 0.000962 | [V]+[I] |
| 6 | Table 4.1 compares all four architectures' val loss | `paper_overall` is on a different target (TRI) | [V] code |
| 7 | "N_test ≈ 59,260 windows" (§9) | **57,875** test windows; 59,261 is the number of test *rows* | [V] |
| 8 | "Adjacent Pearson S→M ≈ 0.04, M→D ≈ 0.32, D→R ≈ 0.27" | PHASE_1.md itself says 0.027 / 0.213 / −0.016; my within-segment recomputation 0.047 / 0.196 / −0.014. 0.32 looks like the notebook's *max lead-lag* value, not Pearson; 0.27 not reproduced | [V] |
| 9 | "Lag-1 ACF > 0.99 across all 4 echelons" (report §3) | PHASE_1.md: 0.9812–0.9947 (largest segment only). Pooled within-segment: S 0.9755, M 0.9953, D 0.9881, R 0.9927 | [V] |
| 10 | "H=10 min lowers persistence to R² ≈ 0.90" | Pooled mean 0.901 ✓, but per node 0.859 / 0.964 / 0.883 / 0.898; Manufacturer stays 0.96. "Defensible task" is fair; "≈0.90" hides spread | [V] |
| 11 | L1: file chosen "to avoid cross-year drift between 2016 and 2018" | The "2018_train" file spans **2015-01-28 → 2018-12-19** with a 428-day gap (Dec 2016 → Feb 2018); train partition contains 2015–16 *and* 2018 data | [V] |
| 12 | Directed model "wins" (headline) | First valid evidence for RQ1 is missing from the report: persistence val MSE is **0.001421**, so dir/lstm are ≈ 36–37% below persistence on val. That is the real result to report | [V] persistence / report numbers |
| 13 | "All 20 checkpoints loaded with `weights_only=True`, zero NaNs, 2.33 h on A100, 685 epochs" | Arithmetic consistent (sum of mean wall-clocks × 5 = 8,388 s ✓) but **unverifiable from the repo** (M8) | [V] arithmetic / [unverifiable] |
| 14 | "`explainability.py`: Integrated Gradients & Delta-attribution" | IG yes; Δ-attribution (explain f(x) − y_t) is not implemented | [V] |
| 15 | PHASE_1 §6 paper citation | Wrong authors/title/conference number (M1) | [V] |
| 16 | Repo root named `Supply_chain_alret_system/` in the inventory | Actual repo is `Supply-Guard` (typo "alret" also in the name) | [V] |
| 17 | "100% data leakage prevention" | My checks found none, but "100%" is not provable. Note val/test windows deliberately borrow input context from the preceding partition (14 of 57,875 test windows start in val) — inputs only, targets stay in-partition, which is acceptable | [V] |

---

## 6. Tech stack compliance (`MASTER_TECHSTACK.md` / `PLAN_A`)

| Spec | Actual | Verdict |
|---|---|---|
| Hand-written `GraphConv`, symmetric + directed | ✓ `graph_layers.py` | ✓ |
| `STGCNLSTM` residual, shared per-node LSTM; `LSTMBaseline`; `PaperHybridOverall` | ✓ | ✓ |
| Sigmoid removed from paper model | ✓ (Gate 3 test) | ✓ |
| Persistence + Ridge-AR(10) + LSTM baselines | implemented in `evaluate.py`; Ridge/persistence ran OK [V] | ✓ (alpha untuned) |
| Single-tensor `seq [B,L,5]` contract | ✓ | ✓ |
| Segment-aware windows, ffill(limit=5), sort/dedup | ✓ | ✓ |
| Fixed timestamp format | ✓ `%m/%d/%Y %I:%M:%S %p`, 0 failures [V] | ✓ |
| Train-only scaler, saved | code ✓ (`dataset.py:119-124`); saved file not inspectable | ✓ (unverified artifact) |
| `requirements.lock` via `pip freeze` | UTF-16, global env | ✗ (M7) |
| MLflow optional | not in `requirements.txt` | minor |
| Severity tiers + **confusion matrix** + macro-F1 | accuracy + macro-F1 only; no confusion matrix output | partial |
| "% MSE improvement vs persistence ± std" (Plan Phase 5.2) | not computed | missing |
| Plots (pred vs truth, error by node) | none | missing |
| NEW_AIM §6.1 lists a **Standalone GCN** baseline (the paper's "GNN-only") | not in Plan A, not implemented | **inconsistency between aim and plan — decide** |
| Δ-attribution, deletion test (Phase 6) | not implemented | not yet due |
| Python 3.14-first / 3.12 fallback | I tested 3.12.3 only | not tested |

---

## 7. Fidelity to the base paper

| Aspect | Paper (text) | SupplyGuard | Justified? |
|---|---|---|---|
| Fusion GCN + LSTM | concat then FC | `paper_overall` concat then FC (scalar) | ✓ as baseline; "re-implementation" is the right word |
| Target | "Supply Chain Disruption Risk Score" (softmax or regression) | TRI (scalar) in baseline; per-node in extension | ✓ declared extension |
| Split | 80:20 chronological | 80:10:10 chronological | ✓ (val needed for early stopping) |
| Scaling | MinMax on all numeric attributes | MinMax, **train-only** | ✓ (fixes leakage) |
| Inputs | RIs, total cost, **stability category** | RIs + total cost; category excluded | ✓ plausible (global label, near-target), but state it explicitly |
| LSTM | 2 layers + dropout between | 2 layers, dropout 0.2 | ✓ |
| Training | Adam, MSE, **50 epochs, dynamic LR schedule**, grad clip, early stopping | Adam, MSE, ≤50 epochs, clip 1.0, patience 10, **no LR scheduler** | scheduler missing (m9) |
| Baselines | LSTM-only, GNN-only, ARIMA | persistence, Ridge-AR(10), LSTM; **no GNN-only, no ARIMA** | Ridge ≈ AR substitute is fine; GNN-only is missing (aim/plan mismatch) |
| Horizon | not stated in text | t+5 steps | ✓ declared |
| Metrics | MSE, MAE, R², "88% accuracy" (undefined) | MSE, MAE, RMSE, R² + tercile accuracy/macro-F1 | ✓ strictly better defined |
| "Pooling / 32-64-96 dims / current-step reconstruction" | **not found in text** | assumed in NEW_AIM | **unverified (M2)** |

---

## 8. Phase 4 outputs and integration with Phase 5

**Phase 4 outputs [cannot inspect — not in repo]:** completeness across 20 runs, divergence, early-stop pattern. What I *can* say from the report numbers + my recomputation:
- `lstm`, `st_gcn_lstm_dir`, `paper_overall`: tight across seeds, all below persistence on val (lstm/dir ≈ 0.00090–0.00096 vs persistence 0.001421; `paper_overall` 0.000249 vs persistence-TRI val MSE 0.000357) [V/derived]. No sign of divergence in these.
- `st_gcn_lstm_sym` seeds 43, 44: see C2 — likely failed runs.
- Wall-clock arithmetic consistent.

**Integration (what Phase 5 needs vs what exists):**
- Checkpoint naming: `{model}_seed{seed}.pt` via `ckpt_path`; `evaluate.py` imports the same function [V]. `build_model` handles `st_gcn_lstm_sym/dir` [V]. `state_dict` includes the graph buffers (`A_hat`, `A_down`, …) so loading is consistent [V, saw keys].
- Round trip tested [V]: train 1 epoch → save → `evaluate.main(cfg, seeds=[42])` loaded all four architectures and produced overall/severity CSVs with no errors.
- Data splits: `evaluate.py` rebuilds datasets from raw with the same code, so splits match training **only if** `dataset.py`/`config.py` are unchanged since Colab. `git diff af7230f HEAD -- src training` is empty [V] (and `7de888a → af7230f` changed only `train.py`), so the committed code equals what Colab cloned, provided Colab cloned after `7de888a`.
- Persistence (test): MSE per node S 8.7e-5, M 3.74e-3, D 1.28e-3, R 1.10e-3; TRI R² 0.961 [V]. **Ridge-AR(10) (alpha 1e-3): TRI MSE 3.54e-4 vs persistence 3.97e-4 (−11%), but MAE 0.00637 vs 0.00549 (worse) and tercile accuracy 0.795 vs 0.836 (worse)** [V]. Ridge helps on large errors only; persistence is a hard bar on typical-error and alert-tier metrics.

---

## 9. Top 5 priority fixes before Phase 5

1. **Prove your 20 checkpoints are intact (C1).** Re-compute val MSE for each loaded checkpoint and match `training_summary.csv`; compare sha256 with the original `outputs.zip`. Then make smoke/tests write to temp dirs and block overwrites.
2. **Diagnose and decide on `st_gcn_lstm_sym` seeds 43/44 (C2).** Open their loss CSVs; if collapsed, re-run (≈7 min each on A100) under a stated protocol, and rewrite report §4.2.
3. **Make Gate 5 real (C3).** Skips must fail the gate; the test must import `training.evaluate` and check the metric CSVs; fix the two broken adversarial imports and the Gate 0 fresh-clone dependency.
4. **Harden `evaluate.py` (M3–M5, m2–m3):** batched inference; per-node skill vs persistence and R² on the *change*; per-seed output files; confusion matrices; common TRI space for `paper_overall`; tune Ridge alpha on val.
5. **Correct the documents and lock the repo state (M1, M7, M8, M10):** fix the paper citation and every number in §5; regenerate a UTF-8 project-only lock; commit the small result artifacts (CSVs + checkpoint manifest, or Release).

---

## 10. Phase 5 readiness checklist

| Required | State |
|---|---|
| 20 checkpoints present and verified intact | **Unknown** — not in repo; C1 hazard |
| `scaler.joblib` present, matches rebuilt scaler | **Unknown** (m4) |
| Loss CSVs / summary for all 20 runs, no failed runs | **Unknown**; sym 43/44 suspicious |
| `evaluate.py` runs on baselines + model checkpoints | ✓ [V] (draft) |
| Batched inference (memory-safe) | ✗ (M3) |
| Persistence and Ridge-AR(10) | ✓ (alpha untuned) |
| LSTM / paper / sym / dir across 5 seeds, mean ± std | ✓ code; per-seed files ✗ |
| % improvement vs persistence, skill score, R² of change | ✗ |
| Severity terciles + confusion matrix + macro-F1 | partial (no matrix) |
| Raw-unit metrics | partial (invalid for `paper_overall`, M4) |
| Plots (pred vs truth slice, error by node, skill bars) | ✗ |
| Pre-agreed claims table enforced by a test | ✗ (Gate 8 test checks only string logic) |
| Gate 5 test meaningful | ✗ (C3) |
| Clean-clone reproducibility | ✗ (Gate 0, lock file) |

---

## 11. Suggestions for Phase 5 (credibility-focused)

1. **Freeze the protocol before looking at test numbers.** Commit `evaluate.py` + the claims table (Plan §4), then run once. Tune nothing on test (Ridge alpha, tier cut points on train/val only).
2. **Lead with skill against persistence, per node:** `skill_i = 1 − MSE_model,i / MSE_persistence,i`, plus **R² of the change** `y(t+5) − y(t)`. Levels R² ≈ 0.88–0.96 is inflated by autocorrelation; the change-R² is the honest signal. On val, the best deep models are ≈ 36% below persistence in mean node MSE — expect Supplier to behave differently (its target variance is tiny).
3. **Report MAE and median absolute error next to MSE.** Ridge beats persistence on MSE and loses on MAE and tier accuracy; any model that wins only on MSE is hedging on rare large moves.
4. **Uncertainty that respects autocorrelation:** five seeds capture init noise only. Add a **block bootstrap over contiguous segments** (100 test segments) for CIs on skill scores, and paired per-seed differences (dir − lstm) instead of overlapping ± std bars.
5. **Baselines/controls that make RQ2/RQ3 meaningful:** own-series-only Ridge vs multivariate Ridge (does *any* cross-node info help?); `lstm` (exists); *complete-graph* and *random/permuted-graph* variants of `st_gcn_lstm` (cheap; if they match `dir`, the topology isn't doing the work); optional M–R-edge variant given corr 0.54; the paper's **GNN-only** baseline if you keep NEW_AIM §6.1.
6. **Stratified analysis:** by true Δt (9–11 min vs others), by recent volatility terciles, by partition month, by segment length. Keep failed/collapsed runs in a separate table.
7. **Severity tiers:** persistence already gets 0.836 accuracy / 0.843 macro-F1 — report the **High-tier recall/precision** (alert utility), not just accuracy; add the confusion matrix per node.
8. **TRI handling:** report TRI only in scaled-mean space; compute persistence/Ridge/node-model TRI identically; for `paper_overall` don't convert to raw (M4).
9. **Save per-seed prediction arrays (`.npz`)** and the test timestamps so Phase 6/7 reuse them without re-inference.
10. **Integrity check at start of `evaluate.py`:** recompute val MSE per checkpoint and assert it equals `training_summary.csv` ±1e-6.

---

## 12. Open questions for you

1. Did you run `pytest`, the gate harness or `--smoke` inside the folder holding your extracted Colab outputs? Is `outputs/models/lstm_seed42.pt` the file from the original `outputs.zip` (sha256 / timestamp)?
2. What do `st_gcn_lstm_sym_seed43_loss.csv` and `…seed44_loss.csv` show — flat train loss from epoch 1? how many epochs?
3. Was `paper_overall` intentionally trained without a residual/persistence anchor and on TRI only?
4. Where do the "global mean pooling, 32/64/96, current-step" statements about the paper come from — Fig. 3.1, released code, or inference?
5. Which Colab Python/torch/CUDA versions were used, and was `--batch-size 256` the only override?
6. Where did "S→M 0.04, M→D 0.32, D→R 0.27" come from (lead-lag max?)
7. Do you still want a standalone GCN ("GNN-only") baseline (NEW_AIM §6.1), given Plan A omits it?
8. Who/what produced the PHASE_1.md paper citation — can you regenerate it from the PDF?
9. Is the repo public intentionally (dataset mirror, report paths with a local OneDrive path in `PHASE_1.md`)?

---

## 13. Appendix — what I actually ran

- `git clone`, `git log`, `git diff af7230f HEAD -- src training` (empty).
- `python setup_and_download.py` → sha256 matched; 649,999 rows; span 2015-01-28 → 2018-12-19; 2,363 duplicate stamps; 28 / 182 / 33,375 / 158 / 35,481 nulls in S/M/D/R/Cost; max null run 5,860 (D); max gap 428 d 07:04.
- `python tests/run_phase_tests.py --up-to 4` (Gate 0 blocked → ran each gate separately): G0 4/5 pass, G1 4/4, G2 4/4, G3 6/6, G4 7/7, G5/6/7 all skipped but "PASSED", G8 2/2.
- `pytest tests` (excluding two un-importable files): **57 passed, 10 skipped, 1 failed** (42 s). With them included: 2 collection errors.
- `python -m tests.smoke_test` (synthetic data): ALL CHECKS PASSED.
- Real-data pipeline: 592,599 rows, 1,198 segments, train/val/test windows **460,135 / 57,817 / 57,875**; train 2015-01-28 → 2018-06-19, val → 2018-09-17, test → 2018-12-19; chronological, no overlap of targets.
- Persistence on real partitions, correlations, ACF, horizon-in-minutes distribution, `evaluate.py` on baselines (numbers above).
- Sentinel test for the `--smoke` overwrite; 1-epoch throwaway training for all four architectures to test the load/evaluate path; full-test-set inference OOM.
