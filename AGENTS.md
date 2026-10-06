# SupplyGuard — Agent Guardrails & Operating Rules

These rules apply to any AI assistant or developer modifying this repository.

## 1. Source of Truth
- The definitive specifications are in `docs/PLAN_A_IMPLEMENTATION_PLAN.md` and `docs/MASTER_TECHSTACK.md`.
- Never re-implement archived designs from `old/` or unapproved proposals.
- Plan B extensions (`docs/PLAN_B_IMPLEMENTATION_PLAN.md`) must NOT be started until Plan A Gate 8 is complete and approved.

## 2. Data Pipeline Integrity (Strict Leakage Prevention)
- Never modify the core leakage prevention logic in `src/dataset.py` without explicit user approval.
- Chronological split (80% train, 10% validation, 10% test) is strictly by timestamp order—no shuffling.
- `MinMaxScaler` must be fitted **strictly on the train partition only**, and saved to `outputs/models/scaler.joblib`.
- Target $y(t+H)$ must strictly reside within its respective partition.
- Windows must never cross time gaps greater than `GAP_MAX`.

## 3. Architecture & Interface Constraints
- Single input tensor: models take one tensor `seq [B, L, 5]` (4 echelon risk indices + 1 total cost).
- Derivation of graph inputs must be done inside the model or dynamically from the last step `seq[:, -1, :4]`.
- No external PyTorch Geometric (`torch_geometric`) dependency; all graph convolution layers are implemented natively in PyTorch in `src/models/graph_layers.py`.
- Adjacency matrices must support both `symmetric` (Kipf-Welling) and `directed` (separate upstream/downstream weights) modes.

## 4. Verification & Testing
- Always run the relevant phase tests (`python tests/run_phase_tests.py --phase <N>`) or smoke test (`python -m tests.smoke_test`) after modifying code.
- Never declare a phase complete until all gate checks pass cleanly.

## 5. Claims & Attribution Honesty
- Never claim a model "beats" another or achieves superior accuracy unless `training/evaluate.py` verifies it beyond 1 standard deviation across the 5 seeds.
- Baselines (Persistence and Ridge-AR(10)) must always be benchmarked. If persistence R² is ~0.95–0.99, state honestly whether the deep learning model adds real value.
- Explainability outputs from Integrated Gradients represent local gradient sensitivity, **never** assert them as definitive "root cause" or causal proof.

## 6. Workspace Boundary Restraint
- Never read, modify, or archive files outside the project root (`Supply_chain_alret_system/`).
- Historical or parent directory planning files must not be imported into the repository without explicit user authorization.

## 7. Phase Completion Standard (Documentation, Commit & Push)
At the conclusion of each phase (once verified, tested, and confirmed complete), and **strictly before beginning implementation of the next phase**:
1. **Mandatory Markdown Report**: Author a completion report saved as `PHASE_<N>.md` at the repository root and mirrored at `docs/PHASE_<N>.md`.
2. **Atomic Git Commit**: Stage all phase deliverables and create a semantic commit (e.g., `git commit -m "feat(phase-<N>): complete Phase <N> deliverables and report"`).
3. **Mandatory Git Push**: Immediately execute `git push` to synchronize changes to `origin/main`. Do not begin the next phase until the report is written, committed, and pushed.
## 8. Testing Integrity & Anti-Tautology Rule
- **No Test Tautologies:** Never use mock data to validate identical mock logic. Gate tests must explicitly import and execute actual `src/` pipeline functions.
- **Strict Failure/Skip Default:** Unimplemented tests must `raise unittest.SkipTest("Not implemented")` or `self.fail()`. Never use empty `pass` blocks that falsely signal completion. Gate runners must never report skipped tests as passed.
- **Production Artifact Safety:** Test scripts (e.g., `smoke_test.py`, `--smoke` modes) must never overwrite production artifacts or production checkpoints. Always pass flags like `save_scaler=False` and use isolated temporary directories (`tempfile.mkdtemp()`) during testing.
- **Refactoring Safety Check:** Never delete or rename functions, classes, or configuration constants in `src/` without first searching across all `tests/`, `scripts/`, and `notebooks/` to ensure zero broken dependencies.

## 9. Documentation Honesty (Hallucination Ban)
- **Zero Inflated Claims:** Markdown completion reports (`PHASE_X.md`) must exactly match the state of the codebase. Do not claim 100% completion if future phases/gates are only stubbed.
- **Factual Justifications:** Never invent or guess explanations for data drops, hyperparameter choices, or performance metrics. If the reason is unknown, state it explicitly or ask the user.
- **Citation Accuracy:** Always use exactly verified citations (e.g., *Farzhana et al., IEEE ICCMC 2025*) based on the source of truth, avoiding generic/hallucinated authors.

## 10. Audit & Review Resolution Protocol
When assigned to fix issues from a review document (e.g., `merged_review_phaseX.md`):
1. **Mandatory Fix Plan:** Output a Fix Plan prioritizing issues (Critical -> Major -> Minor) and explicitly listing all "Open Questions" for the user. **Stop and wait for user approval before coding.**
2. **Scope Restraint:** Fix *only* the issues listed. Do not refactor unrelated code or accidentally start the next phase.
3. **Fix Report:** Conclude the cycle by authoring a `fix_report_*.md` document mapping every issue ID to its final status (Fixed / Skipped) and summarizing file changes.

## 11. Windows PowerShell & Environment Execution Invariants
- **No `&&` Command Chaining:** Windows PowerShell 5.1 does not support the `&&` operator (throws syntax error). Always separate chained commands with `;` or execute them as distinct tool calls.
- **Module Execution (`python -m`):** Always invoke project scripts from the repository root using `python -m <module_path>` (e.g., `python -m training.train`), or explicitly ensure `sys.path.insert(0, os.path.abspath("."))` in standalone scripts to avoid `ModuleNotFoundError: No module named 'src'`.
- **UTF-8 Output Encoding:** Never use PowerShell default redirection (`>`) to generate text, lock, or markdown files, as it creates UTF-16-LE files with BOM that break pip and Linux tools. Always write UTF-8 explicitly via Python or `Set-Content -Encoding utf8`.
- **Scratch Scripts for Complex Logic:** Never pass complex multi-line Python scripts with nested double quotes into PowerShell `python -c "..."` (which triggers quote-escaping syntax errors and CP1252 charmap encoding crashes). Write a temporary script in `scratch/` and execute it cleanly.

## 12. Repository Hygiene & `.gitignore` Transparency
- **Structured Gitignore:** Maintain `.gitignore` in clear, commented categories (Data, Outputs/Checkpoints, Whitelisted Results, Python/Virtual Environments, IDE/OS).
- **No Silent Whitelisting:** Never inject ad-hoc `!` whitelisting exceptions into `.gitignore` without explaining the rationale to the user.
- **Binary vs. Text Artifact Separation:** Large binary model weights (`outputs/models/*.pt`) remain git-ignored to prevent repository bloat, while small text/numerical results (`outputs/results/*.csv`, `checkpoints.sha256`) are explicitly tracked for reproducibility.
