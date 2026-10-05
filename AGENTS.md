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
