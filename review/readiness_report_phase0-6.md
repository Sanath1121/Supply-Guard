# SupplyGuard Phase 0-7 Readiness Report

## Verdict
**Not Ready (for a decoupled frontend connection).**
While the project's internal data pipeline, machine learning models, and testing suite are structurally sound and complete (102/102 tests pass), the system currently functions exclusively as a tightly coupled monolithic Streamlit application. There is **no REST API layer** (no FastAPI/Flask endpoints), which means there are no network interfaces, CORS policies, or HTTP response shapes for a separate frontend engineer to connect to. If the intention is to hand this off to an external frontend engineer to build a React/Vue client, the backend requires a new API bridging layer.

## Blockers
- **Missing API Layer for Frontend Handoff**: The backend currently only exposes Python functions (e.g., `app.utils.artifacts.get_model`) and does not expose standard network endpoints (REST/GraphQL). 
- **Missing Network Contracts**: Because there is no API, there is no documentation for HTTP methods, request/response JSON schemas, status codes, CORS configuration, base URLs, or authentication mechanisms.

## Issues List

### Blockers
1. **No Network API Endpoints (Critical)**
   - **Location**: Project architecture (missing FastAPI/Flask layer).
   - **What's wrong**: The prompt specifies handing off the backend to a frontend engineer requiring endpoints, CORS, and ports, but Phase 7 explicitly locked the architecture to Streamlit (a monolithic Python app).
   - **Evidence**: `docs/PHASE_7_FRONTEND_CONTEXT_AND_STREAMLIT_DECISION.md` and `docs/FRONTEND_DOCUMENTATION.md` show the UI is purely Streamlit. `MASTER_TECHSTACK.md` explicitly excludes FastAPI/React in Plan A.
   - **Suggested fix**: Implement a FastAPI wrapper over the core inference functions (`app.utils.artifacts.predict_window` and `explain_service.py`) if a decoupled frontend is required.

### Non-Blockers
2. **`InconsistentVersionWarning` on `MinMaxScaler` (Minor)**
   - **Location**: Pytest output / `app.utils.artifacts.py`
   - **What's wrong**: The `scaler.joblib` artifact was fitted using `scikit-learn` version `1.6.1` (likely on Colab), but the local environment (`requirements.lock`) specifies `1.8.0`. This triggers a warning during loading.
   - **Evidence**: Pytest warning log: `Trying to unpickle estimator MinMaxScaler from version 1.6.1 when using version 1.8.0.`
   - **Suggested fix**: Retrain/refit the scaler locally using `scikit-learn 1.8.0`, or pin the training environment to `1.8.0`.

3. **Unpinned core dependencies in `requirements.txt` (Minor)**
   - **Location**: `requirements.txt`
   - **What's wrong**: The file uses loose inequalities (e.g., `torch>=2.0.0`, `scikit-learn>=1.3.0`) which can lead to reproducibility failures (like the scaler mismatch above) compared to the strict versions in `requirements.lock`.
   - **Evidence**: Inspection of `requirements.txt` vs. `requirements.lock`.
   - **Suggested fix**: Replace `>=` with exact `==` versions aligned with `requirements.lock`.

## Front-End Integration Contract
Currently, the "frontend" is not a separate application; it is built with Streamlit and couples directly with the Python runtime in `app/streamlit_app.py`.
- **Exposed Interfaces**: Python functions located in `app.utils.artifacts` (`get_model`, `get_scaler`, `get_tiers`, `get_test_windows`).
- **Response Shapes**: Python dataclasses and tuples containing NumPy arrays, PyTorch models, and boolean status flags.
- **Error Handling**: Missing data or checkpoints return `is_loaded=False` and a string `warning_message`.
- **CORS/Ports**: None. Streamlit runs on a single port (8501) and handles its own websocket communication.
- **Can a frontend engineer connect?**: A React/Vue engineer **cannot** connect using the documentation that exists today, because there is no API documentation or REST/GraphQL layer. A Python/Streamlit engineer, however, has everything they need.

## Unverified Suspicions
- None. The previous test isolation issues have been fully resolved, and the pipeline executes perfectly.

## Things Verified as Working
- **Full Test Suite E2E**: Ran `python -m pytest tests/`. 102/102 tests passed successfully.
- **Reporting Honesty & Metrics Accuracy**: Verified that `outputs/results/overall_metrics.csv` accurately states `lstm` MSE (0.000719) outperforms `st_gcn_lstm_dir` MSE (0.000749), matching the Phase 5 report claims without hallucination.
- **Dataset Parsing**: `src/dataset.py` handles time gaps correctly and generates segments without leakage.
- **Model Checkpoints**: Weights and metrics exist for all 5 seeds (42-46) and load cleanly with `weights_only=True`.

## Re-running Training/Evaluation
- No deep training re-runs are necessary. The model metrics match identically.
- A quick re-run of the `dataset.py` pipeline (via setup script) might be beneficial to regenerate `scaler.joblib` with `scikit-learn 1.8.0` and silence the pickling warning.

## Open Questions for Me
- The documentation explicitly rejects React/FastAPI and locks to Streamlit. Was the prompt's request for "CORS, ports, and endpoints" assuming Plan B (a decoupled frontend) was already implemented?
- Should I proceed to generate a FastAPI layer to satisfy the API requirement, or are we continuing with the monolithic Streamlit application for the final handoff?
