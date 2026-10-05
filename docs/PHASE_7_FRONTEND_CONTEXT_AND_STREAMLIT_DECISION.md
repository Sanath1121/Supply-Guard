# SupplyGuard — Phase 7 Frontend Architecture, Streamlit Rationale & Handoff Specification

**Project:** SupplyGuard — Real-Time Multi-Echelon Supply Chain Risk Alert System  
**Document Type:** Architectural Decision Record (ADR) & Frontend Developer Handoff Guide  
**Target Audience:** Frontend Developers, Full-Stack Engineers, ML Engineers, Project Evaluators  
**Primary References:** `docs/PLAN_A_IMPLEMENTATION_PLAN.md`, `docs/MASTER_TECHSTACK.md`, `src/config.py`, `src/dataset.py`, `src/explainability.py`  
**Date:** October 5, 2026  
**Status:** Approved Architectural Baseline for Phase 7  

---

## 1. Executive Summary & Document Purpose

Phase 7 of **SupplyGuard** delivers the user-facing demonstration dashboard. Its primary objective is to take the validated PyTorch spatiotemporal models (`STGCNLSTM`, `PaperHybridOverall`, baselines), test-partition temporal tensors, train-fitted scalers, and Integrated Gradients explainability outputs, and present them in a clean, interactive, and academically defensible web interface.

This document serves as the **complete architectural context and handoff guide** for frontend developers. It details:
1. **Why Streamlit was chosen** over competing web stacks (React, Dash, FastAPI+Next.js).
2. **The comprehensive trade-off analysis** (operational reliability vs. custom DOM control).
3. **The exact functional requirements for Phase 7** (the 4 mandatory tabs, data contracts, and design constraints).
4. **The research and enhancement roadmap** for frontend engineers looking to push visual polish, animations, and determine whether a decoupled stack is warranted for future iterations.

---

## 2. The Architectural Journey: Why Streamlit Was Chosen

During the project planning and technical review phases (Reviewer Responses 1, 2, and 3), the team evaluated multiple frontend options. In machine learning research and academic projects, frontend decisions often determine whether a project succeeds or fails during live demonstrations.

```
+---------------------------------------------------------------------------------------------------+
|                                  THE DUAL-STACK INTEGRATION TRAP                                  |
|                                                                                                   |
|  [React / Next.js Frontend]                                                                       |
|         │                                                                                         |
|         ▼  (HTTP / WebSocket JSON Serialization - NaN traps, float precision, CORS issues)        |
|  [FastAPI / Node Backend]                                                                         |
|         │                                                                                         |
|         ▼  (IPC / Tensor Bridge - In-memory PyTorch serialization, autograd lockouts)             |
|  [PyTorch Engine & Scaler]                                                                        |
|                                                                                                   |
|  * FAILURE MODES DURING VIVA: Port conflicts (8000 vs 3000), npm dependency drift, CORS blocks,   |
|    dual-server startup latency, JSON conversion crashes on complex multidimensional tensors.     |
+---------------------------------------------------------------------------------------------------+
                                                VS.
+---------------------------------------------------------------------------------------------------+
|                                   THE STREAMLIT UNIFIED RUNTIME                                   |
|                                                                                                   |
|  [Streamlit UI + Plotly + CSS] ──(Direct In-Memory Call)──> [PyTorch Engine + Scaler + Data]      |
|                                                                                                   |
|  * SINGLE PROCESS: 1 command (`streamlit run app/streamlit_app.py`), zero network latency,       |
|    zero serialization overhead, zero CORS issues, 100% reproducible on any examiner machine.     |
+---------------------------------------------------------------------------------------------------+
```

### 2.1 Core Justifications for Streamlit

1. **Zero-Latency In-Memory Coupling with PyTorch:**
   - In SupplyGuard, when a user selects a historical test window, the application executes:
     $$\hat{y} = \text{model}(X_{\text{seq}}, \mathcal{G})$$
     followed by:
     $$\text{Raw Units} = \text{scaler.inverse\_transform}(\hat{y})$$
     and optionally computes 64-step Integrated Gradients paths:
     $$\text{Attr} = (x - x_0) \times \frac{1}{m} \sum_{k=1}^m \nabla f\left(x_0 + \frac{k}{m}(x - x_0)\right)$$
   - In Streamlit, this runs **directly inside the active Python runtime**. Tensors and Joblib scalers reside in the same memory space. There is zero JSON serialization overhead, zero float32 truncation, and no need to construct complex Pydantic request/response schemas.

2. **Viva Defense & Evaluation Resilience (Single-Command Execution):**
   - The ultimate failure mode in academic defenses is a multi-service crash (e.g., FastAPI backend boots on port 8000, but the React frontend fails to connect due to CORS errors or mismatched Node.js versions on the examiner's laptop).
   - Streamlit boots the complete application via a single command:
     ```bash
     streamlit run app/streamlit_app.py
     ```
   - It requires only the verified `requirements.txt` environment, ensuring 100% cross-platform reproducibility.

3. **High-Performance Built-In Caching:**
   - Streamlit provides `@st.cache_resource` and `@st.cache_data`.
   - The heavy PyTorch checkpoints (`outputs/models/*.pt`), graph adjacency matrices, and `scaler.joblib` are loaded **exactly once** into GPU/CPU memory at startup. Window switching and tab transitions execute in $< 50\text{ ms}$.

4. **Native PyData Ecosystem Integration:**
   - SupplyGuard relies heavily on Plotly Graph Objects (`plotly.graph_objects`), NetworkX directed graph layouts, Pandas DataFrames, and Seaborn confusion matrices. Streamlit renders all of these natively with interactive tooltips, zoom, and pan out-of-the-box.

---

## 3. Technology Trade-Off Analysis: Streamlit vs. Alternatives

The table below summarizes the technical, operational, and development trade-offs between Streamlit and competing architectures:

| Evaluation Dimension | Streamlit (Plan A Choice) | React/Next.js + FastAPI | Plotly Dash | Gradio |
|---|---|---|---|---|
| **Development Effort** | **6–8 hours** (Rapid, pure Python) | 25–40 hours (Dual codebase) | 14–20 hours (Verbose callbacks) | 4–6 hours (Too rigid) |
| **Runtime Architecture** | Single process (Python) | Decoupled (Node.js + Python) | Single process (Python) | Single process (Python) |
| **PyTorch / Scaler Coupling** | Direct memory pointer | REST / WebSocket JSON bridge | Direct memory pointer | Direct memory pointer |
| **Viva Reproduction Risk** | 🟢 **Extremely Low** (1 command) | 🔴 **High** (CORS, Node, ports) | 🟡 **Low–Medium** | 🟢 **Low** |
| **DOM / CSS Customizability** | 🟡 Medium (Custom CSS injection) | 🟢 **Unlimited** (Tailwind, HTML5) | 🟡 Medium | 🔴 Very Low (Template-locked) |
| **Complex Animation Support** | 🟡 Good (Plotly, Lottie, SVG) | 🟢 **Superb** (Framer Motion, D3) | 🟡 Fair | 🔴 Poor |
| **Multi-Tab Dashboard Fit** | 🟢 **Excellent** (`st.tabs`) | 🟢 **Excellent** (React Router) | 🟢 **Excellent** (Dash Core) | 🔴 Poor (Form-oriented) |
| **Interactive Graph Topology**| 🟢 Native via Plotly / PyVis | 🟢 Canvas / SVG / React Flow | 🟢 Native via Plotly | 🔴 Limited |

### Why Alternative Stacks Were Rejected for Plan A:
- **React / Next.js + FastAPI:** Rejected for the core submission because building a separate client-side SPA, managing state (Redux/Zustand), writing serialization pipelines for 4-dimensional tensors, and handling CORS doubles project risk without adding scientific value to the research paper.
- **Plotly Dash:** While powerful, Dash requires a verbose `Input`/`Output`/`State` decorator callback graph that is significantly slower to iterate and refactor than Streamlit’s reactive top-down execution model.
- **Gradio:** Designed for input-form ML demonstrations (e.g., text-prompt $\to$ image). It lacks native support for complex multi-echelon supply chain operational dashboards with dynamic graph topology and temporal scrubbing.

---

## 4. Phase 7 Functional Requirements Specification (The 4 Core Tabs)

The Phase 7 dashboard must implement **four distinct operational tabs**, fed exclusively by real test-partition data and verified models.

```
+---------------------------------------------------------------------------------------------------+
|                                  SUPPLYGUARD DASHBOARD ARCHITECTURE                               |
+---------------------------------------------------------------------------------------------------+
|  HEADER: Offline Historical Replay | Test Window Selector [Slider / Dropdown: 2018-05-12 14:20]   |
+---------------------------------------------------------------------------------------------------+
|  [Tab 1: Echelon Risk Cards]                                                                      |
|  - Supplier Risk: 0.28 (1.12 RI) [LOW]   | Manufacturer Risk: 0.64 (2.75 RI) [HIGH]               |
|  - Distributor Risk: 0.42 (1.80 RI) [MED] | Retailer Risk: 0.35 (1.50 RI) [MED]                   |
|  - Total Risk Index (TRI): 0.422 (Mean Risk across 4 echelons)                                    |
+---------------------------------------------------------------------------------------------------+
|  [Tab 2: Directed Topology Network]                                                               |
|  - Interactive Directed Graph: S ──(35%)──> M ──(45%)──> D ──(20%)──> R                           |
|  - Nodes dynamically colored by risk tier (Green/Orange/Red)                                      |
|  - Edge thickness proportional to computed upstream attribution share                             |
+---------------------------------------------------------------------------------------------------+
|  [Tab 3: "Why This Forecast?" (Explainability)]                                                   |
|  - Feature Importance Bar Chart (Signed: Positive raises risk, Negative lowers risk)              |
|  - Temporal Attention Bar Chart (Time steps t-9 to t)                                             |
|  - Toggle: Full Attribution vs. Delta-Attribution (Network gain beyond persistence)               |
|  - Dynamic NLP Narrative: narrate() sentence explaining primary risk drivers                      |
|  - Axiom of Completeness status caption (|Sum(Attr) - DeltaF| < 0.01)                            |
+---------------------------------------------------------------------------------------------------+
|  [Tab 4: Benchmark Comparison & Empirical Validation]                                             |
|  - Side-by-side Table: Persistence vs Ridge-AR vs LSTM vs PaperHybrid vs ST-GCN-LSTM (sym/dir)    |
|  - Metrics: MSE, MAE, RMSE, R², % Improvement over Persistence (+/- std across 5 seeds)          |
|  - Severity Confusion Matrix & Macro-F1 score                                                     |
+---------------------------------------------------------------------------------------------------+
```

### 4.1 Tab 1: Echelon Risk Health Cards & Replay Scrubber
- **Historical Scrubber:** A timeline slider or window selector allowing examiners to scrub through historical test-partition timestamps (e.g., from `2018-05-01` to `2018-12-19`).
- **4 Echelon Risk Cards:**
  - Dedicated cards for **Supplier**, **Manufacturer**, **Distributor**, and **Retailer**.
  - **Metrics Displayed per Card:**
    1. Normalized Risk Score: $\hat{R}_i \in [0.0, 1.0]$.
    2. Real-World Sensor Units: Unscaled via `scaler.joblib` (e.g., `2.45 RI`).
    3. Severity Badge: Color-coded tier (`Low`, `Medium`, `High`) derived strictly from train-partition terciles.
- **Summary TRI Card:** Displays the derived Total Risk Index:
  $$\text{TRI} = \frac{1}{4} \sum_{i=1}^4 \hat{R}_i$$

### 4.2 Tab 2: Directed Supply Chain Topology Network
- **Graph Structure:** A clean 4-node directed graph representing the linear echelon flow:
  $$\text{Supplier} \longrightarrow \text{Manufacturer} \longrightarrow \text{Distributor} \longrightarrow \text{Retailer}$$
- **Dynamic Node Styling:** Node background color maps to its current predicted tier:
  - `Low`: Emerald Green (`#10B981`)
  - `Medium`: Amber Orange (`#F59E0B`)
  - `High`: Crimson Red (`#EF4444`)
- **Dynamic Edge Styling (Attribution-Weighted):**
  - Directed arrows indicate physical material flow.
  - Edge thickness and opacity map dynamically to the **computed upstream attribution share** from Integrated Gradients (e.g., if Distributor risk is 60% driven by upstream Manufacturer shocks, that connecting edge widens and glows).
- **Technology:** Implemented via Plotly Graph Objects (`go.Scatter` with custom coordinates) or PyVis / NetworkX.

### 4.3 Tab 3: "Why This Forecast?" (Explainability & Attribution Panel)
- **Input Feature Importance:** Horizontal bar chart showing signed contributions of all 5 input features:
  $$[\text{RI}_{\text{Supplier}}, \text{RI}_{\text{Manufacturer}}, \text{RI}_{\text{Distributor}}, \text{RI}_{\text{Retailer}}, \text{Total\_Cost}]$$
  Positive bars (red) indicate factors increasing risk; negative bars (green/blue) indicate factors mitigating risk.
- **Temporal Attention Profile:** Vertical bar chart showing the relative attribution of each step in the sliding window $[t-9, t-8, \dots, t-1, t]$.
- **The $\Delta$-Attribution Toggle (Critical Plan A Feature):**
  - An interactive toggle allowing the user to switch between:
    1. **Full Forecast Attribution ($F(x)$):** Often dominated by the node's own risk at step $t$ due to strong temporal persistence.
    2. **Residual $\Delta$-Attribution ($F(x) - y_t$):** Explains what the neural network specifically learned *beyond* the persistence baseline.
- **Automated Natural Language Narrative:** Displays the output of the verified `narrate()` function:
  > *"Forecast Distributor risk is 0.54 (baseline 0.48). Most influential input: Manufacturer RI (48% of attribution), which elevates the forecast. Most influential time step: t-1 (34%)."*
- **Scientific Audit Caption:** Displays the Integrated Gradients Completeness check:
  $$\left|\sum \text{Attributions} - (F(x) - F(x_0))\right| < 10^{-3}$$
  confirming that the explanation passes axiomatic validation.

### 4.4 Tab 4: Benchmark Comparison & Empirical Validation
- **Performance Leaderboard:** Displays the benchmark comparison across all 5 seeds ($mean \pm std$):
  - Persistence Baseline
  - Ridge-AR(10) Linear Baseline
  - Graph-Free LSTM Baseline
  - Base Paper Re-implementation (`PaperHybridOverall`)
  - Proposed `STGCNLSTM` (Symmetric Mode)
  - Proposed `STGCNLSTM` (Directed Mode)
- **Reported Metrics:** MSE, MAE, RMSE, $R^2$, and percentage MSE improvement over persistence.
- **Severity Classification Quality:** Displays the confusion matrix heatmap and macro-averaged F1 score for the 3 tercile tiers.

---

## 5. Non-Negotiable Operational Constraints & Rules

Frontend developers must adhere strictly to the following architectural rules:

1. **Rule 1: Historical Replay Only (No Fake Sliders):**
   - The dashboard must be clearly labeled: **"Offline Historical Replay Mode"**.
   - Developers must **NOT** create interactive sliders that allow examiners to fabricate arbitrary sensor values. Predictions must strictly be evaluated on actual historical test windows from `test_dataset`. Arbitrary inputs violate the spatiotemporal distribution and produce nonsensical model outputs.
2. **Rule 2: Zero Hardcoded Narrative Strings:**
   - Explanation text must **never** be hardcoded. It must be generated programmatically from the `RiskExplainer` gradient attribution arrays.
3. **Rule 3: Honest Labeling of Causality:**
   - Never use the term *"Root Cause"* in the UI. Integrated Gradients measures **model sensitivity / feature attribution**, not verified physical causation. Label all panels as *"Sensitivity Attribution"*.
4. **Rule 4: Sub-Second Interactive Latency:**
   - Test window selection and tab rendering must complete in under $500\text{ ms}$. All models and datasets must be wrapped in `@st.cache_resource`.

---

## 6. Frontend Developer's Research, Enhancement & Evaluation Guide

For a frontend developer tasked with elevating Phase 7, here is the exact roadmap to research, enhance, and evaluate the dashboard.

### 6.1 Enhancing Streamlit: Visual Polish & Animations

Can Streamlit achieve high-end aesthetic polish and fluid animations? **Yes.** A skilled frontend engineer can achieve near-React visual fidelity within Streamlit using the following techniques:

#### 1. Custom CSS Injection (Glassmorphism & Theming)
Streamlit allows custom CSS injection via `st.markdown("<style>...</style>", unsafe_allow_html=True)`. The frontend developer can implement:
- Modern dark-mode palette: Deep slate background (`#0F172A`), card surfaces (`#1E293B`), borders (`#334155`).
- Frosted-glass metric cards with subtle drop shadows (`box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1)`).
- Glowing risk badges with pulsating CSS keyframe animations for `High` risk status:
  ```css
  @keyframes pulse-red {
    0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
    70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
    100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
  }
  .risk-badge-high {
    animation: pulse-red 2s infinite;
  }
  ```

#### 2. Vector Animations via Lottie (`streamlit-lottie`)
- Integrate lightweight JSON vector animations for supply chain logistics:
  - Supply flow status indicators (moving logistics nodes).
  - Neural network computation indicators.
  - Alert radar pulses during elevated disruption states.

#### 3. Automated Streaming Replay Mode (Plan B3 Preview)
- Instead of static scrubbing, implement auto-play controls:
  - **Play / Pause / Step Forward** buttons.
  - Speed selector: $1\times$, $2\times$, $5\times$.
  - Uses `streamlit-autorefresh` or a generator loop with `st.empty()` placeholders to smoothly advance test windows chronologically, demonstrating real-time risk alert transitions.

#### 4. Animated Graph Visualizations via Plotly
- Use Plotly's transition animations (`layout.updatemenus` with `frame` objects) so that when a user scrubs between time windows, graph nodes smoothly transition colors and edge widths smoothly morph.

---

### 6.2 Decision Rubric: Should We Decouple to React / FastAPI Later?

The frontend developer should conduct a formal feasibility assessment before deciding whether to rewrite the UI in a separate stack. Use this objective checklist:

```
+---------------------------------------------------------------------------------------------------+
|                            FRONTEND TECHNOLOGY DECISION TREE                                      |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  Q1: Does the project need to be deployed to external mobile clients or enterprise SSO?           |
|      ├── YES ──> Consider Decoupled (FastAPI + Next.js / Tailwind)                                |
|      └── NO  ──> Proceed to Q2                                                                    |
|                                                                                                   |
|  Q2: Does the evaluation require sub-10ms WebSockets from live Apache Kafka streams?               |
|      ├── YES ──> Consider Decoupled (FastAPI WebSockets + React Canvas)                           |
|      └── NO  ──> Proceed to Q3                                                                    |
|                                                                                                   |
|  Q3: Is the primary goal a 100% reliable, zero-crash, highly defensible B.Tech viva demo?         |
|      ├── YES ──> 🟢 STAY WITH STREAMLIT (Plan A Core Baseline)                                    |
|      └── NO  ──> Explore Plan B Decoupled Extensions                                              |
|                                                                                                   |
+---------------------------------------------------------------------------------------------------+
```

#### When to Recommend Staying with Streamlit:
- If the required features (4 tabs, historical replay, Plotly topology, Lottie animations, explainability bars, and leaderboards) can all be executed with $< 500\text{ ms}$ latency in Streamlit.
- Because it preserves the **zero-risk single-command demo** guarantee for examiners.

#### When to Propose Decoupling to React/FastAPI (Plan B Post-Submission):
- Only after **Plan A Gate 8 has fully passed** and core viva deliverables are frozen.
- If the team intends to publish a commercial SaaS demonstration with user authentication, custom Canvas/WebGL graph physics, or multi-tenant database support.

---

## 7. Technical Data Contract: Backend-to-Frontend Interface

For frontend developers implementing or modifying `app/streamlit_app.py`, the backend provides the following standardized artifacts and interfaces:

### 7.1 Required File Paths (from `src/config.py`)
```python
MODEL_DIR = "outputs/models"
SCALER_PATH = "outputs/models/scaler.joblib"
METRICS_PATH = "outputs/results/overall_metrics.csv"
SEVERITY_PATH = "outputs/results/severity_metrics.csv"
```

### 7.2 Scaler Unscaling Interface
```python
import joblib

# Load fitted scaler
scaler_data = joblib.load(SCALER_PATH)
scaler = scaler_data["scaler"]  # sklearn MinMaxScaler

# Input: scaled_array of shape [1, 5] -> [RI_S, RI_M, RI_D, RI_R, Total_Cost]
# Output: raw_array in real units
raw_array = scaler.inverse_transform(scaled_array)
```

### 7.3 Model Inference Interface
```python
import torch
from src.models.st_gcn_lstm import build_model
from src.graph_builder import build_graphs

# Load model and graph matrices
graphs = build_graphs()
model = build_model("st_gcn_lstm", cfg)
checkpoint = torch.load("outputs/models/st_gcn_lstm_directed_seed42.pt", weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Inference on a single test window [1, 10, 5]
with torch.no_grad():
    predicted_risks = model(seq_tensor, graphs)  # Returns [1, 4] -> [R_S, R_M, R_D, R_R]
```

### 7.4 Explainability Output Schema (`RiskExplainer`)
```python
from src.explainability import RiskExplainer, narrate

explainer = RiskExplainer(model, train_mean, steps=64)
explanation = explainer.explain(seq_tensor[0], target_node=target_idx, residual_delta=use_delta)

# Returns dictionary with exact schema:
# {
#     "target_node": int,
#     "predicted_risk": float,
#     "baseline_risk": float,
#     "completeness_gap": float,
#     "feature_importance": [float, float, float, float, float], # Sums to 1.0
#     "time_importance": [float] * 10,                          # Sums to 1.0
#     "upstream_share": float,                                  # Attribution from upstream nodes
#     "narrative": str                                          # Natural language explanation
# }
```

---

## 8. Summary Checklist for Frontend Developer Handoff

- [ ] **Review `docs/PLAN_A_IMPLEMENTATION_PLAN.md`:** Understand the project's scientific objectives and the 4 research questions (RQ1–RQ4).
- [ ] **Verify Local Environment:** Confirm `streamlit run app/streamlit_app.py` boots cleanly from the repository root.
- [ ] **Implement Tab 1:** Echelon cards with unscaled raw values, train tercile badges, and historical timestamp scrubber.
- [ ] **Implement Tab 2:** Directed Plotly/NetworkX supply chain graph with tier-colored nodes and attribution-weighted edges.
- [ ] **Implement Tab 3:** Signed feature bars, temporal attention profile, Delta-attribution toggle, and natural language narrative.
- [ ] **Implement Tab 4:** Multi-seed benchmark table with error bars and tercile confusion matrix.
- [ ] **Verify Gate 7 Test:** Run `python tests/run_phase_tests.py --phase 7` and ensure all component checks pass.
- [ ] **Explore Polish & Animation Extensions:** Benchmark Lottie animations, custom CSS theming, and streaming replay controls before assessing whether a decoupled React stack is warranted.
