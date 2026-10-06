# Phase 7 Architectural Specification & Frontend Handoff Report

**Project:** SupplyGuard — Real-Time Multi-Echelon Supply Chain Risk Alert System  
**Phase:** Phase 7 (SupplyGuard Streamlit Dashboard)  
**Status:** 📋 **APPROVED ARCHITECTURAL BASELINE (Specification Formally Locked)**  
**Date:** October 6, 2026  
**Primary References:** `docs/PLAN_A_IMPLEMENTATION_PLAN.md` §3 (Phase 7), `docs/PHASE_7_FRONTEND_CONTEXT_AND_STREAMLIT_DECISION.md`, `docs/MASTER_TECHSTACK.md`

---

## 1. Executive Summary

Phase 7 delivers the user-facing operational dashboard for SupplyGuard. It interfaces directly with the validated PyTorch models (`st_gcn_lstm_dir`, `st_gcn_lstm_sym`, `lstm`, `paper_overall`, and baselines), chronological test partition windows, train-fitted `scaler.joblib`, and Integrated Gradients $\Delta$-attribution tensors.

The frontend runtime architecture has been formally evaluated and locked to **Streamlit** (Architectural Decision Record documented in `docs/PHASE_7_FRONTEND_CONTEXT_AND_STREAMLIT_DECISION.md`).

---

## 2. Core Architectural Justifications for Streamlit

1. **Zero-Latency In-Memory PyTorch Coupling:**
   - Streamlit shares the Python memory space with PyTorch and Scikit-Learn.
   - Eliminates complex REST/WebSocket JSON serialization, preventing float32 precision degradation and NaN serialization failures.
2. **Single-Command Viva Defense Execution:**
   - Evaluators and faculty review committees can launch the complete application with a single cross-platform command:
     ```powershell
     streamlit run app/streamlit_app.py
     ```
   - Zero CORS issues, zero port collisions (e.g. 8000 vs 3000), and zero external Node.js/npm dependencies.
3. **High-Performance In-Memory Caching:**
   - `@st.cache_resource` loads model state dicts and scalers once at startup.
   - Window scrubbing and tab navigation execute with $< 50\text{ ms}$ latency.
4. **Interactive Graph Topology Rendering:**
   - Seamlessly renders Plotly Graph Objects (`plotly.graph_objects`) for multi-echelon network visualization with dynamic edge weighting derived from Integrated Gradients upstream shares.

---

## 3. Four Core Operational Tabs Specification

### Tab 1: Echelon Risk Health Cards
- **Historical Window Scrubbing:** Select historical test window via slider/dropdown labeled *"Offline replay of historical windows"* (prevents hallucinated real-time streaming claims).
- **Echelon Cards:** Real-time risk cards for Supplier, Manufacturer, Distributor, and Retailer displaying:
  - Scaled risk $[0, 1]$ and derived Total Risk Index (TRI).
  - Raw unscaled units via `scaler.inverse_transform`.
  - Severity badge: Low (Green: $\le 0.33$), Medium (Yellow: $0.33\text{--}0.66$), High (Red: $> 0.66$).

### Tab 2: Directed Graph Topology Network
- **Interactive Directed Flow Graph:** $S \to M \to D \to R$.
- **Node Coloring:** Colored by echelon severity tier.
- **Directional Edge Widths:** Proportional to computed Integrated Gradients upstream attribution shares.
- **Feedback Overlay:** Dotted feedback edges ($R \to D \to M \to S$) showing upstream delay propagation.

### Tab 3: Explainability & Sensitivity Analysis ("Why this forecast?")
- **$\Delta$-Attribution Toggle:** Switch between raw risk attribution and residual $\Delta$-attribution ($g(x) = f(x) - y_t$) to inspect dynamic network adjustments beyond persistence.
- **Signed Feature Attribution Bars:** Horizontal bar chart showing whether each feature raises $(+)$ or pulls down $(-)$ risk.
- **Temporal Importance Chart:** Bar chart showing lag-step weighting ($t, t-1, \dots, t-9$).
- **Anti-Causal Narrative:** Dynamic sentence generated via `src/explainability.py::narrate()` strictly avoiding the forbidden phrase "root cause" per `AGENTS.md` Rule 5.
- **Completeness Caption:** Displays empirical completeness gap ($a.\text{sum}() \approx f(x) - f(b)$).

### Tab 4: Benchmark Leaderboard & Confusion Matrix
- **Empirical Multi-Seed Leaderboard:** Displays Test MSE, MAE, and $R^2$ with $\pm 1\text{ std}$ confidence intervals across all evaluated models (`st_gcn_lstm_dir`, `lstm`, `paper_overall`, `st_gcn_lstm_sym`, `ar10_ridge`, `persistence`).
- **Operational Confusion Matrix:** 3-tier severity confusion matrix demonstrating $95.04\%$ 3-class accuracy and 0 false negatives on high-risk events.

---

## 4. Gate 7 Verification Criteria

- **Gate 7 Test:** `python tests/run_phase_tests.py --phase 7`
- **Criteria:** Every tab renders without exceptions for $\ge 3$ distinct historical test windows from a fresh `streamlit run app/streamlit_app.py` process.
