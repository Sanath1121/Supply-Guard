# SupplyGuard: Real-Time Multi-Echelon Supply Chain Risk Alert System

[![SupplyGuard CI](https://github.com/Sanath1121/Supply-Guard/actions/workflows/ci.yml/badge.svg)](https://github.com/Sanath1121/Supply-Guard/actions/workflows/ci.yml)

An empirical re-implementation and spatio-temporal node-level extension of the IEEE ICCMC 2025 hybrid GNN-LSTM model for supply chain risk forecasting with validated gradient attributions.

## Project Aim
SupplyGuard re-implements the IEEE ICCMC 2025 hybrid GCN+LSTM as a paper-style baseline and extends it into a node-level spatio-temporal model that forecasts next-step risk separately across Supplier, Manufacturer, Distributor, and Retailer echelons, deriving overall Total Risk Index (TRI) as their mean. Each forecast is explained with validated Integrated Gradients attributions and visualized in a real-test-window Streamlit dashboard. The project prioritizes rigorous empirical evaluation: benchmarking against persistence baselines, testing whether graph topology adds measurable predictive value, and verifying attribution fidelity via deletion tests.

---

## Dataset
- **Citation:** Banerjee, Heerok; Saparia, Grishma; Ganapathy, Velappa; Garg, Priyanshi; Shenbagaraman, V. M. (2019), *"Time Series Dataset for Risk Assessment in Supply Chain Networks"*, Mendeley Data, V2, [doi:10.17632/gystn6d3r4.2](https://doi.org/10.17632/gystn6d3r4.2) (CC BY 4.0).
- **Partition:** `SCRM_timeSeries_2018_train.csv` (649,999 rows, spanning 2015-01-28 to 2018-12-19).
- **Mirror Repository:** GitHub [`webintellectual/Supply-Chain-Stability-Classifier`](https://github.com/webintellectual/Supply-Chain-Stability-Classifier)
- **Pinned Commit Hash:** `698ec038f7410a426655d73bff990699ead8808c`
- **SHA-256 Checksum:** `d2e71ae7f55fa70ef498fecb9b6db0c9fd59688f17f8ad3c27c7576f09e76ff3`

---

## Repository Structure
```
Supply_chain_alret_system/
├── app/                      # Production Streamlit Dashboard (Dark Analytics)
│   ├── streamlit_app.py      # Thin orchestrator with st.navigation & status strip
│   ├── views/                # Overview, Monitor, Why, Benchmarks, Export, Sandbox
│   ├── ui/                   # Reusable components, Plotly dark theme, SVG topology canvas
│   ├── utils/                # Cached artifacts, results loader, explain service, formatters
│   └── styles/               # custom_theme.css (offline-safe, WCAG AA dark analytics)
├── docs/                     # Implementation plans, ADR, and fix documentation
├── data/raw/                 # Raw dataset (git-ignored)
├── src/                      # Core data pipeline, graph topology, and attribution
│   ├── config.py             # Single source of truth for hyperparameters & paths
│   ├── graph_builder.py
│   ├── dataset.py
│   ├── explainability.py     # RiskExplainer (IG with Delta-attribution mode)
│   └── models/               # PyTorch GCN, ST-GCN-LSTM, and baseline models
├── training/                 # Train and evaluation harnesses
│   ├── train.py              # Multi-seed training with exact checkpoint saving
│   └── evaluate.py           # Multi-seed empirical evaluation & results CSV writer
├── tests/                    # Phase-gated test harness and smoke tests
│   ├── run_phase_tests.py    # Master gate test runner
│   ├── test_phase7_app.py    # Gate 7 dashboard component and guard verification
│   └── test_phase7_smoke_apptest.py  # AppTest headless smoke tests
├── outputs/                  # Figures, checkpoints, and evaluation tables
├── requirements.txt          # Python dependency specifications
└── README.md
```

---

## Streamlit Dashboard

### Launching the Dashboard
```bash
streamlit run app/streamlit_app.py
```

### Universal Status Strip
The top of the dashboard displays an honest, live status strip that inspects the environment:
- **Data Chip**: Indicates whether test-partition windows are loaded (`Test partition ✓`), running in isolated sandbox mode (`Sandbox`), or unavailable (`Missing raw CSV`).
- **Model Chip**: Reports the active architecture, graph mode, seed, and whether real trained checkpoint weights were loaded (`ckpt ✓`) or untrained random initializations are in use (`UNTRAINED weights`).
- **Scaler Chip**: Confirms whether `outputs/models/scaler.joblib` is fitted. If missing, raw unscaled units are hidden to prevent misleading interpretations.
- **Tiers Chip**: Confirms whether severity thresholds reflect empirical train terciles (`Train terciles ✓`) or provisional defaults (`Provisional 0.35/0.65`).
- **Explainer Chip**: Monitors axiomatic completeness of Integrated Gradients path integrals.

### Scientific Integrity & Empty States
Per the project integrity charter:
1. **Zero Fabricated Metrics**: If checkpoints or evaluation CSVs are missing, the UI renders explicit empty states with exact reproduction commands (`python -m training.train`, `python -m training.evaluate`).
2. **Strict Test Partition Replay**: Predictions are made strictly on historical test windows. No sliders allow manual sensor fabrication in the operational dashboard.
3. **Isolated Sandbox Mode**: Experimental shock testing or custom CSV uploads are feature-gated and strictly isolated with persistent warning banners:
   ```bash
   # Enable sandbox mode via environment variable
   export SG_ENABLE_SANDBOX=1
   streamlit run app/streamlit_app.py
   ```

---

## Setup & Verification

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download and Verify Raw Dataset
```bash
python setup_and_download.py
```

### 3. Run Phase Gate Verification Suite
```bash
# Run Gate 7 verification and AppTest smoke tests:
python tests/run_phase_tests.py --phase 7

# Run all completed phase gates (0 to 3, 6, 7):
python tests/run_phase_tests.py --up-to 3
python tests/run_phase_tests.py --phase 6
python tests/run_phase_tests.py --phase 7
```
