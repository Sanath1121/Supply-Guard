# SupplyGuard: Real-Time Multi-Echelon Supply Chain Risk Alert System

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

## Foundational Literature / Base Paper
- **Citation:** Farzhana I. & Dev Harris L. (2025), *"Hybrid GNN-LSTM Model for Real-Time Supply Chain Risk Prediction"*, 2025 8th International Conference on Computing Methodologies and Communication (ICCMC), IEEE, [doi:10.1109/ICCMC65190.2025.11140739](https://doi.org/10.1109/ICCMC65190.2025.11140739).
- **Note on Architecture:** Global mean pooling, vector dimensions (32/64/96), and scalar TRI represent SupplyGuard's architectural interpretations to bridge the paper's hybrid fusion concept to the dataset, not explicit verbatim specifications.

---

## Repository Structure
```
Supply_chain_alret_system/
├── docs/                     # Definitive implementation plans and master tech stack
├── data/raw/                 # Raw dataset (git-ignored)
├── src/                      # Core data pipeline, graph topology, and attribution
│   ├── config.py
│   ├── graph_builder.py
│   ├── dataset.py
│   ├── explainability.py
│   └── models/               # Pure PyTorch GraphConv and ST-GCN-LSTM models
│       ├── graph_layers.py
│       └── st_gcn_lstm.py
├── training/                 # Train and evaluation harnesses
│   ├── train.py
│   └── evaluate.py
├── tests/                    # Phase-gated test harness and smoke tests
│   ├── smoke_test.py
│   ├── test_phase0_setup.py
│   └── run_phase_tests.py
├── outputs/                  # Figures, checkpoints, and evaluation tables
├── AGENTS.md                 # Agent guardrails and source-of-truth invariants
├── requirements.txt          # Python dependency specifications
├── requirements.lock         # Exact pip dependency lock
├── setup_and_download.py     # Hardened dataset downloader & profiling script
└── README.md
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

### 3. Run Gate 0 Verification Suite
```bash
python -m unittest tests.test_phase0_setup
# or via the master phase test runner:
python tests/run_phase_tests.py --phase 0
```
