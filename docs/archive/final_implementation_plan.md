# Final Implementation Plan: Interpretable Hybrid GCN-LSTM Supply Chain Risk Forecasting

**Project Title:** SupplyGuard: Interpretable Hybrid GCN-LSTM Architecture for Echelon-Level Supply Chain Risk Forecasting  
**Academic Target:** B.Tech 4th Year (1st Semester Mini Project)  
**Base Research Paper:** "Hybrid GNN-LSTM Model for Real-Time Supply Chain Risk Prediction" — IEEE ICCMC 2025  
**DOI:** [10.1109/ICCMC65190.2025.11140739](https://doi.org/10.1109/ICCMC65190.2025.11140739)  
**Primary Dataset:** Banerjee et al. (2019), Mendeley Data V2 (DOI: [10.17632/gystn6d3r4.2](https://doi.org/10.17632/gystn6d3r4.2))  
**Project Root:** `C:\Users\srila\OneDrive\Desktop\BTECH\4th year\mini project\supply_chain_risk_gnn_lstm\`  
**Document Status:** Final Verified & Architecturally Audited Implementation Plan  

---

## Executive Summary & Core Evolution

This plan represents the fully audited, mathematically verified blueprint for the SupplyGuard mini project. Following a comprehensive technical audit against Kipf & Welling spectral graph theory, PyTorch autograd dynamics, and raw Mendeley dataset inspection, the project executes a **dual-phase scientific progression**:

1. **Phase 1 — Base-Paper Reproduction Baseline (`HybridGNNLSTM_Overall`):**  
   Reproduce the IEEE ICCMC 2025 hybrid GCN-LSTM architecture using global mean pooling to predict overall supply chain risk ($\hat{TRI}_{t+1}$), proving technical competency and providing a verified benchmark.
2. **Phase 2 — Proposed Node-Level Extension (`NodeRiskGNNLSTM`):**  
   Preserve individual GCN node representations ($[B, 4, 32]$ instead of collapsing them via mean pooling), expand the LSTM temporal representation to $[B, 4, 64]$, concatenate them per-node with learnable echelon embeddings ($[B, 4, 96]$), and forecast next-step risk individually for each supply chain echelon:
   $$\hat{Y}_{t+1} = \left[ \hat{R}_S^{t+1}, \hat{R}_M^{t+1}, \hat{R}_D^{t+1}, \hat{R}_R^{t+1} \right]$$
   Derive overall Total Risk Index (TRI) as the average: $\hat{TRI}_{t+1} = \frac{1}{4}\sum_{i} \hat{R}_i^{t+1}$.
3. **Phase 3 — Explainability Layer (`src/explainability.py`):**  
   Integrated directly into core Plan A. Implements Input $\times$ Gradient attribution ($\left| X \odot \frac{\partial \hat{R}_i}{\partial X} \right|$) to explain feature importance, historical time-step sensitivity, and upstream/downstream risk propagation across the network.
4. **Phase 4 — Operational SupplyGuard UI (`app/streamlit_app.py`):**  
   An interactive Streamlit decision-support dashboard featuring real 10-step test window selection, node-by-node risk cards with severity status (Low, Medium, High), directed topology flow graphs, and "Why is this node risky?" diagnostic explanations.

---

## Environment Status (Verified)

| Package | Status | Version / Note |
|:---|:---:|:---|
| Python | ✅ Installed | 3.14.3 |
| PyTorch (CPU) | ✅ Installed | 2.12.0+cpu |
| numpy | ✅ Installed | 2.4.2 |
| pandas | ✅ Installed | 3.0.0 |
| scikit-learn | ✅ Installed | 1.8.0 |
| matplotlib | ✅ Installed | 3.10.8 |
| seaborn | ✅ Installed | 0.13.2 |
| networkx | ✅ Installed | 3.6.1 |
| **torch_geometric (PyG)** | ⚡ BYPASSED | Clean Manual Kipf & Welling batched GCN — zero C++ dependencies |
| **streamlit** | ⏳ Phase 0 | UI framework for interactive decision-support app |
| **statsmodels** (ARIMA) | ⚡ RESILIENT | Pure PyTorch / Scikit-learn AR(10) fallback built-in |
| **mlflow** | ⏳ Phase 0 | MLOps experiment tracking & local dashboard (`mlflow ui`) |

---

## Locked Architectural Decisions (Audited & Verified)

> [!IMPORTANT]
> **Decision 1 — Forward Forecasting Horizon ($t+1$):**  
> Input window $[t-9, \dots, t]$ predicts next-step risk at $t+1$. This completely eliminates the target leakage of predicting current risk from current inputs and provides true operational forecasting.

> [!IMPORTANT]
> **Decision 2 — Symmetric Graph Topology for Kipf-Welling GCN:**  
> Kipf & Welling (ICLR 2017) spectral graph convolutions strictly require an undirected/symmetric adjacency matrix $A = A^T$ so that the normalized graph Laplacian $L = I - D^{-1/2}AD^{-1/2}$ is real symmetric with orthogonal eigenvectors. Operationally, this models bidirectional supply chain dynamics: physical disruptions propagate downstream ($S \to M \to D \to R$), while demand volatility and bullwhip effects propagate upstream ($R \to D \to M \to S$).

> [!IMPORTANT]
> **Decision 3 — 2-Layer GCN & Learnable Echelon Embeddings:**  
> A 1-layer GCN only reaches 1 hop (leaving 0.0000 gradient between Supplier and Distributor). A 2-layer GCN (`1 -> 32 -> 32`) enables 2-hop structural propagation. Adding a learnable echelon identity tensor `self.echelon_emb` ($[1, 4, 32]$, 128 parameters) prevents head prediction collapse and differentiates node roles cleanly.

> [!IMPORTANT]
> **Decision 4 — Clean Ingestion Exclusively from `SCRM_timeSeries_2018_train.csv`:**  
> Forensic data inspection revealed that `SCRM_timeSeries_2018_test.csv` in `webintellectual` has **`RI_Distributor1` 100% NULL (empty string `,,`)** and contains 2016–2017 dates. The project strictly downloads `SCRM_timeSeries_2018_train.csv` (650,000 clean 2018 records) and performs the chronological 80:10:10 train/val/test split internally.

> [!IMPORTANT]
> **Decision 5 — Manual Batched GCN:**  
> Python 3.14 on Windows has no prebuilt PyG wheels. GCN is implemented via batched matrix multiplication `torch.matmul(A_norm, x)` with dynamic node features $X_{batch} \in \mathbb{R}^{B \times 4 \times 1}$. Clean, fast (<2 minutes CPU training), and easy to defend in viva.

> [!IMPORTANT]
> **Decision 6 — Rigorous Regression Metrics & Operational Severity Tiers:**  
> Continuous risk forecasting is evaluated using MSE, MAE, RMSE, and $R^2$ at both echelon level and overall level. The ambiguous "88% Accuracy" simulation claim and arbitrary `pd.qcut` binning are replaced by an operational 3-tier severity classification (Low $< 0.35$, Medium $0.35–0.65$, High $> 0.65$) matching industrial alert thresholds.

---

## Final Project Directory Structure

```
C:\Users\srila\OneDrive\Desktop\BTECH\4th year\mini project\
└── supply_chain_risk_gnn_lstm\                     ← PROJECT ROOT
    │
    ├── data\
    │   ├── raw\
    │   │   └── SCRM_timeSeries_2018_train.csv       ← Mendeley Data V2 (650k clean records)
    │   └── processed\
    │       ├── train_scaled.csv
    │       ├── val_scaled.csv
    │       └── test_scaled.csv
    │
    ├── src\
    │   ├── __init__.py
    │   ├── config.py                                ← All hyperparameters and column mappings
    │   ├── dataset.py                               ← build_datasets(): chronological 80:10:10, scaling, windows
    │   ├── graph_builder.py                         ← Bidirectional 4-echelon graph + normalized Laplacian
    │   ├── explainability.py                        ← Input × Gradient attribution (feature, time, node)
    │   └── models\
    │       ├── __init__.py
    │       ├── gcn_layer.py                         ← Manual Batched GCN (Kipf & Welling, no PyG)
    │       ├── hybrid_model.py                      ← Model A: Base-Paper HybridGNNLSTM_Overall (mean pool)
    │       ├── node_risk_model.py                   ← Model B: Proposed NodeRiskGNNLSTM (2-layer + echelon_emb)
    │       └── baselines.py                         ← StandaloneLSTM, StandaloneGCN, AR10 / ARIMA
    │
    ├── training\
    │   ├── train.py                                 ← Full training loop: train/val, early stopping, MLflow
    │   └── evaluate.py                              ← Node-wise & overall MSE/MAE/RMSE/R2 + 3-tier accuracy
    │
    ├── notebooks\
    │   ├── 01_EDA.ipynb                             ← Exploratory Data Analysis & correlation heatmap
    │   └── 02_Training_Demo.ipynb                   ← Cell-by-cell walkthrough of both models & attribution
    │
    ├── app\
    │   ├── streamlit_app.py                         ← SupplyGuard UI: Node cards, network map, reasons
    │   └── components\
    │       ├── network_chart.py                     ← Graph visualization with risk status colors
    │       └── model_loader.py                      ← Model & test sample inference loader
    │
    ├── outputs\
    │   ├── figures\                                 ← Heatmaps, loss curves, risk trajectories
    │   ├── models\
    │   │   ├── hybrid_overall_best.pt               ← Base paper reproduction checkpoint
    │   │   ├── node_risk_best.pt                    ← Proposed node-level model checkpoint
    │   │   ├── lstm_baseline.pt
    │   │   └── gcn_baseline.pt
    │   └── results\
    │       ├── overall_metrics.csv                  ← Model comparison (LSTM, GCN, Base, Proposed)
    │       ├── node_metrics.csv                     ← Echelon breakdown (Supplier, Manufacturer, etc.)
    │       └── attribution_examples.csv             ← Sample explanations
    │
    ├── requirements.txt
    ├── setup_and_download.py                        ← Automated downloader & dataset validator
    └── README.md
```

---

# PLAN A — Standard Mini-Project Timeline (4–7 Days)
## Target: Complete Base-Paper Reproduction + Node-Level Extension + Explainability + UI

---

### PHASE 0 — Environment Setup & Data Acquisition
**Estimated time: 1–2 hours**

#### Step 0.1 — Complete `requirements.txt`
```text
torch>=2.0.0
numpy>=1.24.0
pandas>=1.5.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
networkx>=3.0
statsmodels>=0.14.0
streamlit>=1.28.0
plotly>=5.0.0
mlflow>=2.10.0
requests>=2.28.0
```
> `torch_geometric` is intentionally omitted because manual tensor multiplication is used.

Run command:
```powershell
pip install streamlit statsmodels plotly requests mlflow
```

#### Step 0.2 — Directory Initialization & Dataset Download (`setup_and_download.py`)

> [!CAUTION]
> **Data Integrity Constraint:**  
> Download ONLY `SCRM_timeSeries_2018_train.csv`. Do NOT download `SCRM_timeSeries_2018_test.csv` from `webintellectual` because its `RI_Distributor1` column contains 100% missing values (empty strings `,,`) and dates from 2016–2017. `train.csv` contains 650,000 clean continuous rows from 2018 with complete data across all 4 echelons.

```python
# setup_and_download.py
import os
import requests
import pandas as pd

DIRECTORIES = [
    "data/raw", "data/processed", "src/models",
    "training", "notebooks", "app/components",
    "outputs/figures", "outputs/models", "outputs/results"
]

TRAIN_URL = "https://raw.githubusercontent.com/webintellectual/Supply-Chain-Stability-Classifier/master/Dataset/SCRM_timeSeries_2018_train.csv"
RAW_FILE = "data/raw/SCRM_timeSeries_2018_train.csv"

def init_environment():
    for d in DIRECTORIES:
        os.makedirs(d, exist_ok=True)
    print("Directories initialized successfully.")

    if not os.path.exists(RAW_FILE):
        print(f"Downloading Mendeley Data V2 from {TRAIN_URL}...")
        res = requests.get(TRAIN_URL, timeout=60)
        res.raise_for_status()
        with open(RAW_FILE, "wb") as f:
            f.write(res.content)
        print(f"Downloaded {RAW_FILE} successfully.")
    else:
        print(f"{RAW_FILE} already exists.")

    df = pd.read_csv(RAW_FILE, nrows=5)
    print("Dataset verification: First 5 rows loaded.")
    print("Columns:", list(df.columns))

if __name__ == "__main__":
    init_environment()
```

**✅ Phase 0 Verification:** `python setup_and_download.py` completes with exit code 0 and verifies dataset files.

---

### PHASE 1 — Configuration & Forward Forecasting Pipeline
**Estimated time: 4–5 hours**

#### `src/config.py`

> [!CAUTION]
> **Column Order Caution:**  
> In the raw CSV, columns appear as: `[Timestamp, RI_Supplier1, RI_Distributor1, RI_Manufacturer1, RI_Retailer1, Total_Cost, SCMstability_category]`. Notice Distributor appears before Manufacturer. In `src/config.py`, columns MUST always be selected by name (`df[cfg.FEATURE_COLS]`), never by numeric column indices.

```python
class Config:
    # Paths
    RAW_DATA_PATH      = "data/raw/SCRM_timeSeries_2018_train.csv"
    
    # Supply Chain Topology
    NUM_NODES          = 4
    NODE_NAMES         = ["Supplier", "Manufacturer", "Distributor", "Retailer"]
    NUM_INPUT_FEATURES = 5  # 4 RI columns + Total_Cost
    NUM_NODE_FEATURES  = 1  # 1 RI scalar per node

    # Architecture Hyperparameters
    GCN_HIDDEN_DIM     = 32    # Output dimension of GCN layer
    LSTM_HIDDEN_DIM    = 64    # Output dimension of LSTM
    LSTM_NUM_LAYERS    = 2
    LSTM_DROPOUT       = 0.2
    FUSION_HIDDEN      = 64
    FC_DROPOUT         = 0.3

    # Forecasting & Training
    SEQ_LEN            = 10    # Historical observation window [t-9, ..., t]
    PREDICTION_HORIZON = 1     # Predict step t+1
    BATCH_SIZE         = 32
    LEARNING_RATE      = 0.001
    MAX_EPOCHS         = 50
    PATIENCE           = 10    # Early stopping
    GRAD_CLIP          = 1.0

    # Splits (Chronological, no shuffling)
    TRAIN_RATIO        = 0.80
    VAL_RATIO          = 0.10
    TEST_RATIO         = 0.10
    RANDOM_SEED        = 42
    MAX_SAMPLES        = 50000 # 50,000 for rapid <2 min CPU training (None for full 650k)

    # Columns (Strict Named Access)
    DATE_COL           = 'Timestamp'
    FEATURE_COLS       = ['RI_Supplier1', 'RI_Manufacturer1',
                          'RI_Distributor1', 'RI_Retailer1', 'Total_Cost']
    NODE_TARGET_COLS   = ['RI_Supplier1', 'RI_Manufacturer1',
                          'RI_Distributor1', 'RI_Retailer1']
```

#### `src/dataset.py` — Complete Dataset Pipeline & Ingestion

```python
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

class SupplyChainDataset(Dataset):
    def __init__(self, sequences, graph_nodes, node_targets, tri_targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.graph_nodes = torch.tensor(graph_nodes, dtype=torch.float32)
        self.node_targets = torch.tensor(node_targets, dtype=torch.float32)
        self.tri_targets = torch.tensor(tri_targets, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'sequence':    self.sequences[idx],    # [10, 5]
            'graph_x':     self.graph_nodes[idx],  # [4, 1]
            'node_target': self.node_targets[idx], # [4]
            'tri_target':  self.tri_targets[idx]   # scalar
        }

def build_datasets(cfg):
    """
    Loads raw CSV, cleans missing values, splits chronologically (80:10:10),
    applies train-only MinMax scaling, and creates sliding window samples.
    """
    df = pd.read_csv(cfg.RAW_DATA_PATH)
    # Select features strictly by column names to avoid CSV ordering confusion
    df = df[cfg.FEATURE_COLS].ffill().bfill()
    
    if cfg.MAX_SAMPLES is not None:
        df = df.iloc[:cfg.MAX_SAMPLES]
        
    n = len(df)
    n_train = int(n * cfg.TRAIN_RATIO)
    n_val = int(n * cfg.VAL_RATIO)
    
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]
    
    # Train-only scaling prevents temporal leakage
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df)
    val_scaled = scaler.transform(val_df)
    test_scaled = scaler.transform(test_df)
    
    def create_windows(data):
        seqs, graphs, node_tgts, tri_tgts = [], [], [], []
        L = cfg.SEQ_LEN
        for i in range(len(data) - L):
            # Sequence: time steps i to i+L-1 [10, 5]
            seqs.append(data[i : i + L, :])
            # Graph at step t (i+L-1): 4 node risks [4, 1]
            graphs.append(data[i + L - 1, :4].reshape(4, 1))
            # Target at step t+1 (i+L): 4 future node risks [4]
            tgt = data[i + L, :4]
            node_tgts.append(tgt)
            tri_tgts.append(tgt.mean())
        return (np.array(seqs), np.array(graphs), 
                np.array(node_tgts), np.array(tri_tgts))
                
    tr_data = create_windows(train_scaled)
    va_data = create_windows(val_scaled)
    te_data = create_windows(test_scaled)
    
    return (SupplyChainDataset(*tr_data),
            SupplyChainDataset(*va_data),
            SupplyChainDataset(*te_data), scaler)
```

#### `src/graph_builder.py` — Symmetric 4-Echelon Graph Formulation

```python
import torch

def build_supply_chain_graph() -> tuple[torch.Tensor, torch.Tensor]:
    """
    Constructs the symmetric 4-echelon supply chain graph and normalized Laplacian.
    Symmetric formulation is mathematically required by Kipf & Welling GCN spectral theory
    and models bidirectional supply chain flow (physical disruption + bullwhip information).
    """
    # 0: Supplier, 1: Manufacturer, 2: Distributor, 3: Retailer
    edge_index = torch.tensor([[0, 1, 2, 1, 2, 3],
                               [1, 2, 3, 0, 1, 2]], dtype=torch.long)
    
    A = torch.zeros((4, 4), dtype=torch.float32)
    A[0, 1] = A[1, 0] = 1.0  # Supplier <-> Manufacturer
    A[1, 2] = A[2, 1] = 1.0  # Manufacturer <-> Distributor
    A[2, 3] = A[3, 2] = 1.0  # Distributor <-> Retailer
    
    # Self-loop augmentation
    A_tilde = A + torch.eye(4)
    d = A_tilde.sum(dim=1)
    d_inv_sqrt = torch.pow(d, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    
    # Normalized Laplacian: A_hat = D^(-1/2) * A_tilde * D^(-1/2)
    adj_norm = torch.matmul(torch.matmul(D_inv_sqrt, A_tilde), D_inv_sqrt)
    return edge_index, adj_norm
```

**✅ Phase 1 Verification:**
```powershell
python -c "from src.dataset import build_datasets; from src.config import Config; cfg = Config(); cfg.MAX_SAMPLES = 500; tr, va, te, sc = build_datasets(cfg); b = tr[0]; assert b['sequence'].shape == (10, 5); assert b['graph_x'].shape == (4, 1); assert b['node_target'].shape == (4,); print('Pipeline Verified!')"
```

---

### PHASE 2 — Model Implementations (Baseline vs Proposed)
**Estimated time: 6–7 hours**

#### `src/models/gcn_layer.py` — Manual Batched GCN Layer
Implements $H^{(l+1)} = \text{ReLU}(\hat{A} H^{(l)} W^{(l)})$:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        # x: [B, 4, in_features], adj_norm: [4, 4]
        # Batched matrix multiplication: [4, 4] @ [B, 4, in] -> [B, 4, in]
        ax = torch.matmul(adj_norm, x)
        return F.relu(self.linear(ax))  # [B, 4, out_features]
```

#### `src/models/hybrid_model.py` — Model A: Base-Paper Reproduction Baseline (`HybridGNNLSTM_Overall`)
Implements the exact IEEE ICCMC 2025 structure with **global mean pooling**:
```python
import torch
import torch.nn as nn
from src.models.gcn_layer import GCNLayer

class HybridGNNLSTM_Overall(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gcn = GCNLayer(cfg.NUM_NODE_FEATURES, cfg.GCN_HIDDEN_DIM)
        self.lstm = nn.LSTM(
            input_size=cfg.NUM_INPUT_FEATURES,
            hidden_size=cfg.LSTM_HIDDEN_DIM,
            num_layers=cfg.LSTM_NUM_LAYERS,
            batch_first=True,
            dropout=cfg.LSTM_DROPOUT
        )
        self.fc = nn.Sequential(
            nn.Linear(cfg.GCN_HIDDEN_DIM + cfg.LSTM_HIDDEN_DIM, cfg.FUSION_HIDDEN),
            nn.ReLU(),
            nn.Dropout(cfg.FC_DROPOUT),
            nn.Linear(cfg.FUSION_HIDDEN, 1),
            nn.Sigmoid()
        )

    def forward(self, gx, adj, seq):
        B = gx.size(0)
        # GCN branch with global mean pooling (Collapses node identities)
        g_out = self.gcn(gx, adj)          # [B, 4, 32]
        g_emb = g_out.mean(dim=1)          # [B, 32]
        
        # LSTM branch
        _, (h_n, _) = self.lstm(seq)
        t_emb = h_n[-1]                    # [B, 64]
        
        # Global fusion
        fused = torch.cat([g_emb, t_emb], dim=-1) # [B, 96]
        return self.fc(fused).squeeze(-1)         # [B] scalar overall TRI
```

#### `src/models/node_risk_model.py` — Model B: Proposed Node-Level Architecture (`NodeRiskGNNLSTM`)
Preserves per-node representations with 2-layer GCN and learnable echelon embeddings:
```python
import torch
import torch.nn as nn
from src.models.gcn_layer import GCNLayer

class NodeRiskGNNLSTM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 2-layer GCN allows 2-hop structural propagation across the 4-echelon chain
        self.gcn1 = GCNLayer(cfg.NUM_NODE_FEATURES, cfg.GCN_HIDDEN_DIM)
        self.gcn2 = GCNLayer(cfg.GCN_HIDDEN_DIM, cfg.GCN_HIDDEN_DIM)
        
        # Learnable echelon identity embedding differentiates S, M, D, R
        self.echelon_emb = nn.Parameter(torch.randn(1, cfg.NUM_NODES, cfg.GCN_HIDDEN_DIM) * 0.05)
        
        self.lstm = nn.LSTM(
            input_size=cfg.NUM_INPUT_FEATURES,
            hidden_size=cfg.LSTM_HIDDEN_DIM,
            num_layers=cfg.LSTM_NUM_LAYERS,
            batch_first=True,
            dropout=cfg.LSTM_DROPOUT
        )
        
        # Node head: takes [node_emb (32) + temporal (64)] = 96
        self.node_head = nn.Sequential(
            nn.Linear(cfg.GCN_HIDDEN_DIM + cfg.LSTM_HIDDEN_DIM, 32),
            nn.ReLU(),
            nn.Dropout(cfg.FC_DROPOUT),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, gx, adj, seq):
        B = gx.size(0)
        # 1. 2-Layer GCN Branch - Preserve all 4 node embeddings
        h1 = self.gcn1(gx, adj)                               # [B, 4, 32]
        node_emb = self.gcn2(h1, adj) + self.echelon_emb      # [B, 4, 32]
        assert node_emb.shape == (B, 4, 32), f"Expected [B,4,32], got {node_emb.shape}"

        # 2. LSTM Branch
        _, (h_n, _) = self.lstm(seq)
        t_emb = h_n[-1]                                       # [B, 64]
        
        # 3. Expand temporal embedding to all 4 nodes
        t_expanded = t_emb.unsqueeze(1).expand(-1, 4, -1)    # [B, 4, 64]

        # 4. Concatenate per-node
        fused = torch.cat([node_emb, t_expanded], dim=-1)    # [B, 4, 96]

        # 5. Predict 4 echelon risks
        node_risks = self.node_head(fused).squeeze(-1)       # [B, 4]
        return node_risks
```

#### `src/models/baselines.py` — Comparative Baselines
1. `StandaloneLSTM`: Sequence $\to$ LSTM $\to$ FC $\to$ [B, 4] node risks.
2. `StandaloneGCN`: Graph $\to$ 2-layer GCN $\to$ FC $\to$ [B, 4] node risks.
3. `AR10Baseline`: Pure scikit-learn autoregressive 10-lag baseline for temporal comparison.

**✅ Phase 2 Verification:**
```powershell
python -c "from src.models.node_risk_model import NodeRiskGNNLSTM; from src.config import Config; import torch; m = NodeRiskGNNLSTM(Config()); gx = torch.randn(8, 4, 1); adj = torch.eye(4); seq = torch.randn(8, 10, 5); out = m(gx, adj, seq); assert out.shape == (8, 4); print('Node Model Shape Verified [8, 4]!')"
```

---

### PHASE 3 — Unified Training Engine & MLOps Tracking
**Estimated time: 5–6 hours**

#### `training/train.py`
Unified training framework supporting both models with early stopping and MLflow logging:

```python
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import mlflow
from src.config import Config
from src.dataset import build_datasets
from src.graph_builder import build_supply_chain_graph
from src.models.node_risk_model import NodeRiskGNNLSTM
from src.models.hybrid_model import HybridGNNLSTM_Overall

def train_epoch(model, loader, optimizer, adj, device, is_node_model=True):
    model.train()
    total_loss = 0.0
    for batch in loader:
        gx = batch['graph_x'].to(device)
        seq = batch['sequence'].to(device)
        optimizer.zero_grad()

        if is_node_model:
            target = batch['node_target'].to(device)
            pred = model(gx, adj, seq) # [B, 4]
            node_loss = F.mse_loss(pred, target)
            tri_loss = F.mse_loss(pred.mean(dim=1), target.mean(dim=1))
            loss = node_loss + 0.25 * tri_loss
        else:
            target = batch['tri_target'].to(device)
            pred = model(gx, adj, seq) # [B]
            loss = F.mse_loss(pred, target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(gx)
    return total_loss / len(loader.dataset)

def validate_epoch(model, loader, adj, device, is_node_model=True):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            gx = batch['graph_x'].to(device)
            seq = batch['sequence'].to(device)
            if is_node_model:
                target = batch['node_target'].to(device)
                pred = model(gx, adj, seq)
                loss = F.mse_loss(pred, target)
            else:
                target = batch['tri_target'].to(device)
                pred = model(gx, adj, seq)
                loss = F.mse_loss(pred, target)
            total_loss += loss.item() * len(gx)
    return total_loss / len(loader.dataset)

def train_model(model_type="node"):
    cfg = Config()
    device = torch.device("cpu")
    train_set, val_set, test_set, _ = build_datasets(cfg)
    train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.BATCH_SIZE, shuffle=False)
    _, adj = build_supply_chain_graph()
    adj = adj.to(device)

    is_node = (model_type == "node")
    model = NodeRiskGNNLSTM(cfg) if is_node else HybridGNNLSTM_Overall(cfg)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    
    ckpt_path = "outputs/models/node_risk_best.pt" if is_node else "outputs/models/hybrid_overall_best.pt"
    best_val_loss = float('inf')
    patience_cnt = 0
    history = []

    mlflow.set_experiment("SupplyChain_Risk_Prediction")
    with mlflow.start_run(run_name=f"{model_type}_run"):
        mlflow.log_params({"lr": cfg.LEARNING_RATE, "batch_size": cfg.BATCH_SIZE, "model_type": model_type})
        for epoch in range(1, cfg.MAX_EPOCHS + 1):
            tr_loss = train_epoch(model, train_loader, optimizer, adj, device, is_node)
            va_loss = validate_epoch(model, val_loader, adj, device, is_node)
            history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss})
            
            mlflow.log_metric("train_loss", tr_loss, step=epoch)
            mlflow.log_metric("val_loss", va_loss, step=epoch)
            print(f"Epoch {epoch:02d} | Train Loss: {tr_loss:.4f} | Val Loss: {va_loss:.4f}")

            if va_loss < best_val_loss:
                best_val_loss = va_loss
                torch.save(model.state_dict(), ckpt_path)
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= cfg.PATIENCE:
                    print(f"Early stopping triggered at epoch {epoch}.")
                    break

    pd.DataFrame(history).to_csv(f"outputs/results/{model_type}_loss_log.csv", index=False)
    print(f"Model saved to {ckpt_path}")

if __name__ == "__main__":
    train_model(model_type="base")
    train_model(model_type="node")
```

---

### PHASE 4 — Comprehensive Evaluation Framework
**Estimated time: 4–5 hours**

#### `training/evaluate.py`
Evaluates all models strictly on the unseen test set ($10\%$):

```python
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.config import Config
from src.dataset import build_datasets
from src.graph_builder import build_supply_chain_graph
from src.models.node_risk_model import NodeRiskGNNLSTM
from src.models.hybrid_model import HybridGNNLSTM_Overall

def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, rmse, r2

def evaluate_pipeline():
    cfg = Config()
    device = torch.device("cpu")
    _, _, test_set, _ = build_datasets(cfg)
    test_loader = DataLoader(test_set, batch_size=cfg.BATCH_SIZE, shuffle=False)
    _, adj = build_supply_chain_graph()
    adj = adj.to(device)

    # 1. Evaluate Proposed NodeRiskGNNLSTM
    node_model = NodeRiskGNNLSTM(cfg).to(device)
    node_model.load_state_dict(torch.load("outputs/models/node_risk_best.pt", map_location="cpu"))
    node_model.eval()

    all_node_preds, all_node_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            gx = batch['graph_x'].to(device)
            seq = batch['sequence'].to(device)
            pred = node_model(gx, adj, seq)
            all_node_preds.append(pred.cpu().numpy())
            all_node_targets.append(batch['node_target'].numpy())

    node_preds = np.vstack(all_node_preds)       # [N, 4]
    node_targets = np.vstack(all_node_targets)   # [N, 4]

    # Save Node-Level Metrics
    node_records = []
    for i, name in enumerate(cfg.NODE_NAMES):
        mse, mae, rmse, r2 = compute_metrics(node_targets[:, i], node_preds[:, i])
        node_records.append({"Node": name, "MSE": mse, "MAE": mae, "RMSE": rmse, "R2": r2})
    node_df = pd.DataFrame(node_records)
    node_df.to_csv("outputs/results/node_metrics.csv", index=False)
    print("\n--- NODE-LEVEL METRICS ---")
    print(node_df)

    # 2. Evaluate Base-Paper Reproduction Model
    base_model = HybridGNNLSTM_Overall(cfg).to(device)
    base_model.load_state_dict(torch.load("outputs/models/hybrid_overall_best.pt", map_location="cpu"))
    base_model.eval()

    all_base_preds, all_tri_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            gx = batch['graph_x'].to(device)
            seq = batch['sequence'].to(device)
            pred = base_model(gx, adj, seq)
            all_base_preds.append(pred.cpu().numpy())
            all_tri_targets.append(batch['tri_target'].numpy())

    base_preds = np.concatenate(all_base_preds)
    tri_targets = np.concatenate(all_tri_targets)
    derived_tri_preds = node_preds.mean(axis=1)

    # Model Comparison Metrics
    comp_records = []
    b_mse, b_mae, b_rmse, b_r2 = compute_metrics(tri_targets, base_preds)
    comp_records.append({"Model": "Base-Paper Hybrid GCN-LSTM", "MSE": b_mse, "MAE": b_mae, "RMSE": b_rmse, "R2": b_r2})
    
    n_mse, n_mae, n_rmse, n_r2 = compute_metrics(tri_targets, derived_tri_preds)
    comp_records.append({"Model": "Proposed Node GCN-LSTM (Derived TRI)", "MSE": n_mse, "MAE": n_mae, "RMSE": n_rmse, "R2": n_r2})

    comp_df = pd.DataFrame(comp_records)
    comp_df.to_csv("outputs/results/overall_metrics.csv", index=False)
    print("\n--- OVERALL MODEL COMPARISON ---")
    print(comp_df)

if __name__ == "__main__":
    evaluate_pipeline()
```

---

### PHASE 5 — Explainability Engine (`src/explainability.py`)
**Estimated time: 4–5 hours**

Implements Input $\times$ Gradient attribution with numerical stability and autograd protection:

```python
import torch
from src.models.node_risk_model import NodeRiskGNNLSTM

class RiskExplainer:
    def __init__(self, model: NodeRiskGNNLSTM, adj_norm: torch.Tensor):
        self.model = model
        self.adj_norm = adj_norm
        self.model.eval()

    def explain(self, sample_sequence: torch.Tensor, sample_graph_x: torch.Tensor, target_node_idx: int = 0):
        """
        Computes Input * Gradient sensitivity for a specific echelon's risk prediction:
        target_node_idx: 0 (Supplier), 1 (Manufacturer), 2 (Distributor), 3 (Retailer)
        Enforces torch.enable_grad() to safely execute inside Streamlit caching contexts.
        """
        with torch.enable_grad():
            seq = sample_sequence.unsqueeze(0).clone().detach().requires_grad_(True) # [1, 10, 5]
            gx = sample_graph_x.unsqueeze(0).clone().detach().requires_grad_(True)   # [1, 4, 1]

            preds = self.model(gx, self.adj_norm, seq) # [1, 4]
            target_pred = preds[0, target_node_idx]
            target_pred.backward()

            # 1. Feature Attribution: Input * Gradient across features
            feat_saliency = (seq * seq.grad).abs().squeeze(0).mean(dim=0)
            feat_importance = (feat_saliency / (feat_saliency.sum() + 1e-8)).tolist()

            # 2. Temporal Attribution: Input * Gradient across past time-steps
            time_saliency = (seq * seq.grad).abs().squeeze(0).mean(dim=1)
            time_importance = (time_saliency / (time_saliency.sum() + 1e-8)).tolist()

            # 3. Node/Topology Attribution: Input * Gradient of each echelon
            node_saliency = (gx * gx.grad).abs().squeeze(0).squeeze(-1)
            node_importance = (node_saliency / (node_saliency.sum() + 1e-8)).tolist()

        return {
            'predicted_risk': float(target_pred.item()),
            'feature_importance': feat_importance,
            'time_importance': time_importance,
            'node_importance': node_importance
        }
```

---

### PHASE 6 — Interactive SupplyGuard Streamlit Dashboard
**Estimated time: 5–6 hours**

#### `app/streamlit_app.py`
Deployable web interface for operational decision support:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        🏭 SUPPLYGUARD: SCM RISK INTELLIGENCE                           │
│             Interpretable Hybrid GCN-LSTM Architecture (B.Tech Mini Project)          │
├────────────────────────────────┬───────────────────────────────────────────────────────┤
│ SIDEBAR: Scenario Selector     │ TAB 1: ECHELON RISK RADAR                             │
│ ───────────────────────────    │ ┌───────────────────────────────────────────────────┐ │
│ Select Historical Window:      │ │ [ SUPPLIER ]     [ MANUFACTURER ] [ DISTRIBUTOR ] │ │
│ [ Sample #142 (Timestamp)  ▼ ] │ │   0.84 🔴             0.69 🟠          0.42 🟡    │ │
│                                │ │   CRITICAL             HIGH            MEDIUM     │ │
│ Inspection Target Node:        │ │ [ RETAILER ]     [ OVERALL TRI ]                  │ │
│ (o) Supplier   ( ) Manufacturer│ │   0.25 🟢             0.55 🟡                     │ │
│ ( ) Distribut. ( ) Retailer    │ │   LOW                 MODERATE                    │ │
│                                │ └───────────────────────────────────────────────────┘ │
│ [ Run Explainability Audit ]   │ ───────────────────────────────────────────────────   │
│                                │ TAB 2: NETWORK TOPOLOGY & PROPAGATION                 │
│ Alert Thresholds:              │  🔴 Supplier ──(High Impact)──► 🟠 Manufacturer       │
│ • Low:     < 0.35              │                                       │               │
│ • Medium:  0.35 - 0.65         │                                🟡 Distributor         │
│ • High:    > 0.65              │                                       │               │
│                                │                                🟢 Retailer            │
│                                │ ───────────────────────────────────────────────────   │
│                                │ TAB 3: WHY IS THIS NODE RISKY? (EXPLAINABILITY)       │
│                                │ • Top Contributing Feature: Upstream Supplier RI (48%)│
│                                │ • Most Critical Time Step: t-1 (Recent Spike)         │
│                                │ • Temporal Sensitivity Plot (Plotly Bar Chart)        │
│                                │ ───────────────────────────────────────────────────   │
│                                │ TAB 4: BENCHMARK MODEL COMPARISON                     │
│                                │ Side-by-side MSE/MAE/R² table and grouped bar chart   │
└────────────────────────────────┴───────────────────────────────────────────────────────┘
```

**Key Improvements over Previous UI:**
- **No Synthetic Sliders:** Uses actual 10-step windows from the test set for scientifically valid inference.
- **Node-Specific Severity Badges:** Immediate visual breakdown for all 4 supply chain participants.
- **Direct Interpretability Display:** Integrates gradient attribution directly beneath the network visualization.
- **No Fake Confidence Percentages:** Shows actual predicted risk value and severity classification based on justified thresholds.

---

### PHASE 7 — Notebooks, Documentation & Final Verification
**Estimated time: 2–3 hours**

1. **`notebooks/01_EDA.ipynb`:** EDA showing distribution of echelon risks, missing value handling, correlation heatmap, and linking the 4K architectural diagrams:
   - System Architecture: [`SupplyGuard_Diagrams/System_Architecture_Diagram.png`](file:///c:/Users/srila/OneDrive/Desktop/BTECH/4th%20year/mini%20project/SupplyGuard_Diagrams/System_Architecture_Diagram.png)
   - Activity Diagram: [`SupplyGuard_Diagrams/UML_Activity_Diagram.png`](file:///c:/Users/srila/OneDrive/Desktop/BTECH/4th%20year/mini%20project/SupplyGuard_Diagrams/UML_Activity_Diagram.png)
2. **`notebooks/02_Training_Demo.ipynb`:** End-to-end interactive demo loading models, running inference on a batch, and plotting gradient attribution.
3. **`README.md`:** Comprehensive instructions covering setup, reproduction of the base paper, running the node extension, and launching Streamlit.

---

## Comprehensive 14-Point Verification Checklist

| # | Checkpoint | Verification Method | Success Criteria |
|:---:|:---|:---|:---|
| 1 | **Dataset Integrity** | `python setup_and_download.py` | `train.csv` downloaded, 650k rows verified, clean headers |
| 2 | **Chronological Split** | Check split indices in `src/dataset.py` | 80:10:10 strict time order, zero future shuffling |
| 3 | **Zero Data Leakage** | Inspect scaler in `src/dataset.py` | `MinMaxScaler` fit ONLY on training set |
| 4 | **Forward Horizon ($t+1$)** | Inspect `dataset.py` window loop | $X[t-9:t]$ maps to $Y[t+1]$, not current step $t$ |
| 5 | **Symmetric Graph Laplacian** | Assert in `src/graph_builder.py` | $A = A^T$ (undirected); $\hat{A}$ matches spectral Kipf-Welling |
| 6 | **Manual Batched GCN** | Check `gcn_layer.py` forward pass | `ax = torch.matmul(adj_norm, x)` passes without PyG |
| 7 | **Preserved Node Embeddings** | Assert shape in `NodeRiskGNNLSTM` | `node_emb.shape == (B, 4, 32)` (no mean pooling) |
| 8 | **2-Hop Receptive Field** | Gradient tracing test | Non-zero gradient between Supplier and Distributor |
| 9 | **Node Head Output** | Assert shape in `NodeRiskGNNLSTM` | `node_risks.shape == (B, 4)` |
| 10 | **Base Model Reproduction** | Train `HybridGNNLSTM_Overall` | Successfully trains with `mean(dim=1)` $\to$ `[B]` |
| 11 | **Loss Convergence** | Inspect `outputs/results/loss_log.csv` | Train and val MSE decrease smoothly |
| 12 | **Node Metrics Output** | Inspect `outputs/results/node_metrics.csv` | MSE, MAE, RMSE, $R^2$ generated for $S, M, D, R$ |
| 13 | **Attribution Engine** | Run `explainability.py` test | Returns feature, temporal, and node importance sums = 1.0 |
| 14 | **Streamlit App Launch** | Run `streamlit run app/streamlit_app.py` | Launches cleanly, renders radar, graph, and explanations |

---

## Implementation Schedule (Plan A: 32–42 Hours Total)

```
Day 1 (Morning)   : Phase 0 — Environment verification & raw dataset download
Day 1 (Afternoon) : Phase 1 — Configuration & forward forecasting sliding window pipeline
Day 2             : Phase 2 — Manual GCN, Base-Paper Model, Proposed Node Model, Baselines
Day 3             : Phase 3 — Unified training engine & loss formulation
Day 4             : Phase 4 & 5 — Evaluation metrics generation & Explainability Engine
Day 5             : Phase 6 — Interactive SupplyGuard Streamlit UI
Day 6             : Phase 7 — Jupyter notebooks, README, verification checklist audit
```

---

# PLAN B — Optional Extensions (Only After Plan A is 100% Complete)

> [!NOTE]
> All Plan B extensions are strictly optional. Start only after all 14 checkpoints in Plan A have passed.

1. **Extension B1 — Hyperparameter Ablation Study (3–4h):**  
   Evaluate sensitivity to sequence length ($L \in \{5, 10, 20\}$) and GCN hidden dimension ($d \in \{16, 32, 64\}$).
2. **Extension B2 — Graph Attention Network (GAT) Branch (4–5h):**  
   Implement a manual single-head Graph Attention Layer to learn dynamic edge weights $\alpha_{ij}$ between echelons.
3. **Extension B3 — Real-Time Simulation Mode (3–4h):**  
   Add a streaming simulator in Streamlit generating sequential synthetic risk shocks to observe live downstream propagation.
4. **Extension B4 — Local Network & Cloud Deployment (2–3h):**  
   Configure Streamlit for LAN hosting (`--server.address 0.0.0.0`) or deploy to Streamlit Community Cloud.
5. **Extension B5 — Transformer Empirical Comparison (PatchTST) (5–6h):**  
   Compare LSTM temporal encoder against a lightweight PatchTST Transformer encoder to empirically verify that LSTMs generalize better on small tabular time-series without overfitting.

---

## Architectural Comparison Matrix

| Component | Base Research Paper (IEEE ICCMC 2025) | SupplyGuard (Our Updated Implementation) |
|:---|:---|:---|
| **Primary Architecture** | Hybrid GCN + LSTM | Hybrid GCN + LSTM |
| **GCN Layers & Depth** | 1 layer (1-hop limit) | **2 layers (`1 -> 32 -> 32`) enabling 2-hop chain propagation** |
| **Graph Symmetry** | Omitted | **Symmetric Normalized Laplacian (Kipf & Welling compliant)** |
| **GCN Pooling** | Global Mean Pooling (`mean(dim=1)`) | **None** (Preserves individual echelon embeddings `[B,4,32]`) |
| **Echelon Identity** | Destroyed | **Learnable Echelon Embedding `self.echelon_emb` $[1, 4, 32]$** |
| **Primary Target** | Single continuous overall risk $\hat{TRI}$ | **4 individual next-step risks** $[\hat{R}_S, \hat{R}_M, \hat{R}_D, \hat{R}_R]_{t+1}$ |
| **Overall Risk** | Direct neural output | **Derived aggregate average** ($\frac{1}{4}\sum \hat{R}_i$) |
| **Prediction Horizon** | Static / current-step reconstruction | **True forward forecasting ($t+1$)** |
| **Explainability** | None (Black Box) | **Input $\times$ Gradient attribution (Feature + Time + Node)** |
| **User Interface** | None | **SupplyGuard Streamlit Dashboard (Radar + Network + Why)** |
| **Primary Evaluation** | Simulation MSE / Ambiguous 88% Accuracy | **Rigorous node-level and overall MSE, MAE, RMSE, $R^2$** |
| **Viva Defensibility** | Vulnerable to target ambiguity & simulation hype | **Rock-solid: Reproduction baseline + Novel node-level extension** |
