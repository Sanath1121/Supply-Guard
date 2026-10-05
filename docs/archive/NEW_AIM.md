# REVISED PROJECT AIM & ARCHITECTURAL FOUNDATION

**Project Title:** SupplyGuard: Interpretable Hybrid GCN-LSTM Architecture for Echelon-Level Supply Chain Risk Forecasting  
**Academic Level:** B.Tech 4th Year (1st Semester Mini Project)  
**Base Research Paper:** *Hybrid GNN-LSTM Model for Real-Time Supply Chain Risk Prediction*, IEEE ICCMC 2025 (DOI: [10.1109/ICCMC65190.2025.11140739](https://doi.org/10.1109/ICCMC65190.2025.11140739))  
**Primary Dataset:** Banerjee et al. (2019), Mendeley Data V2 (DOI: [10.17632/gystn6d3r4.2](https://doi.org/10.17632/gystn6d3r4.2))  
**Document Status:** Final Approved Project Aim & Technical Scope  

---

## 1. Executive Summary & Official Project Aim

### 1.1 The Official Project Aim Statement
> **"To design, implement, and evaluate an interpretable hybrid Graph Convolutional Network and Long Short-Term Memory (GCN-LSTM) framework that learns both topological multi-echelon dependencies and temporal risk dynamics to forecast next-step individual risk across supply chain nodes (Supplier, Manufacturer, Distributor, Retailer), derives an overall supply chain risk score, and provides post-hoc gradient-based attribution to explain the root drivers of predicted vulnerabilities."**

### 1.2 Core Research Question
> **"Can a hybrid GCN-LSTM architecture learn both structural network dependencies and temporal patterns to forecast echelon-specific next-step risk ($\hat{Y}_{t+1}$), and can model attribution techniques provide interpretable explanations for how upstream risk propagates across the supply chain?"**

---

## 2. Background, Problem Analysis & The Need for Change

### 2.1 What the Base Paper Implemented
The IEEE ICCMC 2025 base paper (*Farzhana I. et al.*) introduced a hybrid neural architecture combining Graph Neural Networks (specifically Graph Convolutional Networks, GCN) and Long Short-Term Memory (LSTM) networks:
1. **Graph Representation:** Encoded structural relationships among supply chain participants via GCN.
2. **Temporal Representation:** Captured time-series historical dynamics via LSTM.
3. **Global Fusion:** Applied **global mean pooling** across all graph node embeddings to reduce the network to a single graph vector ($h_{graph} \in \mathbb{R}^{32}$), concatenated it with the temporal vector ($h_{temporal} \in \mathbb{R}^{64}$), and fed the fused vector ($h_{fused} \in \mathbb{R}^{96}$) into a fully connected layer.
4. **Single Scalar Output:** Predicted a single aggregated continuous risk index for the entire supply chain ($\hat{TRI} \in [0, 1]$).
5. **Reported Results:** Reported simulation performance of $\text{MSE} = 0.12$, $\text{MAE} = 0.08$, $R^2 = 0.92$, and an ambiguous classification "Accuracy = 88%".

### 2.2 Critical Limitations of the Base Paper Approach
While technically pioneering in combining spatial and temporal modalities, the base paper's formulation contains major engineering and practical shortcomings:

| Shortcoming | Base Paper Formulation | Impact / Deficiency |
| :--- | :--- | :--- |
| **1. Information Collapse via Mean Pooling** | GCN node embeddings are compressed via `mean(dim=1)` into one global vector. | Erases the unique identities and vulnerabilities of individual nodes ($S, M, D, R$). |
| **2. Lack of Actionable Granularity** | Generates a single overall score (e.g., "Supply Chain Risk = 0.72"). | Unusable for operational decision-making. A manager cannot tell *who* is in danger or *where* disruption originated. |
| **3. Reconstruction vs Forecasting** | Sliding window at step $t$ predicts risk at current step $t$. | Becomes a self-reconstruction task rather than predicting future risk ($t+1$). |
| **4. Black-Box Predictions** | Model yields risk scores with zero explanation or reasoning. | Neural network predictions cannot be audited or trusted in high-stakes logistics operations. |
| **5. Ambiguous "88% Accuracy" Metric** | Evaluated on continuous simulation data without defined category boundaries. | Fragile to defend during an academic B.Tech viva; regression metrics ($\text{MSE}, \text{MAE}, R^2$) are scientifically sound. |

---

## 3. The New Aim: Dual-Phase Progression & Innovation

Rather than choosing between simply replicating the paper or pursuing an unverified custom model, this project adopts a **structured dual-phase evolutionary approach**:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   PHASE 1: BASE-PAPER REPRODUCTION                     │
│  • Implement exact GCN + LSTM parallel fusion architecture             │
│  • Apply global mean pooling → Predict Overall Risk (TRI)              │
│  • Serves as scientific baseline & proves technical replication        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Evolution / Extension
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  PHASE 2: PROPOSED NODE-LEVEL MODEL                    │
│  • Remove destructive global pooling; preserve [B, 4, 32] GCN tensors  │
│  • Expand temporal embeddings to [B, 4, 64] & concatenate per node     │
│  • Predict next-step risk for EACH echelon: [R_S, R_M, R_D, R_R]       │
│  • Derive overall TRI as aggregate average: TRI = mean(R_nodes)        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Explainability Layer
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│              PHASE 3: INTERPRETABILITY & DECISION SUPPORT              │
│  • Post-hoc gradient-based attribution (∂R_node / ∂Inputs)             │
│  • Pinpoint root cause: Feature importance, temporal spikes, topology  │
│  • Interactive SupplyGuard Streamlit interface for live diagnostics    │
└────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Why This Evolution Matters
In industrial supply chain management, operational teams require actionable alerts:
- **Base paper output:** *"The overall supply chain risk is 0.72."* (Manager is left guessing which supplier or warehouse is failing).
- **SupplyGuard output:**  
  - **Supplier Risk:** `0.84` (CRITICAL) — *Upstream raw material risk spiked 35% across the last 3 time steps.*
  - **Manufacturer Risk:** `0.69` (HIGH) — *Elevated due to direct dependency on high-risk supplier.*
  - **Distributor Risk:** `0.42` (MEDIUM) — *Moderate downstream buffering.*
  - **Retailer Risk:** `0.25` (LOW) — *Currently insulated from upstream shock.*

---

## 4. Mathematical & Architectural Specification

### 4.1 Topology & Graph Formulation
The supply chain is modeled as a 4-node directed echelon graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, where nodes $\mathcal{V} = \{S, M, D, R\}$ correspond to:
1. **$S$:** Supplier
2. **$M$:** Manufacturer
3. **$D$:** Distributor
4. **$R$:** Retailer

The physical flow of goods follows directed edges:
$$\mathcal{E} = \{(S \rightarrow M), (M \rightarrow D), (D \rightarrow R)\}$$

- **Adjacency Matrix ($A \in \mathbb{R}^{4 \times 4}$):**
  $$A = \begin{bmatrix} 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \end{bmatrix}$$
- **Self-Loop Augmented Adjacency:** $\tilde{A} = A + I_4$
- **Degree Matrix:** $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$
- **Symmetric Normalized Adjacency:** $\hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$
- **Graph Convolution Layer:**
  $$H^{(l+1)} = \text{ReLU}\left( \hat{A} H^{(l)} W^{(l)} + b^{(l)} \right)$$

### 4.2 Temporal Sequence Formulation
- **Sliding Observation Window:** 10 historical time steps ($L = 10$).
- **Features per Time Step ($F = 5$):**
  1. $RI\_Supplier1$ (Supplier Risk Index)
  2. $RI\_Manufacturer1$ (Manufacturer Risk Index)
  3. $RI\_Distributor1$ (Distributor Risk Index)
  4. $RI\_Retailer1$ (Retailer Risk Index)
  5. $Total\_Cost$ (Aggregated supply chain operational cost)
- **Input Temporal Tensor:** $X_{seq} \in \mathbb{R}^{B \times 10 \times 5}$
- **LSTM Processing:**
  $$h_t, c_t = \text{LSTM}(x_t, (h_{t-1}, c_{t-1}))$$
  Output temporal representation: $T_{emb} = h_{10} \in \mathbb{R}^{B \times 64}$.

### 4.3 Node-Level Fusion Architecture (`NodeRiskGNNLSTM`)
Instead of collapsing the graph dimension via mean pooling, we preserve the identity of each node:

1. **Node Structural Embeddings:**
   $$X_{graph} \in \mathbb{R}^{B \times 4 \times 1} \xrightarrow{\text{GCN}} G_{emb} \in \mathbb{R}^{B \times 4 \times 32}$$
2. **Temporal Expansion:**
   Broadcast $T_{emb} \in \mathbb{R}^{B \times 64}$ across the 4 nodes:
   $$\tilde{T}_{emb} = \text{Expand}(T_{emb}) \in \mathbb{R}^{B \times 4 \times 64}$$
3. **Per-Node Feature Concatenation:**
   $$Z_i = [G_{emb} \,||\, \tilde{T}_{emb}] \in \mathbb{R}^{B \times 4 \times 96}$$
4. **Node Risk Head:**
   $$\hat{Y}_{t+1} = \text{Linear}(96 \rightarrow 32) \rightarrow \text{ReLU} \rightarrow \text{Dropout}(0.2) \rightarrow \text{Linear}(32 \rightarrow 1) \rightarrow \text{Sigmoid}$$
   $$\hat{Y}_{t+1} = \left[ \hat{R}_S^{t+1}, \hat{R}_M^{t+1}, \hat{R}_D^{t+1}, \hat{R}_R^{t+1} \right] \in \mathbb{R}^{B \times 4}$$
5. **Derived Overall Risk Score (TRI):**
   $$\hat{TRI}_{t+1} = \frac{\hat{R}_S^{t+1} + \hat{R}_M^{t+1} + \hat{R}_D^{t+1} + \hat{R}_R^{t+1}}{4} \in \mathbb{R}^{B \times 1}$$

```
                INPUT DATA (t-9 to t)
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
  Graph at time t                 Temporal Window
   X_graph [B,4,1]                 X_seq [B,10,5]
         │                               │
         ▼                               ▼
   Manual GCN (2-layer)            Stacked LSTM (2-layer)
         │                               │
         ▼                               ▼
   G_emb [B,4,32]                  T_emb [B,64]
   (Preserve Nodes)                      │
         │                         Expand to 4 nodes
         │                               ▼
         │                         T_exp [B,4,64]
         │                               │
         └───────────────┬───────────────┘
                         ▼
             Concat per-node: [B,4,96]
                         │
                         ▼
             Per-Node Regression Head
                         │
                         ▼
       Predicted Next-Step Risks [B, 4]
       [ R_S(t+1), R_M(t+1), R_D(t+1), R_R(t+1) ]
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
   Derived Overall TRI            Gradient Attribution
   TRI = mean(R_nodes)            ∂R_i / ∂Inputs
         │                               │
         └───────────────┬───────────────┘
                         ▼
             SupplyGuard Streamlit UI
```

---

## 5. Explainability Layer Specification

The system incorporates a lightweight, post-hoc gradient attribution mechanism:

### 5.1 Formulation
For a predicted node risk $\hat{R}_i$, compute the sensitivity of the prediction with respect to input features:
$$\text{Attribution}(X) = \left| X \odot \frac{\partial \hat{R}_i}{\partial X} \right|$$

### 5.2 Diagnostic Outputs
1. **Feature Attribution:** Identifies whether the risk spike was driven by the entity's own risk index, upstream supplier volatility, or operational cost shifts.
2. **Temporal Attribution:** Pinpoints which historical time steps ($t, t-1, t-2, \dots$) exerted the greatest influence.
3. **Graph Attribution:** Quantifies the magnitude of risk propagated from upstream parent nodes to downstream child nodes.

---

## 6. Project Scope & Feasibility Boundaries

To ensure 100% completion and flawless viva delivery within the B.Tech timeline, the project establishes explicit scope boundaries:

### 6.1 What IS in Scope (Plan A - 100% Required)
- **Dataset:** Mendeley Data V2 (Banerjee et al., 2019), 4-echelon dynamic time-series.
- **Preprocessing:** Chronological 80:10:10 train/validation/test split, train-only MinMax scaling (zero data leakage).
- **Baselines:** Standalone LSTM, Standalone GCN, Base-Paper Hybrid GCN-LSTM (`HybridGNNLSTM_Overall`).
- **Proposed Core Model:** Node-level Hybrid GCN-LSTM (`NodeRiskGNNLSTM`).
- **Explainability:** Gradient-based feature, temporal, and node attribution.
- **Evaluation:** Strict regression metrics ($\text{MSE}, \text{MAE}, \text{RMSE}, $R^2$) reported per node and overall.
- **Interface:** Interactive Streamlit dashboard showcasing node alerts, network graph heatmaps, and explanation breakdowns.
- **Compute:** 100% CPU-compatible, lightweight PyTorch implementation (training time $< 2$ minutes).

### 6.2 What is EXCLUDED (Preventing Scope Bloat)
- Satellite imagery / computer vision inputs.
- Real-time IoT physical hardware streams.
- Geopolitical news web-scraping / sentiment analysis.
- Distributed streaming architectures (Apache Spark, Apache Kafka).
- Heavy GNN dependencies (PyTorch Geometric C++ binaries on Windows avoided via clean manual tensor GCN).

---

## 7. Phased Implementation Roadmap

| Phase | Milestone Name | Key Deliverables |
| :---: | :--- | :--- |
| **Phase 0** | Environment & Ingestion | Verify PyTorch CPU environment; ingest raw Mendeley V2 CSV; establish project structure. |
| **Phase 1** | Forecasting Pipeline | Implement sliding window dataset ($X[t-9:t] \rightarrow Y[t+1]$); chronological split; train-only scaler. |
| **Phase 2** | Baseline Models | Implement Standalone LSTM and Standalone GCN benchmarks. |
| **Phase 3** | Base-Paper Reproduction | Build `HybridGNNLSTM_Overall` with global mean pooling; evaluate overall $\hat{TRI}_{t+1}$. |
| **Phase 4** | Proposed Node Model | Build `NodeRiskGNNLSTM` preserving $[B, 4, 32]$ node embeddings; predict $[\hat{R}_S, \hat{R}_M, \hat{R}_D, \hat{R}_R]_{t+1}$. |
| **Phase 5** | Explainability Engine | Develop `explainability.py` computing gradient attribution for features, time steps, and graph nodes. |
| **Phase 6** | Comparative Evaluation | Generate node-wise and overall metric tables ($\text{MSE}, \text{MAE}, \text{RMSE}, R^2$); generate prediction vs ground truth plots. |
| **Phase 7** | SupplyGuard Dashboard | Develop multi-tab Streamlit application: Node Risk Dashboard, Network Propagation Map, Explanation Inspector. |
| **Phase 8** | Viva Preparation & Report | Finalize technical documentation, slide deck, and B.Tech mini project report. |

---

## 8. Summary Comparison: Base Paper vs. SupplyGuard

| Attribute | Base Research Paper (IEEE ICCMC 2025) | SupplyGuard (Our Mini Project) |
| :--- | :--- | :--- |
| **Architecture** | Hybrid GCN + LSTM | Hybrid GCN + LSTM |
| **Graph Pooling** | Global Mean Pooling (`mean(dim=1)`) | **None** (Preserves individual node representations) |
| **Primary Target** | Single continuous overall risk $\hat{TRI}$ | **4 individual node risks** $[\hat{R}_S, \hat{R}_M, \hat{R}_D, \hat{R}_R]_{t+1}$ |
| **Overall Risk** | Direct neural scalar output | **Derived aggregate average** ($\frac{1}{4}\sum \hat{R}_i$) |
| **Prediction Horizon** | Static / current-step reconstruction | **True forward forecasting ($t+1$)** |
| **Explainability** | None (Black box) | **Gradient-based attribution (Features + Time + Graph)** |
| **User Interface** | None | **Interactive SupplyGuard Streamlit Dashboard** |
| **Primary Metrics** | Simulation MSE / MAE / Ambiguous 88% Accuracy | **Rigorous node-level and overall MSE, MAE, RMSE, $R^2$** |
| **Academic Value** | Initial conceptual hybrid idea | **Actionable, interpretable decision-support system** |

---

## 9. Conclusion & One-Sentence Verdict

> **"Implement the IEEE ICCMC 2025 base paper's GCN-LSTM architecture as a solid reproduction baseline, but engineer the core network to retain echelon-specific node embeddings, thereby forecasting next-step risk individually for Supplier, Manufacturer, Distributor, and Retailer while generating gradient-based attributions to explain risk propagation in an interactive decision-support system."**
