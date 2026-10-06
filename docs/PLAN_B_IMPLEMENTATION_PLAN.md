# SupplyGuard — Plan B Implementation Plan (Deferred Extensions)

> **STATUS: DEFERRED / BLOCKED**  
> Per **AGENTS.md Rule 1**, Plan B extensions must **NOT** be started until Plan A Gate 8 (End-to-End Pipeline & Claims Audit) is completely finished, empirically verified, and formally approved.

---

## 1. Overview of Plan B
Plan B represents experimental and production enhancements beyond the foundational 4-tier benchmark pipeline established in Plan A. While Plan A focuses on verifying whether spatial graph inductive bias (`STGCNLSTM`) improves multi-horizon risk forecasting over temporal-only baselines (`LSTMBaseline`, `Ridge-AR(10)`, and `Persistence`) on the SCRM dataset, Plan B explores advanced graph neural network architectures and production deployment features.

---

## 2. Deferred Architectural Extensions

### 2.1 Graph Attention Networks (GAT)
- **Concept:** Replace static Kipf-Welling and directed normalized adjacency matrices with dynamic multi-head attention weights $\alpha_{ij}^{(t)}$.
- **Motivation:** In Plan A EDA, empirical cross-correlation showed strong direct correlation between Manufacturer and Retailer ($r = 0.54$) despite the physical DAG lacking an edge. A dynamic GAT layer can learn implicit echelon dependencies without manual graph restructuring.
- **Prerequisite:** Plan A Gate 8 complete.

### 2.2 Multi-Layer Deep Spatio-Temporal Graph Convolutions
- **Concept:** Explore 3-layer and 4-layer GCN architectures with residual skip-connections and layer normalization to capture $> 2$-hop cascading upstream delays.
- **Risk to monitor:** Over-smoothing across the small 4-node echelon graph ($S \leftrightarrow M \leftrightarrow D \leftrightarrow R$).

### 2.3 Dynamic Graph Edge Learning
- **Concept:** Learn the graph adjacency matrix $\mathbf{A}_{dyn} \in \mathbb{R}^{4 \times 4}$ directly from node embeddings via adaptive cosine similarity or parameterized matrix factorization:
  $$\mathbf{A}_{dyn} = \text{Softmax}\left(\text{ReLU}\left(\mathbf{E}_1 \mathbf{E}_2^T\right)\right)$$

---

## 3. Deferred Production & System Extensions

### 3.1 Real-Time Streaming Ingestion
- Streaming sliding window queue using Apache Kafka or Redis Streams.
- Micro-batch inference service with sub-50ms latency SLAs.

### 3.2 Automated Drift Detection & Continuous Retraining
- Kolmogorov-Smirnov (KS) test monitoring on incoming echelon risk distributions.
- Automated retraining triggers upon concept drift detection.

### 3.3 Interactive Counterfactual Simulation
- Interactive dashboard UI allowing supply chain managers to simulate disruptions (e.g., supplier shock $+0.50$) and visualize propagation across downstream echelons over $t+1 \dots t+10$.

---

## 4. Gating Invariants
Under no circumstances may any codebase files or tests for Plan B be implemented until Plan A completes all gates (Gates 0 through 8).
