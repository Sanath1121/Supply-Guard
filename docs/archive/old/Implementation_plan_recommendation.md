# Implementation Plan Recommendation

Yes. After reviewing the current `final_implementation_plan.md`, I would
**not throw it away**. The current plan is a good foundation, but it
needs a few important architectural and scope changes so that it
actually implements the final recommendation: **base-paper GCN+LSTM
first, then a node-level GCN+LSTM model with interpretable
explanations**.

The biggest issue is that the current plan is still fundamentally an
**overall-risk prediction project**. It has dynamic node inputs, but the
model immediately does `mean(dim=1)` and produces one scalar.
fileciteturn2file0L283-L295

## 1. Change the project objective

### Current plan

The current plan's main target is:

\[ TRI = 0.25(RI_S+RI_M+RI_D+RI_R) \]

with one continuous output, plus a secondary classification accuracy
evaluation. fileciteturn2file0L44-L50

### Change it to

Make the **main project objective**:

> Predict the next-step risk of each supply-chain node:
>
> \[ `\hat{Y}`{=tex}\_{t+1}=
> \[`\hat{R}`{=tex}\_S,`\hat{R}`{=tex}\_M,`\hat{R}`{=tex}\_D,`\hat{R}`{=tex}\_R\]
> \]

Then derive overall supply-chain risk:

\[ `\hat{TRI}`{=tex}\_{t+1} = `\frac{
\hat{R}_S+\hat{R}_M+\hat{R}_D+\hat{R}_R
}{4}`{=tex} \]

This gives you exactly what your team originally wanted:

**Supplier risk → Manufacturer risk → Distributor risk → Retailer risk →
Overall risk**

The important change is that the model predicts **future risk**, not
merely reconstructing the current risk.

------------------------------------------------------------------------

# 2. This is the most important correction: change the prediction target

The current dataset design creates a potential target leakage/problem.

The plan says the sliding window contains:

``` text
sequence = [t-9, ..., t]
graph_x  = values at t
target   = TRI at t
```

The current plan explicitly says `graph_x` comes from the **last
time-step**, while `TRI` is computed from those same scaled RI values.
fileciteturn2file0L209-L226

That means the model is being asked, in effect:

> "Given today's risk values, predict today's average risk."

That is much weaker than a real forecasting system.

### Change it to

For every sample:

``` text
Input window:
[t-9, t-8, ..., t]

Target:
[t+1]
```

So:

``` text
sequence  = X[t-9:t+1]
graph_x   = node risks at t
target    = node risks at t+1
```

More precisely:

``` text
sequence: [10, 5]
graph_x:  [4, 1]
node_target: [4]
```

where:

``` text
node_target =
[
  RI_Supplier(t+1),
  RI_Manufacturer(t+1),
  RI_Distributor(t+1),
  RI_Retailer(t+1)
]
```

This single modification makes your project much more defensible
academically.

------------------------------------------------------------------------

# 3. Keep the base-paper model, but rename it as the baseline

The current architecture is:

``` text
GCN → mean pooling → 32
LSTM → 64
concat → 96
FC → 1 risk score
```

That is already explicitly defined in the plan.
fileciteturn2file0L283-L295

**Do not delete this.**

Instead create two models:

### Model A --- `HybridGNNLSTM_Overall`

This is your **base-paper reproduction baseline**.

``` text
GCN [B,4,1]
     ↓
GCN [B,4,32]
     ↓
mean pool
     ↓
[B,32]

LSTM [B,10,5]
     ↓
[B,64]

concat
     ↓
[B,96]
     ↓
FC
     ↓
overall risk
```

This lets you honestly say:

> "First, we reproduced the core hybrid GCN-LSTM formulation represented
> in the base paper."

### Model B --- `NodeRiskGNNLSTM`

This becomes your **actual final project model**.

``` text
                    ┌── GCN ──> [B,4,32]
                    │
Input graph --------┤
                    │
                    └── preserve all 4 node embeddings

Sequence ──> LSTM ──> [B,64]
                       │
                       ↓
                 expand to 4 nodes
                       │
                       ↓
          ┌────────────────────────┐
          │ concatenate per node   │
          │ [B,4,32] + [B,4,64]    │
          └────────────────────────┘
                       │
                       ↓
                  [B,4,96]
                       │
                       ↓
                  Node Head
                       │
                       ↓
                 [B,4] risk
```

This is much better than creating four completely independent LSTMs.

It remains simple, shared, lightweight, CPU-friendly, and easy to
explain in a viva.

------------------------------------------------------------------------

# 4. Change the GCN pooling design

Current plan:

``` python
g_emb = gcn_output.mean(dim=1)
```

The problem is that this destroys the distinction between:

``` text
Supplier
Manufacturer
Distributor
Retailer
```

before prediction. That directly conflicts with your node-level goal.

### Replace

``` python
g_emb = gcn_output.mean(dim=1)
```

with:

``` python
node_emb = gcn_output
```

and retain:

``` text
node_emb.shape = [B, 4, 32]
```

This should become a hard architectural requirement.

### New assertion

``` python
assert node_emb.shape == (B, 4, 32)
```

Then combine each node embedding with the shared temporal embedding:

``` python
temporal = t_emb.unsqueeze(1).expand(-1, 4, -1)
fused = torch.cat([node_emb, temporal], dim=-1)
```

Result:

``` text
[B,4,96]
```

Then:

``` python
node_risk = node_head(fused).squeeze(-1)
```

Result:

``` text
[B,4]
```

That is the core modification I'd make.

------------------------------------------------------------------------

# 5. Change `dataset.py`

The current plan has:

``` python
'target': self.targets[idx]   # scalar
'label': self.labels[idx]
```

fileciteturn2file0L216-L226

Change this to:

``` python
{
    'sequence':   [10, 5],
    'graph_x':    [4, 1],
    'node_target':[4],
    'tri_target': scalar,
    'label':      scalar
}
```

I would actually store both:

``` text
node_target
tri_target
```

because that makes experimentation easier.

### Why keep `tri_target`?

Because your base-paper comparison still needs an overall-risk metric.

So:

``` python
node_target = next_step_RI[4]
tri_target = node_target.mean()
```

This gives you both:

**Node-level prediction**

and

**overall prediction**

from the same sample.

------------------------------------------------------------------------

# 6. Change `config.py`

Current configuration is largely fine. fileciteturn2file0L180-L203

Add:

``` python
NUM_NODES = 4
NUM_INPUT_FEATURES = 5
NUM_NODE_FEATURES = 1
NODE_NAMES = [
    "Supplier",
    "Manufacturer",
    "Distributor",
    "Retailer"
]

PREDICTION_HORIZON = 1
```

And rename:

``` python
TARGET_COL = 'TRI'
```

to something like:

``` python
NODE_TARGET_COLS = [
    'RI_Supplier1',
    'RI_Manufacturer1',
    'RI_Distributor1',
    'RI_Retailer1'
]
```

TRI becomes a **derived evaluation quantity**, rather than the primary
learning target.

------------------------------------------------------------------------

# 7. Change the training loss

Current plan:

``` python
loss = MSE(pred, tgt)
```

where `pred` and `tgt` are scalar overall-risk values.
fileciteturn2file0L330-L350

Change the main loss to:

\[ L\_{node} = MSE(`\hat{Y}`{=tex}*{node},Y*{node}) \]

So:

``` python
node_loss = mse(pred_node, target_node)
```

Optionally derive overall loss:

``` python
pred_tri = pred_node.mean(dim=1)
target_tri = target_node.mean(dim=1)

tri_loss = mse(pred_tri, target_tri)
```

Then:

``` python
loss = node_loss + lambda_tri * tri_loss
```

I would use:

``` python
lambda_tri = 0.25
```

or simply start with:

``` python
loss = node_loss
```

and only add the auxiliary TRI loss if needed.

For a B.Tech implementation, **starting with node MSE alone is
cleaner**.

------------------------------------------------------------------------

# 8. Change the evaluation strategy

The current evaluation function returns:

``` text
Model | MSE | MAE | RMSE | R² | Accuracy
```

fileciteturn2file0L370-L388

That is not enough anymore.

Your final evaluation should report:

### Node-level

  Node             MSE   MAE   RMSE   R²
  -------------- ----- ----- ------ ----
  Supplier                          
  Manufacturer                      
  Distributor                       
  Retailer                          

### Overall

  Model               MSE   MAE   RMSE   R²
  ----------------- ----- ----- ------ ----
  LSTM                                 
  GCN                                  
  Hybrid GCN-LSTM                      
  Node GCN-LSTM                        

This gives you a far stronger results section.

------------------------------------------------------------------------

# 9. Remove the current `qcut` accuracy method

I would change this part.

The current plan uses:

``` python
pd.qcut(tri_values, q=5, labels=[0,1,2,3,4])
```

to convert predicted TRI into the five stability categories.
fileciteturn2file0L375-L380

That is not a good primary evaluation design because the categories are
artificially created from the prediction distribution.

Instead:

### Primary evaluation

Use:

``` text
MSE
MAE
RMSE
R²
```

for continuous node and TRI prediction.

### Secondary classification

Only use `SCMstability_category` if you explicitly define a separate
mapping and justify it.

So I recommend:

> **Remove "88% Accuracy" from the core implementation target.**

Do not design the project around reproducing that number.

That also protects you from an awkward viva question about why your
accuracy differs from the paper.

------------------------------------------------------------------------

# 10. Add the actual explainability module to Plan A

Right now explainability exists only as **Plan B Extension B4**.
fileciteturn3file1

According to our final recommendation, it should move into the **main
project**, but in a simplified form.

You do not need SHAP.

Create:

``` text
src/explainability.py
```

with:

``` python
compute_gradient_attribution(...)
```

It should return:

``` text
feature importance
time-step importance
node importance
```

For example:

``` text
Prediction:
Retailer Risk = 0.82

Main contributing factors:
1. Retailer RI ↑ in last 2 time steps
2. Distributor RI ↑ in last 3 time steps
3. Total Cost volatility
4. Upstream manufacturer risk
```

Important: call these **attributions/contributing inputs**, not "the
reason the model knows."

------------------------------------------------------------------------

# 11. Add node-level attribution, not "GCN attention"

The current B4 text mentions visualizing "per-node attention scores" or
GCN edge weights. fileciteturn3file1

A vanilla GCN does not inherently provide attention weights.

So the clean Plan-A solution is:

### Gradient-based node attribution

Compute how much the output for:

``` text
Supplier
Manufacturer
Distributor
Retailer
```

changes with respect to each node input.

That gives you something like:

``` text
Supplier       ██████████
Manufacturer   ███████
Distributor    █████████
Retailer       ███████████
```

Then visualize it on the graph.

This is much easier than adding GAT just to obtain attention weights.

------------------------------------------------------------------------

# 12. Change the Streamlit UI significantly

The current UI expects one overall risk score and five sliders.
fileciteturn2file0L443-L456

That's not compatible with the new temporal forecasting formulation.

The current `predict_single()` also says:

> "Constructs a synthetic sequence from single input."

fileciteturn2file0L478-L490

**I would remove that.**

A single current observation is not a genuine 10-step sequence.

## New UI

Use one of these two modes:

### Recommended

**Demo Prediction**

Load a real 10-step window from the test dataset.

User selects:

``` text
Date / sample
```

Then the system shows:

``` text
Supplier       0.71  HIGH
Manufacturer   0.64  MEDIUM
Distributor    0.80  HIGH
Retailer       0.58  MEDIUM

Overall TRI    0.68  HIGH
```

Then:

``` text
WHY?
↓
Top contributing time steps/features
↓
Supply-chain graph with highlighted important nodes
```

That is much more scientifically valid.

### Optional

Keep manual sliders, but require the user to enter **10 time steps**,
not one.

I would **not** do that initially. It adds UI complexity without helping
the core research.

------------------------------------------------------------------------

# 13. Streamlit should become node-oriented

Change Tab 1 from:

``` text
Risk Score
Risk Category
Gauge
```

to:

``` text
Node Risk Dashboard
```

Something like:

``` text
SUPPLIER          MANUFACTURER
  0.72               0.61
  HIGH               MEDIUM

DISTRIBUTOR       RETAILER
  0.81               0.58
  HIGH               MEDIUM

          OVERALL TRI
             0.68
             HIGH
```

Then:

``` text
Risk Contribution / Explanation
```

Then:

``` text
Supply Chain Graph
S → M → D → R
```

with important nodes highlighted.

That makes the UI visibly demonstrate your project contribution.

------------------------------------------------------------------------

# 14. Change the project directory structure

Add:

``` text
src/
├── explainability.py
└── models/
    ├── gcn_layer.py
    ├── hybrid_model.py
    ├── node_risk_model.py
    └── baselines.py
```

And:

``` text
outputs/
├── models/
│   ├── hybrid_overall_best.pt
│   └── node_risk_best.pt
├── results/
│   ├── overall_metrics.csv
│   ├── node_metrics.csv
│   └── attribution_examples.csv
└── figures/
```

This makes the distinction between the **paper reproduction** and the
**final contribution** explicit.

------------------------------------------------------------------------

# 15. Change the experiment sequence

The current Plan A jumps directly toward one hybrid model plus
baselines.

I recommend rewriting it as:

### Experiment 1 --- LSTM

``` text
Sequence → LSTM → overall risk
```

### Experiment 2 --- GCN

``` text
Graph → GCN → overall risk
```

### Experiment 3 --- Base Hybrid

``` text
GCN + LSTM → overall risk
```

This is your reproduction.

### Experiment 4 --- Proposed Node Hybrid

``` text
GCN + LSTM → 4 node risks
```

This is your contribution.

### Experiment 5 --- Explainability

``` text
Node prediction
      ↓
gradient attribution
      ↓
feature/time/node explanation
```

This sequence is much stronger for both the implementation and the final
report.

------------------------------------------------------------------------

# 16. Change the verification checklist

The existing checklist still expects a scalar pooled GCN architecture.
For example, it checks `[B,4,32] → pooled [B,32]`.
fileciteturn2file0L519-L536

Replace those with:

  \#   Verification
  ---- -----------------------------------------------
  1    Chronological split is correct
  2    Scaler fitted only on train
  3    Input sequence = `[B,10,5]`
  4    Graph input = `[B,4,1]`
  5    GCN output = `[B,4,32]`
  6    LSTM output = `[B,64]`
  7    Temporal embedding expanded to `[B,4,64]`
  8    Node fusion = `[B,4,96]`
  9    Node prediction = `[B,4]`
  10   Next-step targets are truly future timestamps
  11   Node MSE/MAE/R² generated
  12   Overall TRI metrics generated
  13   Explanation map generated
  14   Streamlit node dashboard works

That is much better aligned with the actual research question.

------------------------------------------------------------------------

# 17. Move B4 into Plan A

Current Plan B contains:

-   B1 Ablation
-   B2 GAT
-   B3 Real-time simulation
-   B4 Explainability
-   B5 Cloud
-   B6 Report
-   B7 Transformer

fileciteturn2file0L555-L630

I would restructure that.

## Plan A --- Required

**Core**

1.  Dataset + preprocessing
2.  LSTM baseline
3.  GCN baseline
4.  Base GCN+LSTM overall model
5.  Node-level GCN+LSTM model
6.  Gradient-based explanation
7.  Streamlit UI
8.  Evaluation + report

## Plan B --- Optional

Keep only:

``` text
B1 Ablation
B2 GAT
B3 Real-time simulation
B5 Deployment
B7 Transformer comparison
```

Move **B4 Explainability into Plan A**.

------------------------------------------------------------------------

# 18. Do not make all of the current Plan B

This is where I would make the largest scope reduction.

The current plan potentially includes:

-   hyperparameter ablation
-   GAT
-   real-time simulation
-   explainability
-   cloud deployment
-   IEEE report
-   Transformer comparison

That's a lot for a mini-project. fileciteturn2file0L555-L732

Your final project should instead be:

> **Base reproduction + node-level prediction + explainability + UI**

That is already a strong B.Tech project.

Everything else should genuinely remain optional.

------------------------------------------------------------------------

# 19. Revised final architecture

I would change the architecture section of the plan to this:

``` text
                 Mendeley Supply Chain Dataset
                            │
                  Chronological 80:10:10
                            │
                    Train-only scaling
                            │
                  Sliding Window (10)
                            │
             ┌──────────────┴──────────────┐
             │                             │
      Graph at time t              Sequence t-9 ... t
         [B,4,1]                       [B,10,5]
             │                             │
             ▼                             ▼
          Manual GCN                     LSTM
             │                             │
             ▼                             ▼
        [B,4,32]                       [B,64]
             │                             │
             │                     Expand to 4 nodes
             │                             │
             └──────────────┬──────────────┘
                            ▼
                      [B,4,96]
                            │
                         Node Head
                            │
                            ▼
                    [B,4] Node Risks
                            │
             ┌──────────────┴──────────────┐
             │                             │
             ▼                             ▼
       Node Metrics                  Mean / TRI
                                     Overall Risk
             │                             │
             └──────────────┬──────────────┘
                            ▼
                       Explanation
             (time + feature + node attribution)
                            │
                            ▼
                     SupplyGuard UI
```

And separately:

``` text
Base-paper reproduction:
GCN → mean pool → LSTM → FC → overall risk
```

This distinction is extremely useful in your viva.

------------------------------------------------------------------------

# 20. Revised project story

Your current plan's story is essentially:

> "We implemented the paper."

The improved project story becomes:

> **Phase 1:** We reproduced the paper's hybrid GCN-LSTM architecture
> for overall supply-chain risk prediction.

> **Phase 2:** We identified that global pooling collapses node-specific
> information, so we preserved node embeddings and extended the
> architecture to produce individual risk predictions for Supplier,
> Manufacturer, Distributor, and Retailer.

> **Phase 3:** We added gradient-based attribution to explain which
> nodes, features, and recent time steps influenced the prediction.

That is a much stronger academic narrative.

------------------------------------------------------------------------

# What I would actually modify in `final_implementation_plan.md`

The changes can be summarized like this:

  -----------------------------------------------------------------------
  Current plan                        Revised plan
  ----------------------------------- -----------------------------------
  Overall TRI is primary target       **Next-step node risks are primary
                                      target**

  Current-step TRI                    **t+1 prediction**

  GCN → mean pooling                  **Preserve `[B,4,32]` node
                                      embeddings**

  One scalar output                   **Four node outputs + derived TRI**

  Accuracy as major metric            **Regression metrics primary**

  `qcut` category mapping             **Remove from core evaluation**

  Explainability = Plan B             **Explainability = Plan A**

  Single synthetic input in UI        **Real 10-step test window**

  One `HybridGNNLSTM`                 **Base model + proposed node
                                      model**

  `predict_single()` synthetic        **window-based inference**
  sequence                            

  11 verification checks              **\~14 node-aware checks**

  Many Plan B extras                  **Only a few optional extensions**
  -----------------------------------------------------------------------

The existing plan already has the right low-level foundation---manual
batched GCN, dynamic graph inputs, chronological split, train-only
scaling, and a modest CPU-friendly architecture---so these are
**surgical changes rather than a rewrite**.
fileciteturn2file0L41-L54

### One more important change

I would also delete the hard-coded UI text:

> `Confidence: 88.4%`

from the plan. fileciteturn2file0L443-L456

Unless you actually implement a calibrated confidence method, a
regression model's output should not be presented as an 88.4% confidence
figure.

------------------------------------------------------------------------

## My recommended final Plan A

The final implementation plan should therefore be:

**Phase 0:** Environment + dataset\
**Phase 1:** Correct forecasting dataset pipeline\
**Phase 2:** LSTM + GCN baselines\
**Phase 3:** Base-paper GCN+LSTM reproduction\
**Phase 4:** Node-level GCN+LSTM extension\
**Phase 5:** Explainability\
**Phase 6:** Evaluation + plots\
**Phase 7:** Streamlit SupplyGuard UI\
**Phase 8:** Report + packaging

That is the version I would actually implement with a B.Tech team.

The crucial point is: **do not build a generic GNN and then replace it.
Build the base-paper GCN+LSTM first, but design the code so the per-node
GCN embeddings are never irreversibly discarded.** That makes the
progression clean, feasible, and defensible.
