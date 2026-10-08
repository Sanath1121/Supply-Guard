"""Build the corrected SupplyGuard paper by editing the original .docx XML in place.

usage: python build_paper.py <original.docx> <project_root> <out.docx>
Every number in Tables IV-VI and in the text comes from project_root/outputs/results and
project_root/docs/paper/evidence; nothing is typed by hand except wording.
"""
import json
import re
import sys
import zipfile

import matplotlib.image as mpimg
import numpy as np
import pandas as pd

SRC, ROOT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
R = ROOT + "/outputs/results/"
EV = json.load(open(ROOT + "/docs/paper/evidence/paper_facts.json", encoding="utf-8"))
IG = pd.read_csv(ROOT + "/docs/paper/evidence/ig_audit_weighted.csv")
ov = pd.read_csv(R + "overall_metrics.csv").set_index("model")
nd = pd.read_csv(R + "node_metrics.csv")
sv = pd.read_csv(R + "severity_metrics.csv").set_index("model")
cm = pd.read_csv(R + "confusion_matrix.csv", index_col=0).values
att = pd.read_csv(R + "attribution_examples.csv")


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def sci(v):
    m, e = f"{v:.1e}".split("e")
    return f"{m} × 10^{int(e)}"


def pm(mean, std, scale, nd_=2):
    if pd.isna(std):
        return f"{mean * scale:.{nd_}f}"
    return f"{mean * scale:.{nd_}f} ± {std * scale:.{nd_}f}"


# ------------------------------------------------------------------ numbers
def row_iv(m):
    r, q = ov.loc[m], sv.loc[m]
    return [pm(r.MSE_mean, r.MSE_std, 1e4), pm(r.MAE_mean, r.MAE_std, 1e3, 3), pm(r.RMSE_mean, r.RMSE_std, 1e2, 3),
            pm(r.R2_mean, r.R2_std, 1, 4), pm(q.accuracy_mean, q.accuracy_std, 100, 1),
            pm(q.macro_F1_mean, q.macro_F1_std, 100, 1)]


nodes = ["Supplier", "Manufacturer", "Distributor", "Retailer"]


def nrow(model, node):
    return nd[(nd.model == model) & (nd.node == node)].iloc[0]


tab5 = []
for n in nodes:
    p, l, d = nrow("persistence", n), nrow("lstm", n), nrow("st_gcn_lstm_dir", n)
    tab5.append([f"{p.MSE_mean * 1e4:.2f}", pm(l.MSE_mean, l.MSE_std, 1e4), pm(d.MSE_mean, d.MSE_std, 1e4),
                 f"{d.skill_score_mean * 100:.1f}"])
mp = np.mean([nrow("persistence", n).MSE_mean for n in nodes])
ml = np.mean([nrow("lstm", n).MSE_mean for n in nodes])
md = np.mean([nrow("st_gcn_lstm_dir", n).MSE_mean for n in nodes])
tab5.append([f"{mp * 1e4:.2f}", f"{ml * 1e4:.2f}", f"{md * 1e4:.2f}", f"{(1 - md / mp) * 100:.1f}"])

sh = EV["shock_top5pct"]
shock_dir = (min(sh["mse_directed_shock_by_seed"]), max(sh["mse_directed_shock_by_seed"]))
calm_dir = (min(sh["mse_directed_calm_by_seed"]), max(sh["mse_directed_calm_by_seed"]))
gap = EV["ig_completeness_abs_gap_raw_mode"]


def ig_share(target, delta):
    g = IG[(IG.target == target) & (IG.delta == delta)][["S", "M", "D", "R", "Cost"]].sum()
    return (g / g.sum()) * 100


own = {"Supplier": "S", "Manufacturer": "M", "Distributor": "D", "Retailer": "R"}
raw_own = [ig_share(t, False)[own[t]] for t in nodes]
raw_cost = [ig_share(t, False)["Cost"] for t in nodes]
del_cost = [ig_share(t, True)["Cost"] for t in nodes]
inactive = {t: (IG[(IG.target == t) & (IG.delta)].total < 1e-9).mean() * 100 for t in nodes}
sup_to_dist = ig_share("Distributor", False)["S"]
rec = np.diag(cm) / cm.sum(1) * 100
acc_cm = np.trace(cm) / cm.sum() * 100
deletion_pass = int(att.deletion_test_passed.sum())
lat_inf, lat_ig = EV["inference_ms_cpu_batch1"], EV["ig_ms_cpu_per_target_64steps"]
w_tr, w_va, w_te = EV["windows_train_val_test"]
rl = EV["row_loss"]
pr = EV["partition_rows_train_val_test"]
t2 = EV["table2_raw_train_mean_std"]
cls = EV["test_class_counts_low_med_high"]
per_node_dir = ov.loc["st_gcn_lstm_dir"]
imp_dir = ov.loc["st_gcn_lstm_dir"].pct_improvement_pers_mean
imp_dir_std = ov.loc["st_gcn_lstm_dir"].pct_improvement_pers_std
imp_lstm = ov.loc["lstm"].pct_improvement_pers_mean
imp_ridge = ov.loc["ar10_ridge"].pct_improvement_pers_mean
d_ld = ov.loc["lstm"].MSE_mean - ov.loc["st_gcn_lstm_dir"].MSE_mean
thr = ov.loc["lstm"].MSE_std + ov.loc["st_gcn_lstm_dir"].MSE_std
rel_ld = d_ld / ov.loc["lstm"].MSE_mean * 100

# ------------------------------------------------------------------ text edits keyed by 1-based paragraph number
T = {}
T[19] = ["Abstract—",
         "Multi-echelon supply networks can transmit an upstream disruption to downstream tiers, yet many risk-forecasting "
         "models predict a single aggregated index and are evaluated at very short horizons where persistence is difficult "
         "to beat. We present SupplyGuard, a spatio-temporal model that forecasts the risk indices of four echelons "
         "(Supplier, Manufacturer, Distributor, Retailer) ten minutes ahead (H = 5 steps). It combines a two-layer relational "
         "graph convolution with separate upstream and downstream messages, a shared two-layer LSTM per node, and a residual "
         "head that predicts the change from the current state; Integrated Gradients provides post hoc, node-level "
         f"attributions. On the Mendeley SCRM time-series dataset ({rl['n_clean_rows_retained']:,} usable rows after cleaning; "
         f"chronological 80/10/10 split; {w_te:,} test windows), the directed model reduces the mean-node MSE to "
         f"{ov.loc['st_gcn_lstm_dir'].MSE_mean * 1e4:.2f} × 10^-4 ± {ov.loc['st_gcn_lstm_dir'].MSE_std * 1e4:.2f} × 10^-4 "
         f"(five seeds), {imp_dir:.0f}% below Naive Persistence ({ov.loc['persistence'].MSE_mean * 1e4:.2f} × 10^-4) and "
         f"{rel_ld:.0f}% below a graph-free LSTM ({ov.loc['lstm'].MSE_mean * 1e4:.2f} × 10^-4). The gain is concentrated on "
         f"shock windows, where its error is about half that of persistence, while on calm windows it is about twice as large. "
         f"Persistence remains the most accurate on three-tier severity classification "
         f"({sv.loc['persistence'].accuracy_mean * 100:.1f}% vs {sv.loc['st_gcn_lstm_dir'].accuracy_mean * 100:.1f}%). "
         f"Integrated Gradients attributions have a median completeness gap of {sci(gap['median'])} and describe local model "
         "sensitivity, not causal effects. Limitations include a confounded graph-free ablation and a single linear topology."]
T[22] = ["Contemporary manufacturing and commerce rely on lean inventory management, just-in-time (JIT) fulfillment and tightly "
         "synchronized transportation networks. These practices reduce holding costs in steady state but leave little buffer "
         "against disturbances. When an upstream tier is disrupted, for example by a production outage, component stockout or "
         "port congestion, the schedule deviation can propagate through intermediate facilities and distribution hubs and "
         "appear as stockouts at the retail tier. This propagation is studied in the supply chain literature as the ripple "
         "effect [1], [7]."]
T[23] = ["In response, operations teams increasingly use continuous telemetry of material transit, machine status, inventory "
         "levels and logistics cost. Converting such a stream into forecasts is typically done with autoregressive estimators, "
         "discrete-event simulators or Long Short-Term Memory (LSTM) networks [4], and machine learning is being applied to "
         "supply chain management more broadly [12]. Two limitations of these formulations motivate this work."]
T[24] = ["First, many formulations, including the hybrid graph-LSTM model that we re-implement as a baseline [10], predict a "
         "single aggregated index of network risk. An aggregate index does not show whether a bottleneck originates at a "
         "supplier or at a distributor, so corrective action cannot be targeted. Second, short-horizon forecasts are easy to "
         "match with a trivial baseline: in the dataset used here, adjacent two-minute observations are very strongly "
         "correlated (lag-1 Pearson r between 0.97 and 0.996 across the four echelons), so predicting that the future equals "
         "the present is difficult to beat. A forecasting model must therefore be judged against persistence."]
T[26] = ["• Fragility of Lean Multi-Tier Chains: ",
         "Lean networks hold limited buffers, so a local supplier bottleneck can propagate to later echelons [1]."]
T[27] = ["• Limited Visibility in Aggregate Indices: ",
         "A single network-wide index does not show which echelon is deteriorating."]
T[28] = ["• The 10-Minute Horizon: ",
         "Extending the forecast from 2 minutes (H = 1) to 10 minutes (H = 5) gives more lead time while remaining short enough "
         "to be compared directly with a persistence baseline."]
T[29] = ["• The Need for Interpretable Output: ",
         "Operators need to know which inputs a forecast is sensitive to. Integrated Gradients gives signed, node-level "
         "sensitivities with a measurable completeness check, but these are not causal explanations."]
T[30] = ["C. Key Methodological Contributions"]
T[31] = ["• Directed Spatio-Temporal Architecture: ",
         "A relational graph convolution with separate self, upstream and downstream weights is applied at every time step, "
         "followed by a shared LSTM per node (Section III-D)."]
T[32] = ["• Concurrent Node-Level Forecasts: ",
         "A single forward pass gives the four echelon forecasts, from which the mean risk index is derived."]
T[33] = ["• Residual Persistence Anchoring: ",
         "The head predicts the change from the current state, so the model starts from the persistence forecast [18]."]
T[34] = ["• Evaluation Against Persistence: ",
         "Five-seed results with standard deviations against Naive Persistence, Ridge-AR(10) and a graph-free LSTM, reported "
         "separately for shock and calm windows, plus a measured completeness check for Integrated Gradients [8]."]
T[35] = ["D. Problem Statement and Formal Framing"]
T[36] = ["We represent the multi-echelon supply network as a directed graph G = (V, E), where |V| = 4 nodes are the Tier-1 "
         "Supplier (S), Manufacturer (M), Regional Distributor (D) and Final Retailer (R), and E = {(S, M), (M, D), (D, R)} "
         "is the direction of material flow. At each time step t the system observes x_t in R^5: the min–max scaled risk "
         "indices of the four echelons and the concurrent Total Cost. Given a lookback window of L = 10 consecutive "
         "observations, the input is X in R^(L x 5). The task is to learn a mapping f_theta from X to the risk vector "
         "y_hat_(t+H) in R^4 at horizon H = 5 steps (nominally 10 minutes), and, for a chosen target node k, to compute an "
         "attribution matrix Attr in R^(L x 5) that measures the local sensitivity of the forecast of node k to each past input."]
T[37] = ["II. LITERATURE SURVEY"]
T[39] = ["Ripple-effect research has been studied through analytical and simulation lenses. Ivanov, Dolgui and Sokolov [1] "
         "analysed how digital technology and Industry 4.0 affect ripple-effect and supply chain risk analytics. Dolgui, "
         "Ivanov and Sokolov [7] reviewed quantitative ripple-effect research, separating it from the bullwhip effect and "
         "grouping the approaches into optimisation, simulation, control-theoretic and reliability methods. Simchi-Levi et "
         "al. [2] describe Time-to-Survive (TTS) and Time-to-Recover (TTR) as measures of how long a network can keep "
         "operating, and how long recovery takes, when a node fails."]
T[41] = ["To handle high-frequency telemetry, researchers adopted recurrent networks. Hochreiter and Schmidhuber [4] introduced "
         "the Long Short-Term Memory (LSTM) network, which mitigates vanishing gradients over long sequences. A plain LSTM "
         "over the stacked input does not encode which facility supplies which. Scarselli et al. [16] introduced the graph "
         "neural network model, and Kipf and Welling [6] simplified it to the Graph Convolutional Network (GCN) for "
         "semi-supervised node classification. Yu, Yin and Zhu [5] combined spatial graph convolutions with temporal "
         "convolutions for traffic forecasting (STGCN), and Li et al. [17] modelled traffic as diffusion on a directed graph "
         "inside a recurrent network. Wu et al. [9] survey graph neural networks, and Velickovic et al. [13] introduced "
         "graph attention networks."]
T[42] = ["Farzhana and Dev Harris [10] fuse a graph convolution with an LSTM to predict supply chain risk on the same "
         "dataset used here. We re-implement that fusion as a baseline (global mean pooling of the graph output to a scalar "
         "risk index) and extend it to node-level forecasting; the pooling and layer sizes of our re-implementation are our "
         "interpretation of the published description, not a verbatim specification."]
T[43] = ["C. Explainability"]
T[44] = ["Attribution methods require care. Adebayo et al. [15] showed that several saliency methods can produce maps that are "
         "insensitive to the model's parameters or to the data labels, which motivates sanity checks for any attribution. "
         "Sundararajan, Taly and Yan [8] proposed Integrated Gradients (IG), which accumulates gradients along a straight "
         "path from a baseline and satisfies Sensitivity and Implementation Invariance; completeness (the attributions sum "
         "to the difference in model output) follows. Lundberg and Lee [19] unify additive feature-attribution methods, an "
         "alternative that we do not use. We apply IG to the directed graph model and report its completeness gap and a "
         "deletion test."]
# Table I (rows of 5 cells: ref, year, method, scope, limitation)
tI = [("[7]", "2018", "Literature review (optimisation, simulation, control, reliability)", "Ripple-effect research",
       "Review; no learned telemetry forecaster"),
      ("[2]", "2014", "Time-to-Survive / Time-to-Recover analysis", "Supplier mapping",
       "Analytical; not designed for minute-level telemetry"),
      ("[10]", "2025", "Hybrid GNN-LSTM", "Mendeley SCRM telemetry",
       "Re-implemented here with a scalar risk output"),
      ("[5]", "2018", "Spatio-temporal graph convolution", "Road sensor networks",
       "Traffic forecasting; not evaluated on supply chains"),
      ("[8]", "2017", "Integrated Gradients", "Image, text and other models",
       "Needs a scalar target and a chosen baseline"),
      ("[17]", "2018", "Diffusion convolutional recurrent network", "Traffic forecasting",
       "Directed diffusion on road graphs; not applied to supply chains")]
for i, row in enumerate(tI):
    for j, c in enumerate(row):
        T[51 + 5 * i + j] = [c]
T[81] = ["D. Research Gap"]
T[82] = ["The work reviewed above leaves three gaps that this paper addresses. First, the hybrid GNN-LSTM for this dataset [10] "
         "outputs one aggregated risk index, so the affected echelon is not identified. Second, spatio-temporal graph models "
         "such as [5] are built for symmetric road networks, while material flow in a supply chain is directed; we therefore "
         "compare a symmetric (Kipf–Welling [6]) and a directed variant. Third, forecasts should be reported against "
         "persistence, which is strong at short horizons. We do not claim that attribution identifies a root cause."]
T[85] = ["SupplyGuard has five stages, shown in Fig. 1 and Fig. 2: cleaning and windowing of the telemetry, two graph-convolution "
         "layers applied at every time step, a shared LSTM per node, a residual head that forecasts the four risk indices five "
         "steps (nominally ten minutes) ahead, and post hoc Integrated Gradients attribution."]
T[87] = ["Fig. 1. SupplyGuard model: two relational graph-convolution layers with self, upstream and downstream messages, a shared "
         "per-node LSTM, a residual head, and the Integrated Gradients module."]
T[89] = ["Fig. 2. Processing pipeline: cleaning, chronological split with train-only scaling, five-seed training, evaluation "
         "against baselines, Integrated Gradients, and the Streamlit dashboard."]
T[91] = ["Figure 3 shows the module structure of the implementation. src/dataset.py builds leakage-safe windows (chronological "
         "split; scaler fitted on the training partition only), src/graph_builder.py builds the symmetric and directed "
         "adjacency matrices, and src/models/graph_layers.py implements the graph convolution natively in PyTorch (no PyTorch "
         "Geometric). src/models/st_gcn_lstm.py defines the proposed model and the neural baselines, training/ trains and "
         "evaluates them, src/explainability.py implements Integrated Gradients, and a Streamlit application reads the saved "
         "outputs."]
T[93] = ["Fig. 3. Module dependencies of the implementation (src/, training/, app/)."]
T[95] = [f"We use the Mendeley time-series dataset for risk assessment in supply chain networks [3] (file "
         f"SCRM_timeSeries_2018_train.csv). It has {rl['n_raw']:,} rows between 28 Jan 2015 and 19 Dec 2018, with records for "
         f"2015, 2016 and 2018 only. Each row holds a timestamp, the risk index of each of the four echelons and Total Cost, "
         f"nominally every two minutes. Cleaning removes {rl['n_duplicates_dropped']:,} duplicate timestamps, forward-fills "
         f"gaps of at most five steps, drops {rl['n_nans_dropped']:,} rows that remain incomplete and "
         f"{rl['n_short_seg_rows_dropped']:,} rows in segments shorter than 15 steps, and splits sequences wherever the time "
         f"gap exceeds six minutes. This leaves {rl['n_clean_rows_retained']:,} rows ({rl['pct_retained']:.1f}%). The cleaned rows "
         f"are split chronologically 80/10/10 ({pr[0]:,} / {pr[1]:,} / {pr[2]:,} rows). A window of L = 10 rows predicts the "
         f"risk indices five rows later, and windows never cross a segment boundary, so every target lies inside its own "
         f"partition; this gives {w_tr:,} training, {w_va:,} validation and {w_te:,} test windows, with the test windows "
         f"spanning 17 Sep to 19 Dec 2018. Because windows are indexed by rows, the horizon is five rows; its median duration "
         f"is 10.0 minutes and its mean 10.8 minutes. The min–max scaler is fitted on the training partition only. Before "
         f"cleaning, 5.1% of Distributor values and 5.5% of Total Cost values are missing."]
T[96] = ["TABLE II", "MENDELEY SCRM TELEMETRY CHANNELS AND TRAINING-PARTITION STATISTICS (RAW UNITS)"]
chan = [("Tier-1 Supplier", "Node S", "Risk index RI_Supplier1", "RI_Supplier1"),
        ("Manufacturer", "Node M", "Risk index RI_Manufacturer1", "RI_Manufacturer1"),
        ("Regional Distributor", "Node D", "Risk index RI_Distributor1", "RI_Distributor1"),
        ("Final Retailer", "Node R", "Risk index RI_Retailer1", "RI_Retailer1"),
        ("Logistics Network", "Edge Set", "Total_Cost", "Total_Cost")]
for i, (a, b, c, key) in enumerate(chan):
    base = 102 + 5 * i
    T[base], T[base + 1], T[base + 2] = [a], [b], [c]
    T[base + 3], T[base + 4] = [f"{t2[key][0]:.3f}"], [f"{t2[key][1]:.3f}"]
T[128] = ["1) Relational graph convolution: Let A in {0,1}^(4x4) with A_(i,j) = 1 if facility i ships to facility j (edges S→M, "
          "M→D, D→R). We define A_down = rownorm(A^T), which averages messages from upstream parents, and A_up = rownorm(A), "
          "which averages messages from downstream children. At each time step the node features H^(0) in R^(4x2) hold each "
          "node's risk index and the Total Cost. Each layer computes"]
T[129] = ["H^(l+1) = ReLU( H^(l) W_self^(l) + A_down H^(l) W_down^(l) + A_up H^(l) W_up^(l) + b^(l) )", " (1)"]
T[130] = ["where W_self, W_down, W_up in R^(F_in x F_out) are learnable, b is a bias and F_out = 32. Two layers are stacked so "
          "that information can travel two hops (S can influence D). A learnable echelon embedding is added to the output, "
          "H^(2) in R^(4x32). The symmetric variant uses ReLU(A_hat H W + b) with A_hat = D~^(-1/2)(A_sym + I)D~^(-1/2) [6]."]
T[131] = ["2) Recurrent temporal memory: For each node the L = 10 embeddings form a sequence in R^(10x32). A two-layer LSTM "
          "(hidden size 64, dropout 0.2) with weights shared across the four nodes maps each node's sequence to its final "
          "hidden state:"]
T[132] = ["h_t^(i) = LSTM( H^(2)_(tau,i) : tau = t-L+1, ..., t )", " (2)"]
T[133] = ["h_t^(i) in R^64. Total Cost enters through the node features; it is not concatenated again after the graph layers."]
T[134] = ["3) Residual persistence head: A perceptron (64→32→1, ReLU, dropout 0.2) maps h_t^(i) to a correction, and the forecast "
          "adds it to the current value [18]:"]
T[135] = ["y_hat_(t+H)^(i) = x_t^(i) + g( h_t^(i) )", " (3)"]
T[136] = ["With a zero correction the model reproduces persistence, so any gain must come from the learned correction."]
T[137] = ["4) Training objective: The four echelons are predicted jointly, a form of multi-task learning [14], by minimising the "
          "mean squared error. No explicit weight decay is used; early stopping on validation loss is the regulariser."]
T[138] = ["L(theta) = (1 / |V|) SUM_(i=1..|V|) ( y_hat_(t+H)^(i) - y_(t+H)^(i) )^2", " (4)"]
T[139] = ["5) Integrated Gradients: For a window X and a target node k, the scalar output is f_k(X) = y_hat_(t+H)^(k). With "
          "baseline X^(0) (the training-mean feature vector repeated over the window) and M = 64 steps, the attribution is "
          "given below. Completeness requires SUM Attr = f_k(X) - f_k(X^(0)); we report the residual. Attributions describe "
          "local sensitivity along this path and are not causal effects."]
T[140] = ["Attr(X) = (X - X^(0)) ⊙ (1 / M) SUM_(m=1..M) ∂f_k / ∂X |_(X^(0) + (m/M)(X - X^(0)))", " (5)"]
alg = ["Algorithm 1: ", "SupplyGuard Forecasting & Attribution"]
T[141] = alg
T[142] = ["Input: ", "window X in R^(Lx5) (columns 1–4 risk indices, column 5 Total Cost), L = 10, H = 5; A_down, A_up; target node k; M = 64; baseline X^(0)"]
T[143] = ["Output: ", "forecast y_hat_(t+H) in R^4; attribution Attr in R^(Lx5); completeness gap"]
steps = ["for tau = t-L+1 to t do",
         "    S_tau <- per-node features [x_tau[i], x_tau[5]] in R^(4x2)",
         "    H1 <- ReLU( S_tau W_self0 + A_down S_tau W_down0 + A_up S_tau W_up0 + b0 ) in R^(4x32)",
         "    H2 <- ReLU( H1 W_self1 + A_down H1 W_down1 + A_up H1 W_up1 + b1 ) + E in R^(4x32)   (E: echelon embedding)",
         "end for",
         "for each node i do   // LSTM weights shared across nodes",
         "    h(i) <- LSTM( H2_(t-L+1,i), ..., H2_(t,i) ) in R^64",
         "    y_hat(i)_(t+H) <- x_t[i] + g( h(i) )",
         "end for",
         "// Post hoc attribution for target node k",
         "X^(0) <- training-mean window",
         "for m = 1 to M do",
         "    X^(m) <- X^(0) + (m/M)(X - X^(0))",
         "    G^(m) <- ∂ y_hat(k)_(t+H) / ∂ X^(m)",
         "end for",
         "Attr <- (X - X^(0)) ⊙ (1/M) SUM_m G^(m)",
         "gap <- | SUM Attr - ( y_hat(k)(X) - y_hat(k)(X^(0)) ) |",
         "return y_hat_(t+H), Attr, gap"]
for i, s in enumerate(steps):
    T[144 + i] = [f"{i + 1}: ", s]
T[163] = ["The model is implemented in PyTorch; the graph layers are written natively (no PyTorch Geometric). The four neural "
          "models (LSTM, ST-GCN symmetric, ST-GCN directed and the scalar paper-style hybrid) were trained with five seeds "
          "(42–46) in a Google Colab session on an NVIDIA A100 GPU (40 GB), batch size 256, with Adam [20] (learning rate "
          "1e-3, no weight decay), gradient-norm clipping at 1.0, at most 50 epochs and early stopping with patience 10 on "
          "validation loss. Training took 140–638 s per run. Reported metrics are means ± sample standard deviation over "
          "the five seeds; Persistence and Ridge-AR(10) are deterministic. Inference and attribution timings in Tables III "
          "and VI were measured on a CPU (PyTorch 2.12, four threads) and vary from run to run."]
T[164] = ["TABLE III", "HYPERPARAMETERS AND MEASURED TIMINGS"]
tIII = [("Lookback Window (L)", "10 steps (20 minutes nominal)", "Window length for every model"),
        ("Lookahead Horizon (H)", "5 rows (nominal 10 minutes)", "Median 10.0 min, mean 10.8 min in the cleaned data"),
        ("Graph Layers", "2 layers, 32 channels", "Two hops let the supplier reach the distributor"),
        ("LSTM (ST-GCN models)", "2 layers, hidden 64, dropout 0.2", "Weights shared across the four nodes"),
        ("Adam Learning Rate / Weight Decay", "1e-3 / none", "Early stopping, patience 10"),
        ("IG Steps (M)", "64", "Completeness gap reported in Table VI"),
        ("Inference Latency (CPU, batch 1)", f"about {lat_inf:.1f} ms", "Measured; varies between runs")]
for i, (a, b, c) in enumerate(tIII):
    T[168 + 3 * i], T[169 + 3 * i], T[170 + 3 * i] = [a], [b], [c]
T[190] = [f"For continuous tracking we report MSE, MAE, RMSE and R² on the mean of the four echelon forecasts (derived 4-node "
          f"mean), plus per-node MSE and the skill relative to persistence, 1 − MSE/MSE_persistence. For alert-style evaluation, "
          f"each node's value is assigned to a Low, Medium or High tier using that node's training-set terciles, and accuracy "
          f"and macro-F1 are computed over all node-window pairs. The test tiers are not balanced (Low {cls[0]:,}, Medium "
          f"{cls[1]:,}, High {cls[2]:,} pairs), and for the Supplier the two thresholds differ by only 0.003 in scaled units. "
          f"We also report error separately on shock windows (the top 5% of windows by mean |y(t+H) − y(t)|; "
          f"{sh['n_windows']:,} windows) and on the remaining calm windows."]
T[193] = ["We compare against: (1) Naive Persistence, y_hat_(t+H) = x_t; (2) Ridge-AR(10), ridge regression [11] on the 50 flattened "
          "lookback features with the penalty tuned on the validation set (alpha = 10); (3) a graph-free LSTM, a two-layer "
          "LSTM over the five input features with a joint four-output head and the same residual connection, trained for "
          "five seeds; and (4) a scalar paper-style GCN–LSTM hybrid re-implementing [10], which predicts only the mean risk "
          "index and is therefore not in Table IV. The LSTM and the ST-GCN differ in more than the graph (a joint head versus "
          "a shared per-node LSTM), so the comparison is not a pure test of message passing."]
T[195] = [f"Table IV reports test results on {w_te:,} windows at H = 5. The directed model has the lowest MSE "
          f"({ov.loc['st_gcn_lstm_dir'].MSE_mean * 1e4:.2f} × 10^-4 ± {ov.loc['st_gcn_lstm_dir'].MSE_std * 1e4:.2f} × 10^-4), "
          f"{imp_dir:.1f}% ± {imp_dir_std:.1f}% below Naive Persistence and {rel_ld:.1f}% below the LSTM "
          f"({ov.loc['lstm'].MSE_mean * 1e4:.2f} × 10^-4). The gap to the LSTM ({d_ld * 1e4:.2f} × 10^-4) exceeds the sum of "
          f"the two standard deviations ({thr * 1e4:.2f} × 10^-4), the criterion we use for claiming an improvement, but the "
          f"LSTM has a slightly lower MAE and the architectures differ, so we do not attribute the gap to graph structure "
          f"alone. The symmetric variant is worse and less stable ({ov.loc['st_gcn_lstm_sym'].MSE_mean * 1e4:.2f} × 10^-4 ± "
          f"{ov.loc['st_gcn_lstm_sym'].MSE_std * 1e4:.2f} × 10^-4); two of its five seeds stopped at the persistence solution "
          f"during training. Ridge-AR(10) improves on persistence by {imp_ridge:.1f}%. On three-tier severity, Naive "
          f"Persistence is the most accurate ({sv.loc['persistence'].accuracy_mean * 100:.1f}% accuracy, "
          f"{sv.loc['persistence'].macro_F1_mean * 100:.1f}% macro-F1); no learned model exceeds it and the directed model is "
          f"lower ({sv.loc['st_gcn_lstm_dir'].accuracy_mean * 100:.1f}% ± {sv.loc['st_gcn_lstm_dir'].accuracy_std * 100:.1f}%)."]
T[196] = ["TABLE IV", f"TEST PERFORMANCE ON {w_te:,} WINDOWS (H = 5, MEAN OF FOUR NODES; MEAN ± STD OVER 5 SEEDS)"]
T[198] = ["MSE (10^-4)"]
iv_rows = [("Naive Persistence", "persistence"), ("Ridge-AR(10)", "ar10_ridge"), ("Graph-free LSTM", "lstm"),
           ("ST-GCN-LSTM (symmetric)", "st_gcn_lstm_sym")]
for i, (nm, key) in enumerate(iv_rows):
    base = 204 + 7 * i
    T[base] = [nm]
    for j, v in enumerate(row_iv(key)):
        T[base + 1 + j] = [v]
T[233] = ["Fig. 4. Per-echelon test MSE (log scale, mean ± std over five seeds); percentages are the directed model's MSE "
          "reduction relative to persistence."]
T[235] = [f"Figure 5 shows 80 consecutive test windows around the largest Retailer shock in the test set. Persistence repeats the "
          f"current value and therefore lags an abrupt change by about the forecast horizon. The directed model (five-seed "
          f"mean) reacts earlier than persistence in the Retailer series, but it does not reproduce the step itself, and "
          f"for the Distributor it produces a short overshoot at the onset. This is a single selected window, shown for "
          f"illustration. Across the {sh['n_windows']:,} shock windows the per-node MSE averaged over the four echelons is "
          f"{sh['mse_persistence_shock']:.4f} for persistence and {shock_dir[0]:.4f}–{shock_dir[1]:.4f} for the directed model "
          f"(five seeds), about half; on the remaining calm windows persistence is better "
          f"({sh['mse_persistence_calm'] * 1e4:.1f} × 10^-4 versus {calm_dir[0] * 1e4:.1f}–{calm_dir[1] * 1e4:.1f} × 10^-4)."]
T[237] = ["Fig. 5. Truth, Persistence, LSTM and directed ST-GCN-LSTM (five-seed means) for the Retailer and the Distributor around the "
          "largest Retailer shock in the test set."]
T[239] = [f"Figure 6 shows the three-tier confusion matrix of the directed model (seed 42) over {cm.sum():,} node-window pairs. "
          f"Row-normalised recall is {rec[0]:.1f}% (Low), {rec[1]:.1f}% (Medium) and {rec[2]:.1f}% (High), and accuracy is "
          f"{acc_cm:.1f}% for this seed (five-seed mean {sv.loc['st_gcn_lstm_dir'].accuracy_mean * 100:.1f}% ± "
          f"{sv.loc['st_gcn_lstm_dir'].accuracy_std * 100:.1f}%). Errors occur mainly between adjacent tiers: "
          f"{cm[1, 2]:,} Medium cases are predicted High and {cm[2, 1]:,} High cases are predicted Medium."]
T[241] = ["Fig. 6. Three-tier severity confusion matrix of the directed ST-GCN-LSTM (seed 42): counts and row-normalised shares."]
T[242] = ["TABLE V", "PER-NODE TEST MSE AND SKILL RELATIVE TO PERSISTENCE (MEAN ± STD OVER 5 SEEDS)"]
for j, h in enumerate(["Supply Chain Node", "Persistence MSE (10^-4)", "LSTM MSE (10^-4)", "ST-GCN-Dir MSE (10^-4)",
                       "Dir. skill vs persistence (%)"]):
    T[243 + j] = [h]
names5 = ["Tier-1 Supplier (Node S)", "Manufacturer (Node M)", "Regional Distributor (Node D)", "Final Retailer (Node R)",
          "Mean of the four nodes"]
for i in range(5):
    T[248 + 5 * i] = [names5[i]]
    for j in range(4):
        T[249 + 5 * i + j] = [tab5[i][j]]
T[274] = ["A. How Much Does the Model Add Over Persistence?"]
T[275] = [f"Persistence is a strong baseline here: R² is {ov.loc['persistence'].R2_mean:.3f} for the mean risk index and adjacent "
          f"observations are correlated at r = 0.97–0.996. The learned models nonetheless reduce the MSE by "
          f"{imp_lstm:.0f}–{imp_dir:.0f}%. The reduction is not uniform: it is largest for the Retailer "
          f"({float(tab5[3][3]):.0f}%) and the Distributor ({float(tab5[2][3]):.0f}%) and smaller for the Supplier "
          f"({float(tab5[0][3]):.0f}%) and the Manufacturer ({float(tab5[1][3]):.0f}%). It comes from shock windows, "
          f"where the error is about half of persistence's; on calm windows the learned correction roughly doubles the error. "
          f"A practical system might therefore use persistence in calm periods and the learned forecast when a change is "
          f"detected; we have not tested such a hybrid. The graph-free LSTM reaches most of the gain "
          f"({imp_lstm:.1f}% versus {imp_dir:.1f}%). The directed model's MSE is below the LSTM's by more than the sum of the "
          f"two seed standard deviations at every node; the largest relative difference is at the Supplier "
          f"({float(tab5[0][1].split(' ')[0]):.2f} × 10^-4 versus {float(tab5[0][2].split(' ')[0]):.2f} × 10^-4). Because the "
          f"architectures differ in more than the graph, we regard graph structure as promising rather than proven."]
T[276] = ["B. Integrated Gradients Attribution"]
T[277] = [f"Figure 7 summarises attribution shares for each target node, as magnitude-weighted shares of |attribution| over 100 "
          f"evenly spaced test windows and five seeds. In the forecast, a node's own current risk index dominates "
          f"({min(raw_own):.0f}–{max(raw_own):.0f}%), followed by Total Cost ({min(raw_cost):.0f}–{max(raw_cost):.0f}%); "
          f"upstream echelons contribute little (the Supplier contributes {sup_to_dist:.1f}% to the Distributor forecast and "
          f"essentially 0% to the Retailer forecast). This largely reflects the residual connection, which copies each node's "
          f"own current value into its forecast. Attributing only the learned correction shifts weight to Total Cost "
          f"({min(del_cost):.0f}–{max(del_cost):.0f}%); for the Supplier and Retailer targets the correction is exactly zero "
          f"in {inactive['Supplier']:.0f}% and {inactive['Retailer']:.0f}% of windows, so those shares rest on the remaining "
          f"windows. We therefore cannot support a claim that supplier variation drives downstream risk. Attributions measure "
          f"the local sensitivity of the trained model, not causal effects."]
T[279] = ["Fig. 7. Share of |Integrated Gradients attribution| by input (S, M, D, R: risk indices; C: Total Cost) for each target "
          "node, for the forecast (blue) and for the learned correction (orange); 100 test windows × 5 seeds."]
T[280] = ["TABLE VI", "INTEGRATED GRADIENTS FIDELITY CHECKS"]
for j, h in enumerate(["Check", "Observed", "Reference", "Status"]):
    T[281 + j] = [h]
tVI = [("Completeness gap, |SUM Attr − Δf| (median / 95th pct / max; n = 2,000)",
        f"{sci(gap['median'])} / {sci(gap['p95'])} / {sci(gap['max'])}", "Ideal: 0 (M = 64, Riemann sum)",
        "Small median, heavy tail"),
       ("Deletion test: top-attributed vs random input (10 high-correction windows)", f"{deletion_pass} of 10 pass",
        "Top input should matter more than a random input", "Windows selected for large corrections"),
       ("Implementation invariance", "Not tested", "—", "Not verified"),
       ("Attribution latency (CPU, one target, 64 steps)", f"about {lat_ig:.0f} ms", "—", "Measured; varies between runs")]
for i, row in enumerate(tVI):
    for j, c in enumerate(row):
        T[285 + 4 * i + j] = [c]
T[301] = ["C. Operational Use and Cautions"]
T[302] = ["• Early warning on shock windows: ",
          "Because the gain over persistence is concentrated on shock windows, the forecast is most useful as a change "
          "detector, not as a continuous replacement for persistence."]
T[303] = ["• Node-level triage: ",
          "Four concurrent forecasts show which echelon is predicted to deteriorate; attributions show which inputs the model "
          "is sensitive to but do not establish cause."]
T[304] = ["• Alert quality: ",
          f"On three-tier severity, persistence is more accurate than the learned models in our evaluation "
          f"({sv.loc['persistence'].accuracy_mean * 100:.1f}% versus {sv.loc['st_gcn_lstm_dir'].accuracy_mean * 100:.1f}%), "
          f"so alert thresholds should be validated before operational use."]
T[306] = ["We identify the following limitations. (1) Only one linear four-echelon topology (S→M→D→R) is evaluated. (2) The data "
          "are offline records from a single dataset, with no 2017 records. (3) Gradient attributions indicate sensitivity, "
          "not causal effects. (4) The graph-free LSTM differs from the ST-GCN in more than the graph, and a control with the "
          "same architecture and an identity adjacency was not run. (5) Severity tiers are defined by per-node terciles; the "
          "Supplier's tiers are very narrow. (6) The symmetric variant was unstable across seeds. (7) Results use one "
          "chronological split, with test data from Sep–Dec 2018. (8) The horizon is five rows, which is not always exactly "
          "ten minutes. (9) The deletion test was run on ten windows selected for large corrections, not on a random sample."]
T[309] = [f"We presented SupplyGuard, a spatio-temporal model that forecasts the risk indices of four supply chain echelons "
          f"concurrently. On the Mendeley SCRM dataset, the directed model reduced the mean-node MSE to "
          f"{ov.loc['st_gcn_lstm_dir'].MSE_mean * 1e4:.2f} × 10^-4 (R² = {ov.loc['st_gcn_lstm_dir'].R2_mean:.3f}), {imp_dir:.0f}% "
          f"below Naive Persistence and {rel_ld:.0f}% below a graph-free LSTM, with the benefit concentrated on shock windows. "
          f"Persistence, however, achieved higher three-tier severity accuracy, and the ablation is confounded, so we do not "
          f"claim that graph structure alone explains the gain. Integrated Gradients attributions have a median completeness "
          f"gap of {sci(gap['median'])} with a heavy tail and describe the model's local sensitivity rather than causal effects; "
          f"in our results each node's forecast is dominated by its own current value and the Total Cost, not by upstream "
          f"echelons."]
T[311] = ["• Dynamic Mesh Topologies: ", "Extend to multi-supplier networks and test attention-based edge weights."]
T[312] = ["• External Telemetry: ", "Add external signals such as weather or port congestion where data are available."]
T[313] = ["• Controlled Ablation and Hybrid Alerts: ",
          "Run an identity-adjacency control with the same architecture, and test a hybrid that uses persistence in calm "
          "periods and the learned correction on detected shocks."]
T[315] = ["The authors thank their project supervisor, Ms. Chandana, Assistant Professor, Department of Computer Science and "
          "Engineering, Anurag University, Hyderabad, for her guidance throughout this work, and the authors of the Mendeley "
          "time-series dataset [3] for making it publicly available."]
T[317] = ['[1] D. Ivanov, A. Dolgui, and B. Sokolov, "The impact of digital technology and Industry 4.0 on the ripple effect and '
          'supply chain risk analytics," Int. J. Prod. Res., 2018, doi: 10.1080/00207543.2018.1488086.']
T[318] = ['[2] D. Simchi-Levi, W. Schmidt, and Y. Wei, "From superstorms to factory fires: Managing unpredictable supply-chain '
          'disruptions," Harvard Business Review, vol. 92, no. 1–2, pp. 96–101, 2014.']
T[319] = ['[3] H. Banerjee, G. Saparia, V. Ganapathy, P. Garg, and V. M. Shenbagaraman, "Time series dataset for risk assessment in '
          'supply chain networks," Mendeley Data, V2, 2019, doi: 10.17632/gystn6d3r4.2.']
T[323] = ['[7] A. Dolgui, D. Ivanov, and B. Sokolov, "Ripple effect in the supply chain: An analysis and recent literature," Int. J. '
          'Prod. Res., vol. 56, no. 1–2, pp. 414–430, 2018, doi: 10.1080/00207543.2017.1387680.']
T[326] = ['[10] I. Farzhana and L. Dev Harris, "Hybrid GNN-LSTM model for real-time supply chain risk prediction," in Proc. 8th Int. '
          'Conf. Comput. Methodologies Commun. (ICCMC), IEEE, 2025, doi: 10.1109/ICCMC65190.2025.11140739.']
T[327] = ['[11] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," J. Mach. Learn. Res., vol. 12, pp. 2825–2830, 2011.']
T[328] = ['[12] H. Min, "Artificial intelligence in supply chain management: Theory and applications," Int. J. Logist. Res. Appl., '
          'vol. 13, no. 1, pp. 13–39, 2010.']
T[332] = ['[16] F. Scarselli, M. Gori, A. C. Tsoi, M. Hagenbuchner, and G. Monfardini, "The graph neural network model," IEEE Trans. '
          'Neural Netw., vol. 20, no. 1, pp. 61–80, Jan. 2009.']

# paragraphs to delete entirely (none: algorithm lines were remapped to 18 slots)
DELETE = set()

# ------------------------------------------------------------------ apply
z = zipfile.ZipFile(SRC)
x = z.read("word/document.xml").decode("utf-8")
PARA = re.compile(r"<w:p[ >].*?</w:p>", re.S)
WT = re.compile(r"<w:t(?: [^>]*)?>(.*?)</w:t>", re.S)


def set_wt(p, texts):
    it = iter(texts)
    return WT.sub(lambda m: '<w:t xml:space="preserve">' + esc(next(it, "")) + "</w:t>", p)


paras = PARA.findall(x)
assert len(paras) == 337, len(paras)
heading_tpl, bullet_tpl = paras[29], paras[25]      # L30 (heading), L26 (bold label + text bullet)
objectives = [("O1. ", "Forecast the risk indices of the four echelons concurrently, ten minutes (H = 5 steps) ahead, from a ten-step window."),
              ("O2. ", "Represent directed material flow with a graph layer that treats upstream and downstream messages separately."),
              ("O3. ", "Benchmark the model against Naive Persistence, Ridge-AR(10) and a graph-free LSTM under one chronological split and five seeds."),
              ("O4. ", "Attach Integrated Gradients attributions to each forecast and report their completeness gap and a deletion test."),
              ("O5. ", "State where the learned model adds value over persistence and where it does not.")]
counter = {"i": 0}


def sub(m):
    counter["i"] += 1
    L = counter["i"]
    p = m.group(0)
    if L in DELETE:
        return ""
    if L in T:
        p = set_wt(p, T[L])
    if L == 29:   # objectives block inserted after the last motivation bullet
        extra = set_wt(heading_tpl, ["B. Objectives"])
        for lab, txt in objectives:
            extra += set_wt(bullet_tpl, [lab, txt])
        p += extra
    return p


x2 = PARA.sub(sub, x)

# Table IV: add the directed row (copy of the symmetric row), after the symmetric row
tbls = list(re.finditer(r"<w:tbl>.*?</w:tbl>", x2, re.S))
t4 = tbls[5]
rows = list(re.finditer(r"<w:tr[ >].*?</w:tr>", t4.group(0), re.S))
last = rows[-1].group(0)
dir_vals = ["ST-GCN-LSTM (directed) — SupplyGuard"] + row_iv("st_gcn_lstm_dir")
new_row = set_wt(last, dir_vals)
t4_new = t4.group(0)[:rows[-1].end()] + new_row + t4.group(0)[rows[-1].end():]
x2 = x2[:t4.start()] + t4_new + x2[t4.end():]

# figures: replace media and fix extents
figs = {}
for n in range(1, 8):
    path = f"{ROOT}/docs/paper/figures/fig{n}.png"
    figs[n] = (open(path, "rb").read(), mpimg.imread(path).shape)
CX = 2651760
def fix_extent(m):
    return m.group(0)
drawings = list(re.finditer(r"<w:drawing>.*?</w:drawing>", x2, re.S))
assert len(drawings) == 7
out_x, last_end = [], 0
for n, d in enumerate(drawings, 1):
    h, w = figs[n][1][:2]
    cy = int(round(CX * h / w))
    s = d.group(0)
    s = re.sub(r'(<wp:extent cx=")\d+(" cy=")\d+(")', lambda m: f"{m.group(1)}{CX}{m.group(2)}{cy}{m.group(3)}", s)
    s = re.sub(r'(<a:ext cx=")\d+(" cy=")\d+(")', lambda m: f"{m.group(1)}{CX}{m.group(2)}{cy}{m.group(3)}", s)
    out_x.append(x2[last_end:d.start()])
    out_x.append(s)
    last_end = d.end()
out_x.append(x2[last_end:])
x2 = "".join(out_x)

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zo:
    for item in z.infolist():
        data = z.read(item.filename)
        if item.filename == "word/document.xml":
            data = x2.encode("utf-8")
        m = re.fullmatch(r"word/media/image(\d)\.png", item.filename)
        if m:
            data = figs[int(m.group(1))][0]
        zo.writestr(item, data)
print("written", OUT)
