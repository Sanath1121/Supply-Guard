"""SupplyGuard Isolated Experimental Sandbox View.

Feature-gated environment for uploading custom CSV files or simulating preset shock scenarios.
Strictly isolated from main test-partition monitoring with persistent warning banners.
"""
from typing import Dict, Any, List, Optional
import os
import pandas as pd
import numpy as np
import streamlit as st

from src.config import Config
from app.ui.components import card, banner, section_header, status_badge, escape
from app.ui.canvas import render_topology_svg
from app.utils.artifacts import ArtifactStatus, get_tiers, get_scaler, predict_window
from app.utils.formatters import compute_tercile_tier, node_tier
from app.utils.explain_service import compute_all_edge_shares


def render_sandbox(
    status: ArtifactStatus,
    model: Any
):
    """Render the isolated sandbox testing view."""
    # Permanent warning banner required by Rule R1
    st.html(banner(
        "sandbox",
        "EXPERIMENTAL SANDBOX MODE ACTIVE",
        "Data ingested or simulated in this view does NOT belong to the untouched historical test partition. "
        "Outputs generated here are for exploratory scenario analysis and stress-testing only."
    ))

    section_header(
        "Isolated Scenario & Shock Testing Sandbox",
        "Evaluate model response to custom CSV time series or synthetic disruption shocks."
    )

    tiers = get_tiers()
    p33, p66 = tiers["p33"], tiers["p66"]

    tab_preset, tab_upload = st.tabs(["Preset Shock Scenarios", "Custom CSV Ingestion"])

    with tab_preset:
        preset_name = st.selectbox(
            "Select Disruption Scenario",
            [
                "Scenario A: Upstream Tier-1 Supplier Shortage",
                "Scenario B: Midstream Manufacturing Assembly Halt",
                "Scenario C: Downstream Retail Demand Surge",
                "Scenario D: Global Logistics Cost Shock"
            ]
        )

        # Build synthetic 10-step sequence based on scenario with stochastic noise
        seq = np.zeros((Config.SEQ_LEN, Config.NUM_INPUT_FEATURES), dtype=np.float32)
        base_line = np.linspace(0.2, 0.35, Config.SEQ_LEN)
        
        for i in range(4):
            # Inject Gaussian noise into the base sequences
            noise = np.random.normal(0.0, 0.03, Config.SEQ_LEN)
            seq[:, i] = np.clip(base_line + noise, 0.0, 1.0)

        if "Supplier" in preset_name:
            spike = np.array([0.65, 0.82, 0.95]) + np.random.normal(0, 0.02, 3)
            seq[-3:, 0] = np.clip(spike, 0.0, 1.0) # Supplier spike
            seq[:, 4] = np.clip(np.linspace(0.3, 0.7, Config.SEQ_LEN) + np.random.normal(0, 0.03, Config.SEQ_LEN), 0.0, 1.0) # Cost rise
        elif "Manufacturing" in preset_name:
            spike = np.array([0.60, 0.78, 0.90]) + np.random.normal(0, 0.02, 3)
            seq[-3:, 1] = np.clip(spike, 0.0, 1.0) # Manufacturer spike
            seq[:, 4] = np.clip(np.linspace(0.2, 0.5, Config.SEQ_LEN) + np.random.normal(0, 0.03, Config.SEQ_LEN), 0.0, 1.0)
        elif "Retail" in preset_name:
            spike = np.array([0.70, 0.85, 0.92]) + np.random.normal(0, 0.02, 3)
            seq[-3:, 3] = np.clip(spike, 0.0, 1.0) # Retailer spike
            seq[:, 4] = np.clip(np.linspace(0.4, 0.8, Config.SEQ_LEN) + np.random.normal(0, 0.03, Config.SEQ_LEN), 0.0, 1.0)
        else: # Global Logistics Cost Shock
            for i in range(4):
                spike = np.array([0.45, 0.55, 0.65]) + np.random.normal(0, 0.02, 3)
                seq[-3:, i] = np.clip(spike, 0.0, 1.0) # Universal moderate risk escalation
            cost_spike = np.array([0.60, 0.80, 0.95, 1.0]) + np.random.normal(0, 0.01, 4)
            seq[-4:, 4] = np.clip(cost_spike, 0.0, 1.0) # Massive cost spike in recent steps
            # Backfill earlier cost to avoid zeros
            seq[:-4, 4] = np.clip(np.linspace(0.2, 0.4, Config.SEQ_LEN-4) + np.random.normal(0, 0.02, Config.SEQ_LEN-4), 0.0, 1.0)

        st.caption(f"Simulating lookback sequence under: **{escape(preset_name)}**")

        if st.button("Run Sandbox Simulation", key="run_sim_btn"):
            preds = np.atleast_1d(predict_window(model, seq))
            st.markdown("#### Forecasted Scenario Impacts")
            
            if status.model_name == "paper_overall" or preds.size == 1:
                tri_val = float(preds[0])
                t_tier = compute_tercile_tier(tri_val, p33, p66)
                st.html(card("Simulated Total Risk Index", f"<div style='font-size: 1.5rem;' class='mono-val'>{tri_val:.3f} ({t_tier})</div>"))
            else:
                cols = st.columns(4)
                for i, n in enumerate(Config.NODE_NAMES):
                    with cols[i]:
                        r = float(preds[i])
                        t = node_tier(r, i, tiers)
                        st.html(card(n, f"<div style='font-size: 1.25rem;' class='mono-val'>{r:.3f}</div><div style='margin-top:4px;'>{status_badge(t, t)}</div>", tone=t))

    with tab_upload:
        st.markdown("Upload a test CSV sequence (max 20 MB, minimum 10 consecutive timestamps).")
        uploaded = st.file_uploader("Upload CSV", type=["csv"], key="sandbox_csv_uploader")
        
        if uploaded is not None:
            # Enforce 20 MB size limit
            if uploaded.size > 20 * 1024 * 1024:
                st.error("Uploaded file exceeds the 20 MB limit.")
            else:
                try:
                    df = pd.read_csv(uploaded)
                    safe_name = escape(uploaded.name)
                    st.success(f"Loaded CSV `{safe_name}` with {len(df)} rows.")

                    # Check required columns
                    req_cols = Config.NODE_TARGET_COLS + [Config.COST_COL]
                    missing = [c for c in req_cols if c not in df.columns]
                    if missing:
                        st.error(f"Missing required columns: {missing}")
                    elif len(df) < Config.SEQ_LEN:
                        st.error(f"Dataset must have at least {Config.SEQ_LEN} rows for lookback window.")
                    else:
                        st.dataframe(df.head(5), use_container_width=True)
                        st.info("Custom CSV successfully verified for sandbox evaluation.")
                        
                        # NEW: Run inference on custom uploaded CSV
                        if st.button("Run Inference on Custom CSV", key="run_upload_sim_btn"):
                            tail_df = df.tail(Config.SEQ_LEN)
                            seq_data = tail_df[req_cols].values.astype(np.float32)
                            # The model works in scaled units; raw SCRM values (risk ~1-3, cost ~10-300)
                            # are mapped with the scaler fitted on the training partition.
                            scaler, scaler_fitted = get_scaler()
                            if scaler_fitted and (seq_data.min() < 0.0 or seq_data.max() > 1.0):
                                seq_data = scaler.transform(seq_data).astype(np.float32)
                                st.caption("Values outside 0–1 detected: input scaled with the training-partition MinMax scaler.")
                            
                            preds = np.atleast_1d(predict_window(model, seq_data))
                            st.markdown("#### Forecasted Custom Scenario Impacts")
                            
                            if status.model_name == "paper_overall" or preds.size == 1:
                                tri_val = float(preds[0])
                                t_tier = compute_tercile_tier(tri_val, p33, p66)
                                st.html(card("Simulated Total Risk Index", f"<div style='font-size: 1.5rem;' class='mono-val'>{tri_val:.3f} ({t_tier})</div>"))
                            else:
                                cols = st.columns(4)
                                for i, n in enumerate(Config.NODE_NAMES):
                                    with cols[i]:
                                        r = float(preds[i])
                                        t = node_tier(r, i, tiers)
                                        st.html(card(n, f"<div style='font-size: 1.25rem;' class='mono-val'>{r:.3f}</div><div style='margin-top:4px;'>{status_badge(t, t)}</div>", tone=t))

                            # Spatiotemporal Risk Cascade Network Topology
                            st.markdown("---")
                            st.markdown("#### Spatiotemporal Risk Cascade")
                            
                            try:
                                model_key = f"{status.model_name}:{status.graph_mode}:{status.seed}"
                                # Calculate the XAI upstream flow between all nodes
                                with st.spinner("Calculating cascading network topology..."):
                                    edge_shares, xai_err = compute_all_edge_shares(model_key, seq_data)

                                # Map the AI predictions to their node names
                                if status.model_name == "paper_overall" or preds.size == 1:
                                    node_risks_dict = {Config.NODE_NAMES[i]: float(preds[0]) for i in range(4)}
                                else:
                                    node_risks_dict = {Config.NODE_NAMES[i]: float(preds[i]) for i in range(4)}

                                # Generate and render the animated SVG
                                svg_html = render_topology_svg(node_risks_dict, tiers, edge_shares, xai_error=xai_err)
                                import streamlit.components.v1 as components
                                components.html(svg_html, height=280)
                            except Exception as ex:
                                st.error(f"Topology visualization unavailable: {escape(str(ex))}")
                except Exception as e:
                    st.error(f"Failed to parse CSV: {escape(str(e))}")
