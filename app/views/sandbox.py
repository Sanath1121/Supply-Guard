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
from app.utils.artifacts import ArtifactStatus, get_tiers, get_scaler, predict_window
from app.utils.formatters import compute_tercile_tier


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
                "Scenario C: Downstream Retail Demand Surge"
            ]
        )

        # Build synthetic 10-step sequence based on scenario
        seq = np.zeros((Config.SEQ_LEN, Config.NUM_INPUT_FEATURES), dtype=np.float32)
        base_line = np.linspace(0.2, 0.35, Config.SEQ_LEN)
        for i in range(4):
            seq[:, i] = base_line

        if "Supplier" in preset_name:
            seq[-3:, 0] = [0.65, 0.82, 0.95] # Supplier spike
            seq[:, 4] = np.linspace(0.3, 0.7, Config.SEQ_LEN) # Cost rise
        elif "Manufacturing" in preset_name:
            seq[-3:, 1] = [0.60, 0.78, 0.90] # Manufacturer spike
            seq[:, 4] = np.linspace(0.2, 0.5, Config.SEQ_LEN)
        else:
            seq[-3:, 3] = [0.70, 0.85, 0.92] # Retailer spike
            seq[:, 4] = np.linspace(0.4, 0.8, Config.SEQ_LEN)

        st.caption(f"Simulating lookback sequence under: **{escape(preset_name)}**")

        if st.button("Run Sandbox Simulation", key="run_sim_btn"):
            preds = predict_window(model, seq)
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
                        t = compute_tercile_tier(r, p33, p66)
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
                except Exception as e:
                    st.error(f"Failed to parse CSV: {escape(str(e))}")
