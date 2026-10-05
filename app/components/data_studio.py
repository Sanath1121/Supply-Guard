"""Dataset Ingestion & Custom Data Studio Component.

Supports:
1. Verified 650k-row Mendeley Benchmark Replay.
2. Custom User CSV Upload with Column Mapping Wizard and Data Profiler.
3. 4 Ready-Made Crisis Presets for 1-click viva demonstrations.
"""
from typing import Dict, Any, List, Optional
import streamlit as st
import pandas as pd
import numpy as np

from app.utils.data_bridge import generate_preset_scenarios, process_custom_csv
from src.config import Config


def render_data_studio(scaler) -> Optional[Dict[str, Any]]:
    """Render the Dataset Ingestion Studio and return custom data if selected."""
    st.markdown("### 📂 Dataset Ingestion & Custom Scenario Studio")
    st.markdown(
        "SupplyGuard supports both the **verified 650k-row Mendeley Data V2 benchmark** "
        "and **custom user CSV uploads** with automated column mapping, missingness profiling, and sliding window generation."
    )

    tab_preset, tab_upload, tab_bench = st.tabs([
        "⚡ 1-Click Crisis Presets",
        "📤 Custom CSV Upload Wizard",
        "🏛️ Verified Benchmark Dataset"
    ])

    custom_result = None

    # TAB 1: 1-Click Crisis Presets
    with tab_preset:
        st.markdown("#### Test Ready-Made Disruption Scenarios")
        st.caption("Instantly inject synthetic spatiotemporal risk shocks to evaluate how the neural network and explainability engine respond.")

        presets = generate_preset_scenarios()
        cols = st.columns(2)

        for i, (name, data) in enumerate(presets.items()):
            col = cols[i % 2]
            with col:
                st.markdown(f"""
                <div class="sg-card" style="margin-bottom: 12px; height: 180px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <h4 style="color: #F1F5F9; margin: 0;">{name}</h4>
                        <span class="badge-pill badge-{'high' if data['expected_alert'] == 'High' else 'low'}">
                            {data['expected_alert']}
                        </span>
                    </div>
                    <p style="color: #94A3B8; font-size: 0.85rem; line-height: 1.4; margin-bottom: 8px;">
                        {data['description']}
                    </p>
                    <div style="font-size: 0.75rem; color: #64748B;">
                        <strong>Shock Origin:</strong> {data['crisis_node'] if data['crisis_node'] else 'None (Nominal)'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"Load '{name}' into Radar", key=f"btn_preset_{i}", use_container_width=True):
                    custom_result = {
                        "mode": "preset",
                        "name": name,
                        "sequence": data["sequence"],
                        "timestamp": f"Preset Event: {name}"
                    }

    # TAB 2: Custom CSV Upload Wizard
    with tab_upload:
        st.markdown("#### Upload Your Own Supply Chain Time-Series Data")
        st.caption("Upload a `.csv` file with historical risk indices or sensor readings. The wizard will map columns and generate valid sliding windows.")

        uploaded_file = st.file_uploader("Select CSV File", type=["csv"], key="csv_uploader")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"CSV uploaded successfully: {len(df):,} rows and {len(df.columns)} columns detected.")

                st.markdown("##### 1. Column Mapping Wizard")
                st.caption("Map your dataset's columns to SupplyGuard's 5 required spatiotemporal inputs:")

                col_options = list(df.columns)
                cm1, cm2, cm3 = st.columns(3)
                with cm1:
                    col_s = st.selectbox("Supplier Risk Column", col_options, index=0 if len(col_options) > 0 else 0)
                    col_m = st.selectbox("Manufacturer Risk Column", col_options, index=min(1, len(col_options)-1))
                with cm2:
                    col_d = st.selectbox("Distributor Risk Column", col_options, index=min(2, len(col_options)-1))
                    col_r = st.selectbox("Retailer Risk Column", col_options, index=min(3, len(col_options)-1))
                with cm3:
                    col_cost = st.selectbox("Total Cost / Financial Column", col_options, index=min(4, len(col_options)-1))

                mapping = {
                    "supplier": col_s,
                    "manufacturer": col_m,
                    "distributor": col_d,
                    "retailer": col_r,
                    "cost": col_cost
                }

                st.markdown("##### 2. Data Health & Profiling Card")
                windows_arr, profile = process_custom_csv(df, mapping, scaler)

                if windows_arr is not None:
                    p1, p2, p3, p4 = st.columns(4)
                    with p1:
                        st.metric("Raw Rows", f"{profile['raw_rows']:,}")
                    with p2:
                        st.metric("Clean Rows", f"{profile['clean_rows']:,}")
                    with p3:
                        st.metric("Missingness Handled", f"{profile['total_nulls']:,} nulls")
                    with p4:
                        st.metric("Valid 10-Step Windows", f"{profile['num_windows']:,}")

                    if st.button("🚀 Load Custom Dataset into Radar", use_container_width=True):
                        custom_result = {
                            "mode": "custom_csv",
                            "name": uploaded_file.name,
                            "windows": windows_arr,
                            "profile": profile
                        }
                else:
                    st.error(profile.get("error", "Failed to process custom CSV."))
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")

    # TAB 3: Verified Benchmark Dataset
    with tab_bench:
        st.markdown("#### Mendeley Data V2 Benchmark Specification")
        st.markdown("""
        <div class="sg-card">
            <h4 style="color: #06B6D4; margin-top: 0;">Banerjee et al. (2019) Time-Series Dataset</h4>
            <ul style="color: #CBD5E1; font-size: 0.88rem; line-height: 1.6;">
                <li><strong>Citation:</strong> Banerjee, Heerok; Saparia, Grishma; Ganapathy, Velappa; Garg, Priyanshi; Shenbagaraman, V. M. (2019), <em>"Time Series Dataset for Risk Assessment in Supply Chain Networks"</em>, Mendeley Data, V2.</li>
                <li><strong>Dataset Partition:</strong> <code>SCRM_timeSeries_2018_train.csv</code> (649,999 records).</li>
                <li><strong>Temporal Span:</strong> January 28, 2015 &rarr; December 19, 2018.</li>
                <li><strong>Chronological Partitioning:</strong> 80% Train, 10% Validation, 10% Unseen Test (Strict time order, no shuffling).</li>
                <li><strong>Sampling Cadence:</strong> Median 2 minutes between readings; segment-aware windowing prevents cross-gap contamination.</li>
                <li><strong>SHA-256 Checksum:</strong> <code>d2e71ae7f55fa70ef498fecb9b6db0c9fd59688f17f8ad3c27c7576f09e76ff3</code>.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    return custom_result
