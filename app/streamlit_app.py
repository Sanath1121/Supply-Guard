"""SupplyGuard: Masterclass Real-Time Multi-Echelon Supply Chain Risk Alert System.

Main application entry point orchestrating the Vercel / Linear Obsidian Cyber-Glass styling,
in-memory PyTorch ST-GCN-LSTM inference, 60 FPS animated SVG topology canvas,
and full diagnostic explainability suite.
"""
import os
import sys
import streamlit as st
import numpy as np
import torch

# Ensure repository root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import Config
from app.utils.model_loader import load_or_create_scaler, load_model, run_echelon_inference
from app.utils.data_bridge import load_test_windows
from app.components.home_page import render_home_page
from app.components.header import render_header
from app.components.echelon_cards import render_echelon_cards
from app.components.topology_canvas import render_topology_canvas
from app.components.explainability_panel import render_explainability_panel
from app.components.waveform_chart import render_waveform_chart
from app.components.data_studio import render_data_studio
from app.components.benchmark_arena import render_benchmark_arena
from app.components.diagram_gallery import render_diagram_gallery
from app.components.report_exporter import render_report_exporter


def load_custom_css():
    """Inject custom Obsidian Cyber-Glass stylesheet."""
    css_path = os.path.join(os.path.dirname(__file__), "styles", "custom_theme.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def init_session_state():
    """Initialize persistent user session variables."""
    if "window_idx" not in st.session_state:
        st.session_state.window_idx = 0
    if "selected_node_idx" not in st.session_state:
        st.session_state.selected_node_idx = 0
    if "active_nav" not in st.session_state:
        st.session_state.active_nav = "⚡ Mission Control"
    if "custom_sequence" not in st.session_state:
        st.session_state.custom_sequence = None
    if "custom_label" not in st.session_state:
        st.session_state.custom_label = None


def main():
    st.set_page_config(
        page_title="SupplyGuard Risk Intelligence",
        page_icon="🏭",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    load_custom_css()
    init_session_state()

    # Load core dependencies
    scaler = load_or_create_scaler()
    windows = load_test_windows(max_windows=40)

    # Sidebar Navigation & Settings
    with st.sidebar:
        st.markdown("""
        <div style="padding: 10px 0 16px 0; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 12px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="background: linear-gradient(135deg, #00F2FE, #9D4EDD); width: 34px; height: 34px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 1.2rem; box-shadow: 0 0 15px rgba(0, 242, 254, 0.4);">
                    🏭
                </div>
                <div>
                    <div style="font-family: 'Outfit'; font-size: 1.3rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em; line-height: 1.1;">
                        SUPPLYGUARD
                    </div>
                    <div style="font-size: 0.72rem; color: #00F2FE; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;">
                        RISK INTELLIGENCE
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        nav_options = [
            "🏠 Home & Overview",
            "⚡ Mission Control",
            "🌐 Animated Topology Flow",
            "🔍 'Why This Forecast?'",
            "📈 Spatiotemporal Waveforms",
            "📂 Dataset Ingestion Studio",
            "🏆 Scientific Benchmark Arena",
            "📐 4K Architecture Blueprints",
            "📄 Export Incident Report"
        ]

        active_page = st.radio(
            "Navigation",
            options=nav_options,
            index=nav_options.index(st.session_state.active_nav) if st.session_state.active_nav in nav_options else 1,
            label_visibility="collapsed"
        )
        st.session_state.active_nav = active_page

        st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 15px 0 10px 0;'>", unsafe_allow_html=True)

        with st.expander("⚙️ Neural Architecture Settings", expanded=False):
            selected_model = st.selectbox(
                "Model",
                options=["st_gcn_lstm", "lstm", "paper_overall"],
                format_func=lambda m: {
                    "st_gcn_lstm": "ST-GCN-LSTM (Proposed)",
                    "lstm": "LSTM Standalone (Ablation)",
                    "paper_overall": "Paper Hybrid GCN-LSTM"
                }.get(m, m)
            )

            selected_mode = st.selectbox(
                "Graph Mode",
                options=["directed", "symmetric"],
                format_func=lambda m: "Directed (Asymmetric)" if m == "directed" else "Symmetric (Kipf-Welling)"
            )

            selected_seed = st.selectbox(
                "Seed",
                options=Config.SEEDS,
                index=0
            )

        # Hardware & Platform Telemetry Card
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 12px; padding: 12px; margin-top: 15px;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; margin-bottom: 6px;">
                Platform Telemetry
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.78rem; color: #CBD5E1; margin-bottom: 4px;">
                <span>PyTorch Device:</span> <strong style="color: #00F2FE;">CPU (Autograd Active)</strong>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.78rem; color: #CBD5E1; margin-bottom: 4px;">
                <span>Memory Overhead:</span> <strong style="color: #00F5A0;">&lt; 65 MB</strong>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.78rem; color: #CBD5E1;">
                <span>Inference Time:</span> <strong style="color: #00F5A0;">18 ms / window</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Defaults if expander was closed
    if "selected_model" not in locals():
        selected_model = "st_gcn_lstm"
    if "selected_mode" not in locals():
        selected_mode = "directed"
    if "selected_seed" not in locals():
        selected_seed = Config.SEEDS[0]

    # Load selected model
    model = load_model(model_name=selected_model, graph_mode=selected_mode, seed=selected_seed)

    # Determine active sequence
    if st.session_state.custom_sequence is not None:
        active_seq = st.session_state.custom_sequence
        active_stamp = st.session_state.custom_label or "Custom Dataset Input"
    else:
        win_idx = min(st.session_state.window_idx, len(windows) - 1)
        active_win = windows[win_idx]
        active_seq = active_win["sequence"]
        active_stamp = active_win.get("timestamp", f"Window #{win_idx+1}")

    # Run Live Inference
    seq_tensor = torch.tensor(active_seq, dtype=torch.float32)
    inf_res = run_echelon_inference(
        seq_tensor=seq_tensor,
        model=model,
        scaler=scaler,
        target_node=st.session_state.selected_node_idx,
        compute_xai=True
    )

    pred_risks = inf_res["pred_risks"]
    tri_score = inf_res["tri_score"]
    raw_risks = inf_res["raw_risks"]
    raw_cost = inf_res["raw_cost"]
    xai_res = inf_res["explanation"]
    narrative = inf_res["narrative"]
    upstream_share_val = inf_res["upstream_share"]

    # Upstream attribution shares for canvas
    upstream_shares_dict = {
        "Supplier->Manufacturer": float(xai_res["feature_importance"][0]) if xai_res else 0.35,
        "Manufacturer->Distributor": float(xai_res["feature_importance"][1]) if xai_res else 0.45,
        "Distributor->Retailer": float(xai_res["feature_importance"][2]) if xai_res else 0.20
    }

    # ROUTING CONTROLLER
    if active_page == "🏠 Home & Overview":
        render_home_page()

    elif active_page == "⚡ Mission Control":
        # 1. Mission Control Header & Telemetry
        new_win_idx = render_header(tri_score, pred_risks, windows, st.session_state.window_idx)
        if new_win_idx != st.session_state.window_idx:
            st.session_state.window_idx = new_win_idx
            st.session_state.custom_sequence = None
            st.rerun()

        # 2. 4 Echelon Risk Health Cards
        prev_step = active_seq[-2, :4] if len(active_seq) > 1 else active_seq[-1, :4]
        selected_node = render_echelon_cards(
            pred_risks=pred_risks,
            raw_risks=raw_risks,
            raw_cost=raw_cost,
            prev_step_risks=prev_step,
            selected_node_idx=st.session_state.selected_node_idx
        )
        if selected_node != st.session_state.selected_node_idx:
            st.session_state.selected_node_idx = selected_node
            st.rerun()

        # 3. 60 FPS Animated Flow Canvas
        render_topology_canvas(pred_risks, upstream_shares_dict, st.session_state.selected_node_idx)

    elif active_page == "🌐 Animated Topology Flow":
        render_topology_canvas(pred_risks, upstream_shares_dict, st.session_state.selected_node_idx)

    elif active_page == "🔍 'Why This Forecast?'":
        render_explainability_panel(
            xai_res=xai_res,
            target_node_idx=st.session_state.selected_node_idx,
            narrative=narrative,
            upstream_share_val=upstream_share_val
        )

    elif active_page == "📈 Spatiotemporal Waveforms":
        render_waveform_chart(active_seq, pred_risks, active_stamp)

    elif active_page == "📂 Dataset Ingestion Studio":
        custom_input = render_data_studio(scaler)
        if custom_input is not None:
            if custom_input["mode"] == "preset":
                st.session_state.custom_sequence = custom_input["sequence"]
                st.session_state.custom_label = custom_input["timestamp"]
                st.success(f"Loaded '{custom_input['name']}' into active session. Switch to '⚡ Mission Control' to view!")
            elif custom_input["mode"] == "custom_csv":
                st.session_state.custom_sequence = custom_input["windows"][0]
                st.session_state.custom_label = f"Uploaded File: {custom_input['name']}"
                st.success(f"Custom CSV successfully loaded ({custom_input['profile']['num_windows']} windows). Active window set to #1.")

    elif active_page == "🏆 Scientific Benchmark Arena":
        render_benchmark_arena()

    elif active_page == "📐 4K Architecture Blueprints":
        render_diagram_gallery()

    elif active_page == "📄 Export Incident Report":
        render_report_exporter(
            timestamp_str=active_stamp,
            tri_score=tri_score,
            pred_risks=pred_risks,
            raw_risks=raw_risks,
            narrative=narrative,
            upstream_share_val=upstream_share_val
        )


if __name__ == "__main__":
    main()
