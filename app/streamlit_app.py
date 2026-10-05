"""SupplyGuard: Spatiotemporal Multi-Echelon Supply Chain Risk Alert System.

Main application entry point. Implements a thin orchestrator with:
- Accessible restrained dark analytics styling
- st.navigation multi-page information architecture
- Universal top Status Strip reporting real artifact status
- Honest inference without fabricated telemetry
"""
import os
import sys
import streamlit as st
import torch

# Ensure repository root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import Config
from app.utils.artifacts import (
    ArtifactStatus,
    get_model,
    get_scaler,
    get_tiers,
    get_test_windows,
)
from app.ui.components import status_strip, clean_html
from app.views.overview import render_overview
from app.views.monitor import render_monitor
from app.views.why import render_why
from app.views.benchmarks import render_benchmarks
from app.views.export import render_export
from app.views.sandbox import render_sandbox


def load_custom_css():
    """Inject restrained dark analytics stylesheet."""
    css_path = os.path.join(os.path.dirname(__file__), "styles", "custom_theme.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state defaults."""
    if "window_idx" not in st.session_state:
        st.session_state["window_idx"] = 0
    if "selected_node_idx" not in st.session_state:
        st.session_state["selected_node_idx"] = 1  # Default to Manufacturer


def main():
    st.set_page_config(
        page_title="SupplyGuard Risk Intelligence",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    load_custom_css()
    init_session_state()

    # 1. Sidebar Brand & Controls
    with st.sidebar:
        st.html(clean_html("""
        <div style="padding: 4px 0 14px 0; border-bottom: 1px solid var(--border); margin-bottom: 14px;">
            <div style="font-size: 1.0625rem; font-weight: 700; color: var(--text); letter-spacing: 0.04em;">SUPPLYGUARD</div>
            <div style="font-size: 0.75rem; color: var(--text-3); margin-top: 2px;">Spatiotemporal Risk Intelligence</div>
        </div>
        """))

        # Reduced motion setting
        reduce_motion = st.checkbox("Reduce motion", value=False, help="Disable CSS animations for accessibility")
        if reduce_motion:
            st.markdown("<style>body, body * { animation: none !important; transition: none !important; }</style>", unsafe_allow_html=True)

        with st.expander("Model Configuration", expanded=True):
            model_name = st.selectbox(
                "Architecture",
                ["st_gcn_lstm", "lstm", "paper_overall"],
                index=0,
                format_func=lambda x: {
                    "st_gcn_lstm": "ST-GCN-LSTM (Spatiotemporal)",
                    "lstm": "LSTM Baseline (Graph-Free)",
                    "paper_overall": "Paper Hybrid (Overall TRI)"
                }.get(x, x),
                key="model_name_select"
            )

            graph_mode = st.selectbox(
                "Graph Topology",
                ["directed", "symmetric"],
                index=0,
                disabled=(model_name != "st_gcn_lstm"),
                help="Directed: S->M->D->R asymmetric weights. Symmetric: Undirected Kipf normalization.",
                key="graph_mode_select"
            )

            seed = st.selectbox(
                "Random Seed",
                Config.SEEDS,
                index=0,
                key="seed_select"
            )

        # Live Measured Hardware Telemetry
        device_str = "CUDA (" + torch.cuda.get_device_name(0) + ")" if torch.cuda.is_available() else "CPU"
        st.html(clean_html(f"""
        <div style="font-size: 0.75rem; color: var(--text-3); padding: 8px 10px; background: var(--surface-2); border-radius: var(--r-sm); border: 1px solid var(--border); margin-top: 8px;">
            <div><strong>Device:</strong> <span class="mono-val">{device_str}</span></div>
            <div style="margin-top: 2px;"><strong>PyTorch:</strong> <span class="mono-val">{torch.__version__}</span></div>
            <div style="margin-top: 2px;"><strong>Horizon:</strong> <span class="mono-val">t+{Config.HORIZON} (10 min)</span></div>
        </div>
        """))

    # 2. Honest Artifact Discovery and Loading
    model, is_loaded, ckpt_path, model_warn = get_model(model_name, graph_mode, seed)
    _, scaler_fitted = get_scaler()
    tiers_info = get_tiers()
    windows, data_source, data_warns = get_test_windows(max_windows=500)

    warnings = []
    if model_warn:
        warnings.append(model_warn)
    warnings.extend(data_warns)

    status = ArtifactStatus(
        model_loaded=is_loaded,
        checkpoint_path=ckpt_path,
        model_name=model_name,
        graph_mode=graph_mode,
        seed=seed,
        scaler_fitted=scaler_fitted,
        tiers_source=tiers_info["source"],
        data_source=data_source,
        warnings=tuple(warnings)
    )

    # 3. Page Definitions with Wrapped Status Strip
    def view_overview():
        status_strip(status)
        render_overview(status)

    def view_monitor():
        status_strip(status)
        render_monitor(status, windows, model)

    def view_why():
        status_strip(status)
        render_why(status, windows, model)

    def view_benchmarks():
        status_strip(status)
        render_benchmarks()

    def view_export():
        status_strip(status)
        render_export(status, windows, model)

    def view_sandbox():
        status_strip(status)
        render_sandbox(status, model)

    p_overview = st.Page(view_overview, title="Overview", url_path="overview", icon=":material/dashboard:", default=True)
    p_monitor = st.Page(view_monitor, title="Monitor", url_path="monitor", icon=":material/analytics:")
    p_why = st.Page(view_why, title="Why this forecast", url_path="why", icon=":material/psychology:")
    p_benchmarks = st.Page(view_benchmarks, title="Benchmarks", url_path="benchmarks", icon=":material/leaderboard:")
    p_export = st.Page(view_export, title="Export report", url_path="export", icon=":material/description:")

    # Store p_why in session state for cross-page navigation from Monitor
    st.session_state["page_why"] = p_why

    pages = [p_overview, p_monitor, p_why, p_benchmarks, p_export]

    # Optional Sandbox Mode (strictly feature-flagged per Rule R1)
    enable_sandbox = getattr(Config, "ENABLE_SANDBOX", False) or os.environ.get("SG_ENABLE_SANDBOX") == "1"
    if enable_sandbox:
        p_sandbox = st.Page(view_sandbox, title="Sandbox", url_path="sandbox", icon=":material/science:")
        pages.append(p_sandbox)

    # 4. Multi-Page Navigation Runner
    pg = st.navigation(pages)
    pg.run()


if __name__ == "__main__":
    main()
