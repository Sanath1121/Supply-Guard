"""SupplyGuard Operational Monitor View.

Features single replay control, real-time TRI gauge, multi-echelon risk cards,
interactive topology canvas with Integrated Gradients edge weights, and
ground-truth trajectory comparisons.
"""
from typing import Dict, Any, List, Optional
import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from src.config import Config
from app.ui.components import card, status_badge, metric_display, banner, empty_state, section_header, escape
from app.ui.canvas import render_topology_svg
from app.ui.plotly_theme import apply_theme, SERIES_COLORS
from app.utils.artifacts import ArtifactStatus, get_tiers, get_scaler, predict_window
from app.utils.explain_service import compute_all_edge_shares
from app.utils.formatters import compute_tercile_tier, get_tier_color


def render_monitor(
    status: ArtifactStatus,
    windows: List[Dict[str, Any]],
    model: Any
):
    """Render the main operational monitor view."""
    section_header(
        "Operational Risk Replay Monitor",
        "Test-partition historical window replay, multi-echelon vulnerability tracking, and spatiotemporal trajectories."
    )

    if not windows or len(windows) == 0:
        empty_state(
            "Test Partition Data Unavailable",
            "The raw SCRM dataset is required to generate test-partition replay windows. "
            "Run the setup command below to acquire the dataset without leakage.",
            "python setup_and_download.py"
        )
        return

    # 1. Replay Controls (Single source of truth)
    total_windows = len(windows)
    if "window_idx" not in st.session_state or st.session_state["window_idx"] >= total_windows:
        st.session_state["window_idx"] = 0

    cur_idx = st.session_state["window_idx"]

    ctl_col1, ctl_col2, ctl_col3, ctl_col4 = st.columns([1, 8, 1, 2], vertical_alignment="center")

    with ctl_col1:
        if st.button("◀", key="prev_win_btn", disabled=(cur_idx <= 0), use_container_width=True):
            st.session_state["window_idx"] = max(0, cur_idx - 1)
            st.rerun()

    with ctl_col2:
        if total_windows > 1:
            new_idx = st.slider(
                "Replay Control (Window Scrub)",
                min_value=0,
                max_value=total_windows - 1,
                value=cur_idx,
                format="Window #%d",
                label_visibility="collapsed",
                key="slider_win_idx"
            )
            if new_idx != cur_idx:
                st.session_state["window_idx"] = new_idx
                st.rerun()
        else:
            st.caption("Single window loaded (#1)")

    with ctl_col3:
        if st.button("▶", key="next_win_btn", disabled=(cur_idx >= total_windows - 1), use_container_width=True):
            st.session_state["window_idx"] = min(total_windows - 1, cur_idx + 1)
            st.rerun()

    with ctl_col4:
        if st.button("Jump to Peak TRI", key="jump_peak_btn", help="Find and jump to the test window with the highest predicted Total Risk Index", use_container_width=True):
            # Compute TRI over all windows to find peak
            best_idx = 0
            best_val = -1.0
            with st.spinner("Finding peak risk window..."):
                for i_w, w_item in enumerate(windows):
                    p_out = predict_window(model, w_item["sequence"])
                    tri_score = float(np.mean(p_out))
                    if tri_score > best_val:
                        best_val = tri_score
                        best_idx = i_w
            st.session_state["window_idx"] = best_idx
            st.rerun()

    current_window = windows[st.session_state["window_idx"]]
    w_id = current_window["window_id"]
    w_ts = current_window["timestamp"]
    seq = current_window["sequence"] # [10, 5]
    ground_truth = current_window.get("ground_truth", None)

    st.caption(f"**Current Window:** #{w_id + 1} of {total_windows} | **Timestamp (t):** {escape(w_ts)}")
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # 2. Run Forward Inference
    t0 = time.perf_counter()
    preds = predict_window(model, seq)
    inference_ms = (time.perf_counter() - t0) * 1000.0

    tiers = get_tiers()
    p33, p66 = tiers["p33"], tiers["p66"]
    scaler, scaler_fitted = get_scaler()

    # Determine if model is scalar (PaperHybridOverall) or node-level
    is_scalar_model = (status.model_name == "paper_overall" or preds.size == 1)

    # 3. Overall TRI and Echelon Cards Layout
    top_col1, top_col2 = st.columns([1, 3], gap="medium")

    with top_col1:
        # TRI Gauge / Summary Card
        tri_val = float(preds[0] if is_scalar_model else np.mean(preds))
        persistence_tri = float(np.mean(seq[-1, :4]))
        tri_delta = tri_val - persistence_tri
        tri_tier = compute_tercile_tier(tri_val, p33, p66)

        tri_body = f"""
        <div style="text-align: center; padding: 8px 0;">
            <div style="margin-bottom: 8px;">{status_badge(tri_tier, tri_tier)}</div>
            <div style="font-size: 2.25rem; font-weight: 800; color: var(--text);" class="mono-val">{tri_val:.3f}</div>
            <div style="font-size: 0.8125rem; color: {'var(--bad)' if tri_delta > 0 else 'var(--ok)'}; margin-top: 4px;" class="mono-val">
                {'▲ +' if tri_delta > 0 else '▼ '}{tri_delta:.3f} vs persistence
            </div>
            <div style="font-size: 0.75rem; color: var(--text-3); margin-top: 10px;">
                Horizon: <span class="mono-val">t+{Config.HORIZON}</span> (10 min)
            </div>
        </div>
        """
        st.markdown(card("Total Risk Index (TRI)", tri_body, tone=tri_tier), unsafe_allow_html=True)

    with top_col2:
        if is_scalar_model:
            notice_html = """
            <div style="padding: 16px; font-size: 0.875rem; color: var(--text-2);">
                <p><strong>Scalar Architecture Note:</strong> The <em>PaperHybridOverall</em> model predicts a single aggregated 
                Total Risk Index for the entire supply chain directly.</p>
                <p style="color: var(--text-3); margin-bottom: 0;">Node-level echelon breakdown and individual node sensitivity attributions 
                are unavailable for this model architecture. Switch to <strong>ST-GCN-LSTM</strong> or <strong>LSTM Baseline</strong> in the sidebar to inspect individual echelons.</p>
            </div>
            """
            st.markdown(card("Base-Paper Model View", notice_html), unsafe_allow_html=True)
        else:
            # 4 Echelon Cards
            e_cols = st.columns(4, gap="small")
            for i, name in enumerate(Config.NODE_NAMES):
                with e_cols[i]:
                    r_val = float(preds[i])
                    e_tier = compute_tercile_tier(r_val, p33, p66)
                    last_step = float(seq[-1, i])
                    e_delta = r_val - last_step

                    # Unscale raw RI if scaler available
                    if scaler_fitted and hasattr(scaler, "data_range_") and hasattr(scaler, "data_min_"):
                        raw_ri = r_val * scaler.data_range_[i] + scaler.data_min_[i]
                        raw_str = f"{raw_ri:.2f} RI"
                    else:
                        raw_str = "Raw unscaled N/A"

                    c_body = f"""
                    <div>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            {status_badge(e_tier, e_tier)}
                        </div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: var(--text);" class="mono-val">{r_val:.3f}</div>
                        <div style="font-size: 0.75rem; color: var(--text-3); margin-top: 2px;">{raw_str}</div>
                        <div style="font-size: 0.75rem; color: {'var(--bad)' if e_delta > 0 else 'var(--ok)'}; margin-top: 4px;" class="mono-val">
                            {'▲ +' if e_delta > 0 else '▼ '}{e_delta:.3f} vs t
                        </div>
                    </div>
                    """
                    st.markdown(card(name, c_body, tone=e_tier), unsafe_allow_html=True)
                    if st.button(f"Inspect {name}", key=f"inspect_node_{name}", use_container_width=True):
                        st.session_state["selected_node_idx"] = i
                        if "page_why" in st.session_state and st.session_state["page_why"] is not None:
                            st.switch_page(st.session_state["page_why"])
                        else:
                            st.rerun()

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # 4. Tabs: [ Topology ] and [ Window Trajectories ]
    tab_topo, tab_traj = st.tabs(["Cascading Network Topology", "Spatiotemporal Trajectories"])

    with tab_topo:
        if is_scalar_model:
            st.info("Topology edge attribution shares require node-level target predictions and are disabled for scalar models.")
        else:
            st.markdown(
                "<div style='font-size: 0.8125rem; color: var(--text-3); margin-bottom: 8px;'>"
                "Edge thickness reflects real Integrated Gradients attribution share from upstream echelons. Node borders represent severity tiers."
                "</div>",
                unsafe_allow_html=True
            )
            # Compute real edge shares lazily
            model_key = f"{status.model_name}:{status.graph_mode}:{status.seed}"
            with st.spinner("Calculating attribution edge shares..."):
                edge_shares, xai_err = compute_all_edge_shares(model_key, seq)

            node_risks_dict = {Config.NODE_NAMES[i]: float(preds[i]) for i in range(4)}
            svg_html = render_topology_svg(node_risks_dict, tiers, edge_shares, xai_error=xai_err)
            st.markdown(svg_html, unsafe_allow_html=True)

    with tab_traj:
        # Build Trajectory Plotly Chart with neutral series colors
        fig = go.Figure()

        time_steps = [f"t-{Config.SEQ_LEN - 1 - t}" if (Config.SEQ_LEN - 1 - t) > 0 else "t" for t in range(Config.SEQ_LEN)]
        forecast_step = f"t+{Config.HORIZON}"
        all_steps = time_steps + [forecast_step]

        # Lookback sequence traces
        for i, name in enumerate(Config.NODE_NAMES):
            color = SERIES_COLORS.get(name, "#94A3B8")
            y_hist = seq[:, i].tolist()
            y_pred = preds[i] if not is_scalar_model else tri_val

            # Historical line
            fig.add_trace(go.Scatter(
                x=time_steps,
                y=y_hist,
                mode="lines+markers",
                name=f"{name} (Observed)",
                line=dict(color=color, width=2),
                marker=dict(size=4),
                hovertemplate=f"<b>{name}</b><br>Step: %{{x}}<br>Scaled: %{{y:.3f}}<extra></extra>"
            ))

            # Forecast point at t+H
            fig.add_trace(go.Scatter(
                x=[time_steps[-1], forecast_step],
                y=[y_hist[-1], y_pred],
                mode="lines+markers",
                name=f"{name} (Forecast)",
                line=dict(color=color, width=2, dash="dot"),
                marker=dict(size=8, symbol="diamond"),
                hovertemplate=f"<b>{name} Forecast</b><br>Step: %{{x}}<br>Score: %{{y:.3f}}<extra></extra>"
            ))

            # Ground truth marker if available
            if ground_truth is not None and len(ground_truth) > i:
                gt_val = float(ground_truth[i])
                err = abs(y_pred - gt_val)
                fig.add_trace(go.Scatter(
                    x=[forecast_step],
                    y=[gt_val],
                    mode="markers",
                    name=f"{name} (Ground Truth)",
                    marker=dict(size=9, symbol="circle-open", line=dict(color=color, width=2)),
                    hovertemplate=f"<b>{name} Actual</b><br>Actual: {gt_val:.3f}<br>Error: {err:.3f}<extra></extra>"
                ))

        # Total Cost line
        fig.add_trace(go.Scatter(
            x=time_steps,
            y=seq[:, 4].tolist(),
            mode="lines",
            name="Total Cost",
            line=dict(color=SERIES_COLORS["Total Cost"], width=1.5, dash="dash"),
            hovertemplate="<b>Total Cost</b><br>Step: %{x}<br>Scaled: %{y:.3f}<extra></extra>"
        ))

        # Layout adjustments
        fig.update_layout(
            title="Spatiotemporal Lookback Window (t-9 .. t) & Forecast (t+5)",
            xaxis_title="Timeline",
            yaxis_title="Standardized Risk Index",
            hovermode="x unified"
        )
        apply_theme(fig, height=380)
        st.plotly_chart(fig, use_container_width=True)

        if ground_truth is not None:
            st.caption("ℹ️ Dotted open circles denote untouched ground-truth values at $t+5$ for error evaluation.")

    # Performance diagnostics expander
    with st.expander("Diagnostics & Latency"):
        st.caption(f"Inference latency: {inference_ms:.2f} ms | Window ID: #{w_id} | Model: {status.model_name}")
