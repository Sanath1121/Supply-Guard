"""SupplyGuard Operational Monitor View.

Features single replay control deck, real-time TRI cockpit gauge, multi-echelon risk cards,
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

    # Replay Control Deck UI
    st.markdown(f"""
    <div style="background: linear-gradient(180deg, rgba(22, 34, 59, 0.7) 0%, rgba(11, 16, 32, 0.85) 100%); 
                border: 1px solid var(--border); border-radius: var(--r-md); padding: 14px 18px; margin-bottom: 16px;
                box-shadow: var(--shadow-md); backdrop-filter: blur(16px);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <div style="font-size: 0.75rem; font-weight: 700; color: var(--text-3); letter-spacing: 0.08em; text-transform: uppercase;">
                TIMELINE SCRUBBER & REPLAY CONTROLS
            </div>
            <div style="display: flex; gap: 8px;">
                <span class="sg-chip" style="font-size: 0.75rem;">Window #{cur_idx + 1} of {total_windows}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    ctl_col1, ctl_col2, ctl_col3, ctl_col4 = st.columns([1, 8, 1, 2.5], vertical_alignment="center")

    with ctl_col1:
        if st.button("◀", key="prev_win_btn", disabled=(cur_idx <= 0), use_container_width=True, help="Previous window (t-1)"):
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
        if st.button("▶", key="next_win_btn", disabled=(cur_idx >= total_windows - 1), use_container_width=True, help="Next window (t+1)"):
            st.session_state["window_idx"] = min(total_windows - 1, cur_idx + 1)
            st.rerun()

    with ctl_col4:
        if st.button("⚡ Jump to Peak TRI", key="jump_peak_btn", help="Find and jump to the test window with the highest predicted Total Risk Index", use_container_width=True):
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

    st.markdown(f"""
    <div style="display: flex; gap: 10px; align-items: center; margin-top: -6px; margin-bottom: 14px;">
        <span style="font-size: 0.8125rem; color: var(--text-3);">Active Window:</span>
        <span class="sg-chip mono-val" style="color: var(--text);">#{w_id + 1}</span>
        <span style="font-size: 0.8125rem; color: var(--text-3); margin-left: 8px;">Timestamp (t):</span>
        <span class="sg-chip mono-val" style="color: var(--accent);">{escape(w_ts)}</span>
    </div>
    """, unsafe_allow_html=True)

    # 2. Run Forward Inference
    t0 = time.perf_counter()
    preds = predict_window(model, seq)
    inference_ms = (time.perf_counter() - t0) * 1000.0

    tiers = get_tiers()
    p33, p66 = tiers["p33"], tiers["p66"]
    scaler, scaler_fitted = get_scaler()

    # Determine if model is scalar (PaperHybridOverall) or node-level
    is_scalar_model = (status.model_name == "paper_overall" or preds.size == 1)

    # 3. Overall TRI Cockpit Instrument and Echelon Cards Layout
    top_col1, top_col2 = st.columns([1.1, 2.9], gap="medium")

    with top_col1:
        # High-Fidelity TRI Cockpit Speedometer Card
        tri_val = float(preds[0] if is_scalar_model else np.mean(preds))
        persistence_tri = float(np.mean(seq[-1, :4]))
        tri_delta = tri_val - persistence_tri
        tri_tier = compute_tercile_tier(tri_val, p33, p66)
        tier_color = get_tier_color(tri_tier)

        # SVG Speedometer calculation (R=75, Arc=235.6)
        clamped_tri = min(1.0, max(0.0, tri_val))
        arc_offset = 235.6 * (1.0 - clamped_tri)

        tri_body = f"""
        <div style="text-align: center; padding: 2px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-size: 0.6875rem; font-weight: 700; color: var(--text-3); letter-spacing: 0.08em; text-transform: uppercase;">
                    COMPOSITE RISK INDEX
                </span>
                {status_badge(tri_tier, tri_tier)}
            </div>

            <!-- Glowing Speedometer Arc Gauge -->
            <svg viewBox="0 0 200 115" width="100%" height="auto" style="display:block; margin: 4px auto;">
                <defs>
                    <filter id="arcGlow" x="-20%" y="-20%" width="140%" height="140%">
                        <feGaussianBlur stdDeviation="3" result="blur" />
                        <feComposite in="SourceGraphic" in2="blur" operator="over" />
                    </filter>
                </defs>
                <!-- Background Arc -->
                <path d="M 25 105 A 75 75 0 0 1 175 105" fill="none" stroke="rgba(148, 163, 184, 0.16)" stroke-width="12" stroke-linecap="round" />
                <!-- Active Arc with Glow -->
                <path d="M 25 105 A 75 75 0 0 1 175 105" fill="none" stroke="{tier_color}" stroke-width="12" stroke-linecap="round"
                      stroke-dasharray="235.6" stroke-dashoffset="{arc_offset:.1f}" filter="url(#arcGlow)" />
                <!-- Center Numeric Readout -->
                <text x="100" y="82" fill="#FFFFFF" font-size="28" font-family="'JetBrains Mono', monospace" font-weight="800" text-anchor="middle">
                    {tri_val:.3f}
                </text>
                <text x="100" y="100" fill="{tier_color}" font-size="11" font-weight="700" text-anchor="middle" letter-spacing="0.08em">
                    {tri_tier.upper()} SEVERITY
                </text>
            </svg>

            <div style="display: flex; justify-content: space-between; font-size: 0.6875rem; color: var(--text-3); margin: 2px 8px 10px 8px;">
                <span>0.0 (Low)</span>
                <span>p33: {p33:.2f}</span>
                <span>p66: {p66:.2f}</span>
                <span>1.0 (High)</span>
            </div>

            <div style="display: flex; justify-content: space-between; align-items: center; padding-top: 8px; border-top: 1px solid var(--border); font-size: 0.8125rem;">
                <span style="color: var(--text-3);">Delta vs Persistence:</span>
                <span class="mono-val" style="font-weight: 700; color: {'var(--bad)' if tri_delta > 0 else 'var(--ok)'};">
                    {'▲ +' if tri_delta > 0 else '▼ '}{tri_delta:.3f}
                </span>
            </div>

            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 5px; font-size: 0.75rem; color: var(--text-3);">
                <span>Forecast Horizon:</span>
                <span class="mono-val" style="color: var(--accent); font-weight: 600;">t+{Config.HORIZON} (10 min)</span>
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
            # 4 Echelon Cards with High-End Layout
            echelon_icons = ["📦", "⚙️", "🚚", "🏪"]
            e_cols = st.columns(4, gap="small")
            for i, name in enumerate(Config.NODE_NAMES):
                with e_cols[i]:
                    r_val = float(preds[i])
                    e_tier = compute_tercile_tier(r_val, p33, p66)
                    last_step = float(seq[-1, i])
                    e_delta = r_val - last_step
                    icon = echelon_icons[i]

                    # Unscale raw RI if scaler available
                    if scaler_fitted and hasattr(scaler, "data_range_") and hasattr(scaler, "data_min_"):
                        raw_ri = r_val * scaler.data_range_[i] + scaler.data_min_[i]
                        raw_str = f"{raw_ri:.2f} RI"
                    else:
                        raw_str = "Raw units N/A"

                    c_body = f"""
                    <div>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <span style="font-size: 1rem;">{icon}</span>
                            {status_badge(e_tier, e_tier)}
                        </div>
                        <div style="font-size: 1.625rem; font-weight: 800; color: #FFFFFF;" class="mono-val">{r_val:.3f}</div>
                        <div style="font-size: 0.75rem; color: var(--text-3); margin-top: 3px;" class="mono-val">{raw_str}</div>
                        <div style="font-size: 0.75rem; color: {'var(--bad)' if e_delta > 0 else 'var(--ok)'}; margin-top: 6px; font-weight: 600;" class="mono-val">
                            {'▲ +' if e_delta > 0 else '▼ '}{e_delta:.3f} vs t
                        </div>
                    </div>
                    """
                    st.markdown(card(name, c_body, tone=e_tier), unsafe_allow_html=True)
                    if st.button(f"Inspect {name} →", key=f"inspect_node_{name}", use_container_width=True):
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
                "<div style='font-size: 0.8125rem; color: var(--text-3); margin-bottom: 10px;'>"
                "Edge thickness reflects real Integrated Gradients attribution share from upstream echelons. Node pods represent severity tiers."
                "</div>",
                unsafe_allow_html=True
            )
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

            # Historical line with smooth spline
            fig.add_trace(go.Scatter(
                x=time_steps,
                y=y_hist,
                mode="lines+markers",
                name=f"{name} (Observed)",
                line=dict(color=color, width=2.5, shape="spline", smoothing=1.2),
                marker=dict(size=5),
                hovertemplate=f"<b>{name}</b><br>Step: %{{x}}<br>Scaled: %{{y:.3f}}<extra></extra>"
            ))

            # Forecast point at t+H
            fig.add_trace(go.Scatter(
                x=[time_steps[-1], forecast_step],
                y=[y_hist[-1], y_pred],
                mode="lines+markers",
                name=f"{name} (Forecast)",
                line=dict(color=color, width=2, dash="dot"),
                marker=dict(size=9, symbol="diamond"),
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
                    marker=dict(size=10, symbol="circle-open", line=dict(color=color, width=2.5)),
                    hovertemplate=f"<b>{name} Actual</b><br>Actual: {gt_val:.3f}<br>Error: {err:.3f}<extra></extra>"
                ))

        # Total Cost line
        fig.add_trace(go.Scatter(
            x=time_steps,
            y=seq[:, 4].tolist(),
            mode="lines",
            name="Total Cost",
            line=dict(color=SERIES_COLORS["Total Cost"], width=1.5, dash="dash", shape="spline", smoothing=1.2),
            hovertemplate="<b>Total Cost</b><br>Step: %{x}<br>Scaled: %{y:.3f}<extra></extra>"
        ))

        # Shaded forecast horizon band
        fig.add_vrect(
            x0=time_steps[-1],
            x1=forecast_step,
            fillcolor="rgba(56, 189, 248, 0.06)",
            layer="below",
            line_width=0,
            annotation_text="FORECAST HORIZON (t+5)",
            annotation_position="top left",
            annotation_font_size=10,
            annotation_font_color="#7DD3FC"
        )

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
