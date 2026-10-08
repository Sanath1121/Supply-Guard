"""SupplyGuard Explainability & Sensitivity View (Why This Forecast).

Provides Integrated Gradients attributions, Delta-vs-persistence decomposition,
per-time-step attribution, a completeness check, and a feature deletion test.
Frames all outputs strictly as axiomatic sensitivity attributions.
"""
from typing import Dict, Any, List, Optional
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from src.config import Config
from src.explainability import FEATURE_NAMES, NODE_NAMES, narrate
from app.ui.components import card, status_badge, banner, empty_state, section_header, escape
from app.ui.plotly_theme import apply_theme
from app.utils.artifacts import ArtifactStatus
from app.utils.explain_service import compute_explanation, run_deletion_test


def render_why(
    status: ArtifactStatus,
    windows: List[Dict[str, Any]],
    model: Any
):
    """Render the Explainability view."""
    section_header(
        "Explainable AI (XAI): Integrated Gradients",
        "Which past inputs each forecast is most sensitive to, with a completeness check and a deletion test. Attributions show sensitivity, not cause."
    )

    if not windows or len(windows) == 0:
        empty_state(
            "Test Partition Data Unavailable",
            "Explainability requires test-partition sequence windows to compute attributions. "
            "Run setup to acquire the dataset.",
            "python setup_and_download.py"
        )
        return

    if status.model_name == "paper_overall":
        st.info("The selected model architecture produces a scalar global TRI forecast. Echelon-specific node attributions are active on ST-GCN-LSTM or LSTM Baseline.")
        return

    # Select current window from session state
    w_idx = st.session_state.get("window_idx", 0)
    if w_idx >= len(windows):
        w_idx = 0
    current_window = windows[w_idx]
    seq = current_window["sequence"]  # [10, 5]

    # Controls row: Node Selector and Full vs. Delta Mode
    c1, c2 = st.columns([1, 1], gap="medium")

    with c1:
        if "selected_node_idx" not in st.session_state:
            st.session_state["selected_node_idx"] = 1  # Default to Manufacturer

        node_opts = list(range(len(NODE_NAMES)))
        target_idx = st.radio(
            "Target Echelon to Diagnose",
            options=node_opts,
            format_func=lambda i: NODE_NAMES[i],
            index=st.session_state["selected_node_idx"],
            horizontal=True,
            key="why_target_node_radio"
        )
        st.session_state["selected_node_idx"] = target_idx

    with c2:
        delta_mode_sel = st.radio(
            "Attribution Mode",
            options=["Full Forecast Attribution", "Δ vs Persistence (Network Adjustment)"],
            index=0,
            horizontal=True,
            help="Full mode explains total predicted risk. Residual delta mode isolates the incremental adjustment made by the network over the last observed state."
        )
        residual_delta = (delta_mode_sel != "Full Forecast Attribution")
        st.caption("Explaining total forecast $f(x) - f(x_0)$" if not residual_delta else "Explaining incremental forecast adjustment: $g(x) = f(x) - y_t$")

    st.html("<div style='height: 12px;'></div>")

    # Compute attribution lazily
    model_key = f"{status.model_name}:{status.graph_mode}:{status.seed}"
    with st.spinner("Executing 64-step Path-Integrated Gradients..."):
        res, err = compute_explanation(model_key, seq, target_idx, residual_delta=residual_delta)

    if err or res is None:
        st.error(f"Integrated Gradients computation failed: {err}")
        return

    # 1. Plain-English Dynamic Narrative & Action Directive
    narration_text = narrate(res, seq_len=Config.SEQ_LEN)
    

    narrative_html = f"""
    <div style="font-size: 0.9375rem; line-height: 1.65; color: var(--text);">
        {escape(narration_text)}
    </div>
    <div style="margin-top: 8px; font-size: 0.75rem; color: var(--text-3); padding-top: 4px;">
        <em>Integrated Gradients</em> (64 steps) measures the forecast's sensitivity to each input relative to a training-mean baseline. It does not establish cause.
    </div>
    """
    st.html(card(f"AI Diagnostic Narrative: {NODE_NAMES[target_idx]}", narrative_html))
    st.html("<div style='height: 16px;'></div>")

    # 2. Charts Row: Signed Feature Attribution and Temporal Profile
    chart_col1, chart_col2 = st.columns([1, 1], gap="medium")

    with chart_col1:
        f_signed = res["feature_signed"]  # [5]
        f_shares = res["feature_importance"]  # [5]

        bar_colors = ["#EF4444" if val >= 0 else "#10B981" for val in f_signed]

        fig_feat = go.Figure()
        fig_feat.add_trace(go.Bar(
            y=FEATURE_NAMES,
            x=f_signed,
            orientation="h",
            marker=dict(color=bar_colors),
            customdata=[f"{s:.1%}" for s in f_shares],
            hovertemplate="<b>%{y}</b><br>Net Push: %{x:.4f}<br>Share of |Attr|: %{customdata}<extra></extra>"
        ))
        fig_feat.update_layout(
            title="Feature Signed Attribution (Push Direction)",
            xaxis_title="Attribution (Positive = Increases Risk, Negative = Decreases)",
            yaxis=dict(autorange="reversed"),
            margin=dict(l=10, r=10, t=35, b=25)
        )
        apply_theme(fig_feat, height=280)
        st.plotly_chart(fig_feat, use_container_width=True)

    with chart_col2:
        t_shares = res["time_importance"]  # [10]
        t_labels = [f"t-{Config.SEQ_LEN - 1 - i}" if (Config.SEQ_LEN - 1 - i) > 0 else "t" for i in range(Config.SEQ_LEN)]

        fig_time = go.Figure()
        fig_time.add_trace(go.Bar(
            x=t_labels,
            y=t_shares,
            marker=dict(color="#3B82F6"),
            hovertemplate="<b>Step %{x}</b><br>Saliency Share: %{y:.1%}<extra></extra>"
        ))
        fig_time.update_layout(
            title="Attribution by Time Step",
            xaxis_title="Lookback Time Step Across Window",
            yaxis_title="Normalized Share of Attribution",
            margin=dict(l=10, r=10, t=35, b=25)
        )
        apply_theme(fig_time, height=280)
        st.plotly_chart(fig_time, use_container_width=True)

    st.html("<div style='height: 16px;'></div>")

    # 3. Completeness Audit and Model Sensitivity (Deletion) Test
    aud_col, del_col = st.columns([1, 1], gap="medium")

    with aud_col:
        gap = res["completeness_gap"]
        pass_gap = (abs(gap) <= 1e-2)
        audit_badge = status_badge("Verified Pass", "low") if pass_gap else status_badge("Axiom Fail", "high")
        gap_color = "var(--ok)" if pass_gap else "var(--bad)"

        audit_html = f"""
        <div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 0.8125rem; color: var(--text-2);">Axiomatic Completeness:</span>
                {audit_badge}
            </div>
            <div style="font-size: 1.5rem; font-weight: 700; color: {gap_color};" class="mono-val">
                {gap:.2e}
            </div>
            <div style="font-size: 0.75rem; color: var(--text-3); margin-top: 6px; line-height: 1.5;">
                Condition: <span class="mono-val">|&sum; attr - (f(x) - f(x0))| &le; 1e-2</span>.
                A small gap means the attributions add up to the change in the model's output from the baseline.
            </div>
        </div>
        """
        st.html(card("Completeness Axiom Audit", audit_html))

    with del_col:
        top_feat_idx = int(np.argmax(res["feature_importance"]))
        neutralize_idx = st.selectbox(
            "Feature to Remove (Deletion Test)",
            options=list(range(len(FEATURE_NAMES))),
            format_func=lambda i: f"{FEATURE_NAMES[i]} ({res['feature_importance'][i]:.0%} share)",
            index=top_feat_idx,
            key="neutralize_feat_select"
        )

        del_res = run_deletion_test(model, seq, target_idx, neutralize_idx, baseline_val=0.0)
        orig_r = del_res["original_risk"]
        mod_r = del_res["modified_risk"]
        d_val = del_res["delta"]

        del_html = f"""
        <div>
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px;">
                <span style="font-size: 0.8125rem; color: var(--text-2);">Forecast Change:</span>
                <span style="font-size: 1.125rem; font-weight: 700; color: {'var(--bad)' if d_val > 0 else 'var(--ok)'};" class="mono-val">
                    {'▲ +' if d_val > 0 else '▼ '}{d_val:.4f}
                </span>
            </div>
            <div style="font-size: 0.8125rem; color: var(--text-2);" class="mono-val">
                {orig_r:.3f} &rarr; {mod_r:.3f}
            </div>
            <div style="font-size: 0.75rem; color: var(--text-3); margin-top: 6px;">
                Set <span class="mono-val">{escape(del_res['feature_name'])}</span> to 0 (scaled minimum) across all 10 lookback steps
                and re-ran the model.
            </div>
        </div>
        """
        st.html(card("Deletion Test", del_html))
