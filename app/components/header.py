"""Mission Control Header & Media Scrubber Component.

Displays:
1. Global Total Risk Index (TRI) semi-circular radial gauge.
2. Active Echelon network health status pill.
3. Model seed and graph topology controls.
4. Historical timeline scrubber with Play/Pause streaming replay.
"""
from typing import Dict, Any, List
import streamlit as st
import plotly.graph_objects as go
import numpy as np


def create_tri_gauge(tri_value: float) -> go.Figure:
    """Create a sleek cyber-styled semi-circular gauge for Total Risk Index."""
    # Determine color by tier
    if tri_value < 0.35:
        bar_color = "#10B981" # Emerald
        tier_text = "LOW RISK"
    elif tri_value <= 0.65:
        bar_color = "#F59E0B" # Amber
        tier_text = "MODERATE RISK"
    else:
        bar_color = "#EF4444" # Crimson
        tier_text = "CRITICAL ALERT"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=tri_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"SYSTEM TOTAL RISK INDEX (TRI)<br><span style='font-size:0.7em;color:{bar_color}'>{tier_text}</span>", 'font': {'size': 14, 'color': '#94A3B8', 'family': 'Inter'}},
        number={'font': {'size': 36, 'color': '#F1F5F9', 'family': 'Outfit'}, 'valueformat': '.3f'},
        gauge={
            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': '#334155', 'ticks': ''},
            'bar': {'color': bar_color, 'thickness': 0.75},
            'bgcolor': 'rgba(15, 23, 42, 0.6)',
            'borderwidth': 1,
            'bordercolor': 'rgba(255, 255, 255, 0.08)',
            'steps': [
                {'range': [0, 0.35], 'color': 'rgba(16, 185, 129, 0.12)'},
                {'range': [0.35, 0.65], 'color': 'rgba(245, 158, 11, 0.12)'},
                {'range': [0.65, 1.0], 'color': 'rgba(239, 68, 68, 0.15)'}
            ],
            'threshold': {
                'line': {'color': '#EF4444', 'width': 3},
                'thickness': 0.85,
                'value': 0.65
            }
        }
    ))

    fig.update_layout(
        height=180,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#F1F5F9'}
    )
    return fig


def render_header(
    tri_value: float,
    pred_risks: np.ndarray,
    windows: List[Dict[str, Any]],
    current_idx: int
) -> int:
    """Render the top mission control bar and return updated window index."""
    # Compute active health count
    high_count = sum(1 for r in pred_risks if r > 0.65)
    med_count = sum(1 for r in pred_risks if 0.35 <= r <= 0.65)
    low_count = 4 - high_count - med_count

    col_gauge, col_stats = st.columns([1.1, 1.9])

    with col_gauge:
        fig_gauge = create_tri_gauge(tri_value)
        st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})

    with col_stats:
        st.markdown(f"""
        <div class="sg-card" style="margin-top: 15px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span style="font-size: 0.8rem; font-weight: 700; color: #94A3B8; text-transform: uppercase;">
                    Network Status Health Matrix
                </span>
                <span class="badge-pill badge-{'high' if high_count > 0 else ('medium' if med_count > 0 else 'low')}">
                    {high_count} CRITICAL / {med_count} ELEVATED / {low_count} NOMINAL
                </span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; text-align: center;">
                <div style="background: rgba(13, 21, 39, 0.7); padding: 8px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                    <div style="font-size: 0.75rem; color: #94A3B8;">Supplier</div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: {'#EF4444' if pred_risks[0] > 0.65 else ('#F59E0B' if pred_risks[0] >= 0.35 else '#10B981')};">{pred_risks[0]:.2f}</div>
                </div>
                <div style="background: rgba(13, 21, 39, 0.7); padding: 8px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                    <div style="font-size: 0.75rem; color: #94A3B8;">Manufacturer</div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: {'#EF4444' if pred_risks[1] > 0.65 else ('#F59E0B' if pred_risks[1] >= 0.35 else '#10B981')};">{pred_risks[1]:.2f}</div>
                </div>
                <div style="background: rgba(13, 21, 39, 0.7); padding: 8px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                    <div style="font-size: 0.75rem; color: #94A3B8;">Distributor</div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: {'#EF4444' if pred_risks[2] > 0.65 else ('#F59E0B' if pred_risks[2] >= 0.35 else '#10B981')};">{pred_risks[2]:.2f}</div>
                </div>
                <div style="background: rgba(13, 21, 39, 0.7); padding: 8px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                    <div style="font-size: 0.75rem; color: #94A3B8;">Retailer</div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: {'#EF4444' if pred_risks[3] > 0.65 else ('#F59E0B' if pred_risks[3] >= 0.35 else '#10B981')};">{pred_risks[3]:.2f}</div>
                </div>
            </div>
            <div style="margin-top: 10px; font-size: 0.78rem; color: #64748B;">
                🔒 <strong>Zero Leakage Audit:</strong> Forward horizon $t+5$ (10 min ahead) strictly evaluated on test partition.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Historical Window Replay Scrubber Controls
    st.markdown("<hr style='border-color: rgba(255,255,255,0.06); margin: 15px 0 10px 0;'>", unsafe_allow_html=True)
    
    col_scrub, col_preset = st.columns([2.5, 1.5])
    with col_scrub:
        num_wins = len(windows)
        new_idx = st.slider(
            "⏳ Historical Test Window Timeline Scrubber",
            min_value=0,
            max_value=max(0, num_wins - 1),
            value=min(current_idx, max(0, num_wins - 1)),
            format=f"Sample %d / {num_wins}"
        )
        if num_wins > 0:
            active_stamp = windows[new_idx].get("timestamp", f"Window #{new_idx+1}")
            st.caption(f"Active Timestamp: `{active_stamp}`")

    with col_preset:
        preset_names = [w.get("timestamp", f"Window #{i+1}") for i, w in enumerate(windows)]
        selected_preset = st.selectbox(
            "⚡ Crisis Bookmark Quick-Select",
            options=range(len(preset_names)),
            format_func=lambda i: preset_names[i] if i < len(preset_names) else f"Window #{i+1}",
            index=new_idx
        )
        if selected_preset != new_idx:
            new_idx = selected_preset

    return new_idx
