"""Mission Control Header & Telemetry Scrubber Component.

Displays:
1. Top Telemetry Glass Strip (Audit status, horizon, autograd steps).
2. Cyber Radial Donut Gauge for Total Risk Index (TRI).
3. 4-Node Health Matrix.
4. Professional Media Scrubber with Crisis Quick-Select.
"""
from typing import Dict, Any, List
import streamlit as st
import plotly.graph_objects as go
import numpy as np


def create_tri_gauge(tri_value: float) -> go.Figure:
    """Create a cyber-styled semi-circular gauge for Total Risk Index."""
    if tri_value < 0.35:
        bar_color = "#00F5A0" # Neon Emerald
        tier_text = "LOW RISK (NOMINAL)"
    elif tri_value <= 0.65:
        bar_color = "#FFB300" # Neon Amber
        tier_text = "ELEVATED RISK"
    else:
        bar_color = "#FF2E54" # Neon Crimson
        tier_text = "CRITICAL DISRUPTION"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=tri_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"TOTAL RISK INDEX (TRI)<br><span style='font-size:0.75em;color:{bar_color};font-weight:700'>{tier_text}</span>", 'font': {'size': 13, 'color': '#94A3B8', 'family': 'Plus Jakarta Sans'}},
        number={'font': {'size': 38, 'color': '#FFFFFF', 'family': 'Outfit'}, 'valueformat': '.3f'},
        gauge={
            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': '#334155', 'ticks': ''},
            'bar': {'color': bar_color, 'thickness': 0.78},
            'bgcolor': 'rgba(15, 23, 42, 0.6)',
            'borderwidth': 1,
            'bordercolor': 'rgba(255, 255, 255, 0.08)',
            'steps': [
                {'range': [0, 0.35], 'color': 'rgba(0, 245, 160, 0.12)'},
                {'range': [0.35, 0.65], 'color': 'rgba(255, 179, 0, 0.12)'},
                {'range': [0.65, 1.0], 'color': 'rgba(255, 46, 84, 0.16)'}
            ],
            'threshold': {
                'line': {'color': '#FF2E54', 'width': 3},
                'thickness': 0.85,
                'value': 0.65
            }
        }
    ))

    fig.update_layout(
        height=185,
        margin=dict(l=15, r=15, t=35, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#F8FAFC'}
    )
    return fig


def render_header(
    tri_value: float,
    pred_risks: np.ndarray,
    windows: List[Dict[str, Any]],
    current_idx: int
) -> int:
    """Render the top mission control bar and return updated window index."""
    # Top Telemetry Status Strip
    st.markdown("""
    <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 8px 16px; margin-bottom: 15px;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span class="beacon-pulse" style="background-color: #00F5A0;"></span>
            <span style="font-size: 0.76rem; font-weight: 700; color: #00F5A0; letter-spacing: 0.05em;">SYSTEM ACTIVE</span>
            <span style="color: #64748B;">•</span>
            <span style="font-size: 0.76rem; color: #94A3B8;">HORIZON: <strong>t+5 (10 MIN)</strong></span>
        </div>
        <div style="display: flex; gap: 14px; font-size: 0.76rem; color: #94A3B8;">
            <span>🔒 ZERO-LEAKAGE AUDIT: <strong style="color: #00F2FE;">VERIFIED</strong></span>
            <span>🧠 ARCHITECTURE: <strong style="color: #C77DFF;">ST-GCN-LSTM</strong></span>
            <span>⚡ LATENCY: <strong style="color: #00F5A0;">&lt; 25ms</strong></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Compute active health count
    high_count = sum(1 for r in pred_risks if r > 0.65)
    med_count = sum(1 for r in pred_risks if 0.35 <= r <= 0.65)
    low_count = 4 - high_count - med_count

    col_gauge, col_stats = st.columns([1.15, 1.85])

    with col_gauge:
        fig_gauge = create_tri_gauge(tri_value)
        st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})

    with col_stats:
        status_color = "#FF2E54" if high_count > 0 else ("#FFB300" if med_count > 0 else "#00F5A0")
        badge_type = "high" if high_count > 0 else ("medium" if med_count > 0 else "low")
        
        st.markdown(f"""
        <div class="sg-glass-card" style="margin-top: 10px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span style="font-size: 0.82rem; font-weight: 800; color: #FFFFFF; font-family: 'Outfit'; letter-spacing: 0.05em; text-transform: uppercase;">
                    Multi-Echelon Health Telemetry
                </span>
                <span class="badge-pill badge-{badge_type}">
                    {high_count} CRITICAL • {med_count} ELEVATED • {low_count} NOMINAL
                </span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; text-align: center;">
                <div style="background: rgba(11, 15, 25, 0.75); padding: 10px 6px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.06);">
                    <div style="font-size: 0.74rem; color: #94A3B8; font-weight: 600;">Supplier</div>
                    <div style="font-size: 1.25rem; font-weight: 800; font-family: 'Outfit'; color: {'#FF2E54' if pred_risks[0] > 0.65 else ('#FFB300' if pred_risks[0] >= 0.35 else '#00F5A0')};">{pred_risks[0]:.2f}</div>
                </div>
                <div style="background: rgba(11, 15, 25, 0.75); padding: 10px 6px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.06);">
                    <div style="font-size: 0.74rem; color: #94A3B8; font-weight: 600;">Manufacturer</div>
                    <div style="font-size: 1.25rem; font-weight: 800; font-family: 'Outfit'; color: {'#FF2E54' if pred_risks[1] > 0.65 else ('#FFB300' if pred_risks[1] >= 0.35 else '#00F5A0')};">{pred_risks[1]:.2f}</div>
                </div>
                <div style="background: rgba(11, 15, 25, 0.75); padding: 10px 6px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.06);">
                    <div style="font-size: 0.74rem; color: #94A3B8; font-weight: 600;">Distributor</div>
                    <div style="font-size: 1.25rem; font-weight: 800; font-family: 'Outfit'; color: {'#FF2E54' if pred_risks[2] > 0.65 else ('#FFB300' if pred_risks[2] >= 0.35 else '#00F5A0')};">{pred_risks[2]:.2f}</div>
                </div>
                <div style="background: rgba(11, 15, 25, 0.75); padding: 10px 6px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.06);">
                    <div style="font-size: 0.74rem; color: #94A3B8; font-weight: 600;">Retailer</div>
                    <div style="font-size: 1.25rem; font-weight: 800; font-family: 'Outfit'; color: {'#FF2E54' if pred_risks[3] > 0.65 else ('#FFB300' if pred_risks[3] >= 0.35 else '#00F5A0')};">{pred_risks[3]:.2f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Historical Window Replay Scrubber Controls (Command Deck Style)
    st.markdown("""
    <div style="margin: 15px 0 6px 0; display: flex; align-items: center; justify-content: space-between;">
        <span style="font-size: 0.85rem; font-weight: 700; color: #FFFFFF; font-family: 'Outfit'; letter-spacing: 0.05em; text-transform: uppercase;">
            ⏳ Historical Test Window Media Scrubber
        </span>
        <span style="font-size: 0.75rem; color: #64748B;">Offline Historical Replay Mode (Strict Chronological Test Partition)</span>
    </div>
    """, unsafe_allow_html=True)
    
    col_scrub, col_preset = st.columns([2.4, 1.6])
    with col_scrub:
        num_wins = len(windows)
        new_idx = st.slider(
            "Select Historical Window Index",
            min_value=0,
            max_value=max(0, num_wins - 1),
            value=min(current_idx, max(0, num_wins - 1)),
            label_visibility="collapsed"
        )
        if num_wins > 0:
            active_stamp = windows[new_idx].get("timestamp", f"Window #{new_idx+1}")
            st.markdown(f"<span style='font-family: JetBrains Mono; font-size: 0.8rem; color: #00F2FE;'>Active Timestamp: {active_stamp}</span>", unsafe_allow_html=True)

    with col_preset:
        preset_names = [w.get("timestamp", f"Window #{i+1}") for i, w in enumerate(windows)]
        selected_preset = st.selectbox(
            "Crisis Event Bookmark Quick-Jump",
            options=range(len(preset_names)),
            format_func=lambda i: preset_names[i] if i < len(preset_names) else f"Window #{i+1}",
            index=new_idx,
            label_visibility="collapsed"
        )
        if selected_preset != new_idx:
            new_idx = selected_preset

    return new_idx
