"""Synchronized Multi-Echelon Risk Waveform Component.

Renders interactive continuous time-series curves across all 4 echelons plus Total Cost
over the 10-step lookback window and the forward forecast horizon.
"""
from typing import Dict, Any
import streamlit as st
import plotly.graph_objects as go
import numpy as np

from src.config import Config


def render_waveform_chart(
    seq_window: np.ndarray,
    pred_risks: np.ndarray,
    timestamp_label: str = ""
):
    """Render the synchronized multi-echelon risk waveform chart."""
    st.markdown("### 📈 Multi-Echelon Spatiotemporal Risk Waveforms")
    st.markdown(
        "Synchronized continuous trajectories across the 10-step historical lookback window "
        "and the forward predicted risk ($t+H$)."
    )

    steps = [f"t-{Config.SEQ_LEN - 1 - i}" if (Config.SEQ_LEN - 1 - i) > 0 else "t" for i in range(Config.SEQ_LEN)]
    all_steps = steps + [f"t+{Config.HORIZON} (Forecast)"]

    colors = {
        "Supplier": "#10B981",       # Emerald
        "Manufacturer": "#F59E0B",   # Amber
        "Distributor": "#EF4444",    # Crimson
        "Retailer": "#06B6D4",       # Cyan
        "Total Cost": "#8B5CF6"      # Purple
    }

    fig = go.Figure()

    # Add historical curves
    for i, name in enumerate(Config.NODE_NAMES):
        hist_vals = list(seq_window[:, i])
        forecast_val = float(pred_risks[i])
        combined_vals = hist_vals + [forecast_val]

        fig.add_trace(go.Scatter(
            x=all_steps,
            y=combined_vals,
            mode='lines+markers',
            name=f"{name} Risk",
            line=dict(color=colors[name], width=2.5),
            marker=dict(size=6)
        ))

    # Add Total Cost
    cost_hist = list(seq_window[:, 4])
    # Total cost forecast placeholder
    cost_combined = cost_hist + [cost_hist[-1]]
    fig.add_trace(go.Scatter(
        x=all_steps,
        y=cost_combined,
        mode='lines',
        name="Normalized Total Cost",
        line=dict(color=colors["Total Cost"], width=1.5, dash='dash')
    ))

    # Add vertical divider line at step t
    fig.add_vline(x="t", line_width=1.5, line_dash="dash", line_color="#64748B")

    fig.update_layout(
        title=dict(text=f"Risk Trajectories (Window: {timestamp_label})", font=dict(size=14, color="#94A3B8")),
        xaxis=dict(title="Time Step (t-9 to t+5)", gridcolor="#1E293B", tickfont=dict(color="#CBD5E1")),
        yaxis=dict(title="Normalized Value [0, 1]", gridcolor="#1E293B", tickfont=dict(color="#CBD5E1"), range=[0, 1.05]),
        hovermode="x unified",
        height=350,
        margin=dict(l=10, r=10, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11, color="#CBD5E1")),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#F1F5F9')
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
