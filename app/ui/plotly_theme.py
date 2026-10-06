"""Plotly Theme Utility for SupplyGuard.

Applies a high-fidelity Enterprise dark analytics layout (Linear / Datadog / Tremor aesthetic)
consistently to all dashboard charts.
"""
import plotly.graph_objects as go


SERIES_COLORS = {
    "Supplier": "#38BDF8",      # Sky Cyan
    "Manufacturer": "#A78BFA",  # Lavender Purple
    "Distributor": "#F472B6",   # Rose Pink
    "Retailer": "#2DD4BF",      # Mint Teal
    "Total Cost": "#94A3B8",    # Slate Grey
    "Persistence": "#64748B"    # Muted Grey
}


def apply_theme(fig: go.Figure, height: int = 320) -> go.Figure:
    """Apply modern enterprise dark analytics styling to a Plotly figure."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=30),
        font=dict(
            family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Inter, sans-serif",
            color="#CBD5E1",
            size=12
        ),
        xaxis=dict(
            gridcolor="rgba(148, 163, 184, 0.08)",
            zerolinecolor="rgba(148, 163, 184, 0.16)",
            tickfont=dict(family="'JetBrains Mono', monospace", color="#94A3B8", size=11),
            title_font=dict(color="#F8FAFC", size=12)
        ),
        yaxis=dict(
            gridcolor="rgba(148, 163, 184, 0.08)",
            zerolinecolor="rgba(148, 163, 184, 0.16)",
            tickfont=dict(family="'JetBrains Mono', monospace", color="#94A3B8", size=11),
            title_font=dict(color="#F8FAFC", size=12)
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="right",
            x=1,
            font=dict(color="#CBD5E1", size=11),
            bgcolor="rgba(15, 23, 42, 0.7)",
            bordercolor="rgba(148, 163, 184, 0.2)",
            borderwidth=1
        ),
        hoverlabel=dict(
            bgcolor="rgba(15, 23, 42, 0.96)",
            bordercolor="rgba(59, 130, 246, 0.5)",
            font=dict(family="'JetBrains Mono', monospace", color="#F8FAFC", size=12)
        )
    )
    return fig
