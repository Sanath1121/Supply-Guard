"""Plotly Theme Utility for SupplyGuard.

Applies the restrained dark analytics styling consistently to all figures.
"""
import plotly.graph_objects as go


SERIES_COLORS = {
    "Supplier": "#38BDF8",      # Sky blue
    "Manufacturer": "#A78BFA",  # Lavender purple
    "Distributor": "#F472B6",   # Rose pink
    "Retailer": "#2DD4BF",      # Teal mint
    "Total Cost": "#94A3B8",    # Slate grey
    "Persistence": "#64748B"    # Dim grey
}


def apply_theme(fig: go.Figure, height: int = 300) -> go.Figure:
    """Apply consistent dark analytics layout to a Plotly figure."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=15, r=15, t=35, b=25),
        font=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
            color="#B6C2D4",
            size=12
        ),
        xaxis=dict(
            gridcolor="rgba(148, 163, 184, 0.12)",
            zerolinecolor="rgba(148, 163, 184, 0.2)",
            tickfont=dict(color="#B6C2D4", size=11),
            title_font=dict(color="#F1F5F9", size=12)
        ),
        yaxis=dict(
            gridcolor="rgba(148, 163, 184, 0.12)",
            zerolinecolor="rgba(148, 163, 184, 0.2)",
            tickfont=dict(color="#B6C2D4", size=11),
            title_font=dict(color="#F1F5F9", size=12)
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#B6C2D4", size=11)
        ),
        hoverlabel=dict(
            bgcolor="#111A2E",
            bordercolor="rgba(148, 163, 184, 0.3)",
            font=dict(family="Inter, sans-serif", color="#F1F5F9")
        )
    )
    return fig
