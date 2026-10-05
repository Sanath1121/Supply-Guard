"""Offline SVG Supply Chain Topology Canvas.

Renders a responsive, accessible SVG representation of the 4-echelon supply chain:
Supplier -> Manufacturer -> Distributor -> Retailer.
Uses real Integrated Gradients per-edge attribution shares to scale edge thickness.
No external fonts, no fixed-height iframes, fully WCAG AA compliant.
"""
from typing import Dict, Any, Optional, Tuple
import html
from app.utils.formatters import get_tier_color, compute_tercile_tier


def render_topology_svg(
    node_risks: Dict[str, float],
    tiers: Dict[str, float],
    edge_shares: Optional[Dict[Tuple[str, str], float]] = None,
    xai_error: Optional[str] = None
) -> str:
    """Generate responsive SVG markup for the 4-node supply chain topology."""
    echelons = [
        ("Supplier", 120, 110),
        ("Manufacturer", 340, 110),
        ("Distributor", 560, 110),
        ("Retailer", 780, 110)
    ]
    p33 = tiers.get("p33", 0.35)
    p66 = tiers.get("p66", 0.65)

    # Calculate tiers and colors for each node
    node_data = {}
    aria_parts = []
    for name, x, y in echelons:
        r_val = float(node_risks.get(name, 0.0))
        t_tier = compute_tercile_tier(r_val, p33, p66)
        color = get_tier_color(t_tier)
        node_data[name] = {"risk": r_val, "tier": t_tier, "color": color, "x": x, "y": y}
        aria_parts.append(f"{name} {r_val:.2f} {t_tier}")

    aria_label = html.escape(f"Supply chain topology: {' -> '.join(aria_parts)}")

    # Edges definition
    edges = [
        ("Supplier", "Manufacturer"),
        ("Manufacturer", "Distributor"),
        ("Distributor", "Retailer")
    ]

    edges_svg = []
    for u, v in edges:
        u_info = node_data[u]
        v_info = node_data[v]
        x1 = u_info["x"] + 55
        y1 = u_info["y"]
        x2 = v_info["x"] - 55
        y2 = v_info["y"]

        share_val = None
        if edge_shares and (u, v) in edge_shares:
            share_val = edge_shares[(u, v)]

        if share_val is not None:
            # Scale stroke width between 2px and 8px based on share (0.0 to 1.0)
            stroke_width = max(2.0, min(8.0, 2.0 + share_val * 6.0))
            share_text = f"{share_val:.0%}"
            stroke_color = "var(--accent)"
            dash_attr = 'stroke-dasharray="6,4" class="sg-flow-line"'
        else:
            stroke_width = 2.0
            share_text = "–" if not xai_error else "N/A"
            stroke_color = "var(--border-strong)"
            dash_attr = ""

        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2 - 12

        edges_svg.append(f"""
        <!-- Edge {u} -> {v} -->
        <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke_color}" stroke-width="{stroke_width:.1f}" {dash_attr} />
        <polygon points="{x2-2},{y2-5} {x2+6},{y2} {x2-2},{y2+5}" fill="{stroke_color}" />
        <rect x="{mid_x-22}" y="{mid_y-10}" width="44" height="18" rx="4" fill="var(--surface-2)" stroke="var(--border)" stroke-width="1" />
        <text x="{mid_x}" y="{mid_y+3}" fill="var(--text-2)" font-size="11" font-family="'JetBrains Mono', monospace" font-weight="600" text-anchor="middle">
            {share_text}
        </text>
        """)

    # Nodes SVG
    nodes_svg = []
    for name, info in node_data.items():
        x = info["x"]
        y = info["y"]
        c = info["color"]
        r = info["risk"]
        tier = info["tier"]

        pulse_circle = ""
        if tier == "High":
            pulse_circle = f'<circle cx="{x}" cy="{y}" r="50" fill="none" stroke="{c}" stroke-width="1.5" class="pulse-high" />'

        nodes_svg.append(f"""
        <!-- Node {name} -->
        <g transform="translate(0, 0)">
            {pulse_circle}
            <circle cx="{x}" cy="{y}" r="45" fill="var(--surface-1)" stroke="{c}" stroke-width="2.5" />
            <circle cx="{x}" cy="{y}" r="38" fill="var(--surface-2)" />
            <text x="{x}" y="{y-8}" fill="var(--text-2)" font-size="11" font-weight="600" text-anchor="middle" letter-spacing="0.04em">
                {name.upper()}
            </text>
            <text x="{x}" y="{y+14}" fill="var(--text)" font-size="16" font-family="'JetBrains Mono', monospace" font-weight="700" text-anchor="middle">
                {r:.3f}
            </text>
            <text x="{x}" y="{y+28}" fill="{c}" font-size="10" font-weight="700" text-anchor="middle" text-transform="uppercase">
                {tier}
            </text>
        </g>
        """)

    notice_svg = ""
    if xai_error:
        safe_err = html.escape(str(xai_error)[:60])
        notice_svg = f"""
        <text x="450" y="200" fill="var(--warn)" font-size="11" text-anchor="middle">
            ⚠️ Sensitivity attribution unavailable: {safe_err}
        </text>
        """

    svg_content = f"""
    <div style="width: 100%; max-width: 900px; margin: 0 auto; overflow: hidden;">
        <svg viewBox="0 0 900 215" width="100%" height="auto" role="img" aria-label="{aria_label}" style="display: block; min-height: 180px; max-height: 240px; background: var(--surface-1); border-radius: var(--r-md); border: 1px solid var(--border);">
            <title>{aria_label}</title>
            <style>
                @media (prefers-reduced-motion: no-preference) {{
                    .sg-flow-line {{
                        animation: sg-dash-flow 1.5s linear infinite;
                    }}
                    @keyframes sg-dash-flow {{
                        from {{ stroke-dashoffset: 20; }}
                        to {{ stroke-dashoffset: 0; }}
                    }}
                }}
            </style>
            {''.join(edges_svg)}
            {''.join(nodes_svg)}
            {notice_svg}
        </svg>
    </div>
    """
    return svg_content
