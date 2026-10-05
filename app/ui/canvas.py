"""Offline SVG Supply Chain Topology Canvas.

Renders a responsive, accessible, high-fidelity SVG representation of the 4-echelon supply chain:
Supplier -> Manufacturer -> Distributor -> Retailer.
Features smooth cubic Bézier flow paths, dual-glow stroke shaders, glowing anchor ports,
and glassmorphic node pods.
100% offline-safe, zero external CDNs, fully WCAG AA compliant.
"""
from typing import Dict, Any, Optional, Tuple
import html
from app.utils.formatters import get_tier_color, compute_tercile_tier
from app.ui.components import clean_html


def render_topology_svg(
    node_risks: Dict[str, float],
    tiers: Dict[str, float],
    edge_shares: Optional[Dict[Tuple[str, str], float]] = None,
    xai_error: Optional[str] = None
) -> str:
    """Generate responsive, high-end SVG markup for the 4-node supply chain topology."""
    echelons = [
        ("Supplier", 125, 115, "📦", "Tier-1 Sourcing"),
        ("Manufacturer", 365, 115, "⚙️", "Assembly & Production"),
        ("Distributor", 605, 115, "🚚", "Freight & Logistics"),
        ("Retailer", 845, 115, "🏪", "Point of Sale")
    ]
    p33 = tiers.get("p33", 0.35)
    p66 = tiers.get("p66", 0.65)

    # Calculate tiers and colors for each node
    node_data = {}
    aria_parts = []
    for name, x, y, icon, desc in echelons:
        r_val = float(node_risks.get(name, 0.0))
        t_tier = compute_tercile_tier(r_val, p33, p66)
        color = get_tier_color(t_tier)
        node_data[name] = {
            "risk": r_val,
            "tier": t_tier,
            "color": color,
            "x": x,
            "y": y,
            "icon": icon,
            "desc": desc
        }
        aria_parts.append(f"{name} {r_val:.2f} {t_tier}")

    aria_label = html.escape(f"Supply chain topology: {' -> '.join(aria_parts)}")

    # Edges definition with smooth cubic Bézier flow paths
    edges = [
        ("Supplier", "Manufacturer"),
        ("Manufacturer", "Distributor"),
        ("Distributor", "Retailer")
    ]

    edges_svg = []
    for u, v in edges:
        u_info = node_data[u]
        v_info = node_data[v]
        x1 = u_info["x"] + 82
        y1 = u_info["y"]
        x2 = v_info["x"] - 82
        y2 = v_info["y"]

        # Cubic Bézier control points
        dx = (x2 - x1) * 0.5
        c_path = f"M {x1} {y1} C {x1 + dx} {y1}, {x2 - dx} {y2}, {x2} {y2}"

        share_val = None
        if edge_shares and (u, v) in edge_shares:
            share_val = edge_shares[(u, v)]

        if share_val is not None:
            stroke_width = max(2.5, min(8.0, 2.5 + share_val * 5.5))
            share_text = f"{share_val:.0%}"
            stroke_color = "#38BDF8"
            glow_path = f'<path d="{c_path}" fill="none" stroke="rgba(56, 189, 248, 0.25)" stroke-width="{stroke_width + 6:.1f}" />'
            dash_attr = 'stroke-dasharray="8,6" class="sg-flow-line"'
        else:
            stroke_width = 2.0
            share_text = "–" if not xai_error else "N/A"
            stroke_color = "rgba(148, 163, 184, 0.3)"
            glow_path = ""
            dash_attr = ""

        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2 - 16

        edges_svg.append(f"""
        <!-- Edge {u} -> {v} (Smooth Cubic Bézier) -->
        {glow_path}
        <path d="{c_path}" fill="none" stroke="{stroke_color}" stroke-width="{stroke_width:.1f}" {dash_attr} />
        <!-- Directional Arrowhead -->
        <polygon points="{x2-4},{y2-6} {x2+6},{y2} {x2-4},{y2+6}" fill="{stroke_color}" />
        
        <!-- Connection Anchor Ports -->
        <circle cx="{x1}" cy="{y1}" r="4" fill="#040711" stroke="{stroke_color}" stroke-width="2" />
        <circle cx="{x2}" cy="{y2}" r="4" fill="#040711" stroke="{stroke_color}" stroke-width="2" />

        <!-- Attribution Share Pill Badge -->
        <g transform="translate({mid_x}, {mid_y})">
            <rect x="-32" y="-12" width="64" height="24" rx="12" fill="rgba(15, 23, 42, 0.95)" stroke="rgba(56, 189, 248, 0.35)" stroke-width="1" />
            <text x="0" y="4" fill="#F8FAFC" font-size="11" font-family="'JetBrains Mono', monospace" font-weight="700" text-anchor="middle">
                {share_text}
            </text>
        </g>
        """)

    # Nodes SVG (Refined Glassmorphic Node Pods with Glowing Portals)
    nodes_svg = []
    for name, info in node_data.items():
        x = info["x"]
        y = info["y"]
        c = info["color"]
        r = info["risk"]
        tier = info["tier"]
        icon = info["icon"]
        desc = info["desc"]

        pulse_halo = ""
        if tier == "High":
            pulse_halo = f'<rect x="{x-86}" y="{y-66}" width="172" height="132" rx="18" fill="none" stroke="{c}" stroke-width="1.5" class="pulse-high" />'

        nodes_svg.append(f"""
        <!-- Node Pod: {name} -->
        <g>
            {pulse_halo}
            <!-- Outer Glow Backdrop -->
            <rect x="{x-80}" y="{y-60}" width="160" height="120" rx="16" fill="{c}10" stroke="{c}" stroke-width="2" />
            <rect x="{x-79}" y="{y-59}" width="158" height="118" rx="15" fill="#0F172A" fill-opacity="0.92" />
            
            <!-- Specular Top Highlight Line -->
            <line x1="{x-65}" y1="{y-59}" x2="{x+65}" y2="{y-59}" stroke="rgba(255, 255, 255, 0.2)" stroke-width="1" />

            <!-- Echelon Header (Icon + Name) -->
            <text x="{x-62}" y="{y-36}" font-size="15">{icon}</text>
            <text x="{x-40}" y="{y-35}" fill="#F8FAFC" font-size="12" font-weight="700" letter-spacing="0.05em">
                {name.upper()}
            </text>

            <!-- Large Crisp Risk Score -->
            <text x="{x}" y="{y+10}" fill="#FFFFFF" font-size="24" font-family="'JetBrains Mono', monospace" font-weight="800" text-anchor="middle">
                {r:.3f}
            </text>

            <!-- Tier Status Pill Badge -->
            <g transform="translate({x}, {y+36})">
                <rect x="-44" y="-10" width="88" height="20" rx="10" fill="{c}22" stroke="{c}66" stroke-width="1" />
                <text x="0" y="4" fill="{c}" font-size="10" font-weight="700" text-anchor="middle" letter-spacing="0.06em">
                    {tier.upper()}
                </text>
            </g>
        </g>
        """)

    notice_svg = ""
    if xai_error:
        safe_err = html.escape(str(xai_error)[:60])
        notice_svg = f"""
        <g transform="translate(480, 218)">
            <rect x="-240" y="-12" width="480" height="24" rx="6" fill="#15203B" stroke="rgba(245, 158, 11, 0.4)" />
            <text x="0" y="4" fill="#FBBF24" font-size="11" font-weight="500" text-anchor="middle">
                ⚠️ Sensitivity attribution unavailable: {safe_err}
            </text>
        </g>
        """

    svg_content = f"""
    <div style="width: 100%; max-width: 1020px; margin: 0 auto; overflow: hidden;">
        <svg viewBox="0 0 980 238" width="100%" height="auto" role="img" aria-label="{aria_label}" style="display: block; min-height: 210px; max-height: 260px; background: linear-gradient(180deg, rgba(15, 23, 42, 0.8) 0%, rgba(4, 7, 17, 0.95) 100%); border-radius: 16px; border: 1px solid rgba(148, 163, 184, 0.16); box-shadow: 0 8px 32px rgba(0,0,0,0.6);">
            <title>{aria_label}</title>
            <defs>
                <pattern id="topoGrid" width="28" height="28" patternUnits="userSpaceOnUse">
                    <path d="M 28 0 L 0 0 0 28" fill="none" stroke="rgba(255, 255, 255, 0.025)" stroke-width="1" />
                </pattern>
            </defs>
            <rect width="100%" height="100%" fill="url(#topoGrid)" />
            <style>
                @media (prefers-reduced-motion: no-preference) {{
                    .sg-flow-line {{
                        animation: sg-flow-dash 1.3s linear infinite;
                    }}
                    @keyframes sg-flow-dash {{
                        from {{ stroke-dashoffset: 28; }}
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
    return clean_html(svg_content)
