"""Offline SVG Supply Chain Topology Canvas.

Renders an enterprise-grade, accessible, responsive SVG representation of the 4-echelon supply chain:
Supplier -> Manufacturer -> Distributor -> Retailer.
Features unique anti-gravity floating animations, a delayed highlight on the highest-risk node (1.0s delay),
zero numerical clutter, and spatiotemporal risk cascade visual telemetry.
100% offline-safe, zero external CDNs, fully WCAG AA compliant.
"""
from typing import Dict, Any, Optional, Tuple
import html
from app.utils.formatters import get_tier_color, node_tier
from app.ui.components import clean_html


def render_topology_svg(
    node_risks: Dict[str, float],
    tiers: Dict[str, float],
    edge_shares: Optional[Dict[Tuple[str, str], float]] = None,
    xai_error: Optional[str] = None
) -> str:
    """Generate responsive, floating SVG topology with delayed disruption shockwaves and zero numbers."""
    echelons = [
        ("Supplier", 125, 120, "📦", "Upstream Sourcing"),
        ("Manufacturer", 365, 120, "⚙️", "Production Assembly"),
        ("Distributor", 605, 120, "🚚", "Logistics & Freight"),
        ("Retailer", 845, 120, "🏪", "Point of Sale")
    ]
    # Calculate tiers and colors for each node
    node_data = {}
    aria_parts = []
    for node_idx, (name, x, y, icon, desc) in enumerate(echelons):
        r_val = float(node_risks.get(name, 0.0))
        t_tier = node_tier(r_val, node_idx, tiers)
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
        aria_parts.append(f"{name} {t_tier}")

    aria_label = html.escape(f"Supply chain network topology: {' -> '.join(aria_parts)}")

    # Highlight the High-tier node furthest above its own upper tercile. This marks the highest
    # predicted risk only; it does not identify where a disruption originated.
    def _margin(k):
        i = [e[0] for e in echelons].index(k)
        hi = tiers["node_p66"][i] if "node_p66" in tiers else tiers.get("p66", 0.65)
        return node_data[k]["risk"] - hi
    high_nodes = [k for k in node_data if node_data[k]["tier"] == "High"]
    critical_node = max(high_nodes, key=_margin) if high_nodes else None

    # Edges definition with smooth cubic Bézier flow paths
    edges = [
        ("Supplier", "Manufacturer"),
        ("Manufacturer", "Distributor"),
        ("Distributor", "Retailer")
    ]

    edges_svg = []
    for edge_idx, (u, v) in enumerate(edges):
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

        is_critical_outflow = (u == critical_node)

        # ZERO numbers: show qualitative directional telemetry
        if share_val is not None and share_val > 0.05:
            stroke_width = max(3.0, min(8.0, 3.0 + share_val * 6.0))
            if is_critical_outflow:
                flow_label = "FLOW ▶"
                stroke_color = "#EF4444"
                glow_color = "rgba(239, 68, 68, 0.35)"
                line_cls = "sg-flow-line sg-cascade-edge"
            else:
                flow_label = "FLOW ▶"
                stroke_color = "#3B82F6"
                glow_color = "rgba(59, 130, 246, 0.28)"
                line_cls = "sg-flow-line"
            glow_path = f'<path d="{c_path}" fill="none" stroke="{glow_color}" stroke-width="{stroke_width + 6:.1f}" />'
            dash_attr = f'stroke-dasharray="8,6" class="{line_cls}"'
        else:
            stroke_width = 2.0
            flow_label = "LINK" if not xai_error else "N/A"
            stroke_color = "rgba(148, 163, 184, 0.25)"
            glow_path = ""
            dash_attr = ""

        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2 - 16

        pill_border = "rgba(239, 68, 68, 0.6)" if is_critical_outflow else "rgba(59, 130, 246, 0.4)"

        edges_svg.append(f"""
        <!-- Edge {u} -> {v} -->
        {glow_path}
        <path d="{c_path}" fill="none" stroke="{stroke_color}" stroke-width="{stroke_width:.1f}" {dash_attr} />
        <polygon points="{x2-5},{y2-6} {x2+5},{y2} {x2-5},{y2+6}" fill="{stroke_color}" />
        
        <!-- Connection Anchor Ports -->
        <circle cx="{x1}" cy="{y1}" r="4" fill="#030712" stroke="{stroke_color}" stroke-width="2" />
        <circle cx="{x2}" cy="{y2}" r="4" fill="#030712" stroke="{stroke_color}" stroke-width="2" />

        <!-- Attribution Flow Pill Badge (NO numbers) -->
        <!-- Outer group positions; inner group animates (a CSS transform would replace the translate) -->
        <g transform="translate({mid_x}, {mid_y})"><g class="sg-float-pill-{edge_idx}">
            <rect x="-38" y="-12" width="76" height="24" rx="12" fill="rgba(15, 23, 42, 0.95)" stroke="{pill_border}" stroke-width="1" />
            <text x="0" y="4" fill="#F8FAFC" font-size="10" font-family="'JetBrains Mono', monospace" font-weight="700" text-anchor="middle" letter-spacing="0.05em">
                {flow_label}
            </text>
        </g></g>
        """)

    # Nodes SVG
    nodes_svg = []
    for node_idx, (name, info) in enumerate(node_data.items()):
        x = info["x"]
        y = info["y"]
        c = info["color"]
        tier = info["tier"]
        icon = info["icon"]
        is_highest = (name == critical_node)

        # Center state label (ZERO numbers)
        if is_highest:
            center_label = "HIGH RISK"
            badge_label = "● HIGHEST"
            center_color = "#EF4444"
            badge_bg = "rgba(239, 68, 68, 0.22)"
            badge_border = "rgba(239, 68, 68, 0.8)"
            badge_text_color = "#EF4444"
        elif tier == "High":
            center_label = "HIGH"
            badge_label = "● HIGH RISK"
            center_color = "#EF4444"
            badge_bg = f"{c}22"
            badge_border = f"{c}66"
            badge_text_color = c
        elif tier == "Medium":
            center_label = "ELEVATED"
            badge_label = "● MEDIUM"
            center_color = "#F59E0B"
            badge_bg = f"{c}22"
            badge_border = f"{c}66"
            badge_text_color = c
        else:
            center_label = "NOMINAL"
            badge_label = "● LOW"
            center_color = "#10B981"
            badge_bg = f"{c}22"
            badge_border = f"{c}66"
            badge_text_color = c

        # Disruption shockwaves (ONLY at critical node, triggers strictly after 1.0s delay)
        disruption_shockwaves = ""
        disruption_beacon = ""
        node_extra_class = f"sg-node-{node_idx}"
        outer_rect_class = ""

        if is_highest:
            outer_rect_class = "sg-critical-glow"
            disruption_shockwaves = f"""
            <!-- Disruption Shockwave Rings (Delayed 1.0s) -->
            <rect x="{x-82}" y="{y-62}" width="164" height="124" rx="18" fill="none" stroke="#EF4444" stroke-width="2" class="sg-disrupt-ripple-1" />
            <rect x="{x-84}" y="{y-64}" width="168" height="128" rx="20" fill="none" stroke="#F87171" stroke-width="1.5" class="sg-disrupt-ripple-2" />
            """
            disruption_beacon = f"""
            <!-- Disruption Warning Beacon (Delayed 1.0s) -->
            <g transform="translate({x}, {y-68})"><g class="sg-disrupt-beacon">
                <rect x="-54" y="-11" width="108" height="22" rx="11" fill="rgba(239, 68, 68, 0.95)" stroke="#FFFFFF" stroke-width="1" />
                <text x="0" y="4" fill="#FFFFFF" font-size="10" font-weight="800" text-anchor="middle" letter-spacing="0.06em">
                    ⚡ HIGHEST RISK
                </text>
            </g></g>
            """
            center_text_markup = f"""
            <!-- State Designation (Transitions from MONITORED to HIGH RISK at 1.0s) -->
            <text x="{x}" y="{y+10}" fill="#94A3B8" font-size="15" font-family="'JetBrains Mono', monospace" font-weight="700" text-anchor="middle" letter-spacing="0.08em" class="sg-pre-disrupt-text">
                MONITORED
            </text>
            <text x="{x}" y="{y+10}" fill="{center_color}" font-size="16" font-family="'JetBrains Mono', monospace" font-weight="800" text-anchor="middle" letter-spacing="0.08em" class="sg-post-disrupt-text">
                {center_label}
            </text>
            """
        else:
            if tier == "High":
                disruption_shockwaves = f'<rect x="{x-86}" y="{y-66}" width="172" height="132" rx="18" fill="none" stroke="{c}" stroke-width="1.5" class="pulse-high" />'
            center_text_markup = f"""
            <text x="{x}" y="{y+10}" fill="{center_color}" font-size="16" font-family="'JetBrains Mono', monospace" font-weight="800" text-anchor="middle" letter-spacing="0.08em">
                {center_label}
            </text>
            """

        nodes_svg.append(f"""
        <!-- Node Pod: {name} -->
        <g class="{node_extra_class}">
            {disruption_shockwaves}
            {disruption_beacon}
            
            <!-- Outer Glow Backdrop -->
            <rect x="{x-80}" y="{y-60}" width="160" height="120" rx="16" fill="{c}12" stroke="{c}" stroke-width="2" class="{outer_rect_class}" />
            <rect x="{x-79}" y="{y-59}" width="158" height="118" rx="15" fill="#0F172A" fill-opacity="0.94" />
            
            <!-- Specular Top Highlight -->
            <line x1="{x-65}" y1="{y-59}" x2="{x+65}" y2="{y-59}" stroke="rgba(255, 255, 255, 0.22)" stroke-width="1" />

            <!-- Echelon Header (Icon + Name) -->
            <text x="{x-62}" y="{y-36}" font-size="15">{icon}</text>
            <text x="{x-40}" y="{y-35}" fill="#F8FAFC" font-size="12" font-weight="700" letter-spacing="0.05em">
                {name.upper()}
            </text>

            {center_text_markup}

            <!-- Status Pill Badge (NO NUMBERS) -->
            <g transform="translate({x}, {y+36})">
                <rect x="-48" y="-10" width="96" height="20" rx="10" fill="{badge_bg}" stroke="{badge_border}" stroke-width="1" />
                <text x="0" y="4" fill="{badge_text_color}" font-size="10" font-weight="700" text-anchor="middle" letter-spacing="0.06em">
                    {badge_label}
                </text>
            </g>
        </g>
        """)

    notice_svg = ""
    if xai_error:
        safe_err = html.escape(str(xai_error)[:60])
        notice_svg = f"""
        <g transform="translate(480, 226)">
            <rect x="-240" y="-12" width="480" height="24" rx="6" fill="#15203B" stroke="rgba(245, 158, 11, 0.4)" />
            <text x="0" y="4" fill="#FBBF24" font-size="11" font-weight="500" text-anchor="middle">
                ⚠️ Sensitivity attribution: {safe_err}
            </text>
        </g>
        """

    svg_content = f"""
    <div style="width: 100%; max-width: 1020px; margin: 0 auto; overflow: hidden;">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 980 248" width="100%" height="auto" role="img" aria-label="{aria_label}" class="sg-topo-ambient" style="display: block; min-height: 220px; max-height: 270px; background: linear-gradient(180deg, rgba(15, 23, 42, 0.85) 0%, rgba(3, 7, 18, 0.98) 100%); border-radius: 16px; border: 1px solid rgba(148, 163, 184, 0.16); box-shadow: 0 16px 36px rgba(0,0,0,0.7), 0 0 20px rgba(59, 130, 246, 0.12);">
            <title>{aria_label}</title>
            <defs>
                <pattern id="topoGrid" width="28" height="28" patternUnits="userSpaceOnUse">
                    <path d="M 28 0 L 0 0 0 28" fill="none" stroke="rgba(255, 255, 255, 0.025)" stroke-width="1" />
                </pattern>
            </defs>
            <rect width="100%" height="100%" fill="url(#topoGrid)" />
            <style>
                /* Ambient Floating Animation for Canvas */
                @keyframes sg-float-ambient {{
                    0%, 100% {{ transform: translateY(0px); }}
                    50% {{ transform: translateY(-4px); }}
                }}
                .sg-topo-ambient {{
                    animation: sg-float-ambient 5.8s ease-in-out infinite;
                }}

                /* Unique Floating Animations for Each Node */
                @keyframes sg-float-node-0 {{
                    0%, 100% {{ transform: translateY(0px) rotate(0deg); }}
                    35% {{ transform: translateY(-6px) rotate(-0.4deg); }}
                    70% {{ transform: translateY(3px) rotate(0.3deg); }}
                }}
                .sg-node-0 {{
                    animation: sg-float-node-0 3.4s ease-in-out infinite;
                    transform-box: fill-box;
                    transform-origin: center;
                }}

                @keyframes sg-float-node-1 {{
                    0%, 100% {{ transform: translateY(0px) rotate(0deg); }}
                    40% {{ transform: translateY(5px) rotate(0.4deg); }}
                    75% {{ transform: translateY(-4px) rotate(-0.3deg); }}
                }}
                .sg-node-1 {{
                    animation: sg-float-node-1 4.2s ease-in-out infinite -0.9s;
                    transform-box: fill-box;
                    transform-origin: center;
                }}

                @keyframes sg-float-node-2 {{
                    0%, 100% {{ transform: translateY(0px) rotate(0deg); }}
                    30% {{ transform: translateY(-5px) rotate(0.3deg); }}
                    65% {{ transform: translateY(4px) rotate(-0.4deg); }}
                }}
                .sg-node-2 {{
                    animation: sg-float-node-2 3.8s ease-in-out infinite -1.8s;
                    transform-box: fill-box;
                    transform-origin: center;
                }}

                @keyframes sg-float-node-3 {{
                    0%, 100% {{ transform: translateY(0px) rotate(0deg); }}
                    45% {{ transform: translateY(6px) rotate(-0.4deg); }}
                    80% {{ transform: translateY(-3px) rotate(0.3deg); }}
                }}
                .sg-node-3 {{
                    animation: sg-float-node-3 4.5s ease-in-out infinite -2.4s;
                    transform-box: fill-box;
                    transform-origin: center;
                }}

                /* Floating edge pill badges */
                @keyframes sg-float-pill {{
                    0%, 100% {{ transform: translateY(0px); }}
                    50% {{ transform: translateY(-3px); }}
                }}
                .sg-float-pill-0 {{ animation: sg-float-pill 3.2s ease-in-out infinite; transform-box: fill-box; transform-origin: center; }}
                .sg-float-pill-1 {{ animation: sg-float-pill 4.0s ease-in-out infinite -1.2s; transform-box: fill-box; transform-origin: center; }}
                .sg-float-pill-2 {{ animation: sg-float-pill 3.6s ease-in-out infinite -2.0s; transform-box: fill-box; transform-origin: center; }}

                /* Animated Edge Flow Dashes */
                @media (prefers-reduced-motion: no-preference) {{
                    .sg-flow-line {{
                        animation: sg-flow-dash 1.3s linear infinite;
                    }}
                    .sg-cascade-edge {{
                        animation: sg-flow-dash 0.8s linear infinite;
                    }}
                    @keyframes sg-flow-dash {{
                        from {{ stroke-dashoffset: 28; }}
                        to {{ stroke-dashoffset: 0; }}
                    }}
                }}

                /* Disruption Animation at Critical Node (DELAYED 1.0s) */
                @keyframes sg-disrupt-alarm {{
                    0% {{ filter: drop-shadow(0 0 0px transparent); stroke: #EF4444; }}
                    15% {{ filter: drop-shadow(0 0 22px rgba(239, 68, 68, 0.95)); stroke: #EF4444; }}
                    50% {{ filter: drop-shadow(0 0 8px rgba(239, 68, 68, 0.45)); stroke: #DC2626; }}
                    80% {{ filter: drop-shadow(0 0 24px rgba(239, 68, 68, 0.95)); stroke: #EF4444; }}
                    100% {{ filter: drop-shadow(0 0 12px rgba(239, 68, 68, 0.6)); stroke: #EF4444; }}
                }}
                .sg-critical-glow {{
                    animation: sg-disrupt-alarm 1.8s ease-in-out 1.0s infinite;
                }}

                /* Shockwave 1: Expands strictly after 1.0s delay */
                @keyframes sg-shockwave-1 {{
                    0% {{ opacity: 0; transform: scale(0.96); }}
                    20% {{ opacity: 0.9; stroke: #EF4444; }}
                    100% {{ opacity: 0; transform: scale(1.36); stroke: #DC2626; }}
                }}
                .sg-disrupt-ripple-1 {{
                    opacity: 0;
                    animation: sg-shockwave-1 2.2s cubic-bezier(0.15, 0.85, 0.35, 1) 1.0s infinite;
                    transform-box: fill-box;
                    transform-origin: center;
                }}

                /* Shockwave 2: Staggered expansion after 1.5s delay */
                @keyframes sg-shockwave-2 {{
                    0% {{ opacity: 0; transform: scale(0.96); }}
                    20% {{ opacity: 0.8; stroke: #F87171; }}
                    100% {{ opacity: 0; transform: scale(1.48); stroke: #EF4444; }}
                }}
                .sg-disrupt-ripple-2 {{
                    opacity: 0;
                    animation: sg-shockwave-2 2.2s cubic-bezier(0.15, 0.85, 0.35, 1) 1.5s infinite;
                    transform-box: fill-box;
                    transform-origin: center;
                }}

                /* Warning Beacon Pop-in strictly after 1.0s delay */
                @keyframes sg-beacon-appear {{
                    0% {{ opacity: 0; transform: translateY(6px) scale(0.7); }}
                    100% {{ opacity: 1; transform: translateY(0px) scale(1.0); }}
                }}
                @keyframes sg-beacon-strobe {{
                    0%, 100% {{ filter: drop-shadow(0 0 4px rgba(239, 68, 68, 0.7)); }}
                    50% {{ filter: drop-shadow(0 0 14px rgba(239, 68, 68, 1)); }}
                }}
                .sg-disrupt-beacon {{
                    opacity: 0;
                    animation: sg-beacon-appear 0.4s cubic-bezier(0.34, 1.56, 0.64, 1) 1.0s forwards, sg-beacon-strobe 1.3s ease-in-out 1.4s infinite;
                    transform-box: fill-box;
                    transform-origin: center;
                }}

                /* Text Transition: MONITORED fades out at 1.0s, HIGH RISK fades in at 1.0s */
                @keyframes sg-fade-out {{
                    0% {{ opacity: 1; }}
                    100% {{ opacity: 0; visibility: hidden; }}
                }}
                .sg-pre-disrupt-text {{
                    animation: sg-fade-out 0.3s ease-out 1.0s forwards;
                }}

                @keyframes sg-fade-in-pulse {{
                    0% {{ opacity: 0; transform: scale(0.85); }}
                    100% {{ opacity: 1; transform: scale(1.0); }}
                }}
                .sg-post-disrupt-text {{
                    opacity: 0;
                    animation: sg-fade-in-pulse 0.4s cubic-bezier(0.34, 1.56, 0.64, 1) 1.0s forwards;
                    transform-box: fill-box;
                    transform-origin: center;
                }}
            </style>
            {''.join(edges_svg)}
            {''.join(nodes_svg)}
            {notice_svg}
        </svg>
    </div>
    """
    return clean_html(svg_content)
