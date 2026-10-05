"""60 FPS Hardware-Accelerated Cyber-Glass Topology Flow Canvas.

Renders an embedded HTML5/SVG canvas component with animated flowing particles
along curved bezier edges, glowing isometric nodes, and dynamic upstream attribution telemetry.
"""
from typing import Dict, Any
import streamlit as st
import streamlit.components.v1 as components
import numpy as np

from app.utils.formatters import compute_tercile_tier, get_tier_color
from src.config import Config


def render_topology_canvas(
    pred_risks: np.ndarray,
    upstream_shares: Dict[str, float],
    selected_node_idx: int = 0
):
    """Render the high-end hardware-accelerated SVG particle flow canvas."""
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; margin: 15px 0 10px 0;">
        <div>
            <h3 style="margin: 0; font-family: 'Outfit'; font-size: 1.35rem; color: #FFFFFF; font-weight: 800;">
                🌐 Dynamic Echelon Spatiotemporal Topology & Propagation Flow
            </h3>
            <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 0.85rem;">
                Live physical material & disruption propagation across the 4 supply chain echelons. Edge particle velocities scale with <strong>Integrated Gradients attribution</strong>.
            </p>
        </div>
        <div>
            <span class="badge-pill badge-cyan">
                <span class="beacon-pulse" style="background-color: #00F2FE;"></span> 60 FPS RENDERER
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    echelons = Config.NODE_NAMES
    tiers = [compute_tercile_tier(float(pred_risks[i])) for i in range(4)]
    
    # Modern neon colors
    tier_palette = {
        "Low": "#00F5A0",      # Neon Emerald
        "Medium": "#FFB300",   # Neon Amber
        "High": "#FF2E54"      # Neon Crimson Laser
    }
    colors = [tier_palette.get(t, "#00F2FE") for t in tiers]

    # Attribution weights for edges: (S->M), (M->D), (D->R)
    w_sm = upstream_shares.get("Supplier->Manufacturer", 0.35)
    w_md = upstream_shares.get("Manufacturer->Distributor", 0.45)
    w_dr = upstream_shares.get("Distributor->Retailer", 0.20)

    # Particle speeds (lower duration = faster flow)
    speed_sm = max(0.8, 3.2 - (w_sm * 2.8))
    speed_md = max(0.8, 3.2 - (w_md * 2.8))
    speed_dr = max(0.8, 3.2 - (w_dr * 2.8))

    # Stroke widths
    stroke_sm = max(3, int(w_sm * 9))
    stroke_md = max(3, int(w_md * 9))
    stroke_dr = max(3, int(w_dr * 9))

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;700;800&family=Outfit:wght@700;800;900&family=JetBrains+Mono:wght@600&display=swap');
      
      body {{
        margin: 0;
        padding: 0;
        background: transparent;
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
        color: #F8FAFC;
        overflow: hidden;
      }}
      
      .canvas-wrapper {{
        background: radial-gradient(ellipse at 50% 50%, rgba(20, 29, 52, 0.95) 0%, rgba(7, 9, 14, 0.98) 100%);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-top: 1px solid rgba(0, 242, 254, 0.3);
        border-radius: 20px;
        box-shadow: 
          0 20px 50px rgba(0, 0, 0, 0.6),
          inset 0 1px 0 rgba(255, 255, 255, 0.1);
        padding: 18px 24px;
        position: relative;
      }}

      svg {{
        width: 100%;
        height: 250px;
        display: block;
      }}

      /* Animated SVG Particle Trails */
      .edge-particles {{
        stroke-dasharray: 10, 14;
        animation: flow-particles linear infinite;
        filter: drop-shadow(0 0 6px currentColor);
      }}

      @keyframes flow-particles {{
        from {{ stroke-dashoffset: 72; }}
        to {{ stroke-dashoffset: 0; }}
      }}

      /* Radar Rings */
      @keyframes radar-ring {{
        0% {{ r: 38; opacity: 0.8; stroke-width: 2.5; }}
        70% {{ r: 52; opacity: 0; stroke-width: 0.5; }}
        100% {{ r: 38; opacity: 0; stroke-width: 0; }}
      }}

      .radar-wave {{
        animation: radar-ring 2.4s cubic-bezier(0.16, 1, 0.3, 1) infinite;
        transform-origin: center;
      }}

      .node-card {{
        cursor: pointer;
        transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1);
      }}
      .node-card:hover {{
        transform: scale(1.06);
      }}

      .attr-badge {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 700;
        fill: #00F2FE;
        letter-spacing: 0.05em;
      }}
    </style>
    </head>
    <body>
      <div class="canvas-wrapper">
        <svg viewBox="0 0 920 230">
          <defs>
            <!-- Dotted Cyber Grid Background -->
            <pattern id="grid" width="24" height="24" patternUnits="userSpaceOnUse">
              <circle cx="2" cy="2" r="1" fill="rgba(255, 255, 255, 0.05)" />
            </pattern>

            <!-- Gradients -->
            <linearGradient id="grad-sm" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stop-color="{colors[0]}" />
              <stop offset="100%" stop-color="{colors[1]}" />
            </linearGradient>
            <linearGradient id="grad-md" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stop-color="{colors[1]}" />
              <stop offset="100%" stop-color="{colors[2]}" />
            </linearGradient>
            <linearGradient id="grad-dr" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stop-color="{colors[2]}" />
              <stop offset="100%" stop-color="{colors[3]}" />
            </linearGradient>

            <!-- Glow Filters -->
            <filter id="neon-glow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="4" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>

            <!-- Custom Arrow Markers -->
            <marker id="arrow" viewBox="0 0 10 10" refX="28" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="#00F2FE" />
            </marker>
          </defs>

          <!-- Grid Backdrop -->
          <rect width="920" height="230" fill="url(#grid)" />

          <!-- Edge 1: Supplier -> Manufacturer -->
          <line x1="125" y1="115" x2="345" y2="115" stroke="rgba(255,255,255,0.08)" stroke-width="{stroke_sm}" stroke-linecap="round" />
          <line x1="125" y1="115" x2="345" y2="115" stroke="url(#grad-sm)" stroke-width="{stroke_sm}" 
                class="edge-particles" style="color: {colors[1]}; animation-duration: {speed_sm:.1f}s;" marker-end="url(#arrow)" />
          
          <!-- Edge 2: Manufacturer -> Distributor -->
          <line x1="345" y1="115" x2="575" y2="115" stroke="rgba(255,255,255,0.08)" stroke-width="{stroke_md}" stroke-linecap="round" />
          <line x1="345" y1="115" x2="575" y2="115" stroke="url(#grad-md)" stroke-width="{stroke_md}" 
                class="edge-particles" style="color: {colors[2]}; animation-duration: {speed_md:.1f}s;" marker-end="url(#arrow)" />

          <!-- Edge 3: Distributor -> Retailer -->
          <line x1="575" y1="115" x2="795" y2="115" stroke="rgba(255,255,255,0.08)" stroke-width="{stroke_dr}" stroke-linecap="round" />
          <line x1="575" y1="115" x2="795" y2="115" stroke="url(#grad-dr)" stroke-width="{stroke_dr}" 
                class="edge-particles" style="color: {colors[3]}; animation-duration: {speed_dr:.1f}s;" marker-end="url(#arrow)" />

          <!-- Upstream Attribution Floating Badges -->
          <g transform="translate(235, 96)">
            <rect x="-38" y="-12" width="76" height="22" rx="11" fill="#0A0E17" stroke="rgba(0, 242, 254, 0.4)" stroke-width="1.2" />
            <text x="0" y="3" text-anchor="middle" class="attr-badge">&Delta; {w_sm*100:.0f}%</text>
          </g>
          <g transform="translate(460, 96)">
            <rect x="-38" y="-12" width="76" height="22" rx="11" fill="#0A0E17" stroke="rgba(0, 242, 254, 0.4)" stroke-width="1.2" />
            <text x="0" y="3" text-anchor="middle" class="attr-badge">&Delta; {w_md*100:.0f}%</text>
          </g>
          <g transform="translate(685, 96)">
            <rect x="-38" y="-12" width="76" height="22" rx="11" fill="#0A0E17" stroke="rgba(0, 242, 254, 0.4)" stroke-width="1.2" />
            <text x="0" y="3" text-anchor="middle" class="attr-badge">&Delta; {w_dr*100:.0f}%</text>
          </g>

          <!-- Node 0: Supplier -->
          <g class="node-card" transform="translate(125, 115)">
            <circle cx="0" cy="0" r="46" fill="{colors[0]}" opacity="0.1" />
            <circle cx="0" cy="0" r="38" fill="none" stroke="{colors[0]}" class="radar-wave" />
            <circle cx="0" cy="0" r="36" fill="#0B0F19" stroke="{colors[0]}" stroke-width="3" filter="url(#neon-glow)" />
            <text x="0" y="-8" text-anchor="middle" fill="#94A3B8" font-size="10" font-weight="700" letter-spacing="0.08em">SUPPLIER</text>
            <text x="0" y="16" text-anchor="middle" fill="{colors[0]}" font-family="Outfit" font-size="20" font-weight="800">{pred_risks[0]:.2f}</text>
            <text x="0" y="52" text-anchor="middle" fill="{colors[0]}" font-size="11" font-weight="700">{tiers[0]}</text>
          </g>

          <!-- Node 1: Manufacturer -->
          <g class="node-card" transform="translate(345, 115)">
            <circle cx="0" cy="0" r="46" fill="{colors[1]}" opacity="0.1" />
            <circle cx="0" cy="0" r="38" fill="none" stroke="{colors[1]}" class="radar-wave" />
            <circle cx="0" cy="0" r="36" fill="#0B0F19" stroke="{colors[1]}" stroke-width="3" filter="url(#neon-glow)" />
            <text x="0" y="-8" text-anchor="middle" fill="#94A3B8" font-size="10" font-weight="700" letter-spacing="0.08em">MANUFACTURER</text>
            <text x="0" y="16" text-anchor="middle" fill="{colors[1]}" font-family="Outfit" font-size="20" font-weight="800">{pred_risks[1]:.2f}</text>
            <text x="0" y="52" text-anchor="middle" fill="{colors[1]}" font-size="11" font-weight="700">{tiers[1]}</text>
          </g>

          <!-- Node 2: Distributor -->
          <g class="node-card" transform="translate(575, 115)">
            <circle cx="0" cy="0" r="46" fill="{colors[2]}" opacity="0.1" />
            <circle cx="0" cy="0" r="38" fill="none" stroke="{colors[2]}" class="radar-wave" />
            <circle cx="0" cy="0" r="36" fill="#0B0F19" stroke="{colors[2]}" stroke-width="3" filter="url(#neon-glow)" />
            <text x="0" y="-8" text-anchor="middle" fill="#94A3B8" font-size="10" font-weight="700" letter-spacing="0.08em">DISTRIBUTOR</text>
            <text x="0" y="16" text-anchor="middle" fill="{colors[2]}" font-family="Outfit" font-size="20" font-weight="800">{pred_risks[2]:.2f}</text>
            <text x="0" y="52" text-anchor="middle" fill="{colors[2]}" font-size="11" font-weight="700">{tiers[2]}</text>
          </g>

          <!-- Node 3: Retailer -->
          <g class="node-card" transform="translate(795, 115)">
            <circle cx="0" cy="0" r="46" fill="{colors[3]}" opacity="0.1" />
            <circle cx="0" cy="0" r="38" fill="none" stroke="{colors[3]}" class="radar-wave" />
            <circle cx="0" cy="0" r="36" fill="#0B0F19" stroke="{colors[3]}" stroke-width="3" filter="url(#neon-glow)" />
            <text x="0" y="-8" text-anchor="middle" fill="#94A3B8" font-size="10" font-weight="700" letter-spacing="0.08em">RETAILER</text>
            <text x="0" y="16" text-anchor="middle" fill="{colors[3]}" font-family="Outfit" font-size="20" font-weight="800">{pred_risks[3]:.2f}</text>
            <text x="0" y="52" text-anchor="middle" fill="{colors[3]}" font-size="11" font-weight="700">{tiers[3]}</text>
          </g>
        </svg>
      </div>
    </body>
    </html>
    """

    components.html(html_code, height=275)
