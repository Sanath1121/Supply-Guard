"""60 FPS Hardware-Accelerated Animated Topology Canvas.

Renders an embedded HTML5/SVG canvas component with animated flowing particles
along supply chain edges, with speeds and widths parameterized by Integrated Gradients attribution.
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
    """Render the 60 FPS hardware-accelerated SVG particle flow canvas."""
    st.markdown("### 🌐 Dynamic Multi-Echelon Topology & Propagation Network")
    st.markdown(
        "Interactive directed graph showing physical material and risk flow from "
        "**Supplier** &rarr; **Manufacturer** &rarr; **Distributor** &rarr; **Retailer**. "
        "Edge flow particle velocities and thicknesses are dynamically driven by **Integrated Gradients upstream attribution**."
    )

    echelons = Config.NODE_NAMES
    tiers = [compute_tercile_tier(float(pred_risks[i])) for i in range(4)]
    colors = [get_tier_color(t) for t in tiers]

    # Attribution weights for edges: (S->M), (M->D), (D->R)
    # Default to balanced flow if not calculated
    w_sm = upstream_shares.get("Supplier->Manufacturer", 0.35)
    w_md = upstream_shares.get("Manufacturer->Distributor", 0.45)
    w_dr = upstream_shares.get("Distributor->Retailer", 0.20)

    # Particle speeds (lower duration = faster flow)
    speed_sm = max(0.8, 3.0 - (w_sm * 2.5))
    speed_md = max(0.8, 3.0 - (w_md * 2.5))
    speed_dr = max(0.8, 3.0 - (w_dr * 2.5))

    # Stroke widths
    stroke_sm = max(2, int(w_sm * 8))
    stroke_md = max(2, int(w_md * 8))
    stroke_dr = max(2, int(w_dr * 8))

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
      body {{
        margin: 0;
        padding: 0;
        background: transparent;
        font-family: 'Inter', -apple-system, sans-serif;
        color: #F1F5F9;
        overflow: hidden;
      }}
      .canvas-container {{
        background: radial-gradient(circle at 50% 50%, rgba(15, 23, 42, 0.95) 0%, rgba(7, 11, 20, 0.98) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.5);
        padding: 20px;
        position: relative;
      }}
      svg {{
        width: 100%;
        height: 240px;
      }}
      
      /* Edge Particle Flow Animations */
      .flow-edge {{
        stroke-dasharray: 8, 8;
        animation: flow-particles linear infinite;
      }}
      
      @keyframes flow-particles {{
        from {{ stroke-dashoffset: 64; }}
        to {{ stroke-dashoffset: 0; }}
      }}

      /* Node Halo Pulses */
      @keyframes pulse-ring {{
        0% {{ r: 38; opacity: 0.8; }}
        50% {{ r: 48; opacity: 0.2; }}
        100% {{ r: 38; opacity: 0.8; }}
      }}

      .pulse-circle {{
        animation: pulse-ring 2.2s infinite ease-in-out;
        transform-origin: center;
      }}

      .node-card {{
        cursor: pointer;
        transition: transform 0.2s ease;
      }}
      .node-card:hover {{
        transform: scale(1.05);
      }}
    </style>
    </head>
    <body>
      <div class="canvas-container">
        <svg viewBox="0 0 900 240">
          <defs>
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
            
            <marker id="arrow" viewBox="0 0 10 10" refX="28" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 1 L 10 5 L 0 9 z" fill="#06B6D4" />
            </marker>
          </defs>

          <!-- Edge 1: Supplier -> Manufacturer -->
          <line x1="120" y1="120" x2="340" y2="120" stroke="rgba(255,255,255,0.1)" stroke-width="{stroke_sm}" />
          <line x1="120" y1="120" x2="340" y2="120" stroke="url(#grad-sm)" stroke-width="{stroke_sm}" 
                class="flow-edge" style="animation-duration: {speed_sm:.1f}s;" marker-end="url(#arrow)" />
          
          <!-- Edge 2: Manufacturer -> Distributor -->
          <line x1="340" y1="120" x2="560" y2="120" stroke="rgba(255,255,255,0.1)" stroke-width="{stroke_md}" />
          <line x1="340" y1="120" x2="560" y2="120" stroke="url(#grad-md)" stroke-width="{stroke_md}" 
                class="flow-edge" style="animation-duration: {speed_md:.1f}s;" marker-end="url(#arrow)" />

          <!-- Edge 3: Distributor -> Retailer -->
          <line x1="560" y1="120" x2="780" y2="120" stroke="rgba(255,255,255,0.1)" stroke-width="{stroke_dr}" />
          <line x1="560" y1="120" x2="780" y2="120" stroke="url(#grad-dr)" stroke-width="{stroke_dr}" 
                class="flow-edge" style="animation-duration: {speed_dr:.1f}s;" marker-end="url(#arrow)" />

          <!-- Upstream Attribution Labels -->
          <text x="230" y="105" text-anchor="middle" fill="#06B6D4" font-size="11" font-weight="600">
            Attr: {w_sm*100:.0f}%
          </text>
          <text x="450" y="105" text-anchor="middle" fill="#06B6D4" font-size="11" font-weight="600">
            Attr: {w_md*100:.0f}%
          </text>
          <text x="670" y="105" text-anchor="middle" fill="#06B6D4" font-size="11" font-weight="600">
            Attr: {w_dr*100:.0f}%
          </text>

          <!-- Node 0: Supplier -->
          <g class="node-card" transform="translate(120, 120)">
            <circle cx="0" cy="0" r="38" fill="none" stroke="{colors[0]}" stroke-width="2" opacity="0.4" class="pulse-circle" />
            <circle cx="0" cy="0" r="34" fill="#0F172A" stroke="{colors[0]}" stroke-width="3" />
            <text x="0" y="-8" text-anchor="middle" fill="#94A3B8" font-size="10" font-weight="600">SUPPLIER</text>
            <text x="0" y="14" text-anchor="middle" fill="{colors[0]}" font-size="16" font-weight="800">{pred_risks[0]:.2f}</text>
            <text x="0" y="48" text-anchor="middle" fill="#64748B" font-size="10">{tiers[0]}</text>
          </g>

          <!-- Node 1: Manufacturer -->
          <g class="node-card" transform="translate(340, 120)">
            <circle cx="0" cy="0" r="38" fill="none" stroke="{colors[1]}" stroke-width="2" opacity="0.4" class="pulse-circle" />
            <circle cx="0" cy="0" r="34" fill="#0F172A" stroke="{colors[1]}" stroke-width="3" />
            <text x="0" y="-8" text-anchor="middle" fill="#94A3B8" font-size="10" font-weight="600">MANUFACTURER</text>
            <text x="0" y="14" text-anchor="middle" fill="{colors[1]}" font-size="16" font-weight="800">{pred_risks[1]:.2f}</text>
            <text x="0" y="48" text-anchor="middle" fill="#64748B" font-size="10">{tiers[1]}</text>
          </g>

          <!-- Node 2: Distributor -->
          <g class="node-card" transform="translate(560, 120)">
            <circle cx="0" cy="0" r="38" fill="none" stroke="{colors[2]}" stroke-width="2" opacity="0.4" class="pulse-circle" />
            <circle cx="0" cy="0" r="34" fill="#0F172A" stroke="{colors[2]}" stroke-width="3" />
            <text x="0" y="-8" text-anchor="middle" fill="#94A3B8" font-size="10" font-weight="600">DISTRIBUTOR</text>
            <text x="0" y="14" text-anchor="middle" fill="{colors[2]}" font-size="16" font-weight="800">{pred_risks[2]:.2f}</text>
            <text x="0" y="48" text-anchor="middle" fill="#64748B" font-size="10">{tiers[2]}</text>
          </g>

          <!-- Node 3: Retailer -->
          <g class="node-card" transform="translate(780, 120)">
            <circle cx="0" cy="0" r="38" fill="none" stroke="{colors[3]}" stroke-width="2" opacity="0.4" class="pulse-circle" />
            <circle cx="0" cy="0" r="34" fill="#0F172A" stroke="{colors[3]}" stroke-width="3" />
            <text x="0" y="-8" text-anchor="middle" fill="#94A3B8" font-size="10" font-weight="600">RETAILER</text>
            <text x="0" y="14" text-anchor="middle" fill="{colors[3]}" font-size="16" font-weight="800">{pred_risks[3]:.2f}</text>
            <text x="0" y="48" text-anchor="middle" fill="#64748B" font-size="10">{tiers[3]}</text>
          </g>
        </svg>
      </div>
    </body>
    </html>
    """

    components.html(html_code, height=270)
