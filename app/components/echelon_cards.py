"""Echelon Operational Risk Health Cards Component.

Displays the 4 dedicated echelon cards (Supplier, Manufacturer, Distributor, Retailer)
with cyber glassmorphism, bold telemetry metrics, unscaled physical values, and prescriptive guidance.
"""
from typing import Dict, Any, List
import streamlit as st
import numpy as np

from app.utils.formatters import format_echelon_card, compute_tercile_tier, get_tier_color
from src.config import Config


PRESCRIPTIVE_GUIDANCE = {
    "Supplier": {
        "High": "🚨 <strong>Action:</strong> Trigger secondary supplier buffer contracts; expedite tier-2 raw material shipment audits.",
        "Medium": "⚠️ <strong>Notice:</strong> Monitor supplier lead-time variance; prepare contingency purchase orders.",
        "Low": "✅ <strong>Nominal:</strong> Supplier sourcing lead times and quality defect rates are stable."
    },
    "Manufacturer": {
        "High": "🚨 <strong>Action:</strong> Rebalance assembly lines; throttle high-defect SKUs by 15%; prioritize WIP buffers.",
        "Medium": "⚠️ <strong>Notice:</strong> Plant capacity approaching 88%; schedule preventative maintenance window.",
        "Low": "✅ <strong>Nominal:</strong> Factory throughput and yield metrics operating within optimal parameters."
    },
    "Distributor": {
        "High": "🚨 <strong>Action:</strong> Reroute regional freight via backup 3PL carriers; activate expedited logistics lanes.",
        "Medium": "⚠️ <strong>Notice:</strong> Warehouse dwell time elevated in hub 4; assess carrier surcharge risk.",
        "Low": "✅ <strong>Nominal:</strong> Distribution fulfillment and freight transit schedules on track."
    },
    "Retailer": {
        "High": "🚨 <strong>Action:</strong> Enforce store stock allocation quotas; dynamically elevate safety stock algorithms.",
        "Medium": "⚠️ <strong>Notice:</strong> Volatile point-of-sale demand spike detected; monitor stockout rate.",
        "Low": "✅ <strong>Nominal:</strong> Store shelf fill-rate and consumer demand elasticity within baseline."
    }
}

ECHELON_ICONS = {
    "Supplier": "📦",
    "Manufacturer": "⚙️",
    "Distributor": "🚚",
    "Retailer": "🛒"
}


def render_echelon_cards(
    pred_risks: np.ndarray,
    raw_risks: Dict[str, float],
    raw_cost: float,
    prev_step_risks: np.ndarray,
    selected_node_idx: int = 0
) -> int:
    """Render the 4 echelon cards in a clean 4-column responsive grid."""
    st.markdown("""
    <div style="margin: 20px 0 10px 0;">
        <h3 style="margin: 0; font-family: 'Outfit'; font-size: 1.35rem; color: #FFFFFF; font-weight: 800;">
            📊 Multi-Echelon Risk Health Cards
        </h3>
        <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 0.85rem;">
            Individual forward-forecasted risk indices with train-tercile severity badges and unscaled real-world units.
        </p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(4)
    echelons = Config.NODE_NAMES
    updated_selection = selected_node_idx

    tier_palette = {
        "Low": "#00F5A0",
        "Medium": "#FFB300",
        "High": "#FF2E54"
    }

    for i, name in enumerate(echelons):
        scaled_val = float(pred_risks[i])
        raw_val = raw_risks.get(name, scaled_val)
        tier = compute_tercile_tier(scaled_val)
        color = tier_palette.get(tier, "#00F2FE")

        # Delta trend from past step t
        prev_val = float(prev_step_risks[i]) if len(prev_step_risks) > i else scaled_val
        delta = scaled_val - prev_val
        trend_symbol = "▲" if delta > 0.01 else ("▼" if delta < -0.01 else "▬")
        trend_color = "#FF2E54" if delta > 0.01 else ("#00F5A0" if delta < -0.01 else "#94A3B8")

        # Visual styling
        is_selected = (selected_node_idx == i)
        glow_border = "border: 2px solid #00F2FE; box-shadow: 0 0 25px rgba(0, 242, 254, 0.3);" if is_selected else ""
        card_class = f"sg-glass-card echelon-{tier.lower()[:3]}"
        icon = ECHELON_ICONS.get(name, "📍")

        with cols[i]:
            st.markdown(f"""
            <div class="{card_class}" style="{glow_border}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div style="display: flex; align-items: center; gap: 6px;">
                        <span style="font-size: 1.1rem;">{icon}</span>
                        <span style="font-size: 0.95rem; font-weight: 800; color: #FFFFFF; font-family: 'Outfit';">
                            {name}
                        </span>
                    </div>
                    <span class="badge-pill badge-{tier.lower()}">
                        {tier}
                    </span>
                </div>
                <div style="display: flex; align-items: baseline; gap: 8px; margin-bottom: 6px;">
                    <span style="font-size: 2.2rem; font-weight: 900; font-family: 'Outfit'; color: {color}; letter-spacing: -0.02em;">
                        {scaled_val:.3f}
                    </span>
                    <span style="font-size: 0.85rem; color: {trend_color}; font-weight: 700; font-family: 'JetBrains Mono';">
                        {trend_symbol} {abs(delta):.3f} &Delta;
                    </span>
                </div>
                <div style="background: rgba(11, 15, 25, 0.6); padding: 6px 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); margin-bottom: 8px;">
                    <span style="font-size: 0.74rem; color: #94A3B8;">Raw Physical Unit:</span>
                    <strong style="color: #FFFFFF; font-size: 0.82rem; font-family: 'JetBrains Mono'; float: right;">{raw_val:.2f} RI</strong>
                </div>
                <div style="font-size: 0.7rem; color: #64748B;">
                    Threshold: &lt;0.35 Low | 0.35-0.65 Med | &gt;0.65 High
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"{'🎯 Inspected' if is_selected else '🔍 Inspect'} {name}", key=f"btn_inspect_{name}", use_container_width=True):
                updated_selection = i

    # Prescriptive guidance alert box
    active_node = echelons[updated_selection]
    active_tier = compute_tercile_tier(float(pred_risks[updated_selection]))
    guidance_msg = PRESCRIPTIVE_GUIDANCE.get(active_node, {}).get(active_tier, "")

    st.markdown(f"""
    <div style="background: rgba(15, 23, 42, 0.85); border-left: 4px solid #00F2FE; border-radius: 10px; padding: 12px 16px; margin-top: 15px; border-top: 1px solid rgba(255,255,255,0.06); border-right: 1px solid rgba(255,255,255,0.06); border-bottom: 1px solid rgba(255,255,255,0.06);">
        <span style="font-size: 0.85rem; color: #00F2FE; font-weight: 700;">DECISION SUPPORT PROTOCOL &bull; [{active_node.upper()}]:</span><br>
        <span style="color: #E2E8F0; font-size: 0.88rem; line-height: 1.5;">{guidance_msg}</span>
    </div>
    """, unsafe_allow_html=True)

    return updated_selection
