"""Echelon Risk Health Cards Component.

Displays the 4 dedicated echelon cards (Supplier, Manufacturer, Distributor, Retailer)
with dual-metric display, tercile badges, trend arrows, and prescriptive action recommendations.
Directly compliant with tests/test_phase7_app.py.
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
        "Low": "✅ <strong>Nominal:</strong> Supplier sourcing lead times and quality rates are stable."
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


def render_echelon_cards(
    pred_risks: np.ndarray,
    raw_risks: Dict[str, float],
    raw_cost: float,
    prev_step_risks: np.ndarray,
    selected_node_idx: int = 0
) -> int:
    """Render the 4 echelon cards in a clean 4-column responsive grid."""
    st.markdown("### 📊 Echelon Operational Risk Health Cards")
    cols = st.columns(4)
    echelons = Config.NODE_NAMES
    updated_selection = selected_node_idx

    for i, name in enumerate(echelons):
        scaled_val = float(pred_risks[i])
        raw_val = raw_risks.get(name, scaled_val)
        tier = compute_tercile_tier(scaled_val)
        card_data = format_echelon_card(name, scaled_val, raw_val, tier)

        # Delta trend from past step t
        prev_val = float(prev_step_risks[i]) if len(prev_step_risks) > i else scaled_val
        delta = scaled_val - prev_val
        trend_symbol = "▲" if delta > 0.01 else ("▼" if delta < -0.01 else "▬")
        trend_color = "#EF4444" if delta > 0.01 else ("#10B981" if delta < -0.01 else "#94A3B8")

        # Visual styling
        card_class = f"sg-card sg-card-{tier.lower()}"
        is_selected = (selected_node_idx == i)
        border_highlight = "border: 2px solid #06B6D4;" if is_selected else ""

        with cols[i]:
            st.markdown(f"""
            <div class="{card_class}" style="{border_highlight}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 0.85rem; font-weight: 700; color: #FFFFFF;">
                        {name}
                    </span>
                    <span class="badge-pill badge-{tier.lower()}">
                        {tier}
                    </span>
                </div>
                <div style="display: flex; align-items: baseline; gap: 8px; margin-bottom: 4px;">
                    <span style="font-size: 1.8rem; font-weight: 800; font-family: 'Outfit'; color: {card_data['color']};">
                        {card_data['scaled_score']}
                    </span>
                    <span style="font-size: 0.85rem; color: {trend_color}; font-weight: 600;">
                        {trend_symbol} {abs(delta):.3f} &Delta;
                    </span>
                </div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 8px;">
                    Raw Physical Unit: <strong style="color: #F1F5F9;">{card_data['raw_metric']}</strong>
                </div>
                <div style="font-size: 0.72rem; color: #64748B; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 6px;">
                    Tercile Threshold: Low &lt;0.35 | Med 0.35-0.65 | High &gt;0.65
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"🔍 Inspect {name}", key=f"btn_inspect_{name}", use_container_width=True):
                updated_selection = i

    # Display prescriptive guidance for selected node
    active_node = echelons[updated_selection]
    active_tier = compute_tercile_tier(float(pred_risks[updated_selection]))
    guidance_msg = PRESCRIPTIVE_GUIDANCE.get(active_node, {}).get(active_tier, "")

    st.markdown(f"""
    <div class="prescriptive-box">
        <strong>Decision Support Guidance for [{active_node}]:</strong><br>
        {guidance_msg}
    </div>
    """, unsafe_allow_html=True)

    return updated_selection
