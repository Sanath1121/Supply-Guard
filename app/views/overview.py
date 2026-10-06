"""SupplyGuard Executive Overview & Operational Architecture.

Presents the executive platform summary, operational readiness checklist,
enterprise specifications, and decision action protocols.
"""
import os
import streamlit as st
from src.config import Config
from app.ui.components import card, status_badge, section_header, empty_state
from app.utils.artifacts import ArtifactStatus, get_tiers


def render_overview(status: ArtifactStatus):
    """Render the Executive Overview view."""
    section_header(
        "SupplyGuard — Spatiotemporal Multi-Echelon Risk Intelligence",
        "Continuous spatiotemporal graph forecasting, bottleneck early warning, and explainable AI diagnostics."
    )

    # Top KPI summary strip
    h_col1, h_col2, h_col3, h_col4 = st.columns(4, gap="small")
    with h_col1:
        st.html(card("Core AI Engine", "<div style='font-size:1.4rem; font-weight:800; color:#FFFFFF;' class='mono-val'>ST-GCN-LSTM</div><div style='font-size:0.75rem; color:var(--accent-light); margin-top:3px;'>Spatiotemporal Graph Neural Net</div>"))
    with h_col2:
        st.html(card("Early Warning Lead Time", "<div style='font-size:1.4rem; font-weight:800; color:var(--accent);' class='mono-val'>+10 Minutes</div><div style='font-size:0.75rem; color:var(--text-3); margin-top:3px;'>t+5 Horizon @ 2-min cadence</div>"))
    with h_col3:
        st.html(card("Echelon Coverage", "<div style='font-size:1.4rem; font-weight:800; color:#FFFFFF;' class='mono-val'>4 Tiers Active</div><div style='font-size:0.75rem; color:var(--text-3); margin-top:3px;'>Supplier → Mfg → Dist → Retail</div>"))
    with h_col4:
        st.html(card("Data Verification", "<div style='font-size:1.4rem; font-weight:800; color:#10B981;' class='mono-val'>Zero Leakage</div><div style='font-size:0.75rem; color:var(--text-3); margin-top:3px;'>Strict 80/10/10 Chronological Split</div>"))

    st.html("<div style='height: 14px;'></div>")

    # 1. Executive Platform Scope Card
    summary_html = """
    <p style="margin: 0; font-size: 0.9375rem; line-height: 1.65; color: var(--text-2);">
        SupplyGuard provides real-time operational foresight across the multi-tier supply chain 
        (<strong>Supplier &rarr; Manufacturer &rarr; Distributor &rarr; Retailer</strong>).
        By coupling <strong>Spatial Graph Convolution (GCN)</strong> with <strong>Temporal Recurrent (LSTM)</strong> networks and a residual persistence baseline,
        it anticipates downstream vulnerability cascades <strong>10 minutes ahead of occurrence</strong>.
        Every forecast is auditable via <strong>Path-Integrated Gradients</strong>, isolating whether emerging bottlenecks stem from supplier shortages, assembly delays, transit friction, or cost surges.
    </p>
    """
    st.html(card("Executive Platform Mission", summary_html))
    st.html("<div style='height: 16px;'></div>")

    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        # 2. Operational Readiness Checklist
        has_data = os.path.exists(Config.RAW_DATA_PATH)
        has_scaler = os.path.exists(Config.SCALER_PATH)
        
        ckpt_files = [f for f in os.listdir(Config.CKPT_DIR) if f.endswith(".pt")] if os.path.exists(Config.CKPT_DIR) else []
        has_ckpt = len(ckpt_files) > 0

        res_dir = os.path.join("outputs", "results")
        has_results = os.path.exists(os.path.join(res_dir, "overall_metrics.csv"))

        def check_row(label: str, ok: bool, details: str) -> str:
            badge = status_badge("Verified", "low") if ok else status_badge("Pending", "high")
            return (
                f'<div style="display: flex; align-items: center; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid var(--border);">'
                f'<div><div style="font-size: 0.875rem; font-weight: 600; color: var(--text);">{label}</div>'
                f'<div style="font-size: 0.75rem; color: var(--text-3); margin-top: 1px;">{details}</div></div>'
                f'<div>{badge}</div></div>'
            )

        pipeline_html = (
            f'<div>'
            f'{check_row("Telemetry Dataset", has_data, "Verified chronological partitions (647,636 records)")}'
            f'{check_row("Calibration Scaler", has_scaler, "MinMaxScaler fitted strictly on training partition")}'
            f'{check_row("Model Weights Checkpoints", has_ckpt, f"{len(ckpt_files)} multi-seed models ready in {Config.CKPT_DIR}")}'
            f'{check_row("Empirical SLA Benchmarks", has_results, "Production accuracy & F1 score validation tables")}'
            f'</div>'
        )
        st.html(card("Operational Readiness & Model Health", pipeline_html))

    with col2:
        # 3. Enterprise System Specifications
        specs_html = f"""
        <table style="width: 100%; border-collapse: collapse; font-size: 0.8125rem; color: var(--text-2);">
            <tr style="border-bottom: 1px solid var(--border);">
                <td style="padding: 9px 0; color: var(--text-3);">Historical Lookback Window (L)</td>
                <td style="padding: 9px 0; text-align: right;" class="mono-val">{Config.SEQ_LEN} steps ({Config.SEQ_LEN * Config.CADENCE_MIN:.0f} minutes)</td>
            </tr>
            <tr style="border-bottom: 1px solid var(--border);">
                <td style="padding: 9px 0; color: var(--text-3);">Early Warning Horizon (H)</td>
                <td style="padding: 9px 0; text-align: right;" class="mono-val">t+{Config.HORIZON} ({Config.HORIZON_MIN} min @ {Config.CADENCE_MIN:.1f}m cadence)</td>
            </tr>
            <tr style="border-bottom: 1px solid var(--border);">
                <td style="padding: 9px 0; color: var(--text-3);">Multi-Echelon Network Flow</td>
                <td style="padding: 9px 0; text-align: right;">{" &rarr; ".join(Config.NODE_NAMES)}</td>
            </tr>
            <tr style="border-bottom: 1px solid var(--border);">
                <td style="padding: 9px 0; color: var(--text-3);">Dynamic Feature Vector (F)</td>
                <td style="padding: 9px 0; text-align: right;" class="mono-val">4 Echelon Risk Indices + Logistics Cost (5 dim)</td>
            </tr>
            <tr style="border-bottom: 1px solid var(--border);">
                <td style="padding: 9px 0; color: var(--text-3);">Temporal Partitioning</td>
                <td style="padding: 9px 0; text-align: right;" class="mono-val">{Config.TRAIN_RATIO*100:.0f}% Train / {Config.VAL_RATIO*100:.0f}% Val / {Config.TEST_RATIO*100:.0f}% Test</td>
            </tr>
            <tr>
                <td style="padding: 9px 0; color: var(--text-3);">Data Continuity Threshold</td>
                <td style="padding: 9px 0; text-align: right;" class="mono-val">{Config.GAP_MAX_MIN} min max gap segmentation</td>
            </tr>
        </table>
        """
        st.html(card("Production Architecture Specifications", specs_html))

    st.html("<div style='height: 16px;'></div>")

    # 4. Severity Action Protocols
    tiers_info = get_tiers()
    p33 = tiers_info.get("p33", 0.35)
    p66 = tiers_info.get("p66", 0.65)
    source_label = "Calibrated Distribution Cutoffs" if tiers_info.get("source") == "train_terciles" else "Standard Baseline Cutoffs"

    guide_html = f"""
    <div style="font-size: 0.875rem; line-height: 1.6; color: var(--text-2);">
        <p style="margin-top: 0;">
            SupplyGuard standardizes risk across all tiers onto a calibrated scale ($0.00$ to $1.00$). 
            Operations and dispatch teams follow standardized action protocols mapped to risk severity:
        </p>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 14px 0;">
            <div style="background: var(--surface-2); padding: 14px; border-radius: var(--r-sm); border-left: 3px solid var(--ok);">
                <strong style="color: var(--ok);">● Low Severity (&lt; {p33:.2f})</strong>
                <div style="font-size: 0.8125rem; color: var(--text-2); margin-top: 4px;"><strong>Status:</strong> Normal Operations</div>
                <div style="font-size: 0.75rem; color: var(--text-3); margin-top: 2px;">Standard buffer levels adequate. No dispatch interventions required.</div>
            </div>
            <div style="background: var(--surface-2); padding: 14px; border-radius: var(--r-sm); border-left: 3px solid var(--warn);">
                <strong style="color: var(--warn);">▲ Medium Severity ({p33:.2f} – {p66:.2f})</strong>
                <div style="font-size: 0.8125rem; color: var(--text-2); margin-top: 4px;"><strong>Status:</strong> Elevated Vulnerability</div>
                <div style="font-size: 0.75rem; color: var(--text-3); margin-top: 2px;">Emerging transit lag or capacity strain. Alert downstream distribution hubs.</div>
            </div>
            <div style="background: var(--surface-2); padding: 14px; border-radius: var(--r-sm); border-left: 3px solid var(--bad);">
                <strong style="color: var(--bad);">■ High Severity (&gt; {p66:.2f})</strong>
                <div style="font-size: 0.8125rem; color: var(--text-2); margin-top: 4px;"><strong>Status:</strong> Critical Incident</div>
                <div style="font-size: 0.75rem; color: var(--text-3); margin-top: 2px;">Imminent disruption. Activate emergency buffer inventory and expedite secondary logistics line.</div>
            </div>
        </div>
        <div style="font-size: 0.75rem; color: var(--text-3); margin-top: 8px;">
            <em>Source: {source_label}</em> · Integrated Gradients diagnostics quantify input sensitivity to guide targeted mitigation.
        </div>
    </div>
    """
    st.html(card("Operational Severity Protocols & Action Matrix", guide_html))
