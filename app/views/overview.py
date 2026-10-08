"""SupplyGuard Executive Overview & Operational Architecture.

Presents the platform summary, readiness checklist, model and data
specifications, and the severity-tier definitions.
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
        SupplyGuard forecasts the risk index of every echelon of a four-tier supply chain
        (<strong>Supplier &rarr; Manufacturer &rarr; Distributor &rarr; Retailer</strong>)
        <strong>10 minutes (5 steps) ahead</strong>, replaying windows from the held-out test period of the Mendeley SCRM dataset.
        It couples <strong>directed graph convolution (GCN)</strong> with a <strong>shared LSTM per node</strong> and a residual head that starts from the persistence forecast.
        Each forecast can be explained with <strong>Integrated Gradients</strong>, which shows how sensitive it is to each past input; this is sensitivity, not proof of cause.
    </p>
    """
    st.html(card("Platform Summary", summary_html))
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
            f'{check_row("Telemetry Dataset", has_data, "Mendeley SCRM series, chronological 80/10/10 partitions")}'
            f'{check_row("Calibration Scaler", has_scaler, "MinMaxScaler fitted strictly on training partition")}'
            f'{check_row("Model Weights Checkpoints", has_ckpt, f"{len(ckpt_files)} multi-seed models ready in {Config.CKPT_DIR}")}'
            f'{check_row("Benchmark Results", has_results, "5-seed MSE, R², accuracy and macro-F1 tables")}'
            f'</div>'
        )
        st.html(card("Readiness & Model Health", pipeline_html))

    with col2:
        # 3. Model and data specifications
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
                <td style="padding: 9px 0; color: var(--text-3);">Input Features (F)</td>
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
        st.html(card("Model & Data Specifications", specs_html))

    st.html("<div style='height: 16px;'></div>")

    # 4. Severity tier definitions (same rule as training/evaluate.py)
    tiers_info = get_tiers()
    if "node_p33" in tiers_info and "node_p66" in tiers_info:
        rows = "".join(
            f'<tr style="border-bottom: 1px solid var(--border);">'
            f'<td style="padding: 8px 0; color: var(--text-3);">{name}</td>'
            f'<td style="padding: 8px 0; text-align: right;" class="mono-val">&lt; {lo:.3f}</td>'
            f'<td style="padding: 8px 0; text-align: right;" class="mono-val">{lo:.3f} – {hi:.3f}</td>'
            f'<td style="padding: 8px 0; text-align: right;" class="mono-val">&gt; {hi:.3f}</td></tr>'
            for name, lo, hi in zip(Config.NODE_NAMES, tiers_info["node_p33"], tiers_info["node_p66"])
        )
        thresholds_html = f"""
        <table style="width: 100%; border-collapse: collapse; font-size: 0.8125rem; color: var(--text-2);">
            <tr style="border-bottom: 1px solid var(--border); color: var(--text-3);">
                <th style="text-align: left; padding: 6px 0;">Echelon</th>
                <th style="text-align: right; padding: 6px 0; color: var(--ok);">● Low</th>
                <th style="text-align: right; padding: 6px 0; color: var(--warn);">▲ Medium</th>
                <th style="text-align: right; padding: 6px 0; color: var(--bad);">■ High</th>
            </tr>
            {rows}
        </table>
        """
        source_label = "Per-echelon terciles of the training partition (scaled risk index)"
    else:
        thresholds_html = (
            f'<div style="font-size: 0.8125rem; color: var(--text-2);">Low &lt; {tiers_info["p33"]:.2f} · '
            f'Medium {tiers_info["p33"]:.2f} – {tiers_info["p66"]:.2f} · High &gt; {tiers_info["p66"]:.2f}</div>'
        )
        source_label = "Provisional cutoffs (training data unavailable)"

    guide_html = f"""
    <div style="font-size: 0.875rem; line-height: 1.6; color: var(--text-2);">
        <p style="margin-top: 0;">
            Each forecast is placed in a Low, Medium or High tier using that echelon's own thresholds,
            the same rule used to compute the accuracy and macro-F1 on the Benchmarks page.
        </p>
        {thresholds_html}
        <div style="font-size: 0.75rem; color: var(--text-3); margin-top: 8px;">
            <em>Source: {source_label}</em>
        </div>
    </div>
    """
    st.html(card("Severity Tiers", guide_html))
