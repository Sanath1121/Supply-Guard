"""SupplyGuard Overview & Architecture View.

Presents the executive project summary, live pipeline status checklist,
dataset architecture specifications, and reading guide.
"""
import os
import streamlit as st
from src.config import Config
from app.ui.components import card, status_badge, section_header, empty_state
from app.utils.artifacts import ArtifactStatus, get_tiers


def render_overview(status: ArtifactStatus):
    """Render the Overview view."""
    section_header(
        "SupplyGuard: Spatiotemporal Multi-Echelon Risk Forecasting",
        "Leak-free spatiotemporal graph forecasting and sensitivity attribution for supply chain risk resilience."
    )

    # Highlights strip
    h_col1, h_col2, h_col3, h_col4 = st.columns(4, gap="small")
    with h_col1:
        st.markdown(card("Architecture", "<div style='font-size:1.5rem; font-weight:800; color:#FFFFFF;' class='mono-val'>ST-GCN-LSTM</div><div style='font-size:0.75rem; color:var(--text-3); margin-top:2px;'>Hybrid Spatial-Temporal</div>"), unsafe_allow_html=True)
    with h_col2:
        st.markdown(card("Forecasting Horizon", "<div style='font-size:1.5rem; font-weight:800; color:var(--accent);' class='mono-val'>t+5 (10 Min)</div><div style='font-size:0.75rem; color:var(--text-3); margin-top:2px;'>2-minute cadence</div>"), unsafe_allow_html=True)
    with h_col3:
        st.markdown(card("Cascading Echelons", "<div style='font-size:1.5rem; font-weight:800; color:#FFFFFF;' class='mono-val'>4 Nodes</div><div style='font-size:0.75rem; color:var(--text-3); margin-top:2px;'>S → M → D → R</div>"), unsafe_allow_html=True)
    with h_col4:
        st.markdown(card("Data Splitting", "<div style='font-size:1.5rem; font-weight:800; color:#34D399;' class='mono-val'>80 / 10 / 10</div><div style='font-size:0.75rem; color:var(--text-3); margin-top:2px;'>Zero temporal leakage</div>"), unsafe_allow_html=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # 1. Executive Summary Card
    summary_html = """
    <p style="margin: 0; font-size: 0.9375rem; line-height: 1.6; color: var(--text-2);">
        SupplyGuard forecasts multi-echelon disruption risk indices (TRI) across four cascading echelons 
        (<strong>Supplier &rarr; Manufacturer &rarr; Distributor &rarr; Retailer</strong>) over a 10-minute horizon (<span class="mono-val">t+5</span> at 2-minute cadence).
        By coupling <strong>Spatial Graph Convolution (GCN)</strong> with <strong>Temporal LSTM</strong> networks and a residual persistence baseline,
        it captures upstream shock propagation without lookahead leakage. Model forecasts are explained via path-integrated gradients
        to quantify echelon sensitivity.
    </p>
    """
    st.markdown(card("Project Scope & Purpose", summary_html), unsafe_allow_html=True)
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        # 2. Pipeline Status Checklist
        has_data = os.path.exists(Config.RAW_DATA_PATH)
        has_scaler = os.path.exists(Config.SCALER_PATH)
        
        # Check checkpoints
        ckpt_files = [f for f in os.listdir(Config.CKPT_DIR) if f.endswith(".pt")] if os.path.exists(Config.CKPT_DIR) else []
        has_ckpt = len(ckpt_files) > 0

        # Check results
        res_dir = os.path.join("outputs", "results")
        has_results = os.path.exists(os.path.join(res_dir, "overall_metrics.csv"))

        def check_row(label: str, ok: bool, details: str) -> str:
            badge = status_badge("Verified", "low") if ok else status_badge("Missing", "high")
            color = "var(--ok)" if ok else "var(--text-3)"
            return f"""
            <div style="display: flex; align-items: center; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid var(--border);">
                <div>
                    <div style="font-size: 0.875rem; font-weight: 500; color: var(--text);">{label}</div>
                    <div style="font-size: 0.75rem; color: var(--text-3);">{details}</div>
                </div>
                <div>{badge}</div>
            </div>
            """

        pipeline_html = f"""
        <div>
            {check_row("1. Raw SCRM Dataset", has_data, f"{Config.RAW_DATA_PATH}")}
            {check_row("2. Train-Fitted Scaler", has_scaler, f"{Config.SCALER_PATH}")}
            {check_row("3. Trained Checkpoints", has_ckpt, f"{len(ckpt_files)} model weights in {Config.CKPT_DIR}")}
            {check_row("4. Empirical Benchmark CSVs", has_results, "outputs/results/overall_metrics.csv")}
        </div>
        """
        st.markdown(card("Pipeline Artifact Checklist", pipeline_html), unsafe_allow_html=True)

    with col2:
        # 3. Dataset & Model Specifications (from Config)
        specs_html = f"""
        <table style="width: 100%; border-collapse: collapse; font-size: 0.8125rem; color: var(--text-2);">
            <tr style="border-bottom: 1px solid var(--border);">
                <td style="padding: 8px 0; color: var(--text-3);">Sequence Lookback (L)</td>
                <td style="padding: 8px 0; text-align: right;" class="mono-val">{Config.SEQ_LEN} steps ({Config.SEQ_LEN * Config.CADENCE_MIN:.0f} min)</td>
            </tr>
            <tr style="border-bottom: 1px solid var(--border);">
                <td style="padding: 8px 0; color: var(--text-3);">Forecasting Horizon (H)</td>
                <td style="padding: 8px 0; text-align: right;" class="mono-val">t+{Config.HORIZON} ({Config.HORIZON_MIN} min @ {Config.CADENCE_MIN:.1f}m cadence)</td>
            </tr>
            <tr style="border-bottom: 1px solid var(--border);">
                <td style="padding: 8px 0; color: var(--text-3);">Echelons & Node Order</td>
                <td style="padding: 8px 0; text-align: right;">{" &rarr; ".join(Config.NODE_NAMES)}</td>
            </tr>
            <tr style="border-bottom: 1px solid var(--border);">
                <td style="padding: 8px 0; color: var(--text-3);">Input Features (F)</td>
                <td style="padding: 8px 0; text-align: right;" class="mono-val">4 Node RIs + Total_Cost (5 dim)</td>
            </tr>
            <tr style="border-bottom: 1px solid var(--border);">
                <td style="padding: 8px 0; color: var(--text-3);">Temporal Split Ratios</td>
                <td style="padding: 8px 0; text-align: right;" class="mono-val">{Config.TRAIN_RATIO*100:.0f}% Train / {Config.VAL_RATIO*100:.0f}% Val / {Config.TEST_RATIO*100:.0f}% Test</td>
            </tr>
            <tr>
                <td style="padding: 8px 0; color: var(--text-3);">Max Gap Split Limit</td>
                <td style="padding: 8px 0; text-align: right;" class="mono-val">{Config.GAP_MAX_MIN} minutes</td>
            </tr>
        </table>
        """
        st.markdown(card("Contract Specifications", specs_html), unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # 4. "How to Read this Dashboard" Guide
    tiers_info = get_tiers()
    p33 = tiers_info.get("p33", 0.35)
    p66 = tiers_info.get("p66", 0.65)
    source_label = "Train-Partition Terciles" if tiers_info.get("source") == "train_terciles" else "Provisional Thresholds"

    guide_html = f"""
    <div style="font-size: 0.875rem; line-height: 1.6; color: var(--text-2);">
        <p style="margin-top: 0;">
            This dashboard operates strictly in <strong>historical replay mode</strong> against the untouched test partition.
            Every score represents a standardized risk index (RI) or aggregated Total Risk Index (TRI).
        </p>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin: 12px 0;">
            <div style="background: var(--surface-2); padding: 12px; border-radius: var(--r-sm); border-left: 3px solid var(--ok);">
                <strong style="color: var(--ok);">● Low Severity</strong>
                <div style="font-size: 0.8125rem; color: var(--text-3); margin-top: 4px;">Score &lt; {p33:.2f} ({source_label}). Stable operational conditions.</div>
            </div>
            <div style="background: var(--surface-2); padding: 12px; border-radius: var(--r-sm); border-left: 3px solid var(--warn);">
                <strong style="color: var(--warn);">▲ Medium Severity</strong>
                <div style="font-size: 0.8125rem; color: var(--text-3); margin-top: 4px;">{p33:.2f} &le; Score &le; {p66:.2f}. Emerging vulnerability or transit delay.</div>
            </div>
            <div style="background: var(--surface-2); padding: 12px; border-radius: var(--r-sm); border-left: 3px solid var(--bad);">
                <strong style="color: var(--bad);">■ High Severity</strong>
                <div style="font-size: 0.8125rem; color: var(--text-3); margin-top: 4px;">Score &gt; {p66:.2f}. Critical cascading risk requiring buffer intervention.</div>
            </div>
        </div>
        <p style="margin-bottom: 0; font-size: 0.8125rem; color: var(--text-3);">
            <em>Note on Interpretability:</em> Feature attributions in the Explainability view reflect mathematical sensitivity 
            computed via path-integrated gradients relative to a training-mean baseline. They quantify which signals drove the model's 
            internal forecast, not definitive physical originating causes.
        </p>
    </div>
    """
    st.markdown(card("Operational Interpretation Guide", guide_html), unsafe_allow_html=True)

    # 5. Architecture Expander
    with st.expander("Neural Architecture Design Details"):
        st.markdown("""
        **Spatio-Temporal Graph Convolutional Network (ST-GCN-LSTM)**
        
        1. **Spatial Representation**: 2-layer Graph Convolutional Network (GCN) propagating node feature embeddings along the directed supply chain graph ($S \\to M \\to D \\to R$).
        2. **Temporal Aggregation**: 2-layer stacked LSTM capturing temporal dependencies across the 10-step sequence window ($t-9 \\dots t$).
        3. **Residual Head**: The model explicitly predicts the delta $\\Delta y = f(X) - y_t$ over the persistence baseline, guaranteeing the network learns meaningful dynamics rather than trivial identity mapping.
        """)
