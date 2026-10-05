"""SupplyGuard Incident Report & Audit Provenance Export View.

Generates complete, reproducible Markdown incident audit reports with
tamper-evident artifact provenance metadata and scientific disclaimers.
"""
from typing import Dict, Any, List, Optional
import hashlib
import os
import streamlit as st

from src.config import Config
from src.explainability import FEATURE_NAMES, NODE_NAMES, narrate
from app.ui.components import card, banner, section_header, provenance_dict, escape
from app.utils.artifacts import ArtifactStatus, get_tiers, get_scaler, predict_window
from app.utils.explain_service import compute_explanation
from app.utils.formatters import compute_tercile_tier


def get_checkpoint_sha256(path: Optional[str]) -> str:
    """Calculate first 8 hex characters of checkpoint SHA-256."""
    if not path or not os.path.exists(path):
        return "UNTRAINED"
    try:
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()[:8]
    except Exception:
        return "ERROR"


def generate_markdown_report(
    status: ArtifactStatus,
    window: Optional[Dict[str, Any]],
    preds: Optional[Any],
    xai_res: Optional[Dict[str, Any]],
    target_node_idx: int,
    delta_mode: bool
) -> str:
    """Generate Markdown text for the incident audit report."""
    prov = provenance_dict(
        status,
        window_id=window["window_id"] if window else None,
        timestamp=window["timestamp"] if window else None,
        delta_mode=delta_mode
    )
    ckpt_hash = get_checkpoint_sha256(status.checkpoint_path)

    # Untrained / Unfitted Warning
    warning_block = ""
    if not status.model_loaded or not status.scaler_fitted:
        warning_block = (
            "> ⚠️ **INTEGRITY WARNING: UNTRAINED WEIGHTS / UNFITTED ARTIFACTS IN USE.**\n"
            "> This report was generated using non-checkpointed or unscaled model parameters. "
            "These metrics are intended strictly for software execution verification and MUST NOT be used for operational decision-making.\n\n"
        )

    tiers = get_tiers()
    p33, p66 = tiers["p33"], tiers["p66"]

    # Forecast summary
    tri_val = float(preds[0] if (status.model_name == "paper_overall" or (preds is not None and preds.size == 1)) else (preds.mean() if preds is not None else 0.0))
    tri_tier = compute_tercile_tier(tri_val, p33, p66)

    echelon_rows = []
    if preds is not None and preds.size > 1:
        for i, name in enumerate(Config.NODE_NAMES):
            r = float(preds[i])
            t = compute_tercile_tier(r, p33, p66)
            echelon_rows.append(f"| {name} | {r:.4f} | {t} |")
    echelon_table = "\n".join(echelon_rows) if echelon_rows else "| Overall TRI | {:.4f} | {} |".format(tri_val, tri_tier)

    # Explanation narrative
    xai_text = "Attribution not computed."
    gap_text = "N/A"
    if xai_res:
        xai_text = narrate(xai_res, seq_len=Config.SEQ_LEN)
        gap_text = f"{xai_res.get('completeness_gap', 0.0):.2e}"

    md_content = f"""# SupplyGuard Incident Audit Report

{warning_block}## 1. Provenance & Execution Context
- **Generated At:** {prov['exported_at']}
- **Git Commit:** `{prov['git_commit']}`
- **Model Architecture:** `{status.model_name}` (Graph Mode: `{status.graph_mode}`, Seed: `{status.seed}`)
- **Checkpoint SHA-256:** `{ckpt_hash}` ({status.checkpoint_path or 'No checkpoint found'})
- **Fitted Scaler:** {'Verified (`scaler.joblib`)' if status.scaler_fitted else 'Missing (Standardized units only)'}
- **Severity Thresholds:** {prov['tiers_source']} (p33 = {p33:.3f}, p66 = {p66:.3f})
- **Data Source:** {prov['data_source']} (Window #{prov['window_id']}, Timestamp: {prov['timestamp']})
- **Attribution Mode:** {prov['attribution_mode']}
- **Scientific Framing:** *{prov['disclaimer']}*

---

## 2. Multi-Echelon Risk Forecast (Horizon: t+5 / 10 Minutes)
- **Total Risk Index (TRI):** **{tri_val:.4f}** ({tri_tier} Severity)

| Echelon | Forecast Risk | Severity Tier |
| :--- | :--- | :--- |
{echelon_table}

---

## 3. Sensitivity Attribution (Target: {NODE_NAMES[target_node_idx] if target_node_idx < len(NODE_NAMES) else 'Overall'})
- **Axiomatic Completeness Gap:** `{gap_text}` (Pass condition: |gap| <= 1e-2)
- **Summary Narrative:**
  > {xai_text}

---

## 4. Standard Response Protocols (Operational Reference)
The following heuristics represent standard supply chain risk management operational actions:
1. **Tier Review:** Verify buffer safety margins across the upstream supplier and manufacturing interface.
2. **Logistics Rerouting:** Prepare secondary carrier transit dispatch for distribution lanes exceeding medium threshold.
3. **Audit Verification:** Cross-check sensor telemetries against physical freight status manifests.

*Report automatically compiled by SupplyGuard Diagnostic Core.*
"""
    return md_content


def render_export(
    status: ArtifactStatus,
    windows: List[Dict[str, Any]],
    model: Any
):
    """Render the Export Report view."""
    section_header(
        "Incident Report Export & Provenance Auditor",
        "Generate reproducible Markdown audit reports with complete metadata and scientific provenance."
    )

    if not windows or len(windows) == 0:
        st.warning("No test-partition window currently loaded. Report will contain default placeholder metrics.")
        current_window = None
        preds = None
    else:
        w_idx = st.session_state.get("window_idx", 0)
        if w_idx >= len(windows):
            w_idx = 0
        current_window = windows[w_idx]
        preds = predict_window(model, current_window["sequence"])

    target_idx = st.session_state.get("selected_node_idx", 1)

    # Explanation result
    xai_res = None
    if current_window is not None and status.model_name != "paper_overall":
        model_key = f"{status.model_name}:{status.graph_mode}:{status.seed}"
        xai_res, _ = compute_explanation(model_key, current_window["sequence"], target_idx, residual_delta=False)

    report_md = generate_markdown_report(
        status,
        current_window,
        preds,
        xai_res,
        target_idx,
        delta_mode=False
    )

    st.markdown("#### Audit Report Preview")
    st.markdown(card("Markdown Source Preview", f"<pre style='font-size: 0.8125rem; color: var(--text-2); white-space: pre-wrap; margin: 0;'>{escape(report_md)}</pre>"), unsafe_allow_html=True)
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    filename = f"SupplyGuard_Incident_Report_w{current_window['window_id'] if current_window else 0}.md"
    st.download_button(
        label="Download Incident Report (.md)",
        data=report_md,
        file_name=filename,
        mime="text/markdown",
        use_container_width=True
    )
