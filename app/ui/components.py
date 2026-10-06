"""Reusable UI Component Library.

Strictly follows the Restrained Dark Analytics design system (Linear / Vercel).
All dynamic user and dataset inputs are escaped using html.escape to prevent injection.
All text meets WCAG AA contrast (>= 4.5:1).
"""
import html
import subprocess
import os
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
import streamlit as st

from app.utils.artifacts import ArtifactStatus


def escape(text: Any) -> str:
    """Safely escape any input into HTML text."""
    if text is None:
        return ""
    return html.escape(str(text))


def clean_html(html_str: Any) -> str:
    """Clean HTML string by stripping leading whitespace on every line.
    
    In CommonMark (Markdown parser used by Streamlit), any line indented by 4 or more
    spaces is parsed as an indented code block (<pre><code>). Furthermore, blank lines
    interrupt HTML block parsing. Stripping leading whitespace from every line and
    omitting blank lines guarantees that CommonMark and st.html render pure styled HTML.
    """
    if not html_str:
        return ""
    lines = [line.strip() for line in str(html_str).strip().splitlines() if line.strip()]
    return "\n".join(lines)


def card(title: str, body_html: str, *, tone: Optional[str] = None) -> str:
    """Render a container card with optional status tone border (low, medium, high)."""
    tone_cls = f" sg-card-{escape(tone.lower())}" if tone else ""
    safe_title = escape(title)
    cleaned_body = clean_html(body_html)
    raw_html = (
        f'<div class="sg-card{tone_cls}">'
        f'<div style="font-size: 0.8125rem; font-weight: 600; color: var(--text-2); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: var(--sp-2);">{safe_title}</div>'
        f'<div>{cleaned_body}</div>'
        f'</div>'
    )
    return clean_html(raw_html)


def status_badge(label: str, tier: str) -> str:
    """Render a status pill badge with accessible glyph + text."""
    tier_lower = tier.lower() if tier else "pending"
    glyphs = {
        "low": "●",
        "medium": "▲",
        "high": "■",
        "pending": "○",
        "supported": "✓",
        "not_supported": "✗",
        "inconclusive": "∼"
    }
    glyph = glyphs.get(tier_lower, "●")
    pulse_cls = " pulse-high" if tier_lower == "high" else ""
    
    return f'<span class="sg-badge sg-badge-{escape(tier_lower)}{pulse_cls}">{glyph} {escape(label)}</span>'


def metric_display(label: str, value: str, delta: Optional[str] = None, unit: Optional[str] = None, help_text: Optional[str] = None) -> str:
    """Render a high-precision metric block with monospace tabular numbers."""
    safe_label = escape(label)
    safe_val = escape(value)
    safe_unit = f'<span style="font-size: 0.875rem; color: var(--text-3); margin-left: 4px;">{escape(unit)}</span>' if unit else ""
    
    delta_html = ""
    if delta is not None:
        delta_color = "var(--text-3)"
        delta_glyph = ""
        try:
            d_val = float(delta)
            if d_val > 0:
                delta_glyph = "▲ +"
                delta_color = "var(--bad)"
            elif d_val < 0:
                delta_glyph = "▼ "
                delta_color = "var(--ok)"
            else:
                delta_glyph = "– "
        except Exception:
            delta_glyph = ""
        delta_html = f'<div style="font-size: 0.75rem; color: {delta_color}; margin-top: 2px;" class="mono-val">{delta_glyph}{escape(delta)} vs last</div>'

    title_attr = f' title="{escape(help_text)}"' if help_text else ""

    raw_html = (
        f'<div style="display: flex; flex-direction: column;"{title_attr}>'
        f'<div style="font-size: 0.75rem; font-weight: 500; color: var(--text-3); margin-bottom: 2px;">{safe_label}</div>'
        f'<div style="font-size: 1.5rem; font-weight: 700; color: var(--text); line-height: 1.2;" class="mono-val">{safe_val}{safe_unit}</div>'
        f'{delta_html}'
        f'</div>'
    )
    return clean_html(raw_html)


def banner(kind: str, title: str, body: str, actions: Optional[str] = None) -> str:
    """Render an informational, warning, error, or sandbox banner.
    
    kind in {'warning', 'error', 'info', 'sandbox'}
    """
    safe_kind = escape(kind.lower())
    safe_title = escape(title)
    safe_body = escape(body)
    actions_html = f'<div style="margin-top: 8px;">{clean_html(actions)}</div>' if actions else ""

    icons = {
        "warning": "⚠️",
        "error": "⛔",
        "info": "ℹ️",
        "sandbox": "🧪",
    }
    icon = icons.get(safe_kind, "ℹ️")

    raw_html = (
        f'<div class="sg-banner sg-banner-{safe_kind}" role="alert">'
        f'<div style="font-size: 1.125rem; line-height: 1;">{icon}</div>'
        f'<div style="flex: 1;">'
        f'<strong style="display: block; margin-bottom: 2px;">{safe_title}</strong>'
        f'<span style="font-size: 0.8125rem;">{safe_body}</span>'
        f'{actions_html}'
        f'</div>'
        f'</div>'
    )
    return clean_html(raw_html)


def empty_state(title: str, body: str, command: Optional[str] = None):
    """Streamlit native render of an empty state with optional reproduction command."""
    safe_title = escape(title)
    safe_body = escape(body)
    html_block = clean_html(
        f'<div class="sg-card" style="text-align: center; padding: var(--sp-6) var(--sp-4); margin: var(--sp-4) 0;">'
        f'<div style="font-size: 1.75rem; margin-bottom: var(--sp-2);">📦</div>'
        f'<h4 style="margin: 0 0 var(--sp-2) 0; color: var(--text); font-size: 1.125rem;">{safe_title}</h4>'
        f'<p style="color: var(--text-2); font-size: 0.875rem; max-width: 600px; margin: 0 auto var(--sp-3) auto;">{safe_body}</p>'
        f'</div>'
    )
    st.html(html_block)
    if command:
        st.caption("Reproduction command:")
        st.code(command, language="bash")


def section_header(title: str, subtitle: Optional[str] = None):
    """Render a consistent, professional section header."""
    safe_title = escape(title)
    sub_html = f'<p style="color: var(--text-2); font-size: 0.875rem; margin: 2px 0 0 0;">{escape(subtitle)}</p>' if subtitle else ""
    html_block = clean_html(
        f'<div style="margin-bottom: var(--sp-4);">'
        f'<h3 style="font-size: 1.25rem; font-weight: 600; color: var(--text); margin: 0;">{safe_title}</h3>'
        f'{sub_html}'
        f'</div>'
    )
    st.html(html_block)


def get_git_commit_hash() -> str:
    """Retrieve git short hash safely."""
    try:
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False)
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"


def status_strip(status: ArtifactStatus, xai_gap: Optional[float] = None, xai_error: Optional[str] = None, delta_mode: bool = False):
    """Render the universal Status Strip chip row at the top of every page per §4.1."""
    # 1. Data chip
    if status.data_source == "test_partition":
        data_cls = "sg-chip-ok"
        data_txt = "Data: Test partition ✓"
    elif status.data_source == "sandbox":
        data_cls = "sg-chip-warn"
        data_txt = "Data: Sandbox"
    else:
        data_cls = "sg-chip-bad"
        data_txt = "Data: Unavailable (missing raw CSV)"

    # 2. Model chip
    if status.model_loaded:
        ckpt_tag = status.checkpoint_path.split(os.sep)[-1][:8] if status.checkpoint_path else "loaded"
        model_cls = "sg-chip-ok"
        model_txt = f"Model: {status.model_name} · {status.graph_mode} · seed {status.seed} · ckpt ✓"
    else:
        model_cls = "sg-chip-bad"
        model_txt = f"Model: {status.model_name} · {status.graph_mode} · UNTRAINED weights"

    # 3. Scaler chip
    if status.scaler_fitted:
        scaler_cls = "sg-chip-ok"
        scaler_txt = "Scaler: Fitted ✓"
    else:
        scaler_cls = "sg-chip-bad"
        scaler_txt = "Scaler: Missing (raw units hidden)"

    # 4. Tiers chip
    if status.tiers_source == "train_terciles":
        tiers_cls = "sg-chip-ok"
        tiers_txt = "Tiers: Train terciles ✓"
    else:
        tiers_cls = "sg-chip-warn"
        tiers_txt = "Tiers: Provisional 0.35/0.65"

    # 5. Explainer chip
    if xai_error:
        xai_cls = "sg-chip-bad"
        xai_txt = f"Explainer: Error ({escape(xai_error[:24])})"
    elif delta_mode:
        xai_cls = "sg-chip-warn"
        xai_txt = "Explainer: Δ vs persistence"
    elif xai_gap is not None:
        if abs(xai_gap) <= 1e-2:
            xai_cls = "sg-chip-ok"
            xai_txt = f"Explainer: OK (gap {xai_gap:.1e})"
        else:
            xai_cls = "sg-chip-bad"
            xai_txt = f"Explainer: Gap {xai_gap:.1e} (>1e-2)"
    else:
        xai_cls = "sg-chip"
        xai_txt = "Explainer: Ready"

    strip_html = clean_html(f"""
    <div class="sg-status-strip">
        <span style="font-weight: 700; color: var(--text-3); font-size: 0.75rem; letter-spacing: 0.06em; margin-right: 4px;">SYSTEM STATUS</span>
        <span class="sg-chip {data_cls}"><span class="sg-chip-dot"></span>{escape(data_txt)}</span>
        <span class="sg-chip {model_cls}"><span class="sg-chip-dot"></span>{escape(model_txt)}</span>
        <span class="sg-chip {scaler_cls}"><span class="sg-chip-dot"></span>{escape(scaler_txt)}</span>
        <span class="sg-chip {tiers_cls}"><span class="sg-chip-dot"></span>{escape(tiers_txt)}</span>
        <span class="sg-chip {xai_cls}"><span class="sg-chip-dot"></span>{escape(xai_txt)}</span>
    </div>
    """)
    st.html(strip_html)

    # If any critical artifact is missing, show an explicit red warning banner
    if not status.model_loaded:
        st.html(banner(
            "error",
            "Untrained Model Weights Active",
            f"No trained checkpoint was found for {status.model_name}/{status.graph_mode}/seed{status.seed}. "
            "Forecasts displayed on this dashboard are generated from randomly initialized weights for structural validation only and MUST NOT be scientifically interpreted."
        ))
    elif len(status.warnings) > 0:
        for w in status.warnings:
            st.html(banner("warning", "System Notice", w))


def provenance_dict(status: ArtifactStatus, window_id: Optional[int] = None, timestamp: Optional[str] = None, delta_mode: bool = False) -> Dict[str, Any]:
    """Compile comprehensive provenance metadata for validation and auditing."""
    return {
        "model_name": status.model_name,
        "graph_mode": status.graph_mode,
        "seed": status.seed,
        "checkpoint": status.checkpoint_path if status.model_loaded else "UNTRAINED",
        "scaler_fitted": status.scaler_fitted,
        "tiers_source": status.tiers_source,
        "data_source": status.data_source,
        "window_id": window_id if window_id is not None else "N/A",
        "timestamp": timestamp or "N/A",
        "attribution_mode": "Delta vs Persistence" if delta_mode else "Full Forecast Attribution",
        "git_commit": get_git_commit_hash(),
        "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "disclaimer": "Sensitivity attribution – not verified causation"
    }
