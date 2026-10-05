"""SupplyGuard Empirical Benchmarks View.

Renders empirical comparison leaderboards, hypothesis verdicts, and classification
metrics strictly from outputs/results/ CSV files.
Displays explicit 'Pending' states when artifacts are not yet generated.
Zero hardcoded metrics or pre-computed verdicts.
"""
from typing import Optional
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from app.ui.components import card, status_badge, empty_state, section_header, escape
from app.ui.plotly_theme import apply_theme
from app.utils.results_loader import (
    load_overall_metrics,
    load_severity_metrics,
    load_node_metrics,
    evaluate_rq_verdicts,
)


def render_benchmarks():
    """Render the Benchmarks and Hypothesis Evaluation view."""
    section_header(
        "Empirical Benchmark Arena & RQ Hypothesis Evaluation",
        "Multi-seed evaluation on the untouched test partition. Metrics strictly read from outputs/results/ CSVs."
    )

    overall_df = load_overall_metrics()
    severity_df = load_severity_metrics()
    node_df = load_node_metrics()
    verdicts = evaluate_rq_verdicts(overall_df)

    # 1. Research Question Hypothesis Verdict Cards
    st.markdown("#### Research Question Hypothesis Scorecard")
    rq_cols = st.columns(4, gap="small")

    rq_keys = ["RQ1", "RQ2", "RQ3", "RQ4"]
    for i, k in enumerate(rq_keys):
        v = verdicts[k]
        state = v["state"]
        with rq_cols[i]:
            badge = status_badge(v["status"], state)
            c_html = f"""
            <div>
                <div style="margin-bottom: 6px;">{badge}</div>
                <div style="font-size: 0.8125rem; font-weight: 600; color: var(--text); margin-bottom: 4px;">{escape(v['title'])}</div>
                <div style="font-size: 0.75rem; color: var(--text-3); min-height: 48px; line-height: 1.4;">{escape(v['hypothesis'])}</div>
                <div style="font-size: 0.75rem; color: var(--text-2); border-top: 1px solid var(--border); padding-top: 4px; margin-top: 6px;">
                    {escape(v['details'])}
                </div>
            </div>
            """
            st.markdown(card(k, c_html), unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # 2. Leaderboard Table
    st.markdown("#### Test-Partition Leaderboard (Derived TRI)")
    if overall_df is None or overall_df.empty:
        empty_state(
            "Empirical Benchmarks Pending",
            "The model comparison metrics have not been generated yet. "
            "To train all models across seeds and compute evaluation tables on the test partition, run the pipeline commands below:",
            "python -m training.train\npython -m training.evaluate"
        )
    else:
        # Format tidy display dataframe
        disp_df = overall_df.copy()
        
        # Format columns with mean ± std
        cols_to_format = ["MSE", "MAE", "RMSE", "R2"]
        for c in cols_to_format:
            if f"{c}_mean" in disp_df.columns and f"{c}_std" in disp_df.columns:
                disp_df[c] = disp_df.apply(
                    lambda row: f"{row[f'{c}_mean']:.4f} ± {row[f'{c}_std']:.4f}", axis=1
                )
            elif f"{c}_mean" in disp_df.columns:
                disp_df[c] = disp_df[f"{c}_mean"].apply(lambda v: f"{v:.4f}")

        display_cols = ["model"] + [c for c in cols_to_format if c in disp_df.columns]
        st.dataframe(
            disp_df[display_cols],
            use_container_width=True,
            hide_index=True
        )

        # Bar chart with error bars (MSE_mean ± MSE_std)
        if "MSE_mean" in overall_df.columns:
            fig_bar = go.Figure()
            
            y_err = overall_df["MSE_std"].tolist() if "MSE_std" in overall_df.columns else None
            error_y_dict = dict(type="data", array=y_err, visible=True, color="#94A3B8") if y_err else None

            fig_bar.add_trace(go.Bar(
                x=overall_df["model"],
                y=overall_df["MSE_mean"],
                error_y=error_y_dict,
                marker=dict(color="#38BDF8"),
                hovertemplate="<b>%{x}</b><br>MSE: %{y:.4f}<extra></extra>"
            ))
            fig_bar.update_layout(
                title="Model Test MSE (Derived TRI with Seed Standard Deviations)",
                xaxis_title="Evaluated Model / Baseline",
                yaxis_title="Mean Squared Error (Scaled)",
                margin=dict(l=15, r=15, t=35, b=25)
            )
            apply_theme(fig_bar, height=300)
            st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # 3. Severity Classification & Confusion Matrix
    st.markdown("#### 3-Tier Severity Classification Performance")
    if severity_df is None or severity_df.empty:
        st.info("Severity tier accuracy and Macro-F1 metrics are pending execution of `python -m training.evaluate`.")
    else:
        sev_display = severity_df.copy()
        if "accuracy_mean" in sev_display.columns and "accuracy_std" in sev_display.columns:
            sev_display["Accuracy"] = sev_display.apply(
                lambda r: f"{r['accuracy_mean']:.3f} ± {r['accuracy_std']:.3f}", axis=1
            )
        if "macro_F1_mean" in sev_display.columns and "macro_F1_std" in sev_display.columns:
            sev_display["Macro-F1"] = sev_display.apply(
                lambda r: f"{r['macro_F1_mean']:.3f} ± {r['macro_F1_std']:.3f}", axis=1
            )

        show_sev_cols = ["model"] + [c for c in ["Accuracy", "Macro-F1"] if c in sev_display.columns]
        st.dataframe(
            sev_display[show_sev_cols],
            use_container_width=True,
            hide_index=True
        )
