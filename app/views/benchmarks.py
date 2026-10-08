"""SupplyGuard Production Benchmarks & Model Validation View.

Renders empirical comparison leaderboards, research-question verdicts, and classification
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


def _fmt_mean_std(mean: float, std: float, kind: str) -> str:
    """Format a metric as 'mean ± std'; deterministic baselines (std = NaN) show the mean only."""
    if kind == "MSE":
        fmt = lambda v: f"{v:.2e}"
    elif kind == "pct":
        fmt = lambda v: f"{v * 100:.1f}%"
    else:
        fmt = lambda v: f"{v:.4f}"
    if std is None or pd.isna(std):
        return fmt(mean)
    return f"{fmt(mean)} ± {fmt(std)}"


def render_benchmarks():
    """Render the Production Benchmarks and Validation view."""
    section_header(
        "Model Benchmarks",
        "5-seed evaluation on the untouched test partition, compared with Naive Persistence, Ridge-AR(10) and a graph-free LSTM."
    )

    overall_df = load_overall_metrics()
    severity_df = load_severity_metrics()
    node_df = load_node_metrics()
    verdicts = evaluate_rq_verdicts(overall_df)

    # 1. Research-question verdicts (computed from outputs/results/)
    st.markdown("#### Research Question Verdicts")
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
                <div style="font-size: 0.8125rem; font-weight: 700; color: #FFFFFF; margin-bottom: 4px;">{escape(v['title'].split(': ', 1)[-1])}</div>
                <div style="font-size: 0.75rem; color: var(--text-3); min-height: 44px; line-height: 1.4;">{escape(v['hypothesis'])}</div>
                <div style="font-size: 0.75rem; color: var(--text-2); border-top: 1px solid var(--border); padding-top: 5px; margin-top: 6px;">
                    {escape(v['details'])}
                </div>
            </div>
            """
            st.html(card(f"Research Question {i+1}", c_html))

    st.html("<div style='height: 20px;'></div>")

    # 2. Leaderboard Table
    st.markdown("#### Model Performance Leaderboard (Test Partition)")
    if overall_df is None or overall_df.empty:
        empty_state(
            "Empirical Benchmark Artifacts Pending",
            "The model comparison metrics have not been generated yet. "
            "To train all models across seeds and compute evaluation tables on the test partition, execute:",
            "python -m training.train\npython -m training.evaluate"
        )
    else:
        disp_df = overall_df.copy()
        
        # Format friendly enterprise model names
        model_display_map = {
            "st_gcn_lstm_directed": "SupplyGuard ST-GCN-LSTM (Directed)",
            "st_gcn_lstm_dir": "SupplyGuard ST-GCN-LSTM (Directed)",
            "st_gcn_lstm_symmetric": "SupplyGuard ST-GCN-LSTM (Symmetric)",
            "st_gcn_lstm_sym": "SupplyGuard ST-GCN-LSTM (Symmetric)",
            "lstm": "Graph-free LSTM",
            "paper_overall": "Base-Paper Hybrid (re-implemented, mean index only)",
            "persistence": "Naive Persistence",
            "ridge_ar": "Ridge-AR(10)",
            "ar10_ridge": "Ridge-AR(10)"
        }
        if "model" in disp_df.columns:
            disp_df["Model Name"] = disp_df["model"].apply(lambda m: model_display_map.get(m, m))

        cols_to_format = ["MSE", "MAE", "RMSE", "R2"]
        for c in cols_to_format:
            if f"{c}_mean" in disp_df.columns and f"{c}_std" in disp_df.columns:
                disp_df[c] = disp_df.apply(lambda row, c=c: _fmt_mean_std(row[f"{c}_mean"], row[f"{c}_std"], c), axis=1)
            elif f"{c}_mean" in disp_df.columns:
                disp_df[c] = disp_df[f"{c}_mean"].apply(lambda v, c=c: _fmt_mean_std(v, float("nan"), c))

        first_col = "Model Name" if "Model Name" in disp_df.columns else "model"
        display_cols = [first_col] + [c for c in cols_to_format if c in disp_df.columns]
        st.dataframe(
            disp_df[display_cols],
            use_container_width=True,
            hide_index=True
        )

        # Bar chart with error bars
        if "MSE_mean" in overall_df.columns:
            fig_bar = go.Figure()
            
            y_err = overall_df["MSE_std"].tolist() if "MSE_std" in overall_df.columns else None
            error_y_dict = dict(type="data", array=y_err, visible=True, color="#94A3B8") if y_err else None

            labels = [model_display_map.get(m, m) for m in overall_df["model"]]

            fig_bar.add_trace(go.Bar(
                x=labels,
                y=overall_df["MSE_mean"],
                error_y=error_y_dict,
                marker=dict(color="#3B82F6"),
                hovertemplate="<b>%{x}</b><br>MSE: %{y:.2e}<extra></extra>"
            ))
            fig_bar.update_layout(
                title="Model Test MSE (Lower is Better — mean ± std over 5 seeds)",
                xaxis_title="Model",
                yaxis_title="Mean Squared Error (scaled units)",
                margin=dict(l=15, r=15, t=35, b=25)
            )
            apply_theme(fig_bar, height=320)
            st.plotly_chart(fig_bar, use_container_width=True)

    st.html("<div style='height: 20px;'></div>")

    # 3. Severity Classification & Incident Detection
    st.markdown("#### Incident Severity Classification Accuracy (3-Tier)")
    if severity_df is None or severity_df.empty:
        st.info("Severity tier accuracy and Macro-F1 metrics are pending execution of `python -m training.evaluate`.")
    else:
        sev_display = severity_df.copy()
        if "model" in sev_display.columns:
            sev_display["Model Name"] = sev_display["model"].apply(lambda m: model_display_map.get(m, m))
        
        if "accuracy_mean" in sev_display.columns and "accuracy_std" in sev_display.columns:
            sev_display["Accuracy"] = sev_display.apply(
                lambda r: _fmt_mean_std(r["accuracy_mean"], r["accuracy_std"], "pct"), axis=1
            )
        if "macro_F1_mean" in sev_display.columns and "macro_F1_std" in sev_display.columns:
            sev_display["Macro-F1"] = sev_display.apply(
                lambda r: _fmt_mean_std(r["macro_F1_mean"], r["macro_F1_std"], "pct"), axis=1
            )

        first_s_col = "Model Name" if "Model Name" in sev_display.columns else "model"
        show_sev_cols = [first_s_col] + [c for c in ["Accuracy", "Macro-F1"] if c in sev_display.columns]
        st.dataframe(
            sev_display[show_sev_cols],
            use_container_width=True,
            hide_index=True
        )
        best = sev_display.loc[sev_display["macro_F1_mean"].idxmax()]
        st.caption(
            f"Tiers use each echelon's training-set terciles. Highest macro-F1: "
            f"{escape(str(best.get(first_s_col, best['model'])))} ({best['macro_F1_mean'] * 100:.1f}%)."
        )
