"""Home & Executive Overview Page Component.

Provides a clean, user-friendly, non-technical walkthrough of the supply chain
risk problem, multi-echelon concepts, and 1-click launchpad shortcuts.
"""
import streamlit as st


def render_home_page():
    """Render the masterclass executive home page and onboarding guide."""
    st.markdown("""
    <div class="hero-container">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
            <span class="badge-pill badge-low" style="font-size: 0.8rem; padding: 0.3rem 0.8rem;">
                <span class="beacon-pulse" style="background-color: #10B981;"></span> OPERATIONAL AI ENGINE
            </span>
            <span style="color: #94A3B8; font-size: 0.85rem;">IEEE ICCMC 2025 Extension • B.Tech Capstone Project</span>
        </div>
        <h1 style="font-size: 2.4rem; font-weight: 800; color: #FFFFFF; margin: 0 0 10px 0;">
            SupplyGuard: Intelligent Spatiotemporal Risk Radar
        </h1>
        <p style="font-size: 1.05rem; color: #CBD5E1; max-width: 900px; line-height: 1.6; margin: 0 0 20px 0;">
            An empirical deep-learning decision support system that forecasts next-step operational risk separately across 
            <strong>Supplier</strong>, <strong>Manufacturer</strong>, <strong>Distributor</strong>, and <strong>Retailer</strong> 
            echelons using hybrid Graph Convolutional Networks (GCN) and Long Short-Term Memory (LSTM) with validated gradient attributions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Core Problem Framing
    st.markdown("### 🌐 The Challenge: Why Traditional Supply Chain Alerts Fail")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="sg-card" style="height: 100%;">
            <h4 style="color: #EF4444; margin-top: 0;">❌ Isolated / Tabular Forecasting Trap</h4>
            <p style="color: #94A3B8; font-size: 0.9rem; line-height: 1.5;">
                Traditional ERP systems treat each factory or warehouse as an isolated silo. They evaluate metrics like local stock or shipment delays independently, failing to detect when an upstream supplier crisis will cascade into downstream retail stockouts 48 hours later.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="sg-card" style="height: 100%;">
            <h4 style="color: #10B981; margin-top: 0;">✅ The Spatiotemporal Network Solution</h4>
            <p style="color: #94A3B8; font-size: 0.9rem; line-height: 1.5;">
                SupplyGuard explicitly models the supply chain as a <strong>directed spatial graph</strong> combined with <strong>temporal LSTM memory</strong>. Disruptions propagating downstream and bullwhip demand shocks traveling upstream are captured jointly across the entire network topology.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 4 Interactive Echelon Concept Cards
    st.markdown("### 🏭 Multi-Echelon Network Anatomy")
    st.markdown("Click on any echelon below to understand its operational role in the supply chain:")

    ec1, ec2, ec3, ec4 = st.columns(4)
    with ec1:
        st.markdown("""
        <div class="sg-card sg-card-low">
            <h4 style="color: #10B981; margin: 0 0 6px 0;">1. Supplier (S)</h4>
            <div style="font-size: 0.78rem; color: #64748B; margin-bottom: 8px;">Tier-1 & Tier-2 Vendors</div>
            <p style="color: #CBD5E1; font-size: 0.85rem; line-height: 1.4;">
                Raw materials, mining, component sourcing, and vendor lead-time reliability.
            </p>
            <div style="color: #94A3B8; font-size: 0.75rem;"><strong>Risk Metric:</strong> Sourcing Delay & Defect Rate</div>
        </div>
        """, unsafe_allow_html=True)

    with ec2:
        st.markdown("""
        <div class="sg-card sg-card-medium">
            <h4 style="color: #F59E0B; margin: 0 0 6px 0;">2. Manufacturer (M)</h4>
            <div style="font-size: 0.78rem; color: #64748B; margin-bottom: 8px;">Assembly & Production</div>
            <p style="color: #CBD5E1; font-size: 0.85rem; line-height: 1.4;">
                Assembly lines, factory throughput, capacity utilization, and WIP inventory.
            </p>
            <div style="color: #94A3B8; font-size: 0.75rem;"><strong>Risk Metric:</strong> Line Stoppage & Yield Loss</div>
        </div>
        """, unsafe_allow_html=True)

    with ec3:
        st.markdown("""
        <div class="sg-card sg-card-high">
            <h4 style="color: #EF4444; margin: 0 0 6px 0;">3. Distributor (D)</h4>
            <div style="font-size: 0.78rem; color: #64748B; margin-bottom: 8px;">Logistics & Fulfillment</div>
            <p style="color: #CBD5E1; font-size: 0.85rem; line-height: 1.4;">
                Regional warehousing, freight routing, multi-modal transit, and 3PL carriers.
            </p>
            <div style="color: #94A3B8; font-size: 0.75rem;"><strong>Risk Metric:</strong> Transit Delay & Carrier Surcharge</div>
        </div>
        """, unsafe_allow_html=True)

    with ec4:
        st.markdown("""
        <div class="sg-card sg-card-low">
            <h4 style="color: #06B6D4; margin: 0 0 6px 0;">4. Retailer (R)</h4>
            <div style="font-size: 0.78rem; color: #64748B; margin-bottom: 8px;">Point-of-Sale / End Customer</div>
            <p style="color: #CBD5E1; font-size: 0.85rem; line-height: 1.4;">
                Stores, e-commerce fulfillment, consumer demand volatility, and stockouts.
            </p>
            <div style="color: #94A3B8; font-size: 0.75rem;"><strong>Risk Metric:</strong> Stockout Rate & Bullwhip Fluctuation</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3 Pillars of SupplyGuard
    st.markdown("### 🔬 Academic & Technical Foundations")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown("""
        <div class="sg-card">
            <div style="font-size: 1.5rem; margin-bottom: 8px;">🧠</div>
            <h4 style="color: #F1F5F9; margin: 0 0 6px 0;">Spatiotemporal Hybrid</h4>
            <p style="color: #94A3B8; font-size: 0.85rem; line-height: 1.5;">
                2-layer Graph Convolutions enable 2-hop structural propagation (Supplier &rarr; Distributor reach), while per-node LSTMs track 10 historical lookback steps.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with p2:
        st.markdown("""
        <div class="sg-card">
            <div style="font-size: 1.5rem; margin-bottom: 8px;">🔍</div>
            <h4 style="color: #F1F5F9; margin: 0 0 6px 0;">Axiomatic Explainability</h4>
            <p style="color: #94A3B8; font-size: 0.85rem; line-height: 1.5;">
                Uses 64-step Integrated Gradients to provide signed sensitivity attributions. Features the novel <strong>&Delta;-Attribution</strong> separating neural gain from persistence.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with p3:
        st.markdown("""
        <div class="sg-card">
            <div style="font-size: 1.5rem; margin-bottom: 8px;">⚖️</div>
            <h4 style="color: #F1F5F9; margin: 0 0 6px 0;">Empirical Benchmarking</h4>
            <p style="color: #94A3B8; font-size: 0.85rem; line-height: 1.5;">
                Rigorously benchmarked against Persistence, Ridge-AR(10), and Graph-Free LSTM across 5 seeds (42–46). Evaluates honest value over strong baselines.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Quick Start Action Box
    st.markdown("""
    <div class="prescriptive-box" style="border-left-color: #10B981;">
        <strong>🚀 Quick Start Tip:</strong> Select <strong>"⚡ Mission Control"</strong> in the sidebar to view live echelon forecasts and historical replay, or head to <strong>"📂 Dataset Studio"</strong> to test custom CSV uploads and preset crisis events.
    </div>
    """, unsafe_allow_html=True)
