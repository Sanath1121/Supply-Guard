"""Home & Executive Overview Page Component.

Vercel / Linear styled product overview page with ambient glowing hero banner,
key metric counters, interactive multi-echelon anatomy cards, and 1-click launchpad.
"""
import streamlit as st


def render_home_page():
    """Render the masterclass executive home page and onboarding guide."""
    # Hero Deck Banner
    st.markdown("""
    <div class="hero-deck">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
            <span class="badge-pill badge-cyan">
                <span class="beacon-pulse" style="background-color: #00F2FE;"></span> IEEE ICCMC 2025 EXTENSION
            </span>
            <span class="badge-pill badge-purple">B.TECH CAPSTONE DEMO</span>
        </div>
        <h1 class="gradient-title" style="font-size: 2.8rem; margin: 0 0 12px 0;">
            SupplyGuard: Intelligent Spatiotemporal Risk Radar
        </h1>
        <p style="font-size: 1.1rem; color: #CBD5E1; max-width: 880px; line-height: 1.6; margin: 0 0 24px 0;">
            An empirical deep-learning decision support system that forecasts next-step operational risk separately across 
            <strong>Supplier</strong>, <strong>Manufacturer</strong>, <strong>Distributor</strong>, and <strong>Retailer</strong> 
            echelons using hybrid Graph Convolutional Networks (GCN) and Long Short-Term Memory (LSTM) with validated gradient attributions.
        </p>
        
        <!-- Live Telemetry KPI Chips -->
        <div style="display: flex; flex-wrap: wrap; gap: 14px; margin-top: 15px;">
            <div style="background: rgba(11, 15, 25, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); padding: 8px 16px; border-radius: 10px;">
                <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase;">Benchmark Records</div>
                <div style="font-size: 1.2rem; font-weight: 800; font-family: 'Outfit'; color: #00F2FE;">649,999</div>
            </div>
            <div style="background: rgba(11, 15, 25, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); padding: 8px 16px; border-radius: 10px;">
                <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase;">Graph Receptive Field</div>
                <div style="font-size: 1.2rem; font-weight: 800; font-family: 'Outfit'; color: #00F5A0;">2-Hop Reach</div>
            </div>
            <div style="background: rgba(11, 15, 25, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); padding: 8px 16px; border-radius: 10px;">
                <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase;">Attribution Steps</div>
                <div style="font-size: 1.2rem; font-weight: 800; font-family: 'Outfit'; color: #C77DFF;">64 Riemann Path</div>
            </div>
            <div style="background: rgba(11, 15, 25, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); padding: 8px 16px; border-radius: 10px;">
                <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase;">Temporal Lookback</div>
                <div style="font-size: 1.2rem; font-weight: 800; font-family: 'Outfit'; color: #FFB300;">10 Lags (20 Min)</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Core Problem vs Solution
    st.markdown("### 🌐 The Core Innovation: Networked Spatiotemporal Intelligence")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="sg-glass-card" style="height: 100%; border-left: 4px solid #FF2E54;">
            <h4 style="color: #FF2E54; margin: 0 0 8px 0; font-family: 'Outfit';">❌ The Siloed Forecasting Failure</h4>
            <p style="color: #94A3B8; font-size: 0.9rem; line-height: 1.6; margin: 0;">
                Traditional ERP and BI tools treat each facility as an isolated time series. When an upstream supplier encounters a raw material embargo, downstream factories only discover the disruption days later after production lines halt.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="sg-glass-card" style="height: 100%; border-left: 4px solid #00F5A0;">
            <h4 style="color: #00F5A0; margin: 0 0 8px 0; font-family: 'Outfit';">✅ The Spatiotemporal Graph GCN-LSTM Solution</h4>
            <p style="color: #94A3B8; font-size: 0.9rem; line-height: 1.6; margin: 0;">
                SupplyGuard represents the linear supply chain as a directed graph. A 2-layer GCN propagates structural embeddings across echelons at every time step, while per-node LSTMs capture temporal velocity, predicting downstream bottlenecks before inventory runs out.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 4 Interactive Echelon Anatomy Cards
    st.markdown("### 🏭 Multi-Echelon Anatomy & Operational Roles")

    ec1, ec2, ec3, ec4 = st.columns(4)
    with ec1:
        st.markdown("""
        <div class="sg-glass-card echelon-low">
            <div style="font-size: 1.6rem; margin-bottom: 6px;">📦</div>
            <h4 style="color: #00F5A0; margin: 0 0 4px 0; font-family: 'Outfit';">1. Supplier</h4>
            <div style="font-size: 0.74rem; color: #64748B; margin-bottom: 8px;">Tier-1 & Tier-2 Vendors</div>
            <p style="color: #CBD5E1; font-size: 0.82rem; line-height: 1.4; margin: 0 0 10px 0;">
                Raw materials, mining, component sourcing, and vendor lead-time reliability.
            </p>
            <span class="badge-pill badge-low" style="font-size: 0.68rem;">Sourcing Variance</span>
        </div>
        """, unsafe_allow_html=True)

    with ec2:
        st.markdown("""
        <div class="sg-glass-card echelon-med">
            <div style="font-size: 1.6rem; margin-bottom: 6px;">⚙️</div>
            <h4 style="color: #FFB300; margin: 0 0 4px 0; font-family: 'Outfit';">2. Manufacturer</h4>
            <div style="font-size: 0.74rem; color: #64748B; margin-bottom: 8px;">Assembly & Production</div>
            <p style="color: #CBD5E1; font-size: 0.82rem; line-height: 1.4; margin: 0 0 10px 0;">
                Assembly lines, factory throughput, capacity utilization, and WIP inventory.
            </p>
            <span class="badge-pill badge-medium" style="font-size: 0.68rem;">Yield & Bottlenecks</span>
        </div>
        """, unsafe_allow_html=True)

    with ec3:
        st.markdown("""
        <div class="sg-glass-card echelon-hig">
            <div style="font-size: 1.6rem; margin-bottom: 6px;">🚚</div>
            <h4 style="color: #FF2E54; margin: 0 0 4px 0; font-family: 'Outfit';">3. Distributor</h4>
            <div style="font-size: 0.74rem; color: #64748B; margin-bottom: 8px;">Logistics & Warehousing</div>
            <p style="color: #CBD5E1; font-size: 0.82rem; line-height: 1.4; margin: 0 0 10px 0;">
                Regional warehousing, freight routing, multi-modal transit, and 3PL carriers.
            </p>
            <span class="badge-pill badge-high" style="font-size: 0.68rem;">Transit Dwell Times</span>
        </div>
        """, unsafe_allow_html=True)

    with ec4:
        st.markdown("""
        <div class="sg-glass-card echelon-low">
            <div style="font-size: 1.6rem; margin-bottom: 6px;">🛒</div>
            <h4 style="color: #00F2FE; margin: 0 0 4px 0; font-family: 'Outfit';">4. Retailer</h4>
            <div style="font-size: 0.74rem; color: #64748B; margin-bottom: 8px;">Point-of-Sale / End User</div>
            <p style="color: #CBD5E1; font-size: 0.82rem; line-height: 1.4; margin: 0 0 10px 0;">
                Stores, e-commerce fulfillment, consumer demand volatility, and stockouts.
            </p>
            <span class="badge-pill badge-cyan" style="font-size: 0.68rem;">Stockout & Bullwhip</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Launchpad Cards
    st.markdown("### 🚀 Module Quick-Launchpad")
    lp1, lp2, lp3 = st.columns(3)
    with lp1:
        st.markdown("""
        <div class="sg-glass-card" style="text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 8px;">⚡</div>
            <h4 style="color: #FFFFFF; margin: 0 0 4px 0; font-family: 'Outfit';">Live Mission Control</h4>
            <p style="color: #94A3B8; font-size: 0.82rem;">Real-time radar gauge, echelon health cards, and historical scrubbing.</p>
        </div>
        """, unsafe_allow_html=True)

    with lp2:
        st.markdown("""
        <div class="sg-glass-card" style="text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 8px;">🔍</div>
            <h4 style="color: #FFFFFF; margin: 0 0 4px 0; font-family: 'Outfit';">Explainability Lab</h4>
            <p style="color: #94A3B8; font-size: 0.82rem;">Signed Integrated Gradients waterfall, temporal profile, and &Delta;-attribution.</p>
        </div>
        """, unsafe_allow_html=True)

    with lp3:
        st.markdown("""
        <div class="sg-glass-card" style="text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 8px;">📂</div>
            <h4 style="color: #FFFFFF; margin: 0 0 4px 0; font-family: 'Outfit';">Dataset Ingestion Studio</h4>
            <p style="color: #94A3B8; font-size: 0.82rem;">Upload custom CSVs or launch 1-click preset disruption shock scenarios.</p>
        </div>
        """, unsafe_allow_html=True)
