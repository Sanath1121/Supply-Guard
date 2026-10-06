# SupplyGuard — Visual Interface Guide & Feature Tour

> **Document Purpose**: A visual, beginner-friendly walkthrough of the SupplyGuard Dashboard with high-fidelity UI previews.  
> **Target Audience**: Business Stakeholders, Supply Chain Executives, Developers, and Evaluators.  
> **Key Analogy**: Think of SupplyGuard as a **weather radar and Google Maps for supply chains**—alerting you to incoming storms 10 minutes before impact.

---

## Table of Contents
1. [The 30-Second Elevator Pitch: What Are We Building?](#1-the-30-second-elevator-pitch-what-are-we-building)
2. [Screen 1: Executive Overview Deck](#2-screen-1-executive-overview-deck)
3. [Screen 2: Operational Risk Monitor (The Live Cockpit)](#3-screen-2-operational-risk-monitor-the-live-cockpit)
4. [Screen 3: Diagnostic XAI (The AI Detective)](#4-screen-3-diagnostic-xai-the-ai-detective)
5. [Screen 4: Production Benchmarks (The Proof / Report Card)](#5-screen-4-production-benchmarks-the-proof--report-card)
6. [Screen 5: Incident Dossier & Sandbox](#6-screen-5-incident-dossier--sandbox)
7. [Quick Reference: How to Explain Each Feature to a Non-Technical Client](#7-quick-reference-how-to-explain-each-feature-to-a-non-technical-client)

---

## 1. The 30-Second Elevator Pitch: What Are We Building?

Imagine a 4-step delivery pipeline:

$$\text{📦 Supplier} \longrightarrow \text{⚙️ Factory (Manufacturer)} \longrightarrow \text{🚚 Logistics (Distributor)} \longrightarrow \text{🏪 Store (Retailer)}$$

In traditional supply chains, if a supplier has an equipment failure, **nobody downstream knows until the factory stops working and trucks run empty**. This creates a devastating domino effect costing millions.

**SupplyGuard solves this**:
* It continuously monitors telemetry from all 4 tiers simultaneously.
* It predicts cascading bottlenecks **10 minutes before they happen**.
* It explains in plain English **who caused the delay** and **what immediate action managers should take**.

---

## 2. Screen 1: Executive Overview Deck

![SupplyGuard Executive Overview](assets/images/01_overview_view.jpg)

### What You Are Seeing in This Screen:

1. **Top KPI Hero Cards (The Vital Stats)**:
   * **Core AI Engine (`ST-GCN-LSTM`)**: Tells the user that our spatiotemporal network brain is active.
   * **Early Warning Lead Time (`+10 Mins`)**: The key business value. Gives operators a 10-minute head start.
   * **Echelon Coverage (`4 Tiers`)**: Full end-to-end visibility from Raw Materials to Retail.
   * **Data Verification (`0 Leakage`)**: Guarantees the AI was tested on untouched historical data without cheating or peeking ahead.

2. **Operational Readiness Checklist (Left Box)**:
   * Like a pre-flight inspection for a pilot.
   * Green checkmarks (✅) verify that data feeds, trained AI weights, and system connectors are online and healthy.

3. **Production Architecture Specifications (Right Box)**:
   * The system contract: lookback window (past 20 minutes), early warning horizon (10 minutes into the future), and data refresh rate (every 2 minutes).

4. **Operational Severity Protocols & Action Matrix (Bottom Box)**:
   * Standardizes what managers must do when an alert triggers:
     * 🟢 **Low Severity (Green)**: Normal operations. Monitor routine metrics.
     * 🟡 **Medium Severity (Amber)**: Emerging transit lag or capacity strain. Alert downstream distribution hubs.
     * 🔴 **High Severity (Red)**: Critical disruption imminent! Activate emergency buffer stock and re-route freight.

---

## 3. Screen 2: Operational Risk Monitor (The Live Cockpit)

![SupplyGuard Risk Monitor](assets/images/02_monitor_view.jpg)

### What You Are Seeing in This Screen:

1. **The Timeline Scrubber (Top Slider & Controls)**:
   * Like a security camera replay bar.
   * Drag the slider or click `◀` / `▶` to travel through time and inspect how risks evolved every 2 minutes.
   * **⚡ "Jump to Peak TRI" Button**: The emergency shortcut! One click instantly jumps to the single worst disaster in the test set.

2. **Total Risk Index (TRI) Speedometer Gauge (Left Gauge)**:
   * The "heart monitor" of the entire supply chain.
   * Ranges from $0.00$ (completely safe) to $1.00$ (total system collapse).
   * In this preview, it reads **`0.742 HIGH SEVERITY`**, triggering an amber-red needle warning.
   * Displays the **Delta ($\Delta$)** compared to right now so you know if things are getting better or worse.

3. **The 4 Echelon Risk Cards (Center Row)**:
   * Individual health cards for **Supplier**, **Manufacturer**, **Distributor**, and **Retailer**.
   * Shows each tier's risk score and color-coded severity tag.
   * Notice how **Supplier (`0.812`)** and **Manufacturer (`0.742`)** are red, while **Distributor (`0.415`)** is amber, and **Retailer (`0.210`)** is still green. The disaster is currently traveling downstream!
   * Click **"Diagnose Node →"** on any card to find out why that tier is struggling.

4. **Cascading Network Topology (Bottom Flow Diagram)**:
   * Visual map showing the 4 steps connected by glowing blue pipes with animated moving pulses.
   * **Attribution Badges on the Pipes (e.g. `68%`, `70%`)**: Reveals the domino effect! It mathematically proves how much risk traveled from the upstream partner to the downstream partner.

---

## 4. Screen 3: Diagnostic XAI (The AI Detective)

![SupplyGuard Diagnostic XAI](assets/images/03_xai_view.jpg)

### What You Are Seeing in This Screen:

Most AI models are "black boxes"—they spit out a number, but cannot explain why. **SupplyGuard provides complete transparency.**

1. **Target Echelon Selector (Top Radio Buttons)**:
   * Choose which company you want to inspect: Supplier, Manufacturer, Distributor, or Retailer.

2. **AI Diagnostic Narrative (Top Card)**:
   * Auto-generated diagnosis in plain English:
     > *"Model predicts HIGH risk for Manufacturer. Primary upstream driver is Supplier delays (contributing 62.4% of total gradient saliency), concentrated 4 minutes ago."*

3. **Action Directive Banner (Blue Highlighted Box)**:
   * Clear, actionable instructions for operations managers:
     > *⚙️ Plant Action: Rebalance batch scheduling; verify backup parts across alternate production lines.*

4. **Feature Signed Attribution (Left Chart)**:
   * **Red bars**: Factors pushing risk **UP** (e.g., supplier material shortages or fuel cost inflation).
   * **Green bars**: Factors keeping risk **DOWN** (e.g., stable retailer inventory).

5. **Temporal Saliency Profile (Right Chart)**:
   * Shows **when** the issue started across time steps $t-9$ to $t_0$.
   * Tells you if this was a sudden lightning shock ($t_0$) or a slow backlog brewing over 15 minutes.

6. **Axiomatic Completeness & Counterfactual Test (Bottom Cards)**:
   * **Completeness Audit**: Mathematical proof verifying that the attribution adds up to 100% of the prediction.
   * **Sensitivity Test**: Allows managers to simulate: *"What if we fix the supplier delay right now? How much will factory risk drop?"*

---

## 5. Screen 4: Production Benchmarks (The Proof / Report Card)

![SupplyGuard Production Benchmarks](assets/images/04_benchmarks_view.jpg)

### What You Are Seeing in This Screen:

This is the evidence you present to clients, management, or evaluators to prove the AI is scientifically validated.

1. **SLA Validation Scorecards (Top 4 Cards)**:
   * **Multi-Node Coordinated Accuracy**: Proves tracking all 4 tiers simultaneously beats single-node models.
   * **Spatiotemporal Topology Gain**: Proves that feeding the physical graph structure into the AI makes it significantly smarter than a standard graph-free LSTM.
   * **+10 Min Early Warning Superiority**: Proves our AI decisively beats naive guessing ("persistence") at the 10-minute horizon.
   * **Critical Incident Recall**: Proves that when severe disruptions happen, the AI catches over 90% of them with minimal false alarms.

2. **Leaderboard Table (Middle Data Grid)**:
   * An empirical comparison table ranking **SupplyGuard ST-GCN-LSTM**, **Ablation LSTM**, **Global Aggregate**, **Ridge-AR**, and **Naive Persistence** across standard error metrics (MSE, MAE, RMSE, and $R^2$) tested across 5 random seeds.

3. **Test MSE Error-Whisker Bar Chart (Bottom Chart)**:
   * Displays prediction error (lower bar = better model).
   * Notice the vertical "whiskers" on each bar: these represent standard error across the 5 independent seeds, proving our AI's lead is statistically rock-solid.

---

## 6. Screen 5: Incident Dossier & Sandbox

* **Incident Dossier (`/export`)**:
  * In a crisis, managers need to brief leadership immediately.
  * One click generates a clean, cryptographically hashed Markdown audit report containing the active timestamp, risk scores, root cause attribution, and recommended actions.
  * Click **"📥 Download Incident Report (.md)"** to save and distribute the briefing.

* **Stress-Test Sandbox (`/sandbox`)**:
  * A safe "flight simulator" environment.
  * Run preset disaster simulations like **"Upstream Supplier Blackout"** or upload custom CSV spreadsheets to test novel what-if scenarios without touching historical data.

---

## 7. Quick Reference: How to Explain Each Feature to a Non-Technical Client

| Feature | Everyday Analogy | What to Say in 5 Seconds |
|---|---|---|
| **Timeline Scrubber** | YouTube Video Timeline | *"Lets you rewind and fast-forward to see how supply chain risks unfold every 2 minutes."* |
| **TRI Speedometer** | Car Speedometer / Heart Rate Monitor | *"One single gauge showing if your entire 4-tier supply chain is safe, strained, or in critical danger."* |
| **Echelon Cards** | Individual Medical Health Cards | *"Gives an individual health check score for your Supplier, Factory, Logistics, and Store."* |
| **Topology Diagram** | Subway Map with Flow Lights | *"Shows the real delivery route, with moving pulses showing exactly how a supplier delay spills into the factory."* |
| **Diagnostic Narrative** | Doctor's Prescription | *"Explains in plain English what went wrong and gives the manager exact steps to fix it."* |
| **Feature Bar Chart** | Nutrition Label / Ingredient Breakdown | *"Shows which specific variables (e.g. freight cost vs parts shortage) caused the fever."* |
| **Benchmark Scorecards** | Consumer Reports / Crash Test Ratings | *"Proves our AI beats standard industry baselines across 5 independent scientific test runs."* |
| **Download Dossier** | One-Click Export PDF | *"Generates a signed incident briefing in one second to email to senior leadership."* |
