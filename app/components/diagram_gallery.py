"""System Blueprints & 4K Architecture Diagram Gallery Component.

Embeds and showcases the 11 Ultra-HD 4K system and UML diagrams
from SupplyGuard_Diagrams/ for direct presentation during viva defenses.
"""
import os
import streamlit as st

DIAGRAM_CATALOG = {
    "1. System Architecture Diagram": {
        "file": "System_Architecture_Diagram.png",
        "desc": "Modular pipeline architecture showing end-to-end data ingestion, Kipf-Welling GCN spatial aggregation, LSTM temporal modeling, and Integrated Gradients XAI layer."
    },
    "2. UML Class Diagram": {
        "file": "UML_Class_Diagram.png",
        "desc": "Object-oriented domain model with 8 core classes, aggregation diamonds, and PyTorch module inheritance."
    },
    "3. UML Use Case Diagram": {
        "file": "UML_Use_Case_Diagram.png",
        "desc": "System boundary illustrating 10 use cases across 4 primary actor roles (Supply Chain Officer, ML Engineer, Risk Analyst, Evaluator)."
    },
    "4. UML Sequence Diagram": {
        "file": "UML_Sequence_Diagram.png",
        "desc": "Detailed execution lifelines showing 11 sequential method calls during live window inference and Integrated Gradients autograd backward passes."
    },
    "5. UML Activity Diagram": {
        "file": "UML_Activity_Diagram.png",
        "desc": "6-stage operational pipeline with decision diamonds evaluating risk thresholds and triggering prescriptive alerts."
    },
    "6. UML Component Diagram": {
        "file": "UML_Component_Diagram.png",
        "desc": "8 modular components with UML interface sockets connecting data pipelines, graph layers, and web presentation."
    },
    "7. UML Deployment Diagram": {
        "file": "UML_Deployment_Diagram.png",
        "desc": "3D perspective node topology mapping client browser, application server runtime, and storage layers."
    },
    "8. UML Object Diagram": {
        "file": "UML_Object_Diagram.png",
        "desc": "Concrete runtime snapshot of echelon state objects during an active supply disruption event."
    },
    "9. UML Package Diagram": {
        "file": "UML_Package_Diagram.png",
        "desc": "6 architectural packages showing clean namespace dependencies."
    },
    "10. UML State Machine Diagram": {
        "file": "UML_State_Machine_Diagram.png",
        "desc": "Supply node risk lifecycle across 7 discrete operational states from Normalcy to Cascade Disruption and Restabilization."
    },
    "11. UML Collaboration Diagram": {
        "file": "UML_Collaboration_Diagram.png",
        "desc": "9 collaborating runtime objects organized in a 3x3 interaction ring with numbered message flow arrows."
    }
}


def render_diagram_gallery():
    """Render the 4K architectural diagram viewer."""
    st.markdown("### 📐 System Blueprints & 4K Architecture Gallery")
    st.markdown(
        "Explore the complete suite of **4K Ultra-HD (3840 × 2160)** design and UML diagrams "
        "documenting SupplyGuard's mathematical, architectural, and operational structure."
    )

    selected_diag = st.selectbox(
        "Select System Blueprint / UML Diagram to Inspect:",
        options=list(DIAGRAM_CATALOG.keys())
    )

    diag_info = DIAGRAM_CATALOG[selected_diag]
    filename = diag_info["file"]

    # Search for diagram path
    possible_paths = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "SupplyGuard_Diagrams", filename)),
        os.path.abspath(os.path.join("..", "SupplyGuard_Diagrams", filename)),
        os.path.abspath(os.path.join("SupplyGuard_Diagrams", filename)),
    ]

    found_path = None
    for p in possible_paths:
        if os.path.exists(p):
            found_path = p
            break

    st.markdown(f"""
    <div class="sg-card" style="margin-bottom: 15px;">
        <h4 style="color: #06B6D4; margin: 0 0 6px 0;">{selected_diag}</h4>
        <p style="color: #CBD5E1; font-size: 0.9rem; margin: 0 0 8px 0;">{diag_info['desc']}</p>
        <span style="font-size: 0.75rem; color: #64748B;">Resolution: <strong>3840 × 2160 (4K Ultra-HD)</strong> | Presentation-Grade Typography</span>
    </div>
    """, unsafe_allow_html=True)

    if found_path:
        st.image(found_path, use_container_width=True, caption=selected_diag)
    else:
        st.warning(f"Diagram file `{filename}` not found in local workspace path.")
