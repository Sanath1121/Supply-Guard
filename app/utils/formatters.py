"""SupplyGuard Metric & Graph Formatting Utilities.

Directly compliant with tests/test_phase7_app.py contracts.
"""
from typing import Dict, Any
import networkx as nx
import numpy as np


def format_echelon_card(node_name: str, scaled_val: float, raw_val: float, tier: str) -> Dict[str, str]:
    """Format data for an echelon risk card.
    
    Complies with Gate 7 test contract in tests/test_phase7_app.py.
    """
    tier_colors = {"Low": "green", "Medium": "orange", "High": "red"}
    return {
        "title": f"{node_name} Risk",
        "scaled_score": f"{scaled_val:.3f}",
        "raw_metric": f"{raw_val:.2f} RI",
        "tier": tier,
        "color": tier_colors.get(tier, "gray"),
    }


def build_supply_chain_digraph(node_tiers: Dict[str, str], upstream_shares: Dict[Any, float]) -> nx.DiGraph:
    """Build directed 4-node supply chain graph with tier colors and edge attribution widths.
    
    Complies with Gate 7 test contract in tests/test_phase7_app.py.
    """
    G = nx.DiGraph()
    echelons = ["Supplier", "Manufacturer", "Distributor", "Retailer"]
    for e in echelons:
        G.add_node(e, tier=node_tiers.get(e, "Low"))

    edges = [("Supplier", "Manufacturer"), ("Manufacturer", "Distributor"), ("Distributor", "Retailer")]
    for u, v in edges:
        weight = upstream_shares.get((u, v), 1.0)
        G.add_edge(u, v, weight=weight)
    return G


def compute_tercile_tier(value: float, p33: float = 0.35, p66: float = 0.65) -> str:
    """Classify risk value into tercile severity tier."""
    if value < p33:
        return "Low"
    elif value <= p66:
        return "Medium"
    else:
        return "High"


def get_tier_color(tier: str) -> str:
    """Hex color mapping for UI consistency."""
    mapping = {
        "Low": "#10B981",      # Emerald Green
        "Medium": "#F59E0B",   # Amber Orange
        "High": "#EF4444",     # Crimson Red
    }
    return mapping.get(tier, "#94A3B8")
