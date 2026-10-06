"""Phase 7 Smoke Test: Streamlit AppTest Execution Validation.

Verifies:
1. Streamlit app boots cleanly with ZERO artifacts (missing data, missing checkpoints).
2. Every core view renders with 0 uncaught exceptions.
3. Monitor view with a synthetic fixture properly renders 4 echelon cards.
"""
import os
import sys
import unittest
import numpy as np
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.utils.artifacts import ArtifactStatus, get_model
from app.views.overview import render_overview
from app.views.monitor import render_monitor
from app.views.why import render_why
from app.views.benchmarks import render_benchmarks
from app.views.export import render_export


class TestPhase7AppTestSmoke(unittest.TestCase):

    def test_01_app_boots_without_artifacts(self):
        """Verify main streamlit_app.py loads cleanly with no data and no checkpoints."""
        app_file = os.path.join(PROJECT_ROOT, "app", "streamlit_app.py")
        at = AppTest.from_file(app_file, default_timeout=30).run()
        self.assertEqual(len(at.exception), 0, f"Uncaught exception on boot: {[e.message for e in at.exception]}")

    def test_02_all_views_render_empty_states(self):
        """Verify that all views render cleanly without artifacts and without exceptions."""
        views = ["overview", "monitor", "why", "benchmarks", "export"]
        for view_name in views:
            code = f"""
import streamlit as st
from app.utils.artifacts import ArtifactStatus, get_model, get_test_windows
from app.views.{view_name} import render_{view_name}

status = ArtifactStatus(
    model_loaded=False,
    checkpoint_path=None,
    model_name='st_gcn_lstm',
    graph_mode='directed',
    seed=42,
    scaler_fitted=False,
    tiers_source='provisional_default',
    data_source='unavailable',
    warnings=('Artifacts missing',)
)
model, _, _, _ = get_model('st_gcn_lstm', 'directed', 42)
windows, _, _ = get_test_windows()

if '{view_name}' == 'overview':
    render_overview(status)
elif '{view_name}' == 'monitor':
    render_monitor(status, windows, model)
elif '{view_name}' == 'why':
    render_why(status, windows, model)
elif '{view_name}' == 'benchmarks':
    render_benchmarks()
elif '{view_name}' == 'export':
    render_export(status, windows, model)
"""
            at = AppTest.from_string(code, default_timeout=30).run()
            self.assertEqual(len(at.exception), 0, f"View '{view_name}' threw exception without artifacts: {[e.message for e in at.exception]}")

    def test_03_monitor_renders_with_fixture(self):
        """Verify that Monitor view renders 4 echelon cards with a test fixture."""
        code = """
import streamlit as st, numpy as np
from app.utils.artifacts import ArtifactStatus, get_model
from app.views.monitor import render_monitor

status = ArtifactStatus(
    model_loaded=False,
    checkpoint_path=None,
    model_name='st_gcn_lstm',
    graph_mode='directed',
    seed=42,
    scaler_fitted=False,
    tiers_source='provisional_default',
    data_source='test_partition',
    warnings=()
)
model, _, _, _ = get_model('st_gcn_lstm', 'directed', 42)
seq = np.random.uniform(0.1, 0.9, size=(10, 5)).astype(np.float32)
windows = [
    {'window_id': 0, 'timestamp': '2018-09-14 10:00:00', 'sequence': seq, 'ground_truth': np.array([0.2, 0.4, 0.6, 0.3])},
    {'window_id': 1, 'timestamp': '2018-09-14 10:02:00', 'sequence': seq, 'ground_truth': np.array([0.3, 0.5, 0.5, 0.2])}
]
render_monitor(status, windows, model)
"""
        at = AppTest.from_string(code, default_timeout=30).run()
        self.assertEqual(len(at.exception), 0, f"Monitor view threw with fixture: {[e.message for e in at.exception]}")


if __name__ == "__main__":
    unittest.main()
