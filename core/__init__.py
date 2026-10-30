"""
Face Detection Application Core Package
"""

import sys
import os

# Ensure repo root and core directory are in sys.path
core_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(core_dir)
for p in [root_dir, core_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from .detector import FaceDetector, DetectionResult
    from .visualizer import FaceVisualizer
    from .utils import load_image, save_image, crop_faces, export_detections_json
except (ImportError, ValueError):
    from detector import FaceDetector, DetectionResult
    from visualizer import FaceVisualizer
    from utils import load_image, save_image, crop_faces, export_detections_json

__all__ = [
    "FaceDetector",
    "DetectionResult",
    "FaceVisualizer",
    "load_image",
    "save_image",
    "crop_faces",
    "export_detections_json",
]

# If Streamlit is mistakenly configured to run core/__init__.py as the entrypoint:
def _run_if_streamlit():
    try:
        import streamlit as st
        # If running inside streamlit script runner
        if st.runtime.exists():
            import app_streamlit
            app_streamlit.main()
    except Exception:
        pass

_run_if_streamlit()
