"""
Streamlit Cloud Entrypoint for VisionAI Face Detection Studio
"""

import sys
import os

# Ensure current directory is in sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app_streamlit import main

if __name__ == "__main__":
    main()
