"""Hugging Face Space entry point.

Expects this folder to contain `src/` (copied from the repo) and `models/serving_bundle.joblib`.
See README.md for the deploy steps.
"""
import os
import sys

# make the bundled `src` package importable and point the app at the local model
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("SBA_BUNDLE", os.path.join(os.path.dirname(__file__), "models", "serving_bundle.joblib"))

from src.app.streamlit_app import main  # noqa: E402

main()
