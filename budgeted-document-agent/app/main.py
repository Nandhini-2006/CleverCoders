import sys
from pathlib import Path

# Ensure the project root directory is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from app.ui.chat import render_chat


st.set_page_config(
    page_title="Budgeted Document Agent",
    page_icon="📄",
    layout="wide"
)

render_chat()