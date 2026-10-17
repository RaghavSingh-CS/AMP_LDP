"""Launch from repository root: python -m streamlit run app/main.py."""

import sys
from pathlib import Path

# Streamlit executes this file as a script, so make package imports independent
# of whether it places app/ or the repository root first on sys.path.
root = str(Path(__file__).resolve().parents[1])
if root not in sys.path:
    sys.path.insert(0, root)

import streamlit as st

from app.navigation import PAGES
from app.views import demo_guide, explorer, overview, performance

st.set_page_config(page_title="AMP | Electricity Theft Risk Profiling", page_icon="⚡", layout="wide")
st.set_option("client.disableDataExport", True)
st.set_option("client.toolbarMode", "viewer")
st.title("AMP")
st.write("**Automated Meter Profiling for Electricity Theft Detection**")
st.sidebar.title("AMP")
st.sidebar.caption("Frozen research · Customer inspection · Performance")
page = st.sidebar.radio("Navigation", PAGES, key="amp_page")
st.sidebar.markdown("[Back to page start](#amp)")
st.sidebar.divider()
st.sidebar.caption("Customer-level research ranking. Research results are read-only.")
if page == "Overview":
    overview.render()
elif page == "Customer Risk Explorer":
    explorer.render()
elif page == "Model Performance":
    performance.render()
else:
    demo_guide.render()
