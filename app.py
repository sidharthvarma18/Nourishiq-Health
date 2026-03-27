"""
app.py — NourishIQ Analytics Dashboard
Run:  streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="NourishIQ Analytics",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/salad.png", width=60)
st.sidebar.title("NourishIQ")
st.sidebar.caption("Data-Driven Decision Dashboard")
st.sidebar.markdown("---")

PAGES = {
    "🏠  Home":                 "page_home",
    "📊  Descriptive Analysis": "page_descriptive",
    "🔍  Diagnostic Analysis":  "page_diagnostic",
    "🤖  Predictive Analysis":  "page_predictive",
    "📋  Association Rules":    "page_arm",
    "📈  Regression Analysis":  "page_regression",
    "🚀  New Lead Scorer":      "page_upload",
}

selection = st.sidebar.radio("Navigate", list(PAGES.keys()))
st.sidebar.markdown("---")
st.sidebar.info("Upload your own survey CSV on the **New Lead Scorer** page to score new customers instantly.")

# ── Route to page ─────────────────────────────────────────────────────────────
page_key = PAGES[selection]

if page_key == "page_home":
    import page_home as pg
elif page_key == "page_descriptive":
    import page_descriptive as pg
elif page_key == "page_diagnostic":
    import page_diagnostic as pg
elif page_key == "page_predictive":
    import page_predictive as pg
elif page_key == "page_arm":
    import page_arm as pg
elif page_key == "page_regression":
    import page_regression as pg
elif page_key == "page_upload":
    import page_upload as pg

pg.show()
