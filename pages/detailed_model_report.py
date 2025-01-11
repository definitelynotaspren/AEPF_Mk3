import streamlit as st
from pathlib import Path

# Page config
st.set_page_config(page_title="AI Model Detailed Report", layout="wide")

# Back button
if st.button("← Back to Analysis"):
    st.markdown('<meta http-equiv="refresh" content="0;url=/analyser">', unsafe_allow_html=True)

# Show report
report_path = st.session_state.get('report_path')
if report_path:
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=800, scrolling=True)
    except Exception as e:
        st.error(f"Could not load report: {e}")
else:
    st.error("No report path specified") 