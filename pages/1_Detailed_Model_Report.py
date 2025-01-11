import streamlit as st
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Detailed Model Report",
    page_icon="📊",
    layout="wide"
)

try:
    # Get report path from session state
    report_path = st.session_state.get('report_path')
    
    if report_path:
        logger.info(f"Loading report from: {report_path}")
        
        # Display the report
        with open(report_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
            
        # Show the report content
        st.title("AI Model Detailed Report")
        st.components.v1.html(html_content, height=800, scrolling=True)
        
        # Back button
        if st.button("← Back to Analysis"):
            try:
                st.switch_page("UI/ui_pages/analyser.py")
            except Exception as e:
                logger.error(f"Back navigation failed: {e}")
                st.error("Could not return to analyser")
    else:
        st.error("No report selected. Please generate a report from the Analyser page first.")
        
except Exception as e:
    logger.error(f"Error in report page: {e}")
    st.error("Error loading report") 