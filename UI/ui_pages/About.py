import streamlit as st

def show():
    """Show the about page content."""
    st.title("About AEPF")
    
    st.write("""
    ### The Adaptive Ethical Prism Framework (AEPF)
    
    AEPF is a comprehensive framework designed to evaluate and guide ethical decision-making 
    in artificial intelligence systems.
    """)
    
    # Contact information
    st.sidebar.info("""
    ### Contact Us
    
    For more information or support:  
    📧 support@aepf.org  
    🌐 www.aepf.org
    """)

if __name__ == "__main__":
    show()

