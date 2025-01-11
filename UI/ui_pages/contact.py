import streamlit as st

def show():
    """Show the contact page content."""
    st.title("Contact Us")
    
    st.write("""
    ### Get in Touch
    
    We're here to help with any questions about the AEPF platform.
    """)
    
    # Contact form
    with st.form("contact_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        subject = st.selectbox(
            "Subject",
            ["General Inquiry", "Technical Support", "Bug Report", "Feature Request"]
        )
        message = st.text_area("Message")
        submitted = st.form_submit_button("Send Message")
        
        if submitted:
            st.success("Thank you for your message! We'll get back to you soon.")
    
    # Contact information
    st.subheader("Other Ways to Reach Us")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("""
        #### Technical Support
        📧 support@aepf.org  
        ⏰ Response time: 24-48 hours
        """)
    
    with col2:
        st.write("""
        #### General Inquiries
        📧 info@aepf.org  
        📞 +1 (555) 123-4567
        """)
    
    # Office hours
    st.write("""
    ### Office Hours
    
    Monday - Friday: 9:00 AM - 5:00 PM (EST)  
    Saturday - Sunday: Closed
    
    *For urgent matters, please use the emergency contact form on our website.*
    """)

if __name__ == "__main__":
    show()



