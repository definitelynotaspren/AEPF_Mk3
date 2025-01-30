import streamlit as st
from datetime import datetime

def show():
    """Display the contact page content."""
    # Clear any previous content
    st.empty()
    
    # Use a container for all content
    with st.container():
        st.title("Contact")
        
        # Initialize session state
        if 'form_submitted' not in st.session_state:
            st.session_state.form_submitted = False
        
        # Display form and handle submission
        if not st.session_state.form_submitted:
            with st.form("contact_form_main"):
                # Contact details
                name = st.text_input("Name")
                email = st.text_input("Email")
                
                # Inquiry type dropdown
                inquiry_type = st.selectbox(
                    "Type of Inquiry",
                    [
                        "Select...",
                        "General Enquiry",
                        "Collaboration Proposal",
                        "Funding Offer",
                        "Technical Suggestions"
                    ]
                )
                
                # Show relevant fields based on inquiry type
                if inquiry_type == "Collaboration Proposal":
                    organization = st.text_input("Organization Name")
                    collaboration_type = st.selectbox(
                        "Collaboration Type",
                        ["Research", "Development", "Implementation", "Other"]
                    )
                    
                elif inquiry_type == "Funding Offer":
                    organization = st.text_input("Organization/Fund Name")
                    funding_amount = st.text_input("Proposed Funding Amount (Optional)")
                    
                elif inquiry_type == "Technical Suggestions":
                    area = st.selectbox(
                        "Technical Area",
                        ["AI Model", "Ethics Framework", "UI/UX", "Documentation", "Other"]
                    )
                
                # Common fields
                message = st.text_area("Message", height=150)
                
                # Submit button
                submitted = st.form_submit_button("Send Message")
                
                if submitted:
                    if name and email and inquiry_type != "Select..." and message:
                        # Create mailto link with form data
                        subject = f"AEPF Contact Form: {inquiry_type}"
                        body = f"""
Name: {name}
Email: {email}
Type: {inquiry_type}

"""
                        # Add type-specific info
                        if inquiry_type == "Collaboration Proposal":
                            body += f"""
Organization: {organization}
Collaboration Type: {collaboration_type}
"""
                        elif inquiry_type == "Funding Offer":
                            body += f"""
Organization: {organization}
Funding Amount: {funding_amount}
"""
                        elif inquiry_type == "Technical Suggestions":
                            body += f"""
Technical Area: {area}
"""
                        
                        body += f"\nMessage:\n{message}"
                        
                        # Create mailto link
                        mailto_link = f"mailto:info@etho-shift.com?subject={subject}&body={body}"
                        
                        # Show success message with email link
                        st.success("Click below to send your message:")
                        st.markdown(f"[Click to Open Email Client]({mailto_link})")
                        st.session_state.form_submitted = True
                    else:
                        st.error("Please fill in all required fields.")

        # Contact email in the form itself
        if not st.session_state.form_submitted:
            st.markdown("---")
            st.markdown("📧 [info@etho-shift.com](mailto:info@etho-shift.com)")

if __name__ == "__main__":
    show()



