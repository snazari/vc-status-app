import streamlit as st
import os

# Import utilities
from utils.auth import show_login_page, require_authentication, logout_user
from utils.data_processing import (
    process_portfolio_csv, process_narrative_csv,
    generate_sample_portfolio_data, generate_sample_narrative_data
)

# Page configuration
st.set_page_config(
    page_title="VC Partners Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.user_role = None

# Show login page if not authenticated
if not st.session_state.authenticated:
    show_login_page()
else:
    # Main app layout
    with st.sidebar:
        st.title("📈 VC Partners Dashboard")
        st.markdown(f"Welcome, **{st.session_state.username}**")
        
        st.markdown("---")
        
        # Navigation menu using radio buttons
        selected = st.radio(
            "Navigation",
            options=["Overview", "Finances", "Key Decisions", "Portfolio", "Market Conditions", "Data Upload"],
            index=0
        )
        
        st.markdown("---")
        
        # Logout button
        if st.button("Logout", use_container_width=True):
            logout_user()
            st.rerun()
        
        st.markdown("---")
        st.caption("© 2024 VC Partners Dashboard")
    
    # Main content area - Import and execute pages dynamically
    if selected == "Overview":
        exec(open('pages/overview.py').read())
    elif selected == "Finances":
        exec(open('pages/finances.py').read())
    elif selected == "Key Decisions":
        exec(open('pages/key_decisions.py').read())
    elif selected == "Portfolio":
        exec(open('pages/portfolio.py').read())
    elif selected == "Market Conditions":
        exec(open('pages/market_conditions.py').read())
    elif selected == "Data Upload":
        exec(open('pages/data_upload.py').read()) 