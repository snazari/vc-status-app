import streamlit as st
import hashlib
from typing import Dict, Optional

# Sample users with hashed passwords (in production, store these securely)
# Default credentials: admin/password123, investor/invest456
USERS = {
    "admin": {
        "password_hash": hashlib.sha256("password123".encode()).hexdigest(),
        "role": "admin"
    },
    "investor": {
        "password_hash": hashlib.sha256("invest456".encode()).hexdigest(),
        "role": "investor"
    }
}

def hash_password(password: str) -> str:
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_credentials(username: str, password: str) -> bool:
    """Verify user credentials."""
    if username in USERS:
        password_hash = hash_password(password)
        return USERS[username]["password_hash"] == password_hash
    return False

def login_user(username: str, password: str) -> bool:
    """Attempt to log in a user."""
    if verify_credentials(username, password):
        st.session_state.authenticated = True
        st.session_state.username = username
        st.session_state.user_role = USERS[username]["role"]
        return True
    return False

def logout_user():
    """Log out the current user."""
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.user_role = None
    # Clear any cached data
    if 'portfolio_data' in st.session_state:
        del st.session_state.portfolio_data
    if 'narrative_data' in st.session_state:
        del st.session_state.narrative_data

def require_authentication():
    """Check if user is authenticated, redirect to login if not."""
    if not st.session_state.get('authenticated', False):
        st.warning("Please log in to access this page.")
        st.stop()

def show_login_page():
    """Display the login page."""
    st.title("🔐 Investor Dashboard Login")
    st.markdown("### Welcome to the Venture Capital Partners Dashboard")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login", use_container_width=True)
            
            if submit:
                if login_user(username, password):
                    st.success("Successfully logged in!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
    
    st.markdown("---")
    st.caption("Default credentials for demo: admin/password123 or investor/invest456") 