import streamlit as st
from utils.auth import require_authentication
from utils.data_processing import (
    process_portfolio_csv, process_narrative_csv,
    generate_sample_portfolio_data, generate_sample_narrative_data
)
import pandas as pd
import io

# Require authentication
require_authentication()

st.title("📤 Data Upload")
st.markdown("### Upload portfolio and narrative data files")

# Instructions
with st.expander("📖 File Format Instructions", expanded=True):
    st.markdown("""
    ### Portfolio Data CSV Format
    Your portfolio CSV should contain the following columns:
    - **equity_symbol** (required): Stock ticker symbol (e.g., AAPL, GOOGL)
    - **week_start** (required): Date in YYYY-MM-DD format
    - **weekly_roi** (required): Weekly return on investment as percentage
    - **sector** (optional): Sector classification
    - **cost_basis** (optional): Initial investment amount
    
    ### Narrative Data CSV Format
    Your narrative CSV should contain the following columns:
    - **equity_symbol** (required): Stock ticker symbol
    - **narrative_md** (required): Markdown-formatted narrative text
    
    ### Sample Files
    You can download sample files below to see the expected format.
    """)

# Sample file downloads
col1, col2 = st.columns(2)

with col1:
    sample_portfolio = generate_sample_portfolio_data()
    csv_portfolio = sample_portfolio.to_csv(index=False)
    st.download_button(
        label="📥 Download Sample Portfolio CSV",
        data=csv_portfolio,
        file_name="sample_portfolio_data.csv",
        mime="text/csv"
    )

with col2:
    sample_narrative = generate_sample_narrative_data()
    csv_narrative = sample_narrative.to_csv(index=False)
    st.download_button(
        label="📥 Download Sample Narrative CSV",
        data=csv_narrative,
        file_name="sample_narrative_data.csv",
        mime="text/csv"
    )

st.markdown("---")

# File upload section
st.subheader("Upload Files")

# Portfolio data upload
st.markdown("### Portfolio Data")
portfolio_file = st.file_uploader(
    "Choose a portfolio CSV file",
    type=['csv'],
    key="portfolio_upload"
)

if portfolio_file is not None:
    try:
        # Process the file
        df = process_portfolio_csv(portfolio_file)
        
        if not df.empty:
            st.success(f"✅ Successfully loaded portfolio data with {len(df)} records and {len(df['equity_symbol'].unique())} unique equities")
            
            # Preview the data
            with st.expander("Preview Portfolio Data"):
                st.dataframe(df.head(20))
            
            # Save to session state
            if st.button("Save Portfolio Data", key="save_portfolio"):
                st.session_state.portfolio_data = df
                st.success("Portfolio data saved to session!")
                st.balloons()
        else:
            st.error("Failed to process portfolio file. Please check the format.")
            
    except Exception as e:
        st.error(f"Error processing portfolio file: {str(e)}")

# Narrative data upload
st.markdown("### Narrative Data")
narrative_file = st.file_uploader(
    "Choose a narrative CSV file",
    type=['csv'],
    key="narrative_upload"
)

if narrative_file is not None:
    try:
        # Process the file
        narratives = process_narrative_csv(narrative_file)
        
        if narratives:
            st.success(f"✅ Successfully loaded narratives for {len(narratives)} equities")
            
            # Preview the narratives
            with st.expander("Preview Narrative Data"):
                for symbol, narrative in list(narratives.items())[:3]:
                    st.markdown(f"**{symbol}:**")
                    st.markdown(narrative[:200] + "..." if len(narrative) > 200 else narrative)
                    st.markdown("---")
            
            # Save to session state
            if st.button("Save Narrative Data", key="save_narrative"):
                st.session_state.narrative_data = narratives
                st.success("Narrative data saved to session!")
                st.balloons()
        else:
            st.error("Failed to process narrative file. Please check the format.")
            
    except Exception as e:
        st.error(f"Error processing narrative file: {str(e)}")

# Current data status
st.markdown("---")
st.subheader("Current Data Status")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Portfolio Data")
    if 'portfolio_data' in st.session_state and not st.session_state.portfolio_data.empty:
        df = st.session_state.portfolio_data
        st.success(f"✅ Loaded: {len(df)} records, {len(df['equity_symbol'].unique())} equities")
        st.markdown(f"**Date Range:** {df['week_start'].min().strftime('%Y-%m-%d')} to {df['week_start'].max().strftime('%Y-%m-%d')}")
        st.markdown(f"**Equities:** {', '.join(df['equity_symbol'].unique()[:5])}" + 
                   (" ..." if len(df['equity_symbol'].unique()) > 5 else ""))
    else:
        st.warning("⚠️ No portfolio data loaded (using sample data)")

with col2:
    st.markdown("### Narrative Data")
    if 'narrative_data' in st.session_state and st.session_state.narrative_data:
        narratives = st.session_state.narrative_data
        st.success(f"✅ Loaded: {len(narratives)} narratives")
        st.markdown(f"**Equities with narratives:** {', '.join(list(narratives.keys())[:5])}" + 
                   (" ..." if len(narratives) > 5 else ""))
    else:
        st.warning("⚠️ No narrative data loaded (using sample data)")

# Clear data option
st.markdown("---")
st.subheader("Data Management")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🗑️ Clear Portfolio Data", use_container_width=True):
        if 'portfolio_data' in st.session_state:
            del st.session_state.portfolio_data
        st.success("Portfolio data cleared!")
        st.rerun()

with col2:
    if st.button("🗑️ Clear Narrative Data", use_container_width=True):
        if 'narrative_data' in st.session_state:
            del st.session_state.narrative_data
        st.success("Narrative data cleared!")
        st.rerun()

with col3:
    if st.button("🔄 Reset to Sample Data", use_container_width=True):
        st.session_state.portfolio_data = generate_sample_portfolio_data()
        narrative_df = generate_sample_narrative_data()
        st.session_state.narrative_data = dict(zip(narrative_df['equity_symbol'], narrative_df['narrative_md']))
        st.success("Reset to sample data!")
        st.rerun()

# Tips
st.markdown("---")
st.info("""
💡 **Tips:**
- Portfolio data should be uploaded first before narrative data
- Ensure dates are in YYYY-MM-DD format
- Weekly ROI should be expressed as percentages (e.g., 2.5 for 2.5%)
- Narrative markdown supports **bold**, *italic*, lists, and headers
- Data is stored in session and will persist until logout
""")

# Admin notice
if st.session_state.get('user_role') == 'admin':
    st.markdown("---")
    st.success("👨‍💼 **Admin Mode**: You have full access to upload and modify all data.") 