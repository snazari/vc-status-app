import streamlit as st
from utils.auth import require_authentication
from utils.data_processing import (
    generate_sample_portfolio_data, generate_sample_narrative_data
)
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Require authentication
require_authentication()

st.title("📈 Market Conditions")
st.markdown("### Contextual narratives and market analysis for portfolio equities")

# Load data
if 'portfolio_data' not in st.session_state:
    df = generate_sample_portfolio_data()
    st.session_state.portfolio_data = df
else:
    df = st.session_state.portfolio_data

if 'narrative_data' not in st.session_state:
    narrative_df = generate_sample_narrative_data()
    narratives = dict(zip(narrative_df['equity_symbol'], narrative_df['narrative_md']))
    st.session_state.narrative_data = narratives
else:
    narratives = st.session_state.narrative_data

# Get unique equities
equity_symbols = df['equity_symbol'].unique()

# Market overview
st.subheader("Market Overview")
col1, col2, col3, col4 = st.columns(4)

# Fetch market indices (simulated for demo)
with col1:
    st.metric("S&P 500", "4,783.45", "+1.23%")
with col2:
    st.metric("NASDAQ", "15,123.89", "+1.87%")
with col3:
    st.metric("VIX", "14.52", "-0.45%")
with col4:
    st.metric("10Y Treasury", "4.25%", "+0.05%")

st.markdown("---")

# Equity selection
selected_equity = st.selectbox(
    "Select an equity to view market conditions",
    options=equity_symbols,
    index=0
)

if selected_equity:
    # Display equity information
    st.subheader(f"{selected_equity} Market Conditions")
    
    # Try to fetch real market data
    try:
        ticker = yf.Ticker(selected_equity)
        info = ticker.info
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_price = info.get('currentPrice', 'N/A')
            st.metric("Current Price", f"${current_price}" if current_price != 'N/A' else 'N/A')
        
        with col2:
            week_52_high = info.get('fiftyTwoWeekHigh', 'N/A')
            week_52_low = info.get('fiftyTwoWeekLow', 'N/A')
            if week_52_high != 'N/A' and week_52_low != 'N/A':
                st.metric("52-Week Range", f"${week_52_low:.2f} - ${week_52_high:.2f}")
            else:
                st.metric("52-Week Range", "N/A")
        
        with col3:
            beta = info.get('beta', 'N/A')
            st.metric("Beta", f"{beta:.2f}" if beta != 'N/A' else 'N/A')
        
        with col4:
            volume = info.get('volume', 'N/A')
            if volume != 'N/A':
                st.metric("Volume", f"{volume:,}")
            else:
                st.metric("Volume", "N/A")
    
    except:
        # Fallback to simulated data if API fails
        st.info("Using simulated market data for demonstration")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", "$145.32")
        with col2:
            st.metric("52-Week Range", "$98.45 - $156.78")
        with col3:
            st.metric("Beta", "1.24")
        with col4:
            st.metric("Volume", "45,234,567")
    
    st.markdown("---")
    
    # Display narrative
    st.subheader("Market Analysis & Narrative")
    
    if selected_equity in narratives:
        # Display the markdown narrative
        st.markdown(narratives[selected_equity], unsafe_allow_html=True)
    else:
        st.warning(f"No narrative available for {selected_equity}")
        st.info("Upload a narrative CSV file in the Data Upload section to add custom narratives.")
    
    # Additional market context
    st.markdown("---")
    st.subheader("Additional Market Context")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Sector Performance")
        # Get sector for the selected equity
        equity_sector = df[df['equity_symbol'] == selected_equity]['sector'].iloc[0]
        st.info(f"{selected_equity} is in the **{equity_sector}** sector")
        
        # Simulated sector performance
        st.markdown("**Sector Performance (YTD)**")
        sector_performance = {
            "Technology": "+15.3%",
            "Finance": "+8.7%",
            "Healthcare": "+12.1%",
            "E-commerce": "+22.5%",
            "Automotive": "-3.2%"
        }
        
        for sector, performance in sector_performance.items():
            if sector == equity_sector:
                st.markdown(f"**{sector}: {performance}** ⭐")
            else:
                st.markdown(f"{sector}: {performance}")
    
    with col2:
        st.markdown("### Recent News Headlines")
        # Simulated news headlines
        headlines = [
            f"{selected_equity} announces Q4 earnings beat expectations",
            f"Analysts upgrade {selected_equity} to 'Buy' rating",
            f"{equity_sector} sector shows strong growth momentum",
            "Fed signals potential rate cuts in 2024"
        ]
        
        for headline in headlines:
            st.markdown(f"• {headline}")

# Market sentiment section
st.markdown("---")
st.subheader("Overall Market Sentiment")

sentiment_data = {
    "Bullish": 45,
    "Neutral": 35,
    "Bearish": 20
}

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Bullish", f"{sentiment_data['Bullish']}%", "+5%")
with col2:
    st.metric("Neutral", f"{sentiment_data['Neutral']}%", "-2%")
with col3:
    st.metric("Bearish", f"{sentiment_data['Bearish']}%", "-3%")

# Economic indicators
st.markdown("---")
st.subheader("Key Economic Indicators")

indicators = {
    "GDP Growth": "2.8%",
    "Inflation (CPI)": "3.2%",
    "Unemployment": "3.7%",
    "Fed Funds Rate": "5.25-5.50%"
}

cols = st.columns(len(indicators))
for idx, (indicator, value) in enumerate(indicators.items()):
    with cols[idx]:
        st.metric(indicator, value) 