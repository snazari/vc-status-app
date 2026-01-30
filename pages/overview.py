import streamlit as st
from utils.auth import require_authentication
from utils.data_processing import calculate_portfolio_metrics, generate_sample_portfolio_data
from utils.visualizations import create_kpi_gauge, create_portfolio_composition_chart
import pandas as pd

# Require authentication
require_authentication()

st.title("📊 Fund Overview & KPIs")
st.markdown("### High-level fund performance metrics and key performance indicators")

# Load portfolio data
if 'portfolio_data' not in st.session_state:
    st.info("No portfolio data loaded. Using sample data for demonstration.")
    df = generate_sample_portfolio_data()
    st.session_state.portfolio_data = df
else:
    df = st.session_state.portfolio_data

# Calculate portfolio metrics
metrics = calculate_portfolio_metrics(df)

# Display main KPIs
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Assets Under Management (AUM)",
        value=f"${metrics['total_aum']:,.2f}",
        delta=f"{metrics['total_return']:.2f}%"
    )

with col2:
    st.metric(
        label="Total Return",
        value=f"{metrics['total_return']:.2f}%",
        delta="vs cost basis"
    )

with col3:
    st.metric(
        label="Avg Weekly Return",
        value=f"{metrics['avg_weekly_return']:.2f}%",
        delta="weekly"
    )

with col4:
    st.metric(
        label="Portfolio Size",
        value=f"{len(df['equity_symbol'].unique())} equities",
        delta=None
    )

st.markdown("---")

# Gauge charts for key metrics
col1, col2, col3 = st.columns(3)

with col1:
    # IRR Gauge (simulated)
    irr_value = metrics['total_return'] * 0.8  # Simplified IRR calculation
    fig_irr = create_kpi_gauge(
        value=irr_value,
        title="Internal Rate of Return (IRR)",
        target=15,
        min_val=-10,
        max_val=50
    )
    st.plotly_chart(fig_irr, use_container_width=True)

with col2:
    # TVPI Gauge (simulated)
    tvpi_value = 1 + (metrics['total_return'] / 100)
    fig_tvpi = create_kpi_gauge(
        value=tvpi_value,
        title="Total Value to Paid-In (TVPI)",
        target=2.5,
        min_val=0,
        max_val=5
    )
    st.plotly_chart(fig_tvpi, use_container_width=True)

with col3:
    # DPI Gauge (simulated)
    dpi_value = tvpi_value * 0.4  # Assuming 40% distributed
    fig_dpi = create_kpi_gauge(
        value=dpi_value,
        title="Distributions to Paid-In (DPI)",
        target=1.0,
        min_val=0,
        max_val=3
    )
    st.plotly_chart(fig_dpi, use_container_width=True)

st.markdown("---")

# Portfolio composition
st.subheader("Portfolio Composition")
col1, col2 = st.columns([2, 1])

with col1:
    fig_composition = create_portfolio_composition_chart(df)
    st.plotly_chart(fig_composition, use_container_width=True)

with col2:
    st.markdown("### Top Performers")
    latest_data = df.groupby('equity_symbol').last().reset_index()
    top_performers = latest_data.nlargest(5, 'cumulative_roi')[['equity_symbol', 'cumulative_roi']]
    for _, row in top_performers.iterrows():
        st.markdown(f"**{row['equity_symbol']}**: {row['cumulative_roi']:.2f}%")
    
    st.markdown("### Worst Performers")
    worst_performers = latest_data.nsmallest(3, 'cumulative_roi')[['equity_symbol', 'cumulative_roi']]
    for _, row in worst_performers.iterrows():
        st.markdown(f"**{row['equity_symbol']}**: {row['cumulative_roi']:.2f}%")

# Quick stats
st.markdown("---")
st.subheader("Quick Statistics")
col1, col2, col3 = st.columns(3)

with col1:
    st.info(f"🏆 Best Performer: {metrics['best_performer']}")

with col2:
    st.warning(f"📉 Worst Performer: {metrics['worst_performer']}")

with col3:
    volatility = df.groupby('equity_symbol')['weekly_roi'].std().mean()
    st.success(f"📊 Avg Volatility: {volatility:.2f}%") 