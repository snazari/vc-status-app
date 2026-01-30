import streamlit as st
from utils.auth import require_authentication
from utils.data_processing import (
    calculate_portfolio_metrics, get_equity_details,
    generate_sample_portfolio_data
)
from utils.visualizations import (
    create_roi_trend_chart, create_portfolio_heatmap,
    create_benchmark_comparison_chart, create_equity_candlestick_chart
)
import pandas as pd
import numpy as np
from datetime import datetime

# Require authentication
require_authentication()

st.title("💼 Investment Portfolio Analytics")
st.markdown("### Detailed analysis of fund holdings and performance")

# Load portfolio data
if 'portfolio_data' not in st.session_state:
    st.info("No portfolio data loaded. Using sample data for demonstration.")
    df = generate_sample_portfolio_data()
    st.session_state.portfolio_data = df
else:
    df = st.session_state.portfolio_data

# Portfolio overview metrics
metrics = calculate_portfolio_metrics(df)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Holdings", f"{len(df['equity_symbol'].unique())}")
with col2:
    st.metric("Portfolio Value", f"${metrics['total_aum']:,.0f}")
with col3:
    st.metric("Total Return", f"{metrics['total_return']:.2f}%")
with col4:
    st.metric("Avg Weekly Return", f"{metrics['avg_weekly_return']:.2f}%")

# Equity selection
st.markdown("---")
equity_symbols = df['equity_symbol'].unique()
selected_equities = st.multiselect(
    "Select equities to analyze",
    options=equity_symbols,
    default=equity_symbols[:5]
)

# ROI Trend Chart
if selected_equities:
    st.subheader("ROI Trend Analysis")
    fig_roi = create_roi_trend_chart(df, selected_equities)
    st.plotly_chart(fig_roi, use_container_width=True)

# Portfolio Heatmap
st.subheader("Weekly Returns Heatmap")
# Limit heatmap to last 12 weeks for better visibility
recent_weeks = df['week_start'].unique()[-12:]
df_recent = df[df['week_start'].isin(recent_weeks)]
fig_heatmap = create_portfolio_heatmap(df_recent)
st.plotly_chart(fig_heatmap, use_container_width=True)

# Performance Table
st.markdown("---")
st.subheader("Portfolio Performance Table")

# Get latest data for each equity
latest_data = df.groupby('equity_symbol').last().reset_index()
performance_df = latest_data[['equity_symbol', 'sector', 'cost_basis', 'current_value', 'weekly_roi', 'cumulative_roi']].copy()

# Add volatility
volatility = df.groupby('equity_symbol')['weekly_roi'].std().reset_index()
volatility.columns = ['equity_symbol', 'volatility']
performance_df = performance_df.merge(volatility, on='equity_symbol')

# Sort by cumulative ROI
performance_df = performance_df.sort_values('cumulative_roi', ascending=False)

# Format for display
performance_df.columns = ['Symbol', 'Sector', 'Cost Basis', 'Current Value', 'Latest Weekly ROI', 'Cumulative ROI', 'Volatility']
for col in ['Cost Basis', 'Current Value']:
    performance_df[col] = performance_df[col].apply(lambda x: f"${x:,.2f}")
for col in ['Latest Weekly ROI', 'Cumulative ROI', 'Volatility']:
    performance_df[col] = performance_df[col].apply(lambda x: f"{x:.2f}%")

# Display table
st.dataframe(
    performance_df,
    use_container_width=True,
    hide_index=True
)

# Download button
csv = performance_df.to_csv(index=False)
st.download_button(
    label="Download Portfolio Data as CSV",
    data=csv,
    file_name=f"portfolio_data_{datetime.now().strftime('%Y%m%d')}.csv",
    mime="text/csv"
)

# Benchmark Comparison
st.markdown("---")
st.subheader("Portfolio vs Benchmark Comparison")

# Generate benchmark data (simulated S&P 500)
portfolio_dates = df.groupby('week_start').first().index
portfolio_returns = []
benchmark_returns = []

for date in portfolio_dates:
    date_data = df[df['week_start'] == date]
    avg_return = date_data.groupby('equity_symbol')['cumulative_roi'].last().mean()
    portfolio_returns.append(avg_return)
    
    # Simulate benchmark with lower volatility
    weeks_elapsed = len(portfolio_returns)
    benchmark_return = weeks_elapsed * 0.3 + np.random.normal(0, 1.5)
    benchmark_returns.append(benchmark_return)

fig_benchmark = create_benchmark_comparison_chart(
    portfolio_returns,
    benchmark_returns,
    [d.strftime('%Y-%m-%d') for d in portfolio_dates]
)
st.plotly_chart(fig_benchmark, use_container_width=True)

# Individual Equity Analysis
st.markdown("---")
st.subheader("Individual Equity Analysis")

selected_equity = st.selectbox("Select an equity for detailed analysis", equity_symbols)

if selected_equity:
    equity_details = get_equity_details(df, selected_equity)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_candlestick = create_equity_candlestick_chart(df, selected_equity)
        st.plotly_chart(fig_candlestick, use_container_width=True)
    
    with col2:
        st.markdown(f"### {selected_equity} Details")
        st.metric("Sector", equity_details['sector'])
        st.metric("Cost Basis", f"${equity_details['cost_basis']:,.2f}")
        st.metric("Current Value", f"${equity_details['current_value']:,.2f}")
        st.metric("Cumulative ROI", f"{equity_details['cumulative_roi']:.2f}%")
        st.metric("Volatility", f"{equity_details['volatility']:.2f}%")
        st.metric("Weeks Held", equity_details['weeks_held'])

# Top/Worst Performers
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏆 Top Performers")
    top_5 = performance_df.head(5)[['Symbol', 'Cumulative ROI']]
    st.dataframe(top_5, use_container_width=True, hide_index=True)

with col2:
    st.subheader("📉 Worst Performers")
    worst_5 = performance_df.tail(5)[['Symbol', 'Cumulative ROI']]
    st.dataframe(worst_5, use_container_width=True, hide_index=True) 