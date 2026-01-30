import streamlit as st
from utils.auth import require_authentication
from utils.visualizations import create_financial_trend_chart
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Require authentication
require_authentication()

st.title("💰 Financial Health")
st.markdown("### Fund operational financials and cash flow analysis")

# Generate sample financial data
np.random.seed(42)
months = pd.date_range(end=datetime.now(), periods=12, freq='ME')
revenue_data = np.random.uniform(500000, 800000, 12)
expense_data = np.random.uniform(400000, 600000, 12)

# Calculate key financial metrics
current_revenue = revenue_data[-1]
current_expenses = expense_data[-1]
burn_rate = current_expenses - current_revenue
cash_on_hand = 5000000  # Simulated
runway_months = abs(cash_on_hand / burn_rate) if burn_rate > 0 else float('inf')

# Display main financial KPIs
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Cash on Hand",
        value=f"${cash_on_hand:,.0f}",
        delta=None
    )

with col2:
    st.metric(
        label="Monthly Burn Rate",
        value=f"${abs(burn_rate):,.0f}",
        delta="per month" if burn_rate > 0 else "positive cash flow"
    )

with col3:
    st.metric(
        label="Runway",
        value=f"{runway_months:.1f} months" if runway_months != float('inf') else "∞ months",
        delta=None
    )

with col4:
    profit_margin = ((current_revenue - current_expenses) / current_revenue) * 100
    st.metric(
        label="Profit Margin",
        value=f"{profit_margin:.1f}%",
        delta="current month"
    )

st.markdown("---")

# Revenue vs Expenses chart
st.subheader("Revenue vs Expenses Trend")
fig_financial = create_financial_trend_chart(
    revenue_data.tolist(),
    expense_data.tolist(),
    [d.strftime('%Y-%m') for d in months]
)
st.plotly_chart(fig_financial, use_container_width=True)

# Financial summary table
st.markdown("---")
st.subheader("Monthly Financial Summary")

# Create summary dataframe
financial_df = pd.DataFrame({
    'Month': months,
    'Revenue': revenue_data,
    'Expenses': expense_data,
    'Net Income': revenue_data - expense_data,
    'Cash Flow': np.cumsum(revenue_data - expense_data) + cash_on_hand
})

# Format the dataframe for display
financial_df['Month'] = financial_df['Month'].dt.strftime('%B %Y')
for col in ['Revenue', 'Expenses', 'Net Income', 'Cash Flow']:
    financial_df[col] = financial_df[col].apply(lambda x: f"${x:,.0f}")

# Display with color coding
st.dataframe(
    financial_df,
    use_container_width=True,
    hide_index=True
)

# Financial projections
st.markdown("---")
st.subheader("Financial Projections")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Next Quarter Projections")
    st.metric("Projected Revenue", f"${np.mean(revenue_data) * 3:,.0f}")
    st.metric("Projected Expenses", f"${np.mean(expense_data) * 3:,.0f}")
    st.metric("Projected Net Income", f"${(np.mean(revenue_data) - np.mean(expense_data)) * 3:,.0f}")

with col2:
    st.markdown("### Key Financial Ratios")
    expense_ratio = np.mean(expense_data) / np.mean(revenue_data)
    st.metric("Expense Ratio", f"{expense_ratio:.2f}")
    
    growth_rate = ((revenue_data[-1] - revenue_data[0]) / revenue_data[0]) * 100
    st.metric("Revenue Growth Rate", f"{growth_rate:.1f}%")
    
    avg_monthly_burn = np.mean(expense_data - revenue_data)
    st.metric("Avg Monthly Burn", f"${abs(avg_monthly_burn):,.0f}")

# Download financial data
st.markdown("---")
csv = financial_df.to_csv(index=False)
st.download_button(
    label="Download Financial Data as CSV",
    data=csv,
    file_name=f"financial_data_{datetime.now().strftime('%Y%m%d')}.csv",
    mime="text/csv"
) 