import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List, Optional

def create_kpi_gauge(value: float, title: str, target: float = None, 
                     min_val: float = 0, max_val: float = 100) -> go.Figure:
    """Create a gauge chart for KPI display."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20}},
        delta={'reference': target} if target else None,
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [min_val, max_val * 0.5], 'color': 'lightgray'},
                {'range': [max_val * 0.5, max_val * 0.75], 'color': 'gray'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': target if target else max_val * 0.9
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def create_roi_trend_chart(df: pd.DataFrame, equity_symbols: List[str] = None) -> go.Figure:
    """Create ROI trend line chart for selected equities."""
    if equity_symbols:
        df_filtered = df[df['equity_symbol'].isin(equity_symbols)]
    else:
        df_filtered = df
    
    fig = go.Figure()
    
    for symbol in df_filtered['equity_symbol'].unique():
        equity_data = df_filtered[df_filtered['equity_symbol'] == symbol]
        fig.add_trace(go.Scatter(
            x=equity_data['week_start'],
            y=equity_data['cumulative_roi'],
            mode='lines+markers',
            name=symbol,
            line=dict(width=2),
            marker=dict(size=4)
        ))
    
    fig.update_layout(
        title="Cumulative ROI Trend",
        xaxis_title="Date",
        yaxis_title="Cumulative ROI (%)",
        hovermode='x unified',
        height=400,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def create_portfolio_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create a heatmap of weekly returns by equity."""
    # Pivot data for heatmap
    pivot_df = df.pivot_table(
        values='weekly_roi',
        index='equity_symbol',
        columns='week_start',
        aggfunc='mean'
    )
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns.strftime('%Y-%m-%d'),
        y=pivot_df.index,
        colorscale='RdYlGn',
        zmid=0,
        text=np.round(pivot_df.values, 2),
        texttemplate='%{text}%',
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Weekly Returns Heatmap",
        xaxis_title="Week",
        yaxis_title="Equity",
        height=400,
        xaxis={'tickangle': -45}
    )
    
    return fig

def create_financial_trend_chart(revenue_data: List[float], expense_data: List[float], 
                                dates: List[str]) -> go.Figure:
    """Create a line chart showing revenue vs expenses over time."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=revenue_data,
        mode='lines+markers',
        name='Revenue',
        line=dict(color='green', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=expense_data,
        mode='lines+markers',
        name='Expenses',
        line=dict(color='red', width=3),
        marker=dict(size=8)
    ))
    
    # Add profit/loss area
    profit_loss = [r - e for r, e in zip(revenue_data, expense_data)]
    colors = ['green' if p > 0 else 'red' for p in profit_loss]
    
    fig.add_trace(go.Bar(
        x=dates,
        y=profit_loss,
        name='Profit/Loss',
        marker_color=colors,
        opacity=0.3,
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="Revenue vs Expenses",
        xaxis_title="Month",
        yaxis=dict(title="Amount ($)", side='left'),
        yaxis2=dict(title="Profit/Loss ($)", overlaying='y', side='right'),
        hovermode='x unified',
        height=400,
        showlegend=True
    )
    
    return fig

def create_portfolio_composition_chart(df: pd.DataFrame) -> go.Figure:
    """Create a pie chart showing portfolio composition by sector."""
    latest_data = df.groupby('equity_symbol').last().reset_index()
    sector_values = latest_data.groupby('sector')['current_value'].sum().reset_index()
    
    fig = px.pie(
        sector_values, 
        values='current_value', 
        names='sector',
        title='Portfolio Composition by Sector',
        hole=0.4
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Value: $%{value:,.2f}<br>Percentage: %{percent}<extra></extra>'
    )
    
    fig.update_layout(height=400)
    
    return fig

def create_benchmark_comparison_chart(portfolio_roi: List[float], benchmark_roi: List[float], 
                                    dates: List[str]) -> go.Figure:
    """Create a comparison chart between portfolio and benchmark performance."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=portfolio_roi,
        mode='lines+markers',
        name='Portfolio',
        line=dict(color='blue', width=3),
        marker=dict(size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=benchmark_roi,
        mode='lines+markers',
        name='S&P 500 Benchmark',
        line=dict(color='gray', width=2, dash='dash'),
        marker=dict(size=4)
    ))
    
    # Add outperformance/underperformance shading
    diff = [p - b for p, b in zip(portfolio_roi, benchmark_roi)]
    
    fig.add_trace(go.Scatter(
        x=dates + dates[::-1],
        y=portfolio_roi + benchmark_roi[::-1],
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)' if sum(diff) > 0 else 'rgba(255,0,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title="Portfolio vs Benchmark Performance",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        hovermode='x unified',
        height=400,
        showlegend=True
    )
    
    return fig

def create_equity_candlestick_chart(df: pd.DataFrame, equity_symbol: str) -> go.Figure:
    """Create a candlestick chart for individual equity performance."""
    equity_data = df[df['equity_symbol'] == equity_symbol].copy()
    
    # Simulate OHLC data based on weekly ROI
    equity_data['open'] = equity_data['cost_basis'].iloc[0]
    equity_data['close'] = equity_data['current_value']
    equity_data['high'] = equity_data['current_value'] * 1.02
    equity_data['low'] = equity_data['current_value'] * 0.98
    
    fig = go.Figure(data=[go.Candlestick(
        x=equity_data['week_start'],
        open=equity_data['open'],
        high=equity_data['high'],
        low=equity_data['low'],
        close=equity_data['close'],
        name=equity_symbol
    )])
    
    fig.update_layout(
        title=f"{equity_symbol} Price Performance",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=400,
        xaxis_rangeslider_visible=False
    )
    
    return fig 