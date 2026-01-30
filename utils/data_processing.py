import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from typing import Optional, Dict, List

@st.cache_data
def process_portfolio_csv(file) -> pd.DataFrame:
    """Process uploaded portfolio CSV file."""
    try:
        df = pd.read_csv(file)
        
        # Ensure required columns exist
        required_cols = ['equity_symbol', 'week_start', 'weekly_roi']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Convert week_start to datetime
        df['week_start'] = pd.to_datetime(df['week_start'])
        
        # Sort by equity and date
        df = df.sort_values(['equity_symbol', 'week_start'])
        
        # Calculate cumulative ROI for each equity
        df['cumulative_roi'] = df.groupby('equity_symbol')['weekly_roi'].apply(
            lambda x: ((1 + x/100).cumprod() - 1) * 100
        ).reset_index(drop=True)
        
        # Add default values for optional columns if not present
        if 'sector' not in df.columns:
            df['sector'] = 'Unknown'
        if 'cost_basis' not in df.columns:
            df['cost_basis'] = 100  # Default cost basis
        
        # Calculate current value based on cost basis and cumulative ROI
        df['current_value'] = df['cost_basis'] * (1 + df['cumulative_roi'] / 100)
        
        return df
    except Exception as e:
        st.error(f"Error processing portfolio CSV: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def process_narrative_csv(file) -> Dict[str, str]:
    """Process uploaded narrative CSV file."""
    try:
        df = pd.read_csv(file)
        
        # Ensure required columns exist
        if 'equity_symbol' not in df.columns or 'narrative_md' not in df.columns:
            raise ValueError("Narrative CSV must contain 'equity_symbol' and 'narrative_md' columns")
        
        # Create dictionary of narratives keyed by equity symbol
        narratives = dict(zip(df['equity_symbol'], df['narrative_md']))
        return narratives
    except Exception as e:
        st.error(f"Error processing narrative CSV: {str(e)}")
        return {}

def calculate_portfolio_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate overall portfolio metrics."""
    if df.empty:
        return {
            'total_aum': 0,
            'total_return': 0,
            'avg_weekly_return': 0,
            'best_performer': None,
            'worst_performer': None
        }
    
    # Get latest data for each equity
    latest_data = df.groupby('equity_symbol').last().reset_index()
    
    # Calculate metrics
    total_cost_basis = latest_data['cost_basis'].sum()
    total_current_value = latest_data['current_value'].sum()
    total_return = ((total_current_value - total_cost_basis) / total_cost_basis) * 100 if total_cost_basis > 0 else 0
    
    # Average weekly return across all equities
    avg_weekly_return = df.groupby('equity_symbol')['weekly_roi'].mean().mean()
    
    # Best and worst performers by cumulative ROI
    best_performer = latest_data.loc[latest_data['cumulative_roi'].idxmax(), 'equity_symbol'] if not latest_data.empty else None
    worst_performer = latest_data.loc[latest_data['cumulative_roi'].idxmin(), 'equity_symbol'] if not latest_data.empty else None
    
    return {
        'total_aum': total_current_value,
        'total_return': total_return,
        'avg_weekly_return': avg_weekly_return,
        'best_performer': best_performer,
        'worst_performer': worst_performer
    }

def get_equity_details(df: pd.DataFrame, equity_symbol: str) -> Dict:
    """Get detailed information for a specific equity."""
    equity_data = df[df['equity_symbol'] == equity_symbol]
    
    if equity_data.empty:
        return {}
    
    latest = equity_data.iloc[-1]
    
    return {
        'symbol': equity_symbol,
        'sector': latest['sector'],
        'cost_basis': latest['cost_basis'],
        'current_value': latest['current_value'],
        'cumulative_roi': latest['cumulative_roi'],
        'latest_weekly_roi': latest['weekly_roi'],
        'weeks_held': len(equity_data),
        'volatility': equity_data['weekly_roi'].std()
    }

def generate_sample_portfolio_data() -> pd.DataFrame:
    """Generate sample portfolio data for demonstration."""
    np.random.seed(42)
    
    equities = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM']
    sectors = ['Technology', 'Technology', 'Technology', 'E-commerce', 'Automotive', 'Technology', 'Technology', 'Finance']
    
    data = []
    start_date = datetime.now() - timedelta(weeks=52)
    
    for equity, sector in zip(equities, sectors):
        cost_basis = np.random.uniform(10000, 50000)
        
        for week in range(52):
            week_date = start_date + timedelta(weeks=week)
            weekly_roi = np.random.normal(0.5, 2.5)  # Average 0.5% weekly return with 2.5% std dev
            
            data.append({
                'equity_symbol': equity,
                'sector': sector,
                'week_start': week_date,
                'weekly_roi': weekly_roi,
                'cost_basis': cost_basis
            })
    
    df = pd.DataFrame(data)
    
    # Sort by equity and date
    df = df.sort_values(['equity_symbol', 'week_start'])
    
    # Calculate cumulative ROI for each equity
    df['cumulative_roi'] = df.groupby('equity_symbol')['weekly_roi'].apply(
        lambda x: ((1 + x/100).cumprod() - 1) * 100
    ).reset_index(drop=True)
    
    # Calculate current value based on cost basis and cumulative ROI
    df['current_value'] = df['cost_basis'] * (1 + df['cumulative_roi'] / 100)
    
    return df

def generate_sample_narrative_data() -> pd.DataFrame:
    """Generate sample narrative data for demonstration."""
    narratives = {
        'AAPL': """## Apple Inc. (AAPL)
        
**Market Position**: Apple continues to dominate the premium smartphone market with strong iPhone 15 sales.

**Key Developments**:
- Vision Pro headset showing promising early adoption
- Services revenue growing at 15% YoY
- Strong expansion in India market

**Outlook**: Positive momentum expected to continue through Q4.""",
        
        'GOOGL': """## Alphabet Inc. (GOOGL)

**AI Leadership**: Google's Gemini AI model showing strong competitive positioning against OpenAI.

**Cloud Growth**: Google Cloud revenue up 28% YoY, gaining market share from AWS.

**Regulatory Concerns**: Ongoing antitrust cases may impact future operations.""",
        
        'MSFT': """## Microsoft Corporation (MSFT)

**Azure Dominance**: Cloud infrastructure services maintaining strong growth trajectory.

**AI Integration**: Copilot features driving Office 365 subscription upgrades.

**Gaming Division**: Activision Blizzard acquisition fully integrated, boosting gaming revenue."""
    }
    
    data = [{'equity_symbol': symbol, 'narrative_md': narrative} 
            for symbol, narrative in narratives.items()]
    
    return pd.DataFrame(data) 