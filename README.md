# VC Partners Dashboard

A comprehensive Streamlit-based investor dashboard for venture capital partners, providing insights into fund performance, financial health, key decisions, portfolio analytics, and market conditions.

## Features

- 🔐 **Secure Authentication**: Password-protected access with role-based permissions
- 📊 **Overview Dashboard**: High-level fund metrics including AUM, IRR, TVPI, and DPI
- 💰 **Financial Analytics**: Operational financial health, burn rate, and runway analysis
- 📅 **Decision Timeline**: Track key milestones, investments, and strategic decisions
- 💼 **Portfolio Analytics**: Detailed ROI analysis, heatmaps, and benchmark comparisons
- 📈 **Market Conditions**: Custom narratives and real-time market data for portfolio equities
- 📤 **Data Upload**: Easy CSV upload for portfolio and narrative data

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd status_app
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Login with default credentials:
   - Admin: `admin` / `password123`
   - Investor: `investor` / `invest456`

## Data Format

### Portfolio Data CSV
Required columns:
- `equity_symbol`: Stock ticker (e.g., AAPL, GOOGL)
- `week_start`: Date in YYYY-MM-DD format
- `weekly_roi`: Weekly return as percentage

Optional columns:
- `sector`: Sector classification
- `cost_basis`: Initial investment amount

### Narrative Data CSV
Required columns:
- `equity_symbol`: Stock ticker
- `narrative_md`: Markdown-formatted narrative text

## Project Structure

```
status_app/
├── app.py                 # Main application entry point
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── utils/                # Utility modules
│   ├── __init__.py
│   ├── auth.py          # Authentication functions
│   ├── data_processing.py # Data processing utilities
│   └── visualizations.py # Plotly chart functions
└── pages/               # Application pages
    ├── __init__.py
    ├── overview.py      # Overview/KPI dashboard
    ├── finances.py      # Financial health page
    ├── key_decisions.py # Milestones timeline
    ├── portfolio.py     # Portfolio analytics
    ├── market_conditions.py # Market data and narratives
    └── data_upload.py   # Data upload interface
```

## Key Features

### 1. Overview Dashboard
- Real-time KPIs with gauge charts
- Portfolio composition analysis
- Top/worst performer tracking
- Quick statistics overview

### 2. Financial Health
- Revenue vs expenses tracking
- Burn rate and runway calculations
- Monthly financial summaries
- Financial projections

### 3. Key Decisions Timeline
- Chronological milestone tracking
- Category and impact filtering
- Export functionality
- Statistics dashboard

### 4. Portfolio Analytics
- Interactive ROI trend charts
- Weekly returns heatmap
- Benchmark comparisons
- Individual equity analysis
- Export to CSV functionality

### 5. Market Conditions
- Real-time market data integration
- Custom markdown narratives
- Sector performance tracking
- Economic indicators

### 6. Data Upload
- Drag-and-drop CSV upload
- Data validation and preview
- Sample file downloads
- Session-based data storage

## Security

- Passwords are hashed using SHA-256
- Session-based authentication
- Automatic logout functionality
- Role-based access control ready

## Deployment

The app can be deployed on:

1. **Streamlit Community Cloud** (Recommended)
   - Free hosting with HTTPS
   - Easy GitHub integration

2. **AWS Elastic Beanstalk**
   - Scalable infrastructure
   - Custom domain support

3. **Docker + Nginx**
   - Full control over deployment
   - Custom SSL certificates

## Development

To add new features:

1. Create new page modules in `pages/`
2. Add utility functions in `utils/`
3. Update navigation in `app.py`
4. Follow the existing authentication pattern

## Testing

Run basic tests:
```bash
python -m pytest tests/  # (tests to be added)
```
## Enhancements (future work)

Port to Flask app. 