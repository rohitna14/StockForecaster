Ultimate AI Stock Forecaster - Streamlit Web App

Overview
Ultimate AI Stock Forecaster is a stock prediction platform built with Python, using 15+ ML/DL models to forecast stock prices.
It features a cosmic-themed UI, real-time data, technical indicators, and intelligent trading recommendations.

Development Period: June 2025 - September 2025
Tech Stack: Python, Streamlit, Scikit-learn, TensorFlow, Plotly, Yahoo Finance API

Features
- 15+ AI Models: ARIMA, SARIMA, Random Forest, XGBoost, LightGBM, CatBoost, LSTM, GRU, CNN, Hybrid
- Real-time stock data fetching with technical indicators (RSI, MACD, Bollinger Bands, SMA/EMA)
- Risk Analysis: Sharpe Ratio, Sortino Ratio, Volatility, Max Drawdown, VaR
- Interactive Visualizations: Candlestick charts, model comparisons
- Intelligent Recommendations based on model performance
- SQLite Integration for stock symbol management

Setup & Installation
1. Clone or create project directory:
   mkdir stock-forecaster
   cd stock-forecaster
2. Create a virtual environment:
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
3. Install dependencies:
   pip install streamlit pandas numpy scikit-learn xgboost lightgbm catboost tensorflow statsmodels plotly yfinance
4. Ensure stocks.db SQLite database is in the project root (fallback symbols included if missing)
5. Run the application:
   streamlit run app.py
   Open http://localhost:8501 in your browser

Configuration
- Forecast Horizon: 1-30 days (default: 7 days)
- Model Categories: Time Series (ARIMA, SARIMA), ML (Random Forest, XGBoost, LightGBM, CatBoost, SVR), Deep Learning (LSTM, Bi-LSTM, GRU, RNN, 1D CNN, CNN-LSTM Hybrid)

How It Works
1. Data Acquisition: Fetch stock data and calculate 50+ features
2. Model Training: Train 15+ AI models using 80/20 split, feature preprocessing
3. Prediction & Analysis: Evaluate models, identify champion model, generate recommendations, risk metrics
4. Visualization: Interactive charts for indicators, model performance, risk

Usage Instructions
- Select Stock: Dropdown from SQLite DB or default list
- Click Analyze: Fetch data and metrics
- Navigate Tabs: Charts, AI Training Lab, Prediction Dashboard, Risk & Forecast
- Interpret Results: R² Score, Recommendations, Risk Metrics

Troubleshooting
Issue: Import errors | Solution: Ensure dependencies installed
Issue: No stock data | Solution: Check internet/Yahoo Finance availability
Issue: Model training failures | Solution: Reduce data period or number of models
Issue: Memory issues | Solution: Restart app, use smaller data periods
Issue: SQLite errors | Solution: App will use default stock list as fallback

Author & Disclaimer
Rohit Nikumbh
Built as part of AI/ML development project (June 2025 – September 2025)
Disclaimer: For educational purposes only. Predictions are uncertain. Consult financial advisors before investing.
