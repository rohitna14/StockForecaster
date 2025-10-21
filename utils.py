# utils.py - Utility functions for data handling, risk analysis, and recommendations
import sqlite3
import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class StockDatabase:
    def __init__(self, db_path='stocks.db'):
        self.db_path = db_path

    def get_all_stocks(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            if tables:
                table_name = tables[0][0]
                query = f"SELECT Symbol, Name FROM {table_name}"
                df = pd.read_sql_query(query, conn)
                conn.close()
                return df
            return pd.DataFrame()
        except:
            return pd.DataFrame({
                'Symbol': ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'NFLX', 'AMD', 'INTC'],
                'Name': ['Apple Inc', 'Tesla Inc', 'Microsoft Corp', 'Google LLC', 'Amazon.com Inc',
                         'Meta Platforms Inc', 'NVIDIA Corp', 'Netflix Inc', 'AMD Inc', 'Intel Corp']
            })


def fetch_stock_data(symbol, period="2y"):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        if data.empty:
            return None

        # Technical Indicators
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['SMA_50'] = data['Close'].rolling(50).mean()
        data['SMA_200'] = data['Close'].rolling(200).mean()
        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()

        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()

        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(20).mean()
        bb_std = data['Close'].rolling(20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)

        # Volume indicators
        data['Volume_SMA'] = data['Volume'].rolling(20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']

        # Returns
        data['Daily_Return'] = data['Close'].pct_change()
        data['Cumulative_Return'] = (1 + data['Daily_Return']).cumprod()

        # Momentum
        data['Momentum_10'] = data['Close'] / data['Close'].shift(10) - 1
        data['Momentum_20'] = data['Close'] / data['Close'].shift(20) - 1

        return data.dropna()
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return None


def analyze_stock_characteristics(stock_data):
    """Analyze stock to determine best models"""
    df = stock_data['Close'].dropna()

    # Check stationarity
    adf_result = adfuller(df)
    is_stationary = adf_result[1] < 0.05

    # Check volatility
    returns = stock_data['Daily_Return'].dropna()
    volatility = returns.std() * np.sqrt(252)
    is_high_volatility = volatility > 0.4

    # Check trend
    trend_strength = abs(df.iloc[-60:].mean() / df.iloc[-260:-200].mean() - 1)
    has_strong_trend = trend_strength > 0.15

    # Check seasonality
    if len(df) >= 365:
        weekly_pattern = df.groupby(
            df.index.dayofweek).mean().std() / df.mean()
        has_seasonality = weekly_pattern > 0.02
    else:
        has_seasonality = False

    return {
        'is_stationary': is_stationary,
        'is_high_volatility': is_high_volatility,
        'has_strong_trend': has_strong_trend,
        'has_seasonality': has_seasonality,
        'volatility': volatility,
        'trend_strength': trend_strength
    }


def calculate_risk_metrics(stock_data):
    try:
        returns = stock_data['Daily_Return'].dropna()
        if len(returns) < 30:
            return None

        annual_volatility = returns.std() * np.sqrt(252) * 100
        sharpe_ratio = (returns.mean() * 252 - 0.02) / \
            (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max * 100
        max_drawdown = drawdowns.min()

        var_95 = np.percentile(returns, 5) * 100
        cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100

        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std(
        ) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (returns.mean() * 252 - 0.02) / \
            downside_volatility if downside_volatility > 0 else 0

        return {
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'returns': returns,
            'drawdowns': drawdowns
        }
    except Exception as e:
        print(f"Risk metrics calculation failed: {e}")
        return None


def generate_advanced_recommendation(risk_metrics, performance, characteristics):
    if not risk_metrics or not performance:
        return "HOLD", "hold", "Insufficient data"

    score = 0
    factors = []

    # Best model performance
    best_r2 = max([p['R2'] for p in performance.values()])
    if best_r2 > 0.7:
        score += 3
        factors.append(f"<b>Strong AI prediction (R²={best_r2:.3f})</b>")
    elif best_r2 > 0.4:
        score += 1
        factors.append(f"<b>Moderate prediction accuracy</b>")

    # Risk metrics
    if risk_metrics['sharpe_ratio'] > 1.5:
        score += 3
        factors.append("<b>Excellent risk-adjusted returns</b>")
    elif risk_metrics['sharpe_ratio'] > 1.0:
        score += 2
    elif risk_metrics['sharpe_ratio'] > 0.5:
        score += 1

    if risk_metrics['sortino_ratio'] > 1.5:
        score += 2
        factors.append("<b>Excellent downside protection</b>")

    if risk_metrics['annual_volatility'] < 20:
        score += 2
        factors.append("<b>Low volatility</b>")
    elif risk_metrics['annual_volatility'] < 35:
        score += 1

    if risk_metrics['max_drawdown'] > -15:
        score += 2
        factors.append("<b>Strong drawdown control</b>")
    elif risk_metrics['max_drawdown'] > -25:
        score += 1

    # Stock characteristics
    if characteristics['has_strong_trend'] and not characteristics['is_high_volatility']:
        score += 2
        factors.append("<b>Strong positive trend</b>")

    if characteristics['is_high_volatility']:
        score -= 2
        factors.append("<b>High volatility risk</b>")

    # Final recommendation
    reasoning = " | ".join(factors[:4])

    if score >= 8:
        return "STRONG BUY 🚀🚀🚀", "buy", reasoning
    elif score >= 5:
        return "BUY 📈", "buy", reasoning
    elif score >= 2:
        return "HOLD ⚖️", "hold", reasoning
    elif score >= -1:
        return "REDUCE 📉", "sell", reasoning
    else:
        return "SELL 🛑", "sell", reasoning
