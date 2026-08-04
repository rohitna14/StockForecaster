# app.py - Main Streamlit application for ULTIMATE AI Stock Forecasting Tool
from sklearn.metrics import mean_absolute_error, r2_score
from models import prepare_features_advanced, train_time_series_models, train_ml_models, train_deep_learning_models, evaluate_all_models
from utils import StockDatabase, fetch_stock_data, analyze_stock_characteristics, calculate_risk_metrics, generate_advanced_recommendation
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import warnings
warnings.filterwarnings('ignore')


# 🌌 COSMIC UI CONFIGURATION
st.set_page_config(page_title="🌌 AI Stock Forecaster Tool",
                   page_icon="🚀", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    
    .main {
        background: linear-gradient(135deg, #000000 0%, #0a0e27 25%, #16213e 50%, #1a1a2e 75%, #000000 100%);
        color: white;
        font-family: 'Orbitron', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #000000 0%, #0a0e27 25%, #16213e 50%, #1a1a2e 75%, #000000 100%);
        animation: gradientShift 15s ease infinite;
    }
    @keyframes gradientShift {
        0% { filter: hue-rotate(0deg); }
        50% { filter: hue-rotate(30deg); }
        100% { filter: hue-rotate(0deg); }
    }
    .cosmic-header {
        text-align: center;
        padding: 40px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        border-radius: 25px;
        margin-bottom: 30px;
        box-shadow: 0 0 80px rgba(102, 126, 234, 0.8);
        animation: glow 3s ease-in-out infinite;
        border: 3px solid rgba(255, 255, 255, 0.3);
    }
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 40px rgba(102, 126, 234, 0.6); }
        50% { box-shadow: 0 0 80px rgba(102, 126, 234, 1); }
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
        padding: 20px;
        border-radius: 15px;
        border: 2px solid rgba(102, 126, 234, 0.5);
        text-align: center;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px 0 rgba(102, 126, 234, 0.6);
    }
    .metric-card h2, .metric-card h3, .metric-card p {
        color: white !important;
        margin: 5px 0;
    }
    .metric-card h2 {
        font-size: 2em;
        font-weight: bold;
    }
    .metric-card h3 {
        font-size: 1.5em;
    }
    .metric-card p {
        font-size: 1.1em;
        opacity: 0.9;
    }
    .model-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        padding: 20px;
        border-radius: 12px;
        border: 2px solid rgba(102, 126, 234, 0.4);
        margin: 10px 0;
        backdrop-filter: blur(10px);
        color: white;
    }
    .model-card strong, .model-card p, .model-card span {
        color: white !important;
        font-size: 1.1em;
    }
    .champion-model {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        padding: 25px;
        border-radius: 15px;
        border: 3px solid #FFD700;
        color: #000;
        font-weight: bold;
        box-shadow: 0 0 40px rgba(255, 215, 0, 0.8);
        animation: championGlow 2s ease-in-out infinite;
        margin: 20px 0;
    }
    .champion-model h2, .champion-model p {
        color: #000 !important;
        font-size: 1.3em;
    }
    @keyframes championGlow {
        0%, 100% { box-shadow: 0 0 20px rgba(255, 215, 0, 0.6); }
        50% { box-shadow: 0 0 40px rgba(255, 215, 0, 1); }
    }
    .recommendation-buy {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        padding: 25px;
        border-radius: 20px;
        color: white;
        border: 3px solid #00ff88;
        box-shadow: 0 0 50px rgba(0, 255, 136, 0.6);
    }
    .recommendation-hold {
        background: linear-gradient(135deg, #ff9a00 0%, #ffcc00 100%);
        padding: 25px;
        border-radius: 20px;
        color: white;
        border: 3px solid #ffaa00;
        box-shadow: 0 0 50px rgba(255, 170, 0, 0.6);
    }
    .recommendation-sell {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        padding: 25px;
        border-radius: 20px;
        color: white;
        border: 3px solid #ff4444;
        box-shadow: 0 0 50px rgba(255, 68, 68, 0.6);
    }
    .stock-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 25px;
        border-radius: 20px;
        border: 2px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        margin: 15px 0;
    }
    .explanation-box {
        background: rgba(102, 126, 234, 0.2);
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 15px 0;
        color: white;
    }
    .explanation-box h4 {
        color: #667eea !important;
        margin-top: 0;
        font-size: 1.3em;
    }
    .explanation-box p, .explanation-box li {
        color: white !important;
        font-size: 1.1em;
        line-height: 1.6;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
        text-shadow: 0 0 10px rgba(102, 126, 234, 0.8);
        color: white !important;
    }
    .training-result {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.2), rgba(0, 200, 100, 0.2));
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #00ff88;
        margin: 10px 0;
        color: white;
    }
    .training-result h3, .training-result p {
        color: white !important;
        font-size: 1.2em;
    }
</style>
""", unsafe_allow_html=True)


def display_stock_metrics(stock_data):
    """Display current stock metrics in beautiful cards"""
    current_price = stock_data['Close'].iloc[-1]
    prev_price = stock_data['Close'].iloc[-2]
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100

    if len(stock_data) > 252:
        ytd_return = (current_price / stock_data['Close'].iloc[-252] - 1) * 100
    else:
        ytd_return = (current_price / stock_data['Close'].iloc[0] - 1) * 100

    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h2>${current_price:.2f}</h2>
            <p><strong>Current Price</strong></p>
            <h3 style="color: {'#00ff88' if price_change_pct >= 0 else '#ff4444'};">{price_change_pct:+.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{ytd_return:+.1f}%</h2>
            <p><strong>YTD Return</strong></p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{stock_data['Volume'].iloc[-1]/1e6:.1f}M</h2>
            <p><strong>Volume</strong></p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        rsi_val = stock_data['RSI'].iloc[-1]
        rsi_color = '#ff4444' if rsi_val > 70 else '#00ff88' if rsi_val < 30 else '#ffaa00'
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color:{rsi_color};">{rsi_val:.1f}</h2>
            <p><strong>RSI</strong></p>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        avg_vol = stock_data['Volume'].rolling(20).mean().iloc[-1]
        vol_ratio = stock_data['Volume'].iloc[-1] / avg_vol
        st.markdown(f"""
        <div class="metric-card">
            <h2>{vol_ratio:.2f}x</h2>
            <p><strong>Vol Ratio</strong></p>
        </div>
        """, unsafe_allow_html=True)


def display_advanced_chart(stock_data):
    """Display advanced stock chart with technical indicators"""
    st.markdown("<div class='stock-card'>", unsafe_allow_html=True)

    # Advanced chart
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(
            '<b>Price Action with Bollinger Bands</b>', '<b>MACD</b>', '<b>Volume</b>')
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=stock_data.index[-200:],
        open=stock_data['Open'][-200:],
        high=stock_data['High'][-200:],
        low=stock_data['Low'][-200:],
        close=stock_data['Close'][-200:],
        name='Price'
    ), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=stock_data.index[-200:], y=stock_data['BB_Upper'][-200:],
                             line=dict(color='rgba(250, 128, 114, 0.5)', width=1), name='BB Upper'), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock_data.index[-200:], y=stock_data['BB_Lower'][-200:],
                             line=dict(color='rgba(173, 216, 230, 0.5)', width=1), name='BB Lower',
                             fill='tonexty', fillcolor='rgba(173, 216, 230, 0.1)'), row=1, col=1)

    # SMAs
    fig.add_trace(go.Scatter(x=stock_data.index[-200:], y=stock_data['SMA_20'][-200:],
                             line=dict(color='orange', width=2), name='SMA 20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock_data.index[-200:], y=stock_data['SMA_50'][-200:],
                             line=dict(color='cyan', width=2), name='SMA 50'), row=1, col=1)

    # MACD
    fig.add_trace(go.Scatter(x=stock_data.index[-200:], y=stock_data['MACD'][-200:],
                             line=dict(color='#00ff88', width=2), name='MACD'), row=2, col=1)
    fig.add_trace(go.Scatter(x=stock_data.index[-200:], y=stock_data['MACD_Signal'][-200:],
                             line=dict(color='#ff4444', width=2), name='Signal'), row=2, col=1)

    # Volume
    colors = ['#00ff88' if stock_data['Close'].iloc[i] >= stock_data['Open'].iloc[i]
              else '#ff4444' for i in range(-200, 0)]
    fig.add_trace(go.Bar(x=stock_data.index[-200:], y=stock_data['Volume'][-200:],
                         marker_color=colors, name='Volume'), row=3, col=1)

    fig.update_layout(
        height=900,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.3)',
        font=dict(color='white'),
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def display_model_training_tab():
    """Display the AI model training tab"""
    st.markdown("<div class='stock-card'>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: white; font-size: 2em;'>🤖 <b>AI MODEL TRAINING LAB</b></h2>",
                unsafe_allow_html=True)

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Prepare features
    status_text.text("🔧 Engineering features...")
    features = prepare_features_advanced(st.session_state.stock_data)

    if features:
        # Train all models
        status_text.text("🚀 Initiating AI training sequence...")
        time.sleep(0.5)

        # Time Series Models
        ts_models = train_time_series_models(
            st.session_state.stock_data, progress_bar, status_text)

        # ML Models
        ml_models = train_ml_models(
            features, progress_bar, status_text)

        # Deep Learning Models
        dl_models = train_deep_learning_models(
            features, progress_bar, status_text)

        progress_bar.progress(1.0)
        status_text.text("✅ All models trained successfully!")

        # Store in session state
        st.session_state.ml_models = ml_models
        st.session_state.dl_models = dl_models
        st.session_state.ts_models = ts_models
        st.session_state.features = features

        time.sleep(1)

        total_models = len(ml_models) + len(dl_models) + len(ts_models)

        st.markdown(f"""
        <div class="training-result">
            <h3 style="margin: 0;">🎉 <b>Successfully Trained {total_models} AI Models!</b></h3>
            <p style="margin: 10px 0 0 0;"><b>Ready for prediction analysis</b></p>
        </div>
        """, unsafe_allow_html=True)

        # Explanation boxes
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="explanation-box">
            <h4>📚 <b>What Just Happened?</b></h4>
            <p><strong>Feature Engineering:</strong> We created over 50 intelligent features from the raw stock data including price lags, moving averages, momentum indicators, and technical signals.</p>
            <p><strong>Model Training:</strong> Each AI model learned different patterns in the data:</p>
            <ul style="color: white; font-size: 1.1em;">
                <li><strong>Time Series Models (ARIMA, SARIMA)</strong>: These capture trends and seasonal patterns in prices over time</li>
                <li><strong>Machine Learning Models (Random Forest, XGBoost, etc.)</strong>: These learn complex relationships between features to predict returns</li>
                <li><strong>Deep Learning Models (LSTM, GRU, CNN)</strong>: These neural networks capture sequential patterns and non-linear relationships</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # Show model summary with predictions
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h2>{len(ts_models)}</h2>
                <p><b>Time Series Models</b></p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h2>{len(ml_models)}</h2>
                <p><b>ML Models</b></p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h2>{len(dl_models)}</h2>
                <p><b>Deep Learning Models</b></p>
            </div>
            """, unsafe_allow_html=True)

        # Quick preview of one model's performance
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<h3 style='color: white; font-size: 1.5em;'>🔍 <b>Quick Model Preview</b></h3>", unsafe_allow_html=True)

        if ml_models:
            preview_model_name = list(ml_models.keys())[0]
            preview_model = ml_models[preview_model_name]
            preview_pred = preview_model.predict(features['X_test'])
            preview_actual = features['y_test'].values

            fig_preview = go.Figure()
            fig_preview.add_trace(go.Scatter(
                y=preview_actual[:50],
                mode='lines+markers',
                name='Actual Returns',
                line=dict(color='#00ff88', width=3)
            ))
            fig_preview.add_trace(go.Scatter(
                y=preview_pred[:50],
                mode='lines+markers',
                name=f'{preview_model_name} Predictions',
                line=dict(color='#667eea', width=3, dash='dash')
            ))
            fig_preview.update_layout(
                title=f"<b>{preview_model_name} - Actual vs Predicted Returns (First 50 Test Days)</b>",
                xaxis_title="<b>Test Period Days</b>",
                yaxis_title="<b>Daily Return</b>",
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.3)',
                font=dict(color='white', size=14),
                showlegend=True,
                legend=dict(font=dict(size=14))
            )
            fig_preview.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
            fig_preview.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
            st.plotly_chart(fig_preview, use_container_width=True)

            # Calculate quick stats
            preview_mae = mean_absolute_error(
                preview_actual[:50], preview_pred[:50])
            preview_r2 = r2_score(preview_actual[:50], preview_pred[:50])

            st.markdown(f"""
            <div class="explanation-box">
                <h4>📊 <b>Understanding This Preview</b></h4>
                <p><strong>Green Line (Actual):</strong> The real daily returns that occurred in the market</p>
                <p><strong>Purple Dashed Line (Predicted):</strong> What the {preview_model_name} AI model predicted would happen</p>
                <p><strong>Accuracy Score (R²):</strong> <b>{preview_r2:.3f}</b> - Higher is better (1.0 = perfect, 0.0 = random)</p>
                <p><strong>Average Error (MAE):</strong> <b>{preview_mae:.4f}</b> - Lower is better (average percentage prediction error)</p>
                <p>💡 <em><b>When the lines follow each other closely, the model is accurate!</b></em></p>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.error("❌ <b>Insufficient data for training</b>")

    st.markdown("</div>", unsafe_allow_html=True)


def display_prediction_dashboard():
    """Display the prediction dashboard tab"""
    st.markdown("<div class='stock-card'>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: white; font-size: 2em;'>📈 <b>PREDICTION DASHBOARD</b></h2>",
                unsafe_allow_html=True)

    if hasattr(st.session_state, 'ml_models'):
        # Evaluate all models
        performance = evaluate_all_models(
            st.session_state.ml_models,
            st.session_state.dl_models,
            st.session_state.ts_models,
            st.session_state.features,
            st.session_state.stock_data
        )

        if performance:
            # Find champion
            champion = max(performance.items(), key=lambda x: x[1]['R2'])

            st.markdown(f"""
            <div class="champion-model">
                <h2 style="color: #000;">👑 <b>BEST PERFORMING MODEL: {champion[0]}</b></h2>
                <p style="color: #000; font-size: 1.2em;"><b>Accuracy Score (R²):</b> <strong>{champion[1]['R2']:.4f}</strong> | 
                <b>Avg Error (RMSE):</b> <strong>{champion[1]['RMSE']:.6f}</strong> | 
                <b>Absolute Error (MAE):</b> <strong>{champion[1]['MAE']:.6f}</strong></p>
            </div>
            """, unsafe_allow_html=True)

            # Store performance
            st.session_state.performance = performance

            # Show detailed model comparisons and visualizations
            display_model_comparisons(performance, champion)
        else:
            st.warning("⚠️ <b>Model evaluation in progress...</b>")
    else:
        st.info("👆 <b>Please train models in the AI TRAINING LAB tab first!</b>")

    st.markdown("</div>", unsafe_allow_html=True)


def display_model_comparisons(performance, champion):
    """Display detailed model comparisons and visualizations"""
    from sklearn.metrics import mean_absolute_error, r2_score

    # Explanation of metrics
    st.markdown("""
    <div class="explanation-box">
        <h4>📖 <b>Understanding Model Accuracy Metrics</b></h4>
        <p><strong>R² Score (Accuracy):</strong> Measures how well predictions match reality (0 to 1, where 1 is perfect)</p>
        <ul style="color: white; font-size: 1.1em;">
            <li><b>Above 0.7</b> = Excellent prediction capability</li>
            <li><b>0.4-0.7</b> = Good, useful for forecasting</li>
            <li><b>Below 0.4</b> = Weak, needs improvement</li>
        </ul>
        <p><strong>RMSE (Root Mean Square Error):</strong> Average size of prediction errors (lower is better)</p>
        <p><strong>MAE (Mean Absolute Error):</strong> Average percentage the model is "off" by (lower is better)</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Show champion predictions visually
    st.markdown("<h3 style='color: white; font-size: 1.8em;'>🎯 <b>Best Model Predictions vs Reality</b></h3>",
                unsafe_allow_html=True)

    champion_pred = champion[1]['predictions']
    actual_returns = st.session_state.features['y_test'].values
    test_dates = st.session_state.features['test_dates']

    fig_champion = make_subplots(
        rows=2, cols=1,
        subplot_titles=('<b>Predicted vs Actual Daily Returns</b>',
                        '<b>Prediction Accuracy Over Time</b>'),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )

    # Plot 1: Predictions vs Actual
    fig_champion.add_trace(go.Scatter(
        x=test_dates[:len(champion_pred)],
        y=actual_returns[:len(champion_pred)],
        mode='lines',
        name='Actual Returns',
        line=dict(color='#00ff88', width=2)
    ), row=1, col=1)

    fig_champion.add_trace(go.Scatter(
        x=test_dates[:len(champion_pred)],
        y=champion_pred,
        mode='lines',
        name=f'{champion[0]} Predictions',
        line=dict(color='#667eea', width=2, dash='dash')
    ), row=1, col=1)

    # Plot 2: Prediction errors
    errors = np.abs(actual_returns[:len(champion_pred)] - champion_pred)
    fig_champion.add_trace(go.Bar(
        x=test_dates[:len(champion_pred)],
        y=errors,
        name='Prediction Error',
        marker_color='rgba(255, 68, 68, 0.6)'
    ), row=2, col=1)

    fig_champion.update_layout(
        height=800,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.3)',
        font=dict(color='white', size=13),
        showlegend=True
    )
    fig_champion.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig_champion.update_yaxes(gridcolor='rgba(255,255,255,0.1)')

    st.plotly_chart(fig_champion, use_container_width=True)

    # Performance comparison of ALL models
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: white; font-size: 1.8em;'>🏆 <b>All Models Performance Comparison</b></h3>",
                unsafe_allow_html=True)

    perf_df = pd.DataFrame({
        'Model': list(performance.keys()),
        'R²': [v['R2'] for v in performance.values()],
        'RMSE': [v['RMSE'] for v in performance.values()],
        'MAE': [v['MAE'] for v in performance.values()],
        'Type': [v['type'] for v in performance.values()]
    }).sort_values('R²', ascending=False)

    # Display top models with better formatting
    st.markdown("<h4 style='color: white; font-size: 1.5em;'>📊 <b>Top 10 Models</b></h4>",
                unsafe_allow_html=True)
    for idx, row in perf_df.head(10).iterrows():
        type_emoji = {'ML': '🌲', 'DL': '🧠', 'TS': '🔮'}
        type_name = {'ML': 'Machine Learning',
                     'DL': 'Deep Learning', 'TS': 'Time Series'}

        # Color code based on R² score
        if row['R²'] > 0.7:
            border_color = '#00ff88'
            perf_text = "Excellent"
        elif row['R²'] > 0.4:
            border_color = '#ffaa00'
            perf_text = "Good"
        else:
            border_color = '#ff4444'
            perf_text = "Fair"

        st.markdown(f"""
        <div class="model-card" style="border: 2px solid {border_color};">
            <strong style="font-size: 1.3em;">{type_emoji[row['Type']]} {row['Model']}</strong>
            <span style="background: {border_color}; padding: 3px 10px; border-radius: 5px; margin-left: 10px; color: #000; font-size: 0.9em;"><b>{perf_text}</b></span><br>
            <p style="margin: 10px 0 5px 0; font-size: 1.15em;">
                <strong>Category:</strong> <b>{type_name[row['Type']]}</b><br>
                <strong>Accuracy (R²):</strong> <b>{row['R²']:.4f}</b> | 
                <strong>RMSE:</strong> <b>{row['RMSE']:.6f}</b> | 
                <strong>MAE:</strong> <b>{row['MAE']:.6f}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)


def display_risk_analysis_tab():
    """Display the risk analysis and forecast tab"""
    st.markdown("<div class='stock-card'>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: white; font-size: 2em;'>📉 <b>RISK ANALYSIS & FORECAST</b></h2>",
                unsafe_allow_html=True)

    risk_metrics = calculate_risk_metrics(st.session_state.stock_data)

    if risk_metrics:
        # Risk metrics display
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h2>{risk_metrics['annual_volatility']:.1f}%</h2>
                <p><b>Volatility</b></p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h2>{risk_metrics['sharpe_ratio']:.2f}</h2>
                <p><b>Sharpe Ratio</b></p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h2>{risk_metrics['sortino_ratio']:.2f}</h2>
                <p><b>Sortino Ratio</b></p>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h2>{risk_metrics['max_drawdown']:.1f}%</h2>
                <p><b>Max Drawdown</b></p>
            </div>
            """, unsafe_allow_html=True)
        with col5:
            st.markdown(f"""
            <div class="metric-card">
                <h2>{risk_metrics['var_95']:.2f}%</h2>
                <p><b>VaR (95%)</b></p>
            </div>
            """, unsafe_allow_html=True)

        # Generate recommendation
        if hasattr(st.session_state, 'performance') and hasattr(st.session_state, 'characteristics'):
            recommendation, rec_type, reasoning = generate_advanced_recommendation(
                risk_metrics,
                st.session_state.performance,
                st.session_state.characteristics
            )

            st.markdown("<br>", unsafe_allow_html=True)

            if rec_type == "buy":
                st.markdown(f"""
                <div class="recommendation-buy">
                    <h2 style="font-size: 2em;">🎯 <b>AI RECOMMENDATION: {recommendation}</b></h2>
                    <p style="font-size:1.3em; margin: 15px 0;"><strong>Analysis:</strong> <b>{reasoning}</b></p>
                    <p style="font-size:1.2em;"><strong>Confidence:</strong> <b>Based on {len(st.session_state.performance)} AI models analyzing {len(st.session_state.stock_data)} days of data</b></p>
                </div>
                """, unsafe_allow_html=True)
            elif rec_type == "hold":
                st.markdown(f"""
                <div class="recommendation-hold">
                    <h2 style="font-size: 2em;">⚖️ <b>AI RECOMMENDATION: {recommendation}</b></h2>
                    <p style="font-size:1.3em; margin: 15px 0;"><strong>Analysis:</strong> <b>{reasoning}</b></p>
                    <p style="font-size:1.2em;"><strong>Confidence:</strong> <b>Based on {len(st.session_state.performance)} AI models analyzing {len(st.session_state.stock_data)} days of data</b></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="recommendation-sell">
                    <h2 style="font-size: 2em;">⚠️ <b>AI RECOMMENDATION: {recommendation}</b></h2>
                    <p style="font-size:1.3em; margin: 15px 0;"><strong>Analysis:</strong> <b>{reasoning}</b></p>
                    <p style="font-size:1.2em;"><strong>Confidence:</strong> <b>Based on {len(st.session_state.performance)} AI models analyzing {len(st.session_state.stock_data)} days of data</b></p>
                </div>
                """, unsafe_allow_html=True)

        display_risk_visualizations(risk_metrics)
    else:
        st.error("❌ <b>Insufficient data for risk analysis</b>")

    st.markdown("</div>", unsafe_allow_html=True)


def display_risk_visualizations(risk_metrics):
    """Display risk visualization charts"""
    # Explain risk metrics
    st.markdown("""
    <div class="explanation-box">
        <h4>📚 <b>Understanding Risk Metrics</b></h4>
        <p><strong>Volatility:</strong> How much the price jumps around (lower = more stable)</p>
        <p><strong>Sharpe Ratio:</strong> Return vs risk (above 1.0 is good, above 2.0 is excellent)</p>
        <p><strong>Sortino Ratio:</strong> Like Sharpe but only considers downside risk (higher is better)</p>
        <p><strong>Max Drawdown:</strong> Biggest loss from peak to bottom (how much you could lose)</p>
        <p><strong>VaR (Value at Risk):</strong> Expected worst-case daily loss 95% of the time</p>
    </div>
    """, unsafe_allow_html=True)

    # Risk visualization
    st.markdown("<br>", unsafe_allow_html=True)

    fig_risk = make_subplots(
        rows=2, cols=1,
        subplot_titles=('<b>Cumulative Returns Over Time</b>',
                        '<b>Drawdown Analysis (How Much You Could Lose)</b>'),
        vertical_spacing=0.15
    )

    # Cumulative returns
    cumulative = (1 + risk_metrics['returns']).cumprod() * 100
    fig_risk.add_trace(go.Scatter(
        x=st.session_state.stock_data.index[-len(cumulative):],
        y=cumulative,
        fill='tozeroy',
        fillcolor='rgba(0, 255, 136, 0.2)',
        line=dict(color='#00ff88', width=3),
        name='Cumulative Return (%)'
    ), row=1, col=1)

    # Add 100% baseline
    fig_risk.add_hline(y=100, line_dash="dash", line_color="white",
                       annotation_text="Break-even", row=1, col=1)

    # Drawdown
    fig_risk.add_trace(go.Scatter(
        x=st.session_state.stock_data.index[-len(risk_metrics['drawdowns']):],
        y=risk_metrics['drawdowns'],
        fill='tozeroy',
        fillcolor='rgba(255, 68, 68, 0.3)',
        line=dict(color='#ff4444', width=3),
        name='Drawdown (%)'
    ), row=2, col=1)

    fig_risk.update_layout(
        height=700,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=13),
        showlegend=True
    )
    fig_risk.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig_risk.update_yaxes(gridcolor='rgba(255,255,255,0.1)')

    st.plotly_chart(fig_risk, use_container_width=True)

    st.markdown("""
    <div class="explanation-box">
        <h4>📈 <b>Reading the Risk Charts</b></h4>
        <p><strong>Top Chart (Cumulative Returns):</strong> Shows total profit/loss over time. Starting at 100%, going up means gains, down means losses.</p>
        <p><strong>Bottom Chart (Drawdowns):</strong> Shows periods of losses from peaks. The deeper the red, the bigger the loss from the highest point.</p>
        <p>💡 <em><b>Look for: Steady upward trends with shallow drawdowns = good investment. Big drops = risky periods.</b></em></p>
    </div>
    """, unsafe_allow_html=True)


def main():
    # COSMIC HEADER
    st.markdown("""
    <div class="cosmic-header">
        <h1 style="color: white; margin: 0; font-size: 3.5em;">🌌 <b>ULTIMATE AI STOCK FORECASTER</b></h1>
        <p style="color: white; font-size: 1.4em; margin: 10px 0 0 0;"><b>15+ Advanced Models | Auto-Selection | Real-Time Training</b></p>
        <p style="color: rgba(255,255,255,0.8); font-size: 1em; margin: 5px 0 0 0;"><b>ARIMA • SARIMA • XGBoost • LightGBM • CatBoost • LSTM • GRU • CNN • Transformers</b></p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar for settings
    with st.sidebar:
        st.markdown("### ⚙️ <b>SETTINGS</b>")
        data_period = st.selectbox(
            "Data Period", ["1y", "2y", "5y", "max"], index=1)
        forecast_days = st.slider("Forecast Days", 1, 30, 7)
        st.markdown("---")
        st.markdown("### 📊 <b>MODEL CATEGORIES</b>")
        st.markdown("✅ <b>Time Series:</b> ARIMA, SARIMA")
        st.markdown(
            "✅ <b>ML Models:</b> RF, XGBoost, LightGBM, CatBoost, GBM, SVR")
        st.markdown(
            "✅ <b>Deep Learning:</b> LSTM, GRU, RNN, Bi-LSTM, CNN, CNN-LSTM")

    # Load stocks
    db = StockDatabase()
    stocks_df = db.get_all_stocks()

    options = []
    for _, row in stocks_df.iterrows():
        symbol = row.get('Symbol', '')
        name = row.get('Name', '')
        if symbol and name:
            options.append(f"{symbol} - {name}")

    # Stock selector
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_option = st.selectbox(
            "🔍 SELECT STOCK:",
            options=options if options else [
                "AAPL - Apple Inc", "TSLA - Tesla Inc"],
            index=0
        )
    with col2:
        analyze_button = st.button(
            "🚀 ANALYZE", use_container_width=True, type="primary")

    if selected_option and analyze_button:
        selected_symbol = selected_option.split(' - ')[0]

        # Fetch data
        with st.spinner(f"📡 Fetching {selected_symbol} data..."):
            stock_data = fetch_stock_data(selected_symbol, period=data_period)

        if stock_data is not None:
            # Store data in session state
            st.session_state.stock_data = stock_data

            # Analyze stock characteristics
            characteristics = analyze_stock_characteristics(stock_data)
            st.session_state.characteristics = characteristics

            # Display current metrics
            display_stock_metrics(stock_data)

            # Display characteristics
            st.markdown("<br>", unsafe_allow_html=True)
            char_cols = st.columns(4)
            with char_cols[0]:
                st.info(
                    f"📊 <b>Volatility:</b> {characteristics['volatility']*100:.1f}%")
            with char_cols[1]:
                st.info(
                    f"📈 <b>Trend:</b> {'Strong' if characteristics['has_strong_trend'] else 'Weak'}")
            with char_cols[2]:
                st.info(
                    f"🔄 <b>Stationary:</b> {'Yes' if characteristics['is_stationary'] else 'No'}")
            with char_cols[3]:
                st.info(
                    f"🌊 <b>Seasonal:</b> {'Yes' if characteristics['has_seasonality'] else 'No'}")

            # TABS
            tab1, tab2, tab3, tab4 = st.tabs(
                ["📊 CHARTS", "🤖 AI TRAINING LAB", "📈 PREDICTION DASHBOARD", "📉 RISK & FORECAST"])

            with tab1:
                display_advanced_chart(stock_data)

            with tab2:
                display_model_training_tab()

            with tab3:
                display_prediction_dashboard()

            with tab4:
                display_risk_analysis_tab()

        else:
            st.error(f"❌ Could not fetch data for {selected_symbol}")

    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: rgba(255, 255, 255, 0.05); border-radius: 15px;">
        <p style="color: rgba(255,255,255,0.7); font-size: 1.1em;">⚡ <b>Powered by 15+ Advanced AI Models | Real-Time Analysis | Intelligent Auto-Selection</b></p>
        <p style="color: rgba(255,255,255,0.5); font-size: 1em;"><b>Disclaimer: For educational purposes only. Not financial advice.</b></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
