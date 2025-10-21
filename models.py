# models.py - AI model definitions and training functions
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Conv1D, MaxPooling1D, Flatten, Bidirectional, SimpleRNN
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')


def prepare_features_advanced(stock_data):
    """Prepare advanced features for machine learning models"""
    try:
        df = stock_data.copy()

        # Price-based features
        for lag in [1, 2, 3, 5, 10, 20]:
            df[f'Price_Lag_{lag}'] = df['Close'].shift(lag)

        # Rolling statistics
        for window in [5, 10, 20, 50]:
            df[f'Roll_Mean_{window}'] = df['Close'].rolling(window).mean()
            df[f'Roll_Std_{window}'] = df['Close'].rolling(window).std()
            df[f'Roll_Min_{window}'] = df['Close'].rolling(window).min()
            df[f'Roll_Max_{window}'] = df['Close'].rolling(window).max()

        # Momentum features
        for period in [5, 10, 20]:
            df[f'Momentum_{period}'] = df['Close'] / \
                df['Close'].shift(period) - 1
            df[f'ROC_{period}'] = df['Close'].pct_change(period)

        # Volume features
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

        # Technical indicators as features
        df['RSI_norm'] = df['RSI'] / 100
        df['MACD_diff'] = df['MACD'] - df['MACD_Signal']

        # Target: Next day return
        df['Target'] = df['Close'].shift(-1) / df['Close'] - 1

        df = df.dropna()

        if len(df) < 100:
            return None

        # Select feature columns
        feature_cols = [col for col in df.columns if any(x in col for x in
                                                         ['Lag', 'Roll', 'Momentum', 'Volume', 'RSI', 'MACD', 'ROC'])]

        X = df[feature_cols]
        y = df['Target']

        # Time-based split (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler,
            'feature_names': feature_cols,
            'original_data': df,
            'test_dates': y_test.index
        }
    except Exception as e:
        print(f"Feature preparation error: {str(e)}")
        return None


def train_time_series_models(stock_data, progress_bar, status_text):
    """Train ARIMA, SARIMA, SARIMAX models"""
    models = {}
    prices = stock_data['Close'].values

    try:
        status_text.text("🔮 <b>Training ARIMA...</b>")
        progress_bar.progress(0.1)
        arima = ARIMA(prices, order=(5, 1, 2))
        arima_fit = arima.fit()
        models['ARIMA'] = arima_fit
    except Exception as e:
        print(f"ARIMA training failed: {e}")

    try:
        status_text.text("🌙 <b>Training SARIMA...</b>")
        progress_bar.progress(0.2)
        sarima = SARIMAX(prices, order=(2, 1, 2), seasonal_order=(1, 1, 1, 5))
        sarima_fit = sarima.fit(disp=False)
        models['SARIMA'] = sarima_fit
    except Exception as e:
        print(f"SARIMA training failed: {e}")

    return models


def train_ml_models(features, progress_bar, status_text):
    """Train Machine Learning models"""
    models = {}

    # Random Forest
    try:
        status_text.text("🌲 <b>Training Random Forest...</b>")
        progress_bar.progress(0.3)
        rf = RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(features['X_train'], features['y_train'])
        models['Random Forest'] = rf
    except Exception as e:
        print(f"Random Forest training failed: {e}")

    # Gradient Boosting
    try:
        status_text.text("⚡ <b>Training Gradient Boosting...</b>")
        progress_bar.progress(0.35)
        gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb.fit(features['X_train'], features['y_train'])
        models['Gradient Boosting'] = gb
    except Exception as e:
        print(f"Gradient Boosting training failed: {e}")

    # XGBoost
    try:
        status_text.text("🚀 <b>Training XGBoost...</b>")
        progress_bar.progress(0.4)
        xgb = XGBRegressor(n_estimators=200, learning_rate=0.05,
                           max_depth=8, random_state=42, verbosity=0)
        xgb.fit(features['X_train'], features['y_train'])
        models['XGBoost'] = xgb
    except Exception as e:
        print(f"XGBoost training failed: {e}")

    # LightGBM
    try:
        status_text.text("💡 <b>Training LightGBM...</b>")
        progress_bar.progress(0.45)
        lgbm = LGBMRegressor(
            n_estimators=200, learning_rate=0.05, random_state=42, verbose=-1)
        lgbm.fit(features['X_train'], features['y_train'])
        models['LightGBM'] = lgbm
    except Exception as e:
        print(f"LightGBM training failed: {e}")

    # CatBoost
    try:
        status_text.text("🐱 <b>Training CatBoost...</b>")
        progress_bar.progress(0.5)
        cat = CatBoostRegressor(
            iterations=200, learning_rate=0.05, random_state=42, verbose=0)
        cat.fit(features['X_train'], features['y_train'])
        models['CatBoost'] = cat
    except Exception as e:
        print(f"CatBoost training failed: {e}")

    # SVR
    try:
        status_text.text("🎯 <b>Training SVR...</b>")
        progress_bar.progress(0.55)
        svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        svr.fit(features['X_train'], features['y_train'])
        models['SVR'] = svr
    except Exception as e:
        print(f"SVR training failed: {e}")

    return models


def train_deep_learning_models(features, progress_bar, status_text):
    """Train Deep Learning models"""
    models = {}
    n_features = features['X_train'].shape[1]

    # LSTM
    try:
        status_text.text("🧠 <b>Training LSTM...</b>")
        progress_bar.progress(0.6)
        X_train_lstm = features['X_train'].reshape(
            (features['X_train'].shape[0], 1, n_features))
        X_test_lstm = features['X_test'].reshape(
            (features['X_test'].shape[0], 1, n_features))

        lstm_model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(1, n_features)),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(25),
            Dropout(0.2),
            Dense(1)
        ])
        lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        lstm_model.fit(
            X_train_lstm, features['y_train'], epochs=30, batch_size=32, verbose=0)
        models['LSTM'] = (lstm_model, X_test_lstm)
    except Exception as e:
        print(f"LSTM training failed: {e}")

    # Bidirectional LSTM
    try:
        status_text.text("🔄 <b>Training Bi-LSTM...</b>")
        progress_bar.progress(0.65)
        X_train_bilstm = features['X_train'].reshape(
            (features['X_train'].shape[0], 1, n_features))
        X_test_bilstm = features['X_test'].reshape(
            (features['X_test'].shape[0], 1, n_features))

        bilstm_model = Sequential([
            Bidirectional(LSTM(64, return_sequences=True),
                          input_shape=(1, n_features)),
            Dropout(0.2),
            Bidirectional(LSTM(32)),
            Dropout(0.2),
            Dense(1)
        ])
        bilstm_model.compile(optimizer='adam', loss='mse')
        bilstm_model.fit(
            X_train_bilstm, features['y_train'], epochs=30, batch_size=32, verbose=0)
        models['Bi-LSTM'] = (bilstm_model, X_test_bilstm)
    except Exception as e:
        print(f"Bi-LSTM training failed: {e}")

    # GRU
    try:
        status_text.text("⚙️ <b>Training GRU...</b>")
        progress_bar.progress(0.7)
        X_train_gru = features['X_train'].reshape(
            (features['X_train'].shape[0], 1, n_features))
        X_test_gru = features['X_test'].reshape(
            (features['X_test'].shape[0], 1, n_features))

        gru_model = Sequential([
            GRU(100, return_sequences=True, input_shape=(1, n_features)),
            Dropout(0.2),
            GRU(50),
            Dropout(0.2),
            Dense(1)
        ])
        gru_model.compile(optimizer='adam', loss='mse')
        gru_model.fit(
            X_train_gru, features['y_train'], epochs=30, batch_size=32, verbose=0)
        models['GRU'] = (gru_model, X_test_gru)
    except Exception as e:
        print(f"GRU training failed: {e}")

    # Simple RNN
    try:
        status_text.text("🔁 <b>Training RNN...</b>")
        progress_bar.progress(0.75)
        X_train_rnn = features['X_train'].reshape(
            (features['X_train'].shape[0], 1, n_features))
        X_test_rnn = features['X_test'].reshape(
            (features['X_test'].shape[0], 1, n_features))

        rnn_model = Sequential([
            SimpleRNN(100, return_sequences=True, input_shape=(1, n_features)),
            Dropout(0.2),
            SimpleRNN(50),
            Dropout(0.2),
            Dense(1)
        ])
        rnn_model.compile(optimizer='adam', loss='mse')
        rnn_model.fit(
            X_train_rnn, features['y_train'], epochs=25, batch_size=32, verbose=0)
        models['RNN'] = (rnn_model, X_test_rnn)
    except Exception as e:
        print(f"RNN training failed: {e}")

    # 1D CNN
    try:
        status_text.text("🌊 <b>Training 1D CNN...</b>")
        progress_bar.progress(0.8)
        X_train_cnn = features['X_train'].reshape(
            (features['X_train'].shape[0], n_features, 1))
        X_test_cnn = features['X_test'].reshape(
            (features['X_test'].shape[0], n_features, 1))

        cnn_model = Sequential([
            Conv1D(64, kernel_size=3, activation='relu',
                   input_shape=(n_features, 1)),
            MaxPooling1D(pool_size=2),
            Conv1D(32, kernel_size=3, activation='relu'),
            Flatten(),
            Dense(50, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        cnn_model.compile(optimizer='adam', loss='mse')
        cnn_model.fit(
            X_train_cnn, features['y_train'], epochs=30, batch_size=32, verbose=0)
        models['1D CNN'] = (cnn_model, X_test_cnn)
    except Exception as e:
        print(f"1D CNN training failed: {e}")

    # Hybrid CNN-LSTM
    try:
        status_text.text("🔥 <b>Training CNN-LSTM Hybrid...</b>")
        progress_bar.progress(0.85)
        X_train_hybrid = features['X_train'].reshape(
            (features['X_train'].shape[0], 1, n_features))
        X_test_hybrid = features['X_test'].reshape(
            (features['X_test'].shape[0], 1, n_features))

        hybrid_model = Sequential([
            Conv1D(64, kernel_size=2, activation='relu',
                   input_shape=(1, n_features)),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(25),
            Dropout(0.2),
            Dense(1)
        ])
        hybrid_model.compile(optimizer='adam', loss='mse')
        hybrid_model.fit(
            X_train_hybrid, features['y_train'], epochs=25, batch_size=32, verbose=0)
        models['CNN-LSTM'] = (hybrid_model, X_test_hybrid)
    except Exception as e:
        print(f"CNN-LSTM training failed: {e}")

    return models


def evaluate_all_models(ml_models, dl_models, ts_models, features, stock_data):
    """Evaluate all models and return performance metrics"""
    performance = {}

    # Evaluate ML models
    for name, model in ml_models.items():
        try:
            predictions = model.predict(features['X_test'])
            mae = mean_absolute_error(features['y_test'], predictions)
            rmse = np.sqrt(mean_squared_error(features['y_test'], predictions))
            r2 = r2_score(features['y_test'], predictions)

            performance[name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'predictions': predictions,
                'type': 'ML'
            }
        except Exception as e:
            print(f"ML model {name} evaluation failed: {e}")
            continue

    # Evaluate DL models
    for name, (model, X_test_reshaped) in dl_models.items():
        try:
            predictions = model.predict(X_test_reshaped, verbose=0).flatten()
            mae = mean_absolute_error(features['y_test'], predictions)
            rmse = np.sqrt(mean_squared_error(features['y_test'], predictions))
            r2 = r2_score(features['y_test'], predictions)

            performance[name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'predictions': predictions,
                'type': 'DL'
            }
        except Exception as e:
            print(f"DL model {name} evaluation failed: {e}")
            continue

    # Evaluate Time Series models
    for name, model in ts_models.items():
        try:
            forecast = model.forecast(steps=len(features['y_test']))
            test_prices = stock_data['Close'].values[-len(features['y_test']):]
            predictions = (
                forecast / test_prices[:-1] - 1) if len(forecast) == len(test_prices) - 1 else []

            if len(predictions) > 0:
                mae = mean_absolute_error(
                    features['y_test'][:len(predictions)], predictions)
                rmse = np.sqrt(mean_squared_error(
                    features['y_test'][:len(predictions)], predictions))
                r2 = r2_score(features['y_test']
                              [:len(predictions)], predictions)

                performance[name] = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2,
                    'predictions': predictions,
                    'type': 'TS'
                }
        except Exception as e:
            print(f"Time series model {name} evaluation failed: {e}")
            continue

    return performance
