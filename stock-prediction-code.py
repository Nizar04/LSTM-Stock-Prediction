import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import date, timedelta
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping

def fetch_stock_data(ticker='AAPL', days_back=5000):
    """Fetch historical stock data"""
    today = date.today()
    end_date = today.strftime("%Y-%m-%d")
    start_date = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    data["Date"] = data.index
    data = data[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    data.reset_index(drop=True, inplace=True)
    return data

def visualize_stock_data(data):
    """Create candlestick chart"""
    figure = go.Figure(data=[go.Candlestick(
        x=data["Date"],
        open=data["Open"], 
        high=data["High"],
        low=data["Low"], 
        close=data["Close"]
    )])
    figure.update_layout(
        title=f"Stock Price Analysis", 
        xaxis_rangeslider_visible=False
    )
    figure.show()

def prepare_data(data):
    """Prepare data for LSTM model"""
    features = data[["Open", "High", "Low", "Volume"]]
    target = data["Close"]
    
    # Normalize features
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Create sequences
    X, y = [], []
    for i in range(len(features_scaled) - 60):
        X.append(features_scaled[i:i+60])
        y.append(target[i+60])
    
    return np.array(X), np.array(y), scaler

def create_lstm_model(input_shape):
    """Create LSTM neural network"""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def main():
    # Fetch and visualize data
    data = fetch_stock_data()
    visualize_stock_data(data)
    
    # Prepare data
    X, y, scaler = prepare_data(data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train model
    model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
    
    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    
    # Train model
    history = model.fit(
        X_train, y_train, 
        validation_split=0.2,
        epochs=50, 
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Evaluate model
    loss = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}")

if __name__ == "__main__":
    main()
