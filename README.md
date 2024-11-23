# LSTM Stock Price Prediction

## Overview
This project demonstrates stock price prediction using Long Short-Term Memory (LSTM) neural networks in Python. The model leverages historical stock data to forecast future prices.

## Features
- LSTM neural network for time series forecasting
- Real-time stock data retrieval using yfinance
- Data visualization with Plotly
- Candlestick chart analysis
- Stock price prediction model

## Prerequisites
- Python 3.8+
- Libraries: 
  - pandas
  - numpy
  - scikit-learn
  - keras
  - tensorflow
  - yfinance
  - plotly

## Installation
1. Clone the repository
```bash
git clone https://github.com/Nizar04/LSTM-Stock-Prediction.git
cd stock-prediction-lstm
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage
Run the Jupyter notebook or Python script to:
- Fetch stock data
- Visualize stock prices
- Train LSTM model
- Make price predictions

## Model Performance
- Trained on Apple (AAPL) stock data
- Uses features: Open, High, Low, Volume
- 30 training epochs
- Mean Squared Error loss function

## Future Improvements
- Add more stocks
- Implement cross-validation
- Create interactive visualization
- Add more advanced features

## Disclaimer
Stock predictions are speculative. Always conduct your own research before making investment decisions.

## License
 MIT
