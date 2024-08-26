from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import plotly.graph_objs as go
import yfinance as yf
import requests
from datetime import datetime, timedelta
import os
import schedule
import time
import threading

app = Flask(__name__)

# Global variables
API_KEY = 'MMK7S1NB70W7U85D'
HISTORICAL_DATA_DIR = './stock_data/'
SENTIMENT_DATA_DIR = './sentiment_data/'
TICKERS = ['OXY', 'XOM', 'VRTX', 'NVDA', 'GOOG', 'AMZN']

# Function to download stock data
def download_stock_data(tickers, start_date='2010-01-01'):
    for ticker in tickers:
        stock_data = yf.download(ticker, start=start_date)
        filename = f"{HISTORICAL_DATA_DIR}{ticker}_{datetime.now().strftime('%Y-%m-%d')}.csv"
        stock_data.to_csv(filename)
        print(f"Saved stock data for {ticker} to {filename}")

# Function to get sentiment data
def get_sentiment_data(symbol):
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={API_KEY}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get('feed', [])
    return []

# Function to save sentiment data
def save_sentiment_data(symbol, sentiment_data):
    if not sentiment_data:
        return
    
    processed_data = []
    for item in sentiment_data:
        processed_data.append({
            'symbol': symbol,
            'title': item.get('title', ''),
            'time_published': item.get('time_published', ''),
            'sentiment_score': item.get('overall_sentiment_score', '')
        })
    
    df = pd.DataFrame(processed_data)
    filename = f"{SENTIMENT_DATA_DIR}{symbol}_sentiment_{datetime.now().strftime('%Y-%m-%d')}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved sentiment data for {symbol} to {filename}")

# Function to train model and create plot
def train_model_and_plot(ticker):
    # Load historical data
    historical_data = pd.read_csv(f'{HISTORICAL_DATA_DIR}{ticker}_{datetime.now().strftime("%Y-%m-%d")}.csv')
    
    # Try to load sentiment data, if file doesn't exist, skip it
    try:
        sentiment_data = pd.read_csv(f'{SENTIMENT_DATA_DIR}{ticker}_sentiment_{datetime.now().strftime("%Y-%m-%d")}.csv')
        sentiment_data['time_published'] = pd.to_datetime(sentiment_data['time_published'], format='%Y%m%dT%H%M%S')
        sentiment_data['Date'] = sentiment_data['time_published'].dt.date
        daily_sentiment = sentiment_data.groupby('Date')['sentiment_score'].mean().reset_index()
    except FileNotFoundError:
        print(f"Sentiment file for {ticker} not found, skipping sentiment data merge.")
        daily_sentiment = pd.DataFrame(columns=['Date', 'sentiment_score'])
    
    # Convert 'Date' to datetime in historical data
    historical_data['Date'] = pd.to_datetime(historical_data['Date'])
    daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date'])
    
    # Merge historical data with sentiment data
    merged_data = pd.merge(historical_data, daily_sentiment, on='Date', how='left')
    merged_data['sentiment_score'].fillna(0, inplace=True)
    merged_data['Daily Return'] = merged_data['Close'].pct_change().fillna(0)
    
    # Train the model
    X = merged_data[['Open', 'High', 'Low', 'Volume', 'Daily Return', 'sentiment_score']]
    y = merged_data['Close']
    model = LinearRegression().fit(X, y)
    
    # Prepare data for predicting tomorrow's price
    latest_data = merged_data.iloc[-1]
    tomorrow_features = {
        'Open': latest_data['Close'],
        'High': latest_data['Close'],
        'Low': latest_data['Close'],
        'Volume': latest_data['Volume'],
        'Daily Return': 0,
        'sentiment_score': latest_data['sentiment_score']
    }
    tomorrow_df = pd.DataFrame([tomorrow_features])
    tomorrow_price = model.predict(tomorrow_df)[0]
    
    # Prepare data for plotting
    plot_data = merged_data.tail(30).copy()
    plot_data['Predicted'] = model.predict(plot_data[['Open', 'High', 'Low', 'Volume', 'Daily Return', 'sentiment_score']])
    
    # Use pd.concat instead of append
    next_day_row = pd.DataFrame({
        'Date': [plot_data['Date'].iloc[-1] + pd.Timedelta(days=1)],
        'Close': [np.nan],
        'Predicted': [tomorrow_price]
    })
    
    plot_data = pd.concat([plot_data, next_day_row], ignore_index=True)
    
    # Plotting the results
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['Close'], mode='lines', name='Actual Close'))
    fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['Predicted'], mode='lines', name='Predicted Close'))
    fig.update_layout(title=f'{ticker} Stock Price', xaxis_title='Date', yaxis_title='Price')
    
    return fig, tomorrow_price

# Function to update all data and models
def update_all_data_and_models():
    print("Updating data and models...")
    download_stock_data(TICKERS)
    for ticker in TICKERS:
        sentiment_data = get_sentiment_data(ticker)
        save_sentiment_data(ticker, sentiment_data)
        train_model_and_plot(ticker)
    print("Update complete.")

# Schedule the update
def run_schedule():
    while True:
        schedule.run_pending()
        time.sleep(60)

schedule.every().day.at("07:45").do(update_all_data_and_models)

# Start the scheduler in a separate thread
scheduler_thread = threading.Thread(target=run_schedule)
scheduler_thread.start()

@app.route('/')
def dashboard():
    return render_template('dashboard.html', tickers=TICKERS)

@app.route('/get_data/<ticker>')
def get_data(ticker):
    fig, tomorrow_price = train_model_and_plot(ticker)
    
    # Calculate index risk (using a simple volatility measure)
    historical_data = pd.read_csv(f'{HISTORICAL_DATA_DIR}{ticker}_{datetime.now().strftime("%Y-%m-%d")}.csv')
    returns = historical_data['Close'].pct_change().dropna()
    volatility = returns.std() * (252 ** 0.5)  # Annualized volatility
    index_risk = volatility * 100  # Convert to percentage
    
    return jsonify({
        'plot': fig.to_json(),
        'tomorrow_price': float(tomorrow_price),
        'index_risk': float(index_risk)
    })

if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs(HISTORICAL_DATA_DIR, exist_ok=True)
    os.makedirs(SENTIMENT_DATA_DIR, exist_ok=True)
    
    # Run initial update
    update_all_data_and_models()
    
    app.run(debug=True)