import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error

import os

# logistic regression
def get_stock_data(ticker, start_date='2020-01-01', end_date=datetime.now(), auto_adjust=False):

    # Fetch stock data
    stock_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    stock_data.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    stock_data = stock_data.reset_index()    
    stock_data['Date'] = stock_data['Date'].dt.tz_localize(None)
    
    return stock_data

def aggregate_sentiment(news_df):
    try:
        news_df['date'] = pd.to_datetime(news_df['date'], errors='coerce')
        news_df['date'] = news_df['date'].dt.tz_localize(None) if news_df['date'].dt.tz is not None else news_df['date']
        
        # Filter out rows with NaN dates or compound scores
        news_df = news_df.dropna(subset=['date', 'compound'])
        if news_df.empty:
            print("No valid sentiment data available after filtering.")
            return pd.DataFrame(columns=['date', 'compound'])
        
        sentiment_df = news_df.groupby('date')['compound'].mean().reset_index()
        
        return sentiment_df
    
    except Exception as e:
        print(f"Error in sentiment aggregation: {e}")
        return pd.DataFrame(columns=['date', 'compound'])

def prepare_data(news_df, tickers):
    stock_data = {}
    
    for ticker in tickers:
        ticker_news_df = news_df[news_df['ticker'] == ticker]
        
        # Get stock data
        stock_df = get_stock_data(ticker)
        
        # Aggregate sentiment
        sentiment_df = aggregate_sentiment(ticker_news_df)
        
        print(f"Stock DataFrame for {ticker}:")
        print(stock_df.columns)
        print(stock_df.head())
        print(f"\nSentiment DataFrame for {ticker}:")
        print(sentiment_df.columns)
        print(sentiment_df.head())
        
        # Merge stock data with sentiment data
        try:
            sentiment_df = sentiment_df.set_index('date')
            combined_df = stock_df.join(sentiment_df, on='Date', how='left')
            
            # Reset index to make merging easier
            combined_df = combined_df.reset_index(drop=True)
            
            # Fill NaN sentiment values with 0
            combined_df['compound'] = combined_df['compound'].fillna(0)
            
            # Add target variable: stock movement (1 for up, 0 for down)
            combined_df['target'] = (combined_df['Close'].shift(-1) > combined_df['Close']).astype(int)
            
            # Drop the last row since we can't predict its target
            combined_df = combined_df.iloc[:-1]
            
            # Drop rows with NaN values
            combined_df = combined_df.dropna()
            
            # Select relevant columns
            columns_to_keep = ['Date', 'Open', 'High', 'Low', 'Adj Close', 'Volume', 'compound', 'target']
            combined_df = combined_df[columns_to_keep]
            print("Combined sentiment and stock dataframe:")
            print(combined_df.head())
            
            stock_data[ticker] = combined_df
        
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue
    
    return stock_data

def train_model(stock_data):
    model_results = {}
    classification_reports = {}

    
    for ticker, data in stock_data.items():
        # Prepare features and target
        X = data[['compound', 'Open', 'High', 'Low', 'Volume']]
        y = data['target']
        
        # Ensure we have enough data
        if len(X) < 10:
            print(f"Not enough data for {ticker} to train the model")
            continue
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize and train the model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_test)
        
        # Add prediction results to the data
        prediction_df = pd.DataFrame({
            'Date': data.iloc[X_test.index]['Date'],  # Align the dates with the X_test indices
            'Actual': y_test,
            'Predicted': y_pred
        })
        
        # Display predictions for each ticker
        print(f"\nPredictions for {ticker}:")
        print(prediction_df)

        report = classification_report(y_test, y_pred, output_dict=True)
        classification_reports[ticker] = report
        
        # Store predictions in the results dictionary
        model_results[ticker] = prediction_df
    
    return model_results, classification_reports

# LSTM model
def load_lstm_model(ticker, model_path=None):

    if ticker == 'AAPL':
        model_path = 'LSTM_model.keras'
    elif ticker == 'AMZN':
        model_path = 'AMZN_LSTM_model.keras'
    else:
        raise ValueError(f"No pre-trained model available for {ticker}")
    # Load the model
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading LSTM model: {e}")
        return None, None, None

def prepare_lstm_test_data(ticker, start_date='2020-01-01', end_date=datetime.now()):

    # Fetch stock data
    stock_data = get_stock_data(ticker)
    print("THis is the stock data:", stock_data)
    
    # Extract close prices
    close_prices = stock_data['Adj Close'].values.reshape(-1, 1)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(close_prices)
    
    # Create sequences
    def create_test_sequences(data, seq_length):
        X_test = []
        for i in range(len(data) - seq_length):
            X_test.append(data[i:i+seq_length])
        return np.array(X_test)
    
    # Create sequences with 100 days lookback
    X_test = create_test_sequences(scaled_prices, 100)
    
    return X_test, scaler, close_prices

def lstm_predict(ticker, model_path):

    # Load the model
    model = load_lstm_model(ticker, model_path)
    
    if model is None:
        return None, None, None
    
    # Prepare test data
    X_test, scaler, original_prices = prepare_lstm_test_data(ticker)
    
    # Make predictions
    y_pred_scaled = model.predict(X_test)
    
    # Inverse transform to get actual prices
    y_pred = scaler.inverse_transform(y_pred_scaled)
    
    # Prepare actual prices for comparison (excluding the first 100 days)
    actual_prices = original_prices[100:]
    
    return actual_prices, y_pred.flatten(), scaler

def calculate_lstm_metrics(actual, predicted):

    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    
    return {
        'Mean Absolute Error': mae,
        'Mean Squared Error': mse,
        'Root Mean Squared Error': rmse
    }
def predict_next_7_days(ticker, model_path):
    # Load the LSTM model
    model = load_lstm_model(ticker, model_path)
    if model is None:
        return None, None
    
    # Fetch and preprocess data
    stock_data = get_stock_data(ticker)
    close_prices = stock_data['Adj Close'].values.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(close_prices)
    
    # Use the last 100 days as the initial input sequence
    input_sequence = scaled_prices[-100:].reshape(1, 100, 1)
    predictions = []
    
    # Generate predictions for the next 7 days
    for _ in range(7):
        # Predict the next day
        next_day_scaled = model.predict(input_sequence)
        next_day_price = scaler.inverse_transform(next_day_scaled)  # Scale back to original prices
        predictions.append(next_day_price[0, 0])
        
        # Update the input sequence
        next_day_scaled = next_day_scaled.reshape(1, 1, 1)
        input_sequence = np.append(input_sequence[:, 1:, :], next_day_scaled, axis=1)
    
    # Return actual prices and 7-day predictions
    return close_prices.flatten(), predictions


def train_linear_regression_and_predict(stock_data):
    predictions = {}
    actuals = {}
    y_test_preds = {}
    
    for ticker, data in stock_data.items():
        try:
            # Prepare features and target
            X = data[['compound', 'Open', 'High', 'Low', 'Volume']]
            y = data['Adj Close']
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            
            # Train Linear Regression
            linear_model = LinearRegression()
            linear_model.fit(X_train, y_train)

            # Predict for the test set
            y_pred = linear_model.predict(X_test)
            
            # Predict for the next 7 days using the latest available data
            latest_features = X.tail(7)  # Use the last 7 rows of features
            if len(latest_features) < 7:
                print(f"Not enough data for {ticker} to predict next 7 days.")
                continue
            
            next_7_days_pred = linear_model.predict(latest_features)
            predictions[ticker] = next_7_days_pred
            y_test_preds[ticker] = y_pred
            actuals[ticker] = y_test.values
            
            print(f"\nLinear Regression Predictions for {ticker}: {next_7_days_pred}")
        
        except Exception as e:
            print(f"Error training Linear Regression for {ticker}: {e}")
            continue
    
    return actuals, y_test_preds, predictions

def combine_predictions(linear_preds, lstm_preds, weight_lstm=0.7):
    combined_preds = {}
    
    for ticker in linear_preds.keys():
        if ticker in lstm_preds:
            # Weighted combination of Linear Regression and LSTM predictions
            combined = (
                weight_lstm * np.array(lstm_preds[ticker]) + 
                (1 - weight_lstm) * np.array(linear_preds[ticker])
            )
            combined_preds[ticker] = combined
            print(f"\nCombined Predictions for {ticker}: {combined}")
        else:
            print(f"No LSTM predictions for {ticker}. Using only Linear Regression predictions.")
            combined_preds[ticker] = linear_preds[ticker]
    
    return combined_preds

def lstm_predict_next_7_days(ticker, model_path):
    try:
        # Load the LSTM model
        model = load_lstm_model(ticker, model_path)
        if model is None:
            return None
        
        # Fetch and preprocess data
        stock_data = yf.download(ticker, start='2020-01-01', end=datetime.now())
        close_prices = stock_data['Adj Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_prices = scaler.fit_transform(close_prices)        
        input_sequence = scaled_prices[-100:].reshape(1, 100, 1)
        predictions = []
        
        for _ in range(7):
            next_day_scaled = model.predict(input_sequence)
            next_day_price = scaler.inverse_transform(next_day_scaled)
            predictions.append(next_day_price[0, 0])            
            next_day_scaled = next_day_scaled.reshape(1, 1, 1)
            input_sequence = np.append(input_sequence[:, 1:, :], next_day_scaled, axis=1)
        
        print(f"\nLSTM Predictions for {ticker}: {predictions}")
        return predictions
    
    except Exception as e:
        print(f"Error in LSTM prediction for {ticker}: {e}")
        return None
