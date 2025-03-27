import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import os
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt

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

def tsla_lstm_predict(ticker, model_path):
    # Load the trained LSTM model
    model = tf.keras.models.load_model(model_path)
    if model is None:
        return None, None, None

    # Fetch the latest 200 days of stock data
    stock_data = yf.download(ticker, period="200d", interval="1d")
    
    if stock_data.empty:
        print(f"Error: No data found for {ticker}.")
        return None, None, None
    
    # Extract closing prices
    close_prices = stock_data['Close'].values.reshape(-1, 1)

    # Initialize and fit the scaler
    scaler = joblib.load('scaler.save')  
    scaled_close = scaler.transform(close_prices)

    # Create sequences for prediction
    x_test, y_test = [], []
    time_steps = 100  # Last 100 days as input
    
    # We need to leave at least 7 days at the end for prediction targets
    for i in range(time_steps, len(scaled_close) - 7):
        x_test.append(scaled_close[i-time_steps:i, 0])  # Past 100 days
        y_test.append(scaled_close[i:i+7, 0])  # Next 7 days
    
    x_test, y_test = np.array(x_test), np.array(y_test)
    
    # Reshape X to be [samples, time steps, features]
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    # Make predictions
    y_pred = model.predict(x_test)
    
    # Reshape predictions and actual values for inverse scaling
    # Need to ensure the shapes match the original data shape
    y_pred_reshaped = y_pred.reshape(y_pred.shape[0], y_pred.shape[1])
    y_test_reshaped = y_test.reshape(y_test.shape[0], y_test.shape[1])
    
    # Inverse transform to get actual stock prices
    y_pred_actual = scaler.inverse_transform(y_pred_reshaped)
    y_test_actual = scaler.inverse_transform(y_test_reshaped)
    
    # To match the expected format in your code, we'll return the first day's prediction
    # and the corresponding actual price (you can modify this if needed)
    predicted_prices = y_pred_actual[:, 0]  # First day of each prediction
    actual_prices = y_test_actual[:, 0]  # First day of each actual
    
    return predicted_prices, actual_prices, scaler

def predict_next_7_days_tsla(ticker, model_path):
    SEQ_LENGTH = 60  # Must match training
    
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load('scaler.save')
    
    # Get minimum required data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=SEQ_LENGTH + 60)  # 30-day buffer
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    # Validate data length
    if len(stock_data) < SEQ_LENGTH:
        print(f"Need at least {SEQ_LENGTH} days of data")
        return None, None
    
    # Preprocess
    close_prices = stock_data['Close'].values.reshape(-1, 1)
    scaled_data = scaler.transform(close_prices)
    
    # Create proper input sequence
    input_sequence = scaled_data[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, 1)
    print(f"Scaled Input for Prediction:", input_sequence.flatten()[-5:])
    print("Scaler data_min_:", scaler.data_min_)
    print("Scaler data_max_:", scaler.data_max_) # Print last few inputs

    
    # Predict
    prediction = model.predict(input_sequence)
    next_7_days = scaler.inverse_transform(prediction.reshape(7, 1)).flatten()
    
    # Post-processing
    latest_price = close_prices[-1][0]
    print(f"Latest price: {latest_price:.2f}")
    print(f"Raw predictions: {next_7_days.round(2)}")
    
    
    return close_prices[-30:].flatten().tolist(), next_7_days.tolist()

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
    
# Function to get stock data using yfinance
def get_stock_data(ticker, period='1y'):
    stock_data = yf.download(ticker, period=period)
    return stock_data

# Function to create dataset with lookback days
def create_dataset(dataset, lookback=7):
    X, y = [], []
    for i in range(lookback, len(dataset)):
        X.append(dataset[i-lookback:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

# Function to build and train LSTM model
def build_lstm_model(X_train, y_train, epochs=25, batch_size=32):
    # Initialize the model
    model = Sequential()
    
    # Add LSTM layers with dropout
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.1))
    
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.1))
    
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.1))
    
    model.add(LSTM(units=50))
    model.add(Dropout(0.1))
    
    # Add output layer
    model.add(Dense(units=1))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    
    return model, history

# Function to evaluate the model
def evaluate_model(model, X_test, y_test_actual, scaler):
    # Make predictions
    predictions = model.predict(X_test)
    
    # Inverse transform to get actual prices
    predictions = scaler.inverse_transform(predictions)
    
    # Calculate RMSE
    rmse = math.sqrt(mean_squared_error(y_test_actual, predictions))
    
    return predictions, rmse

# Function to forecast next 7 days
def forecast_next_7_days(model, last_sequence, scaler):
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    # Predict for next 7 days
    for _ in range(7):
        # Reshape for prediction
        current_sequence_reshaped = np.reshape(current_sequence, (1, current_sequence.shape[0], 1))
        
        # Predict next day
        next_day_prediction = model.predict(current_sequence_reshaped)
        
        # Append prediction to list
        future_predictions.append(next_day_prediction[0, 0])
        
        # Update sequence for next prediction (remove first element and add prediction)
        current_sequence = np.append(current_sequence[1:], next_day_prediction[0, 0])
    
    # Convert predictions to original scale
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_predictions)
    
    return future_predictions

# Main function to run the entire process
def run_lstm_stock_prediction(ticker, lookback=7, test_size=0.2, epochs=25, batch_size=32):
    # Get stock data
    print(f"Downloading stock data for {ticker}...")
    df = get_stock_data(ticker)
    
    # Prepare data
    close_prices = df[['Close']].values

    # Split data into train and test sets
    train_size = int(len(close_prices) * (1 - test_size))
    train_data = close_prices[:train_size]
    test_data = close_prices[train_size:]
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data_scaled = scaler.fit_transform(train_data)
    
    # Create datasets with lookback days
    X_train, y_train = create_dataset(train_data_scaled, lookback)
    
    # Reshape input for LSTM [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    # Build and train the model
    print("Building and training LSTM model...")
    model, history = build_lstm_model(X_train, y_train, epochs, batch_size)
    
    # Prepare test data
    dataset_total = np.concatenate((train_data, test_data), axis=0)
    inputs = dataset_total[len(dataset_total) - len(test_data) - lookback:]
    inputs_scaled = scaler.transform(inputs)
    
    # Create test dataset
    X_test = []
    for i in range(lookback, len(inputs_scaled)):
        X_test.append(inputs_scaled[i-lookback:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    # Evaluate model
    print("Evaluating model...")
    predictions, rmse = evaluate_model(model, X_test, test_data, scaler)
    print(f"Actual prices: {test_data.flatten()}")
    print(f"Predicted prices: {predictions.flatten()}")
    
    # Forecast next 7 days
    print("Forecasting next 7 days...")
    last_sequence = inputs_scaled[-lookback:].reshape(-1)
    future_predictions = forecast_next_7_days(model, last_sequence, scaler)
    
    # Create dates for the next 7 days (excluding weekends)
    last_date = df.index[-1]
    future_dates = []
    days_added = 0
    current_date = last_date
    
    while days_added < 7:
        current_date += timedelta(days=1)
        if current_date.weekday() < 5:  # Monday to Friday
            future_dates.append(current_date)
            days_added += 1
    
    print("\n--- 7-Day Price Forecast ---")
    for i, date in enumerate(future_dates):
        print(f"{date.strftime('%Y-%m-%d')}: ${future_predictions[i][0]:.2f}")
    
    # Return data instead of plotting
    return {
        'ticker': ticker,
        'rmse': rmse,
        'actual_prices': test_data,
        'predicted_prices': predictions,
        'future_predictions': future_predictions,
        'future_dates': future_dates,
        'historical_dates': df.index[-30:] if len(close_prices) >= 30 else df.index,
        'historical_prices': close_prices[-30:] if len(close_prices) >= 30 else close_prices,
        'training_loss': history.history['loss']
    }