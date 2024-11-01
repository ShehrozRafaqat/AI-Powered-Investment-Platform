from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_bcrypt import Bcrypt
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from sentiment_analysis_logic import fetch_news, process_news 
from stock_prediction import aggregate_sentiment,predict_next_7_days, prepare_data,train_model, lstm_predict, calculate_lstm_metrics, train_linear_regression_and_predict, lstm_predict_next_7_days, combine_predictions
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import json
import seaborn as sns
import numpy as np
import okama as ok


app = Flask(__name__)
app.secret_key = 'your_secret_key'
bcrypt = Bcrypt(app)

DATABASE = 'investment_platform.db'

# Database connection function
def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

# Home route
@app.route('/')
def home():
    if 'user_id' in session:
        return render_template('dashboard.html', username=session['username'])
    return render_template('index.html')

# Sign-up route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')

        try:
            conn = get_db_connection()
            conn.execute(
                'INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                (username, email, password)
            )
            conn.commit()
            conn.close()
            flash('Sign-up successful! Please log in.', 'success')
            return redirect('/login')
        except sqlite3.IntegrityError:
            flash('Username or email already exists.', 'danger')

    return render_template('signup.html')

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        conn.close()

        if user and bcrypt.check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Logged in successfully!', 'success')
            return redirect('/')
        else:
            flash('Invalid email or password.', 'danger')

    return render_template('login.html')

# Logout route
@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'success')
    return redirect('/')

# Investment profile route
@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        return redirect('/login')

    if request.method == 'POST':
        risk_tolerance = request.form['risk_tolerance']
        preferred_sectors = request.form['preferred_sectors']
        investment_goals = request.form['investment_goals']

        conn = get_db_connection()
        conn.execute(
            '''
            INSERT OR REPLACE INTO investment_profiles (user_id, risk_tolerance, preferred_sectors, investment_goals)
            VALUES (?, ?, ?, ?)
            ''',
            (session['user_id'], risk_tolerance, preferred_sectors, investment_goals)
        )
        conn.commit()
        conn.close()

        flash('Investment profile updated!', 'success')
        return redirect('/profile')

    conn = get_db_connection()
    profile = conn.execute('SELECT * FROM investment_profiles WHERE user_id = ?', (session['user_id'],)).fetchone()
    conn.close()

    return render_template('profile.html', profile=profile)

@app.route('/sentiment', methods=['GET', 'POST'])
def sentiment_analysis():
    if request.method == 'POST':
        tickers = request.form.getlist('tickers')
        
        if not tickers:
            return render_template('sentiment_analysis.html', 
                                 hide_loading=True,
                                 error='Please select at least one ticker to analyze.')
        
        try:
            # Your existing analysis code
            news_tables = fetch_news(tickers)
            df = process_news(news_tables)
            
            # Your plotting code
            mean_df = df.groupby(['ticker', 'date'])['compound'].mean().unstack()
            mean_df = mean_df.transpose()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            mean_df.plot(kind='bar', stacked=False, ax=ax)
            ax.set_title('Average Sentiment Scores by Date and Ticker')
            ax.set_ylabel('Compound Sentiment Score')
            ax.set_xlabel('Date')
            ax.legend(title='Ticker')
            
            img = io.BytesIO()
            plt.tight_layout()
            plt.savefig(img, format='png')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode('utf8')
            
            return render_template('sentiment_analysis.html',
                                 table=df.to_html(classes='table table-bordered table-striped'),
                                 plot_url=plot_url,
                                 hide_loading=True)
                                 
        except Exception as e:
            # Handle any errors and ensure loading overlay is hidden
            return render_template('sentiment_analysis.html',
                                 hide_loading=True,
                                 error=f'An error occurred during analysis: {str(e)}')
    
    return render_template('sentiment_analysis.html')

# Stock Prediction route
@app.route('/stock-prediction', methods=['GET', 'POST'])
def predict_stock():
    if request.method == 'POST':
        model_type = request.form.get('model_type')
        tickers = request.form.getlist('tickers')
        tick = request.form.get('tickers')
        
        if not tickers:
            return render_template('stock_prediction.html', error="Please select at least one ticker.")

        try:
            news_tables = fetch_news(tickers)
            df = process_news(news_tables)
            
            if df.empty:
                return render_template('stock_prediction.html', error="No news data found for the selected tickers.")

            stock_data = prepare_data(df, tickers)
            
            if not stock_data:
                return render_template('stock_prediction.html', error="Could not prepare stock data for prediction.")

            if model_type == "Logistic Regression":
                model_results, classification_reports = train_model(stock_data)
                
                plots = {}
                tables = {}
                for ticker, prediction_df in model_results.items():
                    # Format the DataFrame
                    if isinstance(prediction_df.index, pd.DatetimeIndex):
                        prediction_df.index = prediction_df.index.strftime('%Y-%m-%d')
                    
                    # Round numerical columns to 2 decimal places
                    for col in prediction_df.select_dtypes(include=['float64']).columns:
                        prediction_df[col] = prediction_df[col].round(2)
                    
                    # Generate HTML table
                    tables[ticker] = prediction_df.to_html(
                        classes='table table-bordered table-striped',
                        float_format=lambda x: '{:.2f}'.format(x) if pd.notnull(x) else '',
                        index=True
                    )
                    
                    # Create prediction vs actual plot
                    fig, ax = plt.subplots(figsize=(12, 6))
                    plt.plot(prediction_df.index, prediction_df['Actual'], 'b', label="Original Price")
                    plt.plot(prediction_df.index, prediction_df['Predicted'], 'r', label="Predicted Price")
                    plt.title(f"{ticker} Stock Price Prediction")
                    plt.xlabel('Date')
                    plt.ylabel('Price Movement (Up/Down)')
                    plt.xticks(rotation=45)
                    plt.legend()
                    
                    # Save plot to base64
                    img = io.BytesIO()
                    plt.savefig(img, format='png', bbox_inches='tight')
                    img.seek(0)
                    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
                    plt.close()
                    
                    # Create classification report visualization
                    report_df = pd.DataFrame(classification_reports[ticker]).transpose()
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.heatmap(report_df, annot=True, cmap='YlGnBu', fmt='.2f')
                    plt.title(f"{ticker} Classification Report")
                    
                    # Save classification report plot
                    img_report = io.BytesIO()
                    plt.savefig(img_report, format='png', bbox_inches='tight')
                    img_report.seek(0)
                    report_plot_url = base64.b64encode(img_report.getvalue()).decode('utf8')
                    plt.close()
                    
                    plots[ticker] = {
                        'prediction_plot': plot_url,
                        'report_plot': report_plot_url
                    }
            
                return render_template(
                    'stock_prediction.html',
                    plots=plots,
                    tables=tables,
                    model_type=model_type,
                    tickers=tickers
                )
            elif model_type == "LSTM":
                # Depending on the selected ticker, load the respective LSTM model
                if tick == "AAPL":
                    actual_prices, predicted_prices, scaler = lstm_predict(tick, "LSTM_model.keras")
                    historical_prices, next_7_days = predict_next_7_days(tick, "LSTM_model.keras")
                elif tick == "AMZN":
                    actual_prices, predicted_prices, scaler = lstm_predict(tick, "AMZN_LSTM_model.keras")
                    historical_prices, next_7_days = predict_next_7_days(tick, "AMZN_LSTM_model.keras")

                metrics = calculate_lstm_metrics(actual_prices, predicted_prices)
                metrics_df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
                metrics_table = metrics_df.to_html(classes="table table-bordered", index=False)               

                if actual_prices is not None and predicted_prices is not None:
                # Plot predictions
                    plt.figure(figsize=(12,6))
                    plt.plot(actual_prices, 'b', label="Actual Price")
                    plt.plot(predicted_prices, 'r', label="Predicted Price")
                    plt.title(f"{tick} Stock Price Prediction")
                    plt.xlabel('Time')
                    plt.ylabel('Price')
                    plt.legend()

                    img = io.BytesIO()
                    plt.savefig(img, format='png')
                    img.seek(0)
                    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
                    plt.close()

                    future_dates = [datetime.now() + timedelta(days=i) for i in range(1, 8)]
                    next_7_days_with_dates = list(zip(future_dates, next_7_days))
                    next_7_days_df = pd.DataFrame(next_7_days_with_dates, columns=['Date', 'Predicted Price'])
                    next_7_days_df['Date'] = next_7_days_df['Date'].dt.strftime('%Y-%m-%d')
                    next_7_days_table = next_7_days_df.to_html(classes='table table-bordered')

                    fig_next, ax_next = plt.subplots(figsize=(8, 6))
                    ax_next.plot(range(1, 8), next_7_days, marker='o', color="orange", label="Predicted Prices")
                    ax_next.set_title(f"{tick} Next 7 Days Prediction")
                    ax_next.set_xlabel("Days (1 = Tomorrow)")
                    ax_next.set_ylabel("Price")
                    ax_next.set_xticks(range(1, 8))
                    ax_next.legend()

                    img_next = io.BytesIO()
                    plt.savefig(img_next, format='png')
                    img_next.seek(0)
                    plt.close()
                    plot_next_url = base64.b64encode(img_next.getvalue()).decode('utf8')

                return render_template('stock_prediction.html',
                                    plot_url=plot_url,
                                    metrics_table=metrics_table,
                                    next_7_days_table=next_7_days_table,
                                    plot_next_url=plot_next_url,
                                    model_type=model_type,
                                    ticker=tick)
            
            elif model_type == "Linear Regression":
                if not tickers:
                    return render_template('stock_prediction.html', error="Please select at least one ticker.")

                # Fetch and process news data
                news_tables = fetch_news(tickers)
                df = process_news(news_tables)

                if df.empty:
                    return render_template('stock_prediction.html', error="No news data found for the selected tickers.")

                # Prepare stock data
                stock_data = prepare_data(df, tickers)
                if not stock_data:
                    return render_template('stock_prediction.html', error="Could not prepare stock data for prediction.")

                # Train Linear Regression and make predictions
                lr_actuals, lr_test_preds, linear_results = train_linear_regression_and_predict(stock_data)

                # Get LSTM predictions for next 7 days
                lstm_predictions = {}
                for ticker in tickers:
                    lstm_preds = lstm_predict_next_7_days(ticker, model_path=None)
                    if lstm_preds is not None:
                        lstm_predictions[ticker] = lstm_preds

                # Combine predictions
                final_predictions = combine_predictions(linear_results, lstm_predictions, weight_lstm=0.7)

                # Generate plots and data tables
                combined_plots = {}
                lstm_vs_combined_plots = {}
                actual_vs_predicted_lr_plots = {}
                lstm_actual_vs_predicted_plots = {}

                for ticker, preds in final_predictions.items():
                    plt.figure(figsize=(8, 6))
                    plt.plot(range(1, 8), preds, marker='o', color="orange", label="Predicted Prices")
                    plt.title(f"{ticker} Next 7 Days Prediction")
                    plt.xlabel("Days (1 = Tomorrow)")
                    plt.ylabel("Price")
                    plt.xticks(range(1, 8))
                    plt.legend()

                    img = io.BytesIO()
                    plt.savefig(img, format='png')
                    img.seek(0)
                    plot_combine_url = base64.b64encode(img.getvalue()).decode('utf8')
                    plt.close()

                l = []
                for ticker, preds in final_predictions.items():
                    for day, price in enumerate(preds, 1):
                        l.append([f"Day {day}", f"${price:.2f}"])  # Create a list with two elements per row
                l_df = pd.DataFrame(l, columns=['Days', 'Predicted Price'])
                l_table = l_df.to_html(classes='table table-bordered')

                if tick == "AAPL":
                    actual_prices, predicted_prices, scaler = lstm_predict(tick, "LSTM_model.keras")
                    historical_prices, lstm_next_7_days = predict_next_7_days(tick, "LSTM_model.keras")
                elif tick == "AMZN":
                    actual_prices, predicted_prices, scaler = lstm_predict(tick, "AMZN_LSTM_model.keras")
                    historical_prices, lstm_next_7_days = predict_next_7_days(tick, "AMZN_LSTM_model.keras")
                
                if actual_prices is not None and predicted_prices is not None:
                # Plot predictions
                    plt.figure(figsize=(12,6))
                    plt.plot(actual_prices, 'b', label="Actual Price")
                    plt.plot(predicted_prices, 'r', label="Predicted Price")
                    plt.title(f"{tick} Stock Price Prediction")
                    plt.xlabel('Time')
                    plt.ylabel('Price')
                    plt.legend()

                    img = io.BytesIO()
                    plt.savefig(img, format='png')
                    img.seek(0)
                    plot_lstm_url = base64.b64encode(img.getvalue()).decode('utf8')
                    plt.close()

                for ticker in lr_actuals.keys():
                    if ticker in lr_actuals and ticker in lr_test_preds:
                        actual_prices = lr_actuals[ticker]
                        predicted_prices = lr_test_preds[ticker]
                        # Plot the data
                        plt.figure(figsize=(12, 6))
                        plt.plot(actual_prices, 'b', label="Actual Price")
                        plt.plot(predicted_prices, 'r', label="Linear Regression Predicted Price")
                        plt.title(f"{ticker} Stock Price Prediction")
                        plt.xlabel('Time')
                        plt.ylabel('Price')
                        plt.legend() 

                        img = io.BytesIO()
                        plt.savefig(img, format='png')
                        img.seek(0)
                        plot_linear_url = base64.b64encode(img.getvalue()).decode('utf8')
                        plt.close()

                for ticker, preds in final_predictions.items():
                    plt.figure(figsize=(8, 6))
                    plt.plot(range(1, 8), preds, marker='o', color="orange", label="Combined Predicted Prices")
                    plt.plot(range(1, 8), lstm_next_7_days, marker='o', color="blue", label="LSTM Predicted Prices")
                    plt.title(f"{ticker} Next 7 Days Prediction")
                    plt.xlabel("Days (1 = Tomorrow)")
                    plt.ylabel("Price")
                    plt.xticks(range(1, 8))
                    plt.legend() 

                    img = io.BytesIO()
                    plt.savefig(img, format='png')
                    img.seek(0)
                    plot_lstm_v_combine_url = base64.b64encode(img.getvalue()).decode('utf8')
                    plt.close()                          

                return render_template(
                    'stock_prediction.html',
                    ticker=tick,
                    plot_combine_url=plot_combine_url,
                    l_table=l_table,
                    plot_lstm_url=plot_lstm_url,
                    plot_linear_url=plot_linear_url,
                    plot_lstm_v_combine_url=plot_lstm_v_combine_url,
                    model_type=model_type,
                    )

                
        except Exception as e:
            return render_template('stock_prediction.html', error=str(e))

    return render_template('stock_prediction.html')

# Portfolio Analysis route
@app.route('/portfolio_analysis', methods=['GET'])
def portfolio_page():
    # Render the HTML page for portfolio analysis
    return render_template('porfolio_analysis.html')

@app.route('/portfolio_analysis', methods=['POST'])
def portfolio_analysis():
    try:
        # Get portfolio data from frontend
        data = request.json
        tickers = data['tickers']  # e.g., ['AAPL.US', 'MSFT.US', 'TSLA.US']
        weights = data['weights']  # e.g., [0.4, 0.4, 0.2]
        currency = data.get('currency', 'USD')  # Default to USD
        rebalancing_period = data.get('rebalancing_period', 'month')  # Optional, default to 'month'
        
        # Validate rebalancing_period
        allowed_periods = ['none', 'year', 'half-year', 'quarter', 'month']
        if rebalancing_period not in allowed_periods:
            raise ValueError(f"Invalid rebalancing period. Must be one of {allowed_periods}")
        
        # Create Portfolio instance
        portfolio = ok.Portfolio(assets=tickers, weights=weights, ccy=currency, rebalancing_period=rebalancing_period)
        
        # Get portfolio metrics
        metrics = {
            'expected_return': portfolio.expected_return.mean() * 100,  # As percentage
            'risk': portfolio.risk.mean() * 100,  # As percentage
            'sharpe_ratio': portfolio.sharpe_ratio.mean(),
            'diversification_ratio': portfolio.diversification_ratio.mean(),
            'historical_cagr': portfolio.cagr.mean() * 100,  # Compounded Annual Growth Rate
        }
        
        # Generate Efficient Frontier (if available)
        efficient_frontier = portfolio.efficient_frontier  # Efficient frontier data
        
        return jsonify({'metrics': metrics, 'efficient_frontier': efficient_frontier.to_dict()})
    
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)