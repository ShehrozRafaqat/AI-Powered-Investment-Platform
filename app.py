from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_file
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
import plotly
import plotly.graph_objs as go
import types
from itertools import chain
import requests

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
            print(f"Email: {email}, Password: {password}")
            print(f"User fetched from DB: {user}")
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

@app.route('/chatbot', methods=['POST'])
def chatbot():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized. Please log in.'}), 401

    user_query = request.json.get('query')

    # Replace with your Kaggle notebook's API endpoint
    KAGGLE_NOTEBOOK_API_URL = "https://0d71-35-194-9-32.ngrok-free.app/api"

    try:
        # Send the user query to the Kaggle notebook
        response = requests.post(
            KAGGLE_NOTEBOOK_API_URL,
            json={'query': user_query},
            timeout=10  # Add a timeout
        )

        # Check if the Kaggle notebook responded successfully
        if response.status_code == 200:
            chatbot_response = response.json().get('response')
            return jsonify({'response': chatbot_response})
        else:
            # Get more detailed error information
            error_details = response.text
            print(f"Kaggle API error: Status {response.status_code}, Details: {error_details}")
            return jsonify({
                'response': f'GPU unavailable. Status code: {response.status_code}. Try again later.'
            }), 503

    except requests.exceptions.ConnectionError:
        print("Connection error to Kaggle notebook")
        return jsonify({'response': 'Unable to connect to the chatbot. The service might be down.'}), 503
    except requests.exceptions.Timeout:
        print("Timeout connecting to Kaggle notebook")
        return jsonify({'response': 'The chatbot is taking too long to respond. Try again later.'}), 504
    except requests.exceptions.RequestException as e:
        print(f"Request exception: {e}")
        return jsonify({'response': f'Error connecting to the chatbot: {str(e)}. Try again later.'}), 500
        
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
    print("Portfolio Analysis")
    try:
        # Parse and validate input data
        data = request.json
        print(f"Received data: {data}")
        
        tickers = data['tickers']
        weights = data['weights']
        if isinstance(weights[0], list):
            weights = list(chain.from_iterable(weights))
        weights = [float(w) for w in weights]
        currency = data.get('currency', 'USD')
        rebalancing_period = data.get('rebalancing_period', 'month')

        # Create portfolio
        portfolio = ok.Portfolio(
            assets=tickers,
            weights=weights,
            ccy=currency,
            rebalancing_period=rebalancing_period
        )
        asset_list = ok.AssetList(tickers, ccy=currency)

        # Get wealth indexes data for individual assets
        wealth_indexes = asset_list.wealth_indexes
        
        # Prepare wealth indexes data for Plotly
        wealth_plot_data = {
            'dates': wealth_indexes.index.strftime('%Y-%m-%d').tolist(),
            'series': {}
        }
        
        # Add data for each asset
        for column in wealth_indexes.columns:
            wealth_plot_data['series'][column] = [
                float(val) for val in wealth_indexes[column].values
            ]
        
        # Add portfolio wealth index
        portfolio_wealth = portfolio.wealth_index
        wealth_plot_data['series']['Portfolio'] = [
            float(val) for val in portfolio_wealth.values.flatten()
        ]

        # Calculate metrics with proper series handling
        # For risk_annual, get the last value from the series
        try:
            annual_risk = portfolio.risk_annual.iloc[-1]  # Get the last value from the series
        except Exception as e:
            print(f"Error getting risk_annual: {str(e)}")
            annual_risk = 0.0

        # Get CAGR and Sharpe ratio with error handling
        try:
            cagr = portfolio.get_cagr()
            if isinstance(cagr, (pd.Series, pd.DataFrame)):
                cagr = cagr.iloc[-1]
        except Exception as e:
            print(f"Error getting CAGR: {str(e)}")
            cagr = 0.0

        try:
            sharpe_ratio = portfolio.get_sharpe_ratio()
            if isinstance(sharpe_ratio, (pd.Series, pd.DataFrame)):
                sharpe_ratio = sharpe_ratio.iloc[-1]
        except Exception as e:
            print(f"Error getting Sharpe ratio: {str(e)}")
            sharpe_ratio = 0.0

        # Get diversification ratio with error handling
        try:
            div_ratio = float(portfolio.diversification_ratio)
        except Exception as e:
            print(f"Error getting diversification ratio: {str(e)}")
            div_ratio = 0.0

        # Prepare metrics dictionary with proper rounding
        metrics = {
            'annual_risk': np.round(float(annual_risk) * 100, 2),
            'sharpe_ratio': np.round(float(sharpe_ratio), 2),
            'cagr': np.round(float(cagr) * 100, 2),
            'diversification_ratio': np.round(float(div_ratio), 2)
        }

        # Get historical data with proper error handling
        try:
            historical_data = portfolio.wealth_index
            performance_data = {
                'dates': historical_data.index.strftime('%Y-%m-%d').tolist(),
                'values': [np.round(float(v), 2) for v in historical_data.values.flatten()]
            }
        except Exception as e:
            print(f"Error processing historical data: {str(e)}")
            performance_data = {'dates': [], 'values': []}

        # Format weights data
        weights_data = {
            'assets': tickers,
            'weights': [np.round(float(w) * 100, 2) for w in weights]
        }

        # Convert correlation matrix with error handling
        try:
            corr_matrix = portfolio.assets_ror.corr()
            correlation = {}
            for column in corr_matrix.columns:
                correlation[column] = {
                    str(index): float(value) 
                    for index, value in corr_matrix[column].items()
                }
        except Exception as e:
            print(f"Error processing correlation matrix: {str(e)}")
            correlation = {}

        # Log successful calculations
        print("Successfully calculated portfolio metrics:")
        print(f"Risk Annual: {annual_risk}")
        print(f"CAGR: {cagr}")
        print(f"Sharpe Ratio: {sharpe_ratio}")

        return jsonify({
            'metrics': metrics,
            'performance': performance_data,
            'weights': weights_data,
            'correlation': correlation,
            'wealth_plot_data': wealth_plot_data 
        })

    except Exception as e:
        print(f"Error in portfolio analysis: {str(e)}")
        return jsonify({
            'error': str(e),
            'error_type': type(e).__name__,
            'error_location': 'main_function'
        }), 400   

@app.route('/efficient_frontier', methods=['GET'])
def efficient_frontier_page():
    # Render the HTML page for efficient frontier analysis
    return render_template('efficient_frontier.html')

@app.route('/efficient_frontier', methods=['POST'])
def efficient_frontier_analysis():
    try:
        data = request.json
        tickers = data.get('tickers')
        currency = data.get('currency', 'USD')
        rf_return = data.get('rf_return', 0.05)  # Risk-free rate (default 5%)
        
        # Validate input
        if not tickers or len(tickers) < 2:
            return jsonify({'error': 'At least two tickers are required.'}), 400
        if currency not in ['USD', 'EUR', 'GBP']:
            return jsonify({'error': 'Invalid currency selected.'}), 400
        
        # Create EfficientFrontier object
        try:
            ef = ok.EfficientFrontier(assets=tickers, ccy=currency, full_frontier=True)
        except Exception as e:
            print(f"Error creating EfficientFrontier: {e}")
            return jsonify({'error': 'Failed to create EfficientFrontier.'}), 400
        
        # Generate Efficient Frontier points
        try:
            ef_points = ef.ef_points
            mc_results = ef.get_monte_carlo(n=1000, kind='cagr')
            asset_returns = ef.assets_ror
        except Exception as e:
            print(f"Error generating EF points: {e}")
            return jsonify({'error': 'Failed to generate Efficient Frontier points.'}), 400
        
        # Get individual asset data (annualized risk and return)
        asset_data = []
        for asset in tickers:
            asset_ror = asset_returns[asset]
            asset_risk = asset_ror.std() * (12 ** 0.5)  # Annualized risk
            asset_return = asset_ror.mean() * 12  # Annualized return
            asset_data.append({
                'ticker': asset,
                'risk': asset_risk,
                'return': asset_return
            })
        
        # Calculate CML data
        try:
            tangency_portfolio = ef.get_tangency_portfolio(rf_return=rf_return, rate_of_return="cagr")
            tangency_portfolio['Weights'] = tangency_portfolio['Weights'].tolist()
            cml_risks = [0, tangency_portfolio['Risk']]  # Risk starts at 0 (risk-free asset)
            cml_returns = [rf_return, tangency_portfolio['Rate_of_return']]  # Return starts at rf_return
        except Exception as e:
            print(f"Error calculating CML: {e}")
            tangency_portfolio = None
            cml_risks = []
            cml_returns = []

        # Generate Transition Map data
        try:
            # Helper function to extract data from the Axes object
            def extract_transition_map_data(ax):
                data = {}
                for line in ax.get_lines():  # Get all lines plotted on the Axes
                    label = line.get_label()  # Asset ticker (label)
                    x_data = line.get_xdata().tolist()  # X-axis data (Risk or CAGR)
                    y_data = line.get_ydata().tolist()  # Y-axis data (Weights)
                    data[label] = {"x": x_data, "y": y_data}
                return data

            # Plot Transition Map for Risk
            ax_risk = ef.plot_transition_map(x_axe='risk')
            transition_map_risk = extract_transition_map_data(ax_risk)

            # Plot Transition Map for CAGR
            ax_cagr = ef.plot_transition_map(x_axe='cagr')
            transition_map_cagr = extract_transition_map_data(ax_cagr)

        except Exception as e:
            print(f"Error generating Transition Map: {e}")
            transition_map_risk = {}
            transition_map_cagr = {}
        
        # Prepare output data
        frontier_data = {
            'risks': list(ef_points['Risk']),
            'returns': list(ef_points['CAGR']),
            'mc_risks': list(mc_results['Risk']),
            'mc_returns': list(mc_results['CAGR']),
            'assets': asset_data,
            'cml_risks': cml_risks,
            'cml_returns': cml_returns,
            'tangency_portfolio': tangency_portfolio,
            'transition_map_risk': transition_map_risk,
            'transition_map_cagr': transition_map_cagr,
        }
        
        return jsonify({'frontier': frontier_data})
    
    except Exception as e:
        print(f"Error in efficient frontier analysis: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/compare_assets', methods=['GET'])
def compare_assets_page():
    return render_template('compare_assests.html')

@app.route('/compare_assets', methods=['POST'])
def compare_assets():
    try:
        data = request.json
        tickers = data.get('tickers')
        metric = data.get('metric', 'returns')
        currency = data.get('currency', 'USD')
        
        # Validate input
        if not tickers or len(tickers) < 1:
            return jsonify({'error': 'At least one ticker is required.'}), 400
        
        # Create AssetList object - WITHOUT specifying dates at first
        try:
            asset_list = ok.AssetList(
                assets=tickers, 
                ccy=currency
            )
            
            print(f"Asset list using dates: first_date={asset_list.first_date}, last_date={asset_list.last_date}")
            
        except Exception as e:
            print(f"Error creating AssetList: {e}")
            return jsonify({'error': f'Failed to create AssetList: {str(e)}'}), 400
        
        try:
            # Get asset names for better labeling
            asset_names = asset_list.names
            
            # Dictionary to store time series for each asset
            series_data = {}
            dates = []
            
            if metric == 'returns':
                # For returns, use wealth_indexes which shows cumulative returns over time
                wealth_indexes = asset_list.wealth_indexes
                
                # Convert DataFrame to the format needed for plotting
                dates = wealth_indexes.index.strftime('%Y-%m-%d').tolist()
                
                # For each asset, get its wealth index time series
                for ticker in tickers:
                    if ticker in wealth_indexes.columns:
                        display_name = asset_names.get(ticker, ticker)
                        series_data[display_name] = wealth_indexes[ticker].tolist()
                
            elif metric == 'volatility':
                # For volatility, use risk_annual which gives expanding window risk
                risk_series = asset_list.risk_annual
                
                # Convert DataFrame to the format needed for plotting
                dates = risk_series.index.strftime('%Y-%m-%d').tolist()
                
                # For each asset, get its risk time series
                for ticker in tickers:
                    if ticker in risk_series.columns:
                        display_name = asset_names.get(ticker, ticker)
                        series_data[display_name] = risk_series[ticker].tolist()
                
            elif metric == 'sharpe_ratio':
                # For Sharpe ratio, we need to calculate it for different periods
                # We'll use rolling windows to show how Sharpe ratio evolves
                # First get rolling returns and risk for a 12-month window
                rolling_returns = asset_list.get_rolling_cagr(window=12)
                rolling_risk = asset_list.get_rolling_risk_annual(window=12)
                
                # Calculate Sharpe ratio manually for each period
                sharpe_series = (rolling_returns - 0) / rolling_risk  # Using 0 as risk-free rate
                
                # Convert DataFrame to the format needed for plotting
                dates = sharpe_series.index.strftime('%Y-%m-%d').tolist()
                
                # For each asset, get its Sharpe ratio time series
                for ticker in tickers:
                    if ticker in sharpe_series.columns:
                        display_name = asset_names.get(ticker, ticker)
                        series_data[display_name] = sharpe_series[ticker].tolist()
            
            else:
                return jsonify({'error': f'Unsupported metric: {metric}'}), 400
            
            if not series_data:
                return jsonify({'error': 'No valid data found for the selected tickers'}), 400
                
            # Return the time series data
            return jsonify({
                'dates': dates,
                'series': series_data,
                'metric': metric,
                'currency': currency
            })
            
        except Exception as e:
            print(f"Error in metric calculation: {e}")
            return jsonify({'error': f'Failed to calculate {metric}: {str(e)}'}), 400
            
    except Exception as e:
        print(f"Error in asset comparison: {e}")
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)