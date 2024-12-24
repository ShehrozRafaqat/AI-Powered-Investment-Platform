from datetime import datetime
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as t
from sentiment_analysis_logic import fetch_news, process_news
from stock_prediction import aggregate_sentiment,predict_next_7_days, prepare_data,train_model, lstm_predict, calculate_lstm_metrics, train_linear_regression_and_predict, lstm_predict_next_7_days, combine_predictions

# Streamlit UI
def main():
    st.sidebar.title("Stock Sentiment & Prediction App")
    page = st.sidebar.radio("Select a Page", ("Sentiment Analysis", "Stock Prediction"))
    
    if page == "Sentiment Analysis":
        st.title('Stock Sentiment Analysis')
        st.write('Select the tickers to analyze news sentiment.')
        # Multi-select for tickers
        tickers = st.multiselect(
            'Choose tickers',
            ['AMZN', 'AMD', 'FB', 'GOOGL', 'AAPL', 'TSLA'],
            default=['AMZN', 'AMD']
        )        
        if tickers:
            st.write(f"Analyzing sentiment for: {', '.join(tickers)}")
            news_tables = fetch_news(tickers)
            df = process_news(news_tables)

            st.subheader('Sentiment Data')
            st.write(df)
            
             # Plot the sentiment analysis chart
            st.subheader('Sentiment Analysis Chart')
            mean_df = df.groupby(['ticker', 'date'])['compound'].mean().unstack()
            mean_df = mean_df.transpose()
            plt.figure(figsize=(10, 8))
            mean_df.plot(kind='bar', stacked=False) 
            plt.title('Average Sentiment Scores by Date and Ticker')
            plt.ylabel('Compound Sentiment Score')
            plt.xlabel('Date')
            plt.legend(title='Ticker')
            plt.tight_layout()
            st.pyplot(plt)
        else:
            st.write("Please select at least one ticker to start.")

    elif page == "Stock Prediction":
        model_type = st.sidebar.selectbox(
        "Select Prediction Model", 
        ["Logistic Regression", "Linear Regression", "LSTM"])
        if model_type == "Logistic Regression":
            st.title("Stock Prediction Based on Sentiment")
            tickers = st.multiselect('Select Tickers for Prediction', ['AMZN', 'AMD', 'FB', 'GOOGL', 'AAPL', 'TSLA'])
            if tickers:
                try:
                    news_tables = fetch_news(tickers)
                    df = process_news(news_tables)
                    
                    # Check if we have data
                    if df.empty:
                        st.error("No news data found for the selected tickers.")
                        return
                    
                    stock_data = prepare_data(df, tickers)
                    
                    # Check if stock_data is empty
                    if not stock_data:
                        st.error("Could not prepare stock data for prediction.")
                        return
                    
                    model_results, classification_reports = train_model(stock_data)
                    
                    st.write("Model Results for Each Ticker (1) means price will go up and (0) means price will go down:")
                    for ticker, prediction_df in model_results.items():
                        st.subheader(f"Results for {ticker}:")
                        st.write(prediction_df)

                        if 'Actual' in prediction_df.columns and 'Predicted' in prediction_df.columns:
                            plt.figure(figsize=(14, 8))
                            plt.plot(prediction_df['Actual'], 'b', label = "Original Price")
                            plt.plot(prediction_df['Predicted'], 'r', label = "Predicted Price")
                            # Title and labels
                            plt.title(f"{ticker} Stock Price Prediction", fontsize=16)
                            plt.xlabel('Date', fontsize=14)
                            plt.ylabel('Price Movement (Up/Down)', fontsize=14)
                            plt.legend()
                            st.pyplot(plt)
                    # Display classification results as DataFrame
                    for ticker, report in classification_reports.items():
                        st.subheader(f"Classification Report for {ticker}:")
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df.style.highlight_max(axis=0))  
                except Exception as e:
                    st.error(f"An error occurred: {e}")

        elif model_type == "Linear Regression":
            st.title("Stock Prediction Based on Sentiment")
            tickers = st.multiselect('Select Tickers for Prediction', ['AMZN', 'AAPL'])
            if tickers:
                try:
                    news_tables = fetch_news(tickers)
                    df = process_news(news_tables)
                    
                    # Check if we have data
                    if df.empty:
                        st.error("No news data found for the selected tickers.")
                        return
                    
                    stock_data = prepare_data(df, tickers)
                    
                    # Check if stock_data is empty
                    if not stock_data:
                        st.error("Could not prepare stock data for prediction.")
                        return
                    
                    lr_actuals, lr_test_preds, linear_results = train_linear_regression_and_predict(stock_data)

                    # Get LSTM predictions for next 7 days
                    lstm_predictions = {}
                    for ticker in tickers:
                        print(f"\nMaking LSTM predictions for {ticker}...")
                        lstm_preds = lstm_predict_next_7_days(ticker, model_path=None)
                        if lstm_preds is not None:
                            lstm_predictions[ticker] = lstm_preds
                    
                    # combine predictions
                    final_predictions = combine_predictions(linear_results, lstm_predictions, weight_lstm=0.7)

                     # Sidebar description
                    st.sidebar.header("User Guide")
                    st.sidebar.write("""
                    - Predictions are for the **next 7 days**.
                    - Predictions are made for selected stock tickers.
                    - The displayed predictions are weighted combinations of:
                    - **Linear Regression** (based on sentiment + stock features).
                    - **LSTM** (based on historical stock prices).
                    """)
                     # Plot combined predictions for the next 7 days
                    st.subheader("Linear regression and LSTM combined next 7 days prediction plot")
                    for ticker, preds in final_predictions.items():
                        plt.figure(figsize=(8, 6))
                        plt.plot(range(1, 8), preds, marker='o', color="orange", label="Predicted Prices")
                        plt.title(f"{ticker} Next 7 Days Prediction")
                        plt.xlabel("Days (1 = Tomorrow)")
                        plt.ylabel("Price")
                        plt.xticks(range(1, 8))
                        plt.legend()
                        st.pyplot(plt)
                    # Display results for each stock ticker
                    st.header("Final Predictions")
                    for ticker, preds in final_predictions.items():
                        st.subheader(f"{ticker} - Next 7-Day Predictions")
                        for day, price in enumerate(preds, 1):
                            st.write(f"Day {day}: ${price:.2f}")
                    
                    if ticker == "AAPL":
                        lstm_actual_prices, lstm_predicted_prices, scaler = lstm_predict(ticker, "LSTM_model.keras")
                        LSTM_historical_prices, lstm_next_7_days = predict_next_7_days(ticker, "LSTM_model.keras")

                        
                    elif ticker == "AMZN":
                        lstm_actual_prices, lstm_predicted_prices, scaler = lstm_predict(ticker, "AMZN_LSTM_model.keras")
                        LSTM_historical_prices, lstm_next_7_days = predict_next_7_days(ticker, "AMZN_LSTM_model.keras")


                    st.header("Actual vs predicted price plot for LSTM")
                    if lstm_actual_prices is not None and lstm_predicted_prices is not None:
                        # Plot predictions
                        plt.figure(figsize=(12,6))
                        plt.plot(lstm_actual_prices, 'b', label="Actual Price")
                        plt.plot(lstm_predicted_prices, 'r', label="LSTM Predicted Price")
                        plt.title(f"{ticker} Stock Price Prediction")
                        plt.xlabel('Time')
                        plt.ylabel('Price')
                        plt.legend()
                        st.pyplot(plt)

                    st.header("Actual vs predicted price plot for linear regression")
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
                            st.pyplot(plt)

                    st.subheader("Linear regression and LSTM combined vs LSTM alone next 7 days prediction plot")
                    for ticker, preds in final_predictions.items():
                        # Plot predictions for each ticker
                        plt.figure(figsize=(8, 6))
                        plt.plot(range(1, 8), preds, marker='o', color="orange", label="Combined Predicted Prices")
                        plt.plot(range(1, 8), lstm_next_7_days, marker='o', color="blue", label="LSTM Predicted Prices")
                        plt.title(f"{ticker} Next 7 Days Prediction")
                        plt.xlabel("Days (1 = Tomorrow)")
                        plt.ylabel("Price")
                        plt.xticks(range(1, 8))
                        plt.legend()
                        st.pyplot(plt)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
        
        elif model_type == "LSTM":
            st.title("LSTM Stock Prediction")
        
            ticker = st.selectbox(
                "Select Stock for LSTM Prediction", 
                ["AAPL", "AMZN"]
            )
            
            if ticker == "AAPL":
                actual_prices, predicted_prices, scaler = lstm_predict(ticker, "LSTM_model.keras")
                historical_prices, next_7_days = predict_next_7_days(ticker, "LSTM_model.keras")

            elif ticker == "AMZN":
                actual_prices, predicted_prices, scaler = lstm_predict(ticker, "AMZN_LSTM_model.keras")
                historical_prices, next_7_days = predict_next_7_days(ticker, "AMZN_LSTM_model.keras")

            
            if actual_prices is not None and predicted_prices is not None:
                # Plot predictions
                plt.figure(figsize=(12,6))
                plt.plot(actual_prices, 'b', label="Actual Price")
                plt.plot(predicted_prices, 'r', label="Predicted Price")
                plt.title(f"{ticker} Stock Price Prediction")
                plt.xlabel('Time')
                plt.ylabel('Price')
                plt.legend()
                st.pyplot(plt)
                
                # Calculate and display metrics
                metrics = calculate_lstm_metrics(actual_prices, predicted_prices)
                st.subheader("Prediction Metrics")
                for metric, value in metrics.items():
                    st.write(f"{metric}: {value:.2f}")
                
                # Plot next 7 days predictions
                st.subheader("Next 7 Days Prediction")
                future_dates = [datetime.now() + pd.Timedelta(days=i) for i in range(1, 8)]
                plt.figure(figsize=(8, 6))
                plt.plot(range(1, 8), next_7_days, marker='o', color="orange", label="Predicted Prices")
                plt.title(f"{ticker} Next 7 Days Prediction")
                plt.xlabel("Days (1 = Tomorrow)")
                plt.ylabel("Price")
                plt.xticks(range(1, 8))
                plt.legend()
                st.pyplot(plt)

                 # Display next 7 days predictions
                st.subheader("Predicted Prices for the Next 7 Days")
                for date, price in zip(future_dates, next_7_days):
                    st.write(f"{date.strftime('%Y-%m-%d')}: ${price:.2f}")

if __name__ == '__main__':
    main()