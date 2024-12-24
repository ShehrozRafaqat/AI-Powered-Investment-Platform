# Sentiment-analysis-and-stock-price-prediction-app
# Introduction
The Stock Sentiment & Prediction App is a Streamlit-based application designed for analyzing news sentiment and predicting stock prices. It integrates sentiment analysis with machine learning models to provide insights into stock price movements.
# Features
* Sentiment Analysis
  - Purpose: Analyzes news articles for selected stock tickers and determines their sentiment scores.
  - Inputs:Stock tickers (e.g., AMZN, AMD, FB).
  - Outputs:
    - Sentiment scores (positive, neutral, or negative) displayed in tabular form.
    - A bar chart showing average sentiment scores by date and ticker.
* Stock Prediction
  - Purpose: Predicts stock price movements and values based on sentiment analysis and historical data.
  - Models:
    - Logistic Regression: Predicts price movements (up/down).
    - Linear Regression: Predicts stock prices using sentiment features.
    - LSTM: Predicts stock prices based on historical data.
  - Outputs:
    - Actual vs. Predicted price plots for different models.
    - Predictions for the next 7 days (LSTM and combined models).
* Visualizations
  - Sentiment trends grouped by ticker and date.
  - Actual vs. predicted prices for each model.
  - Combined predictions for the next 7 days.
# Setup Instructions
## 1. Clone the repository ##
```
git clone https://github.com/Eihaab-cmyk/Sentiment-analysis-and-stock-price-prediction-app
```
## 2. Install Python dependencies ##
```
pip install -r requirements.txt
```
## 3. Run the app ##
```
streamlit run sentiment_analysis_app.py
```
## Built With
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-red?style=for-the-badge&logo=keras&logoColor=white)
![Numpy](https://img.shields.io/badge/Numpy-blue?style=for-the-badge&logo=numpy&logoColor=white)
![pandas](https://img.shields.io/badge/Pandas-green?style=for-the-badge&logo=pandas&logoColor=white)
# Usage Guide
* Sentiment Analysis:
  - Select stock tickers and view sentiment scores in tabular and chart formats.
* Stock Prediction:
  - Choose prediction models (Logistic Regression, Linear Regression, or LSTM).
  - Get predictions and visualizations for the next 7 days.
* Visualizations:
  - Explore interactive charts for sentiment trends and stock predictions.
# Evaluation Metrics
* Sentiment Analysis: Accuracy and correlation with market movement.
* Stock Prediction:
  - Logistic Regression: Accuracy, F1 Score.
  - Linear Regression and LSTM: Mean Squared Error (MSE), Mean Absolute Error (MAE).
* Future Enhancements
  - Real-time stock price integration.
  - Advanced sentiment analysis with transformer models.
  - Support for more stock exchanges.
  - Enhanced visualization features and interactive dashboards.
# Contributors
Developer: Muhammad Eihaab
