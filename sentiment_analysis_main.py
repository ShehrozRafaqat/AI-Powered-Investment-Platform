from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Finviz URL and tickers
finviz_url = 'https://finviz.com/quote.ashx?t='

tickers = ['AMZN', 'AMD', 'GOOG']

# Scrape data
news_tables = {}
for ticker in tickers:
    url = finviz_url + ticker.upper()  # Convert to uppercase
    req = Request(url=url, headers={'user-agent': 'my-app'})
    try:
        response = urlopen(req)
        html = BeautifulSoup(response, 'html.parser')
        news_table = html.find(id='news-table')
        news_tables[ticker] = news_table
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        continue

# Parse data
parsed_data = []
for ticker, news_table in news_tables.items():
    for row in news_table.findAll('tr'):
        title = row.a.text
        date_data = row.td.text.strip().split(' ')
        if len(date_data) == 1:
            time = date_data[0]
        else:
            date = date_data[0]
            time = date_data[1]
        parsed_data.append([ticker, date, time, title])

df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])

# Adjust "Today" and "Yesterday"
today = datetime.now().date()
df['date'] = df['date'].apply(
    lambda x: today if x == "Today" else (today - timedelta(days=1) if x == "Yesterday" else x)
)

# Convert to datetime
df['date'] = pd.to_datetime(df['date'])

# Sentiment Analysis
vader = SentimentIntensityAnalyzer()
df['compound'] = df['title'].apply(lambda title: vader.polarity_scores(title)['compound'])

# Group by ticker and date, calculating the mean of the "compound" column
mean_df = df.groupby(['ticker', 'date'])['compound'].mean().unstack()

# Transpose the DataFrame so that dates are the index and tickers are columns
mean_df = mean_df.transpose()

# Plot the results
plt.figure(figsize=(10, 8))
mean_df.plot(kind='bar', stacked=False) 
plt.title('Average Sentiment Scores by Date and Ticker')
plt.ylabel('Compound Sentiment Score')
plt.xlabel('Date')
plt.legend(title='Ticker')
plt.tight_layout()
plt.show()

print(mean_df)
