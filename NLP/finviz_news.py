import finvizfinance.quote as ff
import pandas as pd


def fetch_finviz_news(symbol):
    """
    Fetch news articles for a given stock symbol from Finviz.
    """
    fnews = ff.finvizfinance(symbol)
    news_data = fnews.ticker_news()
    try:
        return news_data
    except Exception as e:
        print(f"Error fetching news for {symbol}: {e}")
        return pd.DataFrame()


def main():
    symbols = ["AAPL", "MSFT", "GOOGL"]
    for symbol in symbols:
        print(f"Fetching news for {symbol}")
        news_df = fetch_finviz_news(symbol)
        if not news_df.empty:
            csv_path = f"/Users/lodoloro/programs/MQS/MQSMaster/NLP/articles/finviz_news_{symbol}.csv"
            news_df.to_csv(csv_path, index=False)
            print(f"Saved news articles to {csv_path}")
        else:
            print(f"No news articles found for {symbol}")


if __name__ == "__main__":
    main()
