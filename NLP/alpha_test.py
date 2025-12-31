import os
from datetime import datetime

import dotenv
import requests
from tqdm import tqdm

dotenv.load_dotenv()
ALPHA_KEY = os.getenv("ALPHA_KEY")


def scrape_alpha(
    ticker=["AAPL"],
    time_from="20251201T1200",
    time_to="20251231T1200",
    apikey=ALPHA_KEY,
):
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={','.join(ticker)}&time_from={time_from}&time_to={time_to}&apikey={apikey}"

    r = requests.get(url)
    news = r.json()
    for news in tqdm(
        news.get("feed", []),
        desc="Scraping Alpha Vantage News articles...",
        total=len(news.get("feed", [])),
    ):
        title = news.get("title", "N/A")
        summary = news.get("summary", "N/A")
        pub_date = news.get("time_published", "N/A")

        # Format the timestamp from API format (e.g., "20251231T120000Z") to readable format
        if pub_date != "N/A":
            try:
                pub_date = datetime.strptime(pub_date, "%Y%m%dT%H%M%S").strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            except ValueError as e:
                print(f"Error parsing date {pub_date}: {e}")
                
        url = news.get("url", "N/A")

        yield {
            "publishedDate": pub_date,
            "title": title,
            "content": summary,
            "site": url,
        }


def main():
    import pandas as pd

    row = scrape_alpha(
        ticker=["AAPL"],
        time_from="20251201T1200",
        time_to="20251231T1200",
        apikey=ALPHA_KEY,
    )
    path = os.path.dirname(__file__) + "/articles"
    new_df = pd.DataFrame(row)
    with open(f"{path}/AAPL_alpha_news.csv", "w", encoding="utf-8") as f:
        new_df.to_csv(f, index=False, date_format="%Y-%m-%d %H:%M:%S")
    return


if __name__ == "__main__":
    main()
