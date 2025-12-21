from email.mime import message
import re
from urllib import response


def scrape_article_content(symbol):
    """
    Scrape the full article content from the given URL using Selenium and BeautifulSoup.
    Returns the article text or an empty string if scraping fails.
    """

    url = f'https://finance.yahoo.com/quote/{symbol}/latest-news/'

    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By

        driver = webdriver.Chrome()
        driver.get(url)

        news_stream = driver.find_element(By.CSS_SELECTOR, 'div[data-testid="news-stream"]')
        text = news_stream.text
        # Get first story
        first_story = driver.find_element(By.CSS_SELECTOR, 'section[data-testid="storyitem"]')

        # Get story title within first item
        title = first_story.find_element(By.TAG_NAME, 'h3').text
        print(f"Title: {title}")
         # Get story link within first item
        print(f"Message received: {text}")

    except Exception as e:
        print(f"Session error: {e}. Cannot scrape article content.")
        return ""
    try:
        if driver:

            driver.quit()
            return text
    except Exception as e:
        print(f"Error scraping {url}: {e}")
    driver.quit()
    return 'none'

def main():
    content = scrape_article_content("AAPL")
    print("Content for AAPL:")  # Print first 500 characters
    print(content[:500])
    return

if __name__ == "__main__":
    main()