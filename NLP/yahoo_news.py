#!/usr/bin/env python3
"""
news_scraper.py
Scrape full article content from news URLs using Selenium and BeautifulSoup.
"""


# todo : implement error handling and content extraction logic
def scrape_article_content(symbol):
    """
    Scrape the full article content from the given URL using Selenium and BeautifulSoup.
    Returns the article text or an empty string if scraping fails.
    """

    url = f"https://finance.yahoo.com/quote/{symbol}/latest-news/"

    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By

        driver = webdriver.Chrome()
        driver.get(url)
        driver.implicitly_wait(10)
        # Get first story
        stories = driver.find_elements(
            By.CSS_SELECTOR, 'section[data-testid="storyitem"]'
        )
        story = stories[0]
        # Get story title within first item
        title = story.find_element(By.TAG_NAME, "h3").text
        print(f"Scraped article title: {title}")
        # get link to story
        link = story.find_element(By.TAG_NAME, "a").get_attribute("href")
        print(f"Link: {link}")

    except Exception as e:
        print(f"Session error: {e}. Cannot scrape article content.")
        return ""
    try:
        if driver:
            for x in stories:
                link = x.find_element(By.TAG_NAME, "a").get_attribute("href")
                title = x.find_element(By.TAG_NAME, "h3").text
                print(f"Visiting link: {link} with title: {title}")
            driver.quit()
    except Exception as e:
        print(f"Error scraping {url}: {e}")
    driver.quit()
    return "none"


def main():
    scrape_article_content("AAPL")
    return


if __name__ == "__main__":
    main()
