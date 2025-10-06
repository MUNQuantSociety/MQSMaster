import os
import pandas as pd
from textblob import TextBlob
from datetime import datetime

class LocalAgenticNewsAI:
    """
    Simulates an agentic AI that:
    - Loads or collects articles
    - Analyzes sentiment locally
    - Summarizes results
    """
    def __init__(self, news_csv_path=None):
        self.memory = pd.DataFrame(columns=['date', 'title', 'content', 'polarity', 'subjectivity'])
        self.news_csv_path = news_csv_path
        if news_csv_path and os.path.exists(news_csv_path):
            self.load_articles(news_csv_path)
    
    # -----------------------------
    # Memory / Knowledge
    # -----------------------------
    def load_articles(self, path):
        self.memory = pd.read_csv(path)
        print(f"[AGENT] Loaded {len(self.memory)} articles from {path}")

    def add_article(self, title, content, date=None):
        date = date or datetime.now().strftime("%Y-%m-%d")
        new_article = pd.DataFrame([{
            'date': date,
            'title': title,
            'content': content,
            'polarity': None,
            'subjectivity': None
        }])
        self.memory = pd.concat([self.memory, new_article], ignore_index=True)
        print(f"[AGENT] Added article: {title}")

    # -----------------------------
    # Task Planning / Reasoning
    # -----------------------------
    def analyze_sentiment(self):
        if self.memory.empty:
            print("[AGENT] No articles to analyze.")
            return
        self.memory['polarity'] = self.memory['content'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        self.memory['subjectivity'] = self.memory['content'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
        print("[AGENT] Sentiment analysis completed.")

    def summarize_daily(self):
        if self.memory.empty:
            return pd.DataFrame()
        summary = self.memory.groupby('date').agg({
            'polarity': 'mean',
            'subjectivity': 'mean',
            'title': 'count'
        }).rename(columns={'title': 'num_articles'}).reset_index()
        print("[AGENT] Generated daily summary.")
        return summary

    # -----------------------------
    # Persistence / Output
    # -----------------------------
    def save_memory(self, path):
        self.memory.to_csv(path, index=False)
        print(f"[AGENT] Memory saved to {path}")

    # -----------------------------
    # Agentic behavior
    # -----------------------------
    def run(self):
        """
        Simple autonomous workflow:
        1. Analyze sentiment
        2. Summarize daily
        3. Save results
        """
        print("[AGENT] Running autonomous workflow...")
        self.analyze_sentiment()
        summary = self.summarize_daily()
        self.save_memory("agent_memory.csv")
        print("[AGENT] Workflow completed.")
        return summary

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    agent = LocalAgenticNewsAI("aapl_articles.csv")
    
    # Add articles manually
    agent.add_article("Apple hits new highs", "Apple stock surges as earnings beat estimates")
    agent.add_article("Investors cautious", "Market uncertainty affects Apple stock", "2025-10-05")
    
    # Run autonomous agent
    daily_summary = agent.run()
    
    print(daily_summary)






