import re

import pandas as pd

if __name__ == "__main__":
    df_1 = pd.read_csv("/Users/lodoloro/programs/MQS/MQSMaster/NLP/articles/AAPL.csv")
    df_2 = pd.read_csv(
        "/Users/lodoloro/programs/MQS/MQSMaster/NLP/articles/finviz_news_AAPL.csv"
    )

    cleaned_titles_1 = set()
    cleaned_titles_2 = set()
    for x in df_1["title"]:
        # Strip special characters and spaces
        x_normalized = re.sub(r"[^a-zA-Z0-9]", "", x).lower()
        cleaned_titles_1.add(x_normalized)

    for y in df_2["Title"]:
        y_normalized = re.sub(r"[^a-zA-Z0-9]", "", y).lower()
        cleaned_titles_2.add(y_normalized)
    duplicates = cleaned_titles_1.intersection(cleaned_titles_2)
    print(f"Found {len(duplicates)}/{len(cleaned_titles_1) + len(cleaned_titles_2)} Duplicate titles.")
    for dup in duplicates:
        print(dup)
    #match duplicates to original titles
    print("\nMatching duplicates to original titles:")
    for dup in duplicates:
        for x in df_1["title"]:
            x_normalized = re.sub(r"[^a-zA-Z0-9]", "", x).lower()
            if x_normalized == dup:
                print(f"- {x}")
