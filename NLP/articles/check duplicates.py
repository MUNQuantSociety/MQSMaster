import re

import pandas as pd


def check_duplicates():
    df_1 = pd.read_csv(
        "/Users/lodoloro/programs/MQS/MQSMaster/NLP/articles/AAPL_alpha_news.csv"
    )
    df_2 = pd.read_csv(
        "/Users/lodoloro/programs/MQS/MQSMaster/NLP/articles/AAPL_finviz_news.csv"
    )
    df_3 = pd.read_csv(
        "/Users/lodoloro/programs/MQS/MQSMaster/NLP/articles/AAPL_yahoo_news.csv"
    )
    cleaned_titles_1 = set()
    cleaned_titles_2 = set()
    cleaned_titles_3 = set()
    for x in df_1["title"]:
        # Strip special characters and spaces
        x_normalized = re.sub(r"[^a-zA-Z0-9]", "", x).lower()
        cleaned_titles_1.add(x_normalized)
    for y in df_2["title"]:
        y_normalized = re.sub(r"[^a-zA-Z0-9]", "", y).lower()
        cleaned_titles_2.add(y_normalized)
    for z in df_3["title"]:
        z_normalized = re.sub(r"[^a-zA-Z0-9]", "", z).lower()
        cleaned_titles_3.add(z_normalized)
    duplicates = cleaned_titles_1.intersection(cleaned_titles_2).intersection(
        cleaned_titles_3
    )
    print(
        f"Found {len(duplicates)}/{len(cleaned_titles_1) + len(cleaned_titles_2) + len(cleaned_titles_3)} Duplicate titles."
    )
    # match duplicates to original titles
    print("\nDuplicates Titles:")
    print(duplicates)
    print("----------------")
    cleaned_dups = set()
    for dup in duplicates:
        for x in df_1["title"]:
            x_normalized = re.sub(r"[^a-zA-Z0-9]", "", x).lower()
            if x_normalized == dup:
                cleaned_dups.add(x)
        for y in df_2["Title"]:
            y_normalized = re.sub(r"[^a-zA-Z0-9]", "", y).lower()
            if y_normalized == dup:
                cleaned_dups.add(y)
        for z in df_3["title"]:
            z_normalized = re.sub(r"[^a-zA-Z0-9]", "", z).lower()
            if z_normalized == dup:
                cleaned_dups.add(z)
    for n, title in enumerate(cleaned_dups, 0):
        print(f" {n} - {title}")
    return duplicates


if __name__ == "__main__":
    check_duplicates()
