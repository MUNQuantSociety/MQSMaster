# NLP Sentiment Analysis Pipeline

## Fetching Articles

1. Open a terminal and run:

python -m NLP.fetch_articles.py <TICKER> <START_DATE> <END_DATE>

- Replace `<TICKER>` with the stock symbol (e.g., `AAPL`).
- Replace `<START_DATE>` and `<END_DATE>` with the desired date range in `YYYY-MM-DD` format.

**Example:**
python -m NLP.fetch_articles AAPL 2024-01-01 2025-01-01


## Adding Models

1. Download the required model `.zip` file from [this link]( https://drive.google.com/drive/u/4/folders/1v7NjSuyFq4CTIctrw1bSv13JzkkMg1l8).
2. Extract the contents into the `NLP/models` folder.

> Contact one of the DataInfra Members if you are unnable to access the link.

## Executing the Model

1. Open `visualise_NLP.ipynb` in a Jupyter Notebook environment.
2. Run each cell in order to process the data and visualize the results. 

**Example:**

- Start with:
  - `# ─── Cell 1: Compute & Save Per-Article & Daily Sentiment Scores ───────────────`
  - Followed by: `# ─── Cell 2: Fetch Stock & Market Prices ───────────────────────────────`

- Continue until all **4 cells** have been executed.

> You may revisit specific cells (e.g., Cell 1 or Cell 2) as needed to refresh outputs.
