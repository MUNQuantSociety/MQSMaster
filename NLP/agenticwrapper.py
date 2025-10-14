#Hopefully final
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset as HFDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ================================================================
# STEP 1: Merge CSV Files for Ticker
# ================================================================

def merge_csv_files(ticker_symbol, base_path='', output_file=None):
    """
    Merge three CSV files:
    - {ticker}.csv: publishedDate, title, content, site
    - {ticker}_article_scores.csv: timestamp, sentiment, title, site
    - {ticker}_CAPM.csv: date, alpha, beta, stock_return, expected_return
    """
    if output_file is None:
        output_file = f'{ticker_symbol}_merged.csv'

    article_file = os.path.join(base_path, f'{ticker_symbol}.csv')
    score_file = os.path.join(base_path, f'{ticker_symbol}_article_scores.csv')
    capm_file = os.path.join(base_path, f'{ticker_symbol}_CAPM.csv')

    print(f"Loading files for {ticker_symbol}...")

    try:
        df_articles = pd.read_csv(article_file)
        df_scores = pd.read_csv(score_file)
        df_capm = pd.read_csv(capm_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

    print(f"Initial shapes: Articles={df_articles.shape}, Scores={df_scores.shape}, CAPM={df_capm.shape}")

    # Extract date columns
    article_date_col = 'publishedDate' if 'publishedDate' in df_articles.columns else df_articles.columns[0]
    score_date_col = 'timestamp' if 'timestamp' in df_scores.columns else df_scores.columns[0]
    capm_date_col = 'date' if 'date' in df_capm.columns else df_capm.columns[0]

    df_articles['date'] = pd.to_datetime(df_articles[article_date_col], errors='coerce')
    df_scores['date'] = pd.to_datetime(df_scores[score_date_col], errors='coerce')
    df_capm['date'] = pd.to_datetime(df_capm[capm_date_col], errors='coerce')

    # Extract relevant columns
    article_col = 'content' if 'content' in df_articles.columns else 'title'
    sentiment_col = 'sentiment' if 'sentiment' in df_scores.columns else df_scores.select_dtypes(include=[np.number]).columns[0]
    alpha_col = 'alpha' if 'alpha' in df_capm.columns else df_capm.select_dtypes(include=[np.number]).columns[0]

    # Prepare clean dataframes
    df_articles_clean = df_articles[['date', article_col]].copy()
    df_articles_clean.columns = ['date', 'article']

    df_scores_clean = df_scores[['date', sentiment_col]].copy()
    df_scores_clean.columns = ['date', 'sentiment_score']

    df_capm_clean = df_capm[['date', alpha_col]].copy()
    df_capm_clean.columns = ['date', 'alpha']

    # Drop invalid dates
    df_articles_clean = df_articles_clean.dropna(subset=['date'])
    df_scores_clean = df_scores_clean.dropna(subset=['date'])
    df_capm_clean = df_capm_clean.dropna(subset=['date'])

    # Convert to date only (no time)
    df_articles_clean['date'] = df_articles_clean['date'].dt.date
    df_scores_clean['date'] = df_scores_clean['date'].dt.date
    df_capm_clean['date'] = df_capm_clean['date'].dt.date

    # Aggregate by date
    df_articles_agg = df_articles_clean.groupby('date')['article'].apply(
        lambda x: ' '.join(x.dropna().astype(str))
    ).reset_index()
    df_scores_agg = df_scores_clean.groupby('date')['sentiment_score'].mean().reset_index()
    df_capm_agg = df_capm_clean.groupby('date')['alpha'].mean().reset_index()

    # Merge
    merged = df_articles_agg.merge(df_scores_agg, on='date', how='inner')
    merged = merged.merge(df_capm_agg, on='date', how='inner')
    merged = merged.dropna()
    merged = merged[merged['article'].str.strip().str.len() > 0]

    print(f"Merged: {merged.shape[0]} samples from {merged['date'].min()} to {merged['date'].max()}")

    output_path = os.path.join(base_path, output_file)
    merged.to_csv(output_path, index=False)
    print(f"Saved to {output_path}\n")

    return merged


# ================================================================
# STEP 2: Correlation Agent (Policy Network)
# ================================================================

class CorrelationAgent:
    """
    Agent that learns sentiment-alpha correlation through:
    1. Supervised Learning
    2. Evolutionary Fine-tuning
    3. RL Bandit-style Fine-tuning
    """

    def __init__(self, model_name="yiyanghkust/finbert-tone"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load base transformer
        base_model = AutoModel.from_pretrained(model_name)
        hidden_size = base_model.config.hidden_size

        # Build policy network with dual heads
        self.policy_network = nn.ModuleDict({
            'encoder': base_model,
            'sentiment_head': nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1),
                nn.Tanh()  # [-1, 1]
            ),
            'alpha_head': nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1)
            )
        }).to(self.device)

    def forward(self, input_ids, attention_mask):
        outputs = self.policy_network['encoder'](input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        sentiment_pred = self.policy_network['sentiment_head'](pooled).squeeze(-1)
        alpha_pred = self.policy_network['alpha_head'](pooled).squeeze(-1)
        return sentiment_pred, alpha_pred

    @torch.no_grad()
    def predict(self, text: str):
        """Predict sentiment and alpha for a given text"""
        self.policy_network.eval()
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        sentiment, alpha = self.forward(inputs['input_ids'], inputs['attention_mask'])
        return sentiment.item(), alpha.item()

    # Parameter manipulation for evolution
    def get_parameters(self, mode="head"):
        """Extract parameters for evolution"""
        parts = []
        for name, param in self.policy_network.named_parameters():
            if mode == "head" and ("sentiment_head" in name or "alpha_head" in name):
                parts.append(param.data.detach().cpu().numpy().astype(np.float32).ravel())
            elif mode == "all":
                parts.append(param.data.detach().cpu().numpy().astype(np.float32).ravel())
        return np.concatenate(parts) if parts else np.array([], dtype=np.float32)

    def set_parameters(self, flat_params, mode="head"):
        """Set parameters from flat array"""
        pointer = 0
        for name, param in self.policy_network.named_parameters():
            should_set = False
            if mode == "head" and ("sentiment_head" in name or "alpha_head" in name):
                should_set = True
            elif mode == "all":
                should_set = True

            if should_set:
                shape = tuple(param.data.shape)
                size = int(np.prod(shape))
                chunk = flat_params[pointer:pointer+size]
                pointer += size
                values = chunk.reshape(shape).astype(np.float32)
                with torch.no_grad():
                    param.data.copy_(torch.from_numpy(values).to(param.data.dtype).to(self.device))

    @torch.no_grad()
    def evaluate_fitness(self, dataloader):
        """Fitness = negative correlation loss (maximize correlation)"""
        self.policy_network.eval()
        all_sent_preds, all_alpha_preds = [], []
        all_sent_true, all_alpha_true = [], []

        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            sentiment_pred, alpha_pred = self.forward(batch['input_ids'], batch['attention_mask'])

            all_sent_preds.extend(sentiment_pred.cpu().numpy())
            all_alpha_preds.extend(alpha_pred.cpu().numpy())
            all_sent_true.extend(batch['sentiment_score'].cpu().numpy())
            all_alpha_true.extend(batch['alpha'].cpu().numpy())

        # Fitness = correlation between predicted sentiment and true alpha
        corr = np.corrcoef(all_sent_preds, all_alpha_true)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0


# ================================================================
# STEP 3: Dataset
# ================================================================

class CorrelationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        encoding = self.tokenizer(
            row['article'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'sentiment_score': torch.tensor(row['sentiment_score'], dtype=torch.float32),
            'alpha': torch.tensor(row['alpha'], dtype=torch.float32)
        }


# ================================================================
# STEP 4: Three-Phase Training Pipeline
# ================================================================

def train_hybrid_agent(ticker_symbol, base_path='', model_name='yiyanghkust/finbert-tone'):
    """
    Three-phase hybrid training:
    1. Supervised Learning
    2. Evolutionary Fine-tuning
    3. RL Bandit Fine-tuning
    """

    # Load merged data
    merged_path = os.path.join(base_path, f'{ticker_symbol}_merged.csv')
    df = pd.read_csv(merged_path)

    # Normalize
    if df['sentiment_score'].min() >= 0:
        df['sentiment_score'] = (df['sentiment_score'] - 0.5) * 2
    df['alpha'] = (df['alpha'] - df['alpha'].mean()) / df['alpha'].std()

    # Split
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    print(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # ========================================
    # PHASE 1: SUPERVISED LEARNING
    # ========================================
    print("\n" + "="*60)
    print("PHASE 1: SUPERVISED LEARNING")
    print("="*60)

    agent = CorrelationAgent(model_name=model_name)

    train_dataset = CorrelationDataset(train_df, agent.tokenizer)
    val_dataset = CorrelationDataset(val_df, agent.tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    optimizer = AdamW(agent.policy_network.parameters(), lr=2e-5, weight_decay=0.01)

    best_val_loss = float('inf')
    sl_epochs = 3

    for epoch in range(sl_epochs):
        agent.policy_network.train()
        train_losses = []

        for batch in train_loader:
            batch = {k: v.to(agent.device) for k, v in batch.items()}
            sentiment_pred, alpha_pred = agent.forward(batch['input_ids'], batch['attention_mask'])

            # Multi-objective loss
            loss_sent = F.mse_loss(sentiment_pred, batch['sentiment_score'])
            loss_alpha = F.mse_loss(alpha_pred, batch['alpha'])

            # Correlation loss
            sent_norm = (sentiment_pred - sentiment_pred.mean()) / (sentiment_pred.std() + 1e-8)
            alpha_true_norm = (batch['alpha'] - batch['alpha'].mean()) / (batch['alpha'].std() + 1e-8)
            corr_loss = -(sent_norm * alpha_true_norm).mean()

            loss = 0.3 * loss_sent + 0.3 * loss_alpha + 0.4 * corr_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.policy_network.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        # Validation
        val_fitness = agent.evaluate_fitness(val_loader)
        avg_train_loss = np.mean(train_losses)

        print(f"Epoch {epoch+1}/{sl_epochs}: Train Loss={avg_train_loss:.4f}, Val Corr={val_fitness:.4f}")

        if avg_train_loss < best_val_loss:
            best_val_loss = avg_train_loss
            torch.save(agent.policy_network.state_dict(), f'{ticker_symbol}_supervised.pt')

    sl_fitness = agent.evaluate_fitness(val_loader)
    print(f"Supervised Learning Correlation: {sl_fitness:.4f}")

    # ========================================
    # PHASE 2: EVOLUTIONARY FINE-TUNING
    # ========================================
    print("\n" + "="*60)
    print("PHASE 2: EVOLUTIONARY FINE-TUNING")
    print("="*60)

    population_size = 6
    num_generations = 5
    mutation_rate = 0.03
    mutation_scale = 0.02
    elitism_count = 2
    tournament_size = 3

    # Initialize population
    population = []
    for i in range(population_size):
        new_agent = CorrelationAgent(model_name=model_name)
        new_agent.policy_network.load_state_dict(agent.policy_network.state_dict())

        if i > 0:  # Add diversity
            params = new_agent.get_parameters(mode="head")
            if params.size > 0:
                noise_mask = (np.random.rand(len(params)) < mutation_rate)
                noise = np.random.normal(0, mutation_scale, noise_mask.sum()).astype(np.float32)
                params[noise_mask] += noise
                new_agent.set_parameters(params, mode="head")

        population.append(new_agent)

    best_fitness_history = [sl_fitness]
    avg_fitness_history = [sl_fitness]

    for generation in range(num_generations):
        print(f"\nGeneration {generation+1}/{num_generations}")
        fitness_scores = []

        for i, pop_agent in enumerate(population):
            fitness = pop_agent.evaluate_fitness(val_loader)
            fitness_scores.append(fitness)
            print(f"  Agent {i+1}: {fitness:.4f}")

        best_fitness = float(np.max(fitness_scores))
        avg_fitness = float(np.mean(fitness_scores))
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)

        print(f"Best: {best_fitness:.4f}, Avg: {avg_fitness:.4f}")

        # Tournament selection
        selected_parents = []
        for _ in range(population_size - elitism_count):
            tournament_idx = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fit = [fitness_scores[i] for i in tournament_idx]
            winner = tournament_idx[int(np.argmax(tournament_fit))]
            selected_parents.append(population[winner])

        # Elitism
        new_population = []
        elite_idx = np.argsort(fitness_scores)[-elitism_count:]
        for idx in elite_idx:
            new_population.append(population[int(idx)])

        # Crossover + Mutation
        for _ in range(population_size - elitism_count):
            parent1, parent2 = random.sample(selected_parents, 2)
            child = CorrelationAgent(model_name=model_name)

            p1 = parent1.get_parameters(mode="head")
            p2 = parent2.get_parameters(mode="head")

            if p1.size > 0 and p2.size > 0:
                alpha = random.random()
                child_params = alpha * p1 + (1 - alpha) * p2

                # Mutation
                mut_mask = (np.random.rand(len(child_params)) < mutation_rate)
                mut_vals = np.random.normal(0, mutation_scale, mut_mask.sum()).astype(np.float32)
                child_params[mut_mask] += mut_vals

                child.set_parameters(child_params, mode="head")

            new_population.append(child)

        population = new_population

    # Select best
    final_fitness = [agent.evaluate_fitness(val_loader) for agent in population]
    best_idx = int(np.argmax(final_fitness))
    best_agent = population[best_idx]
    evo_fitness = float(final_fitness[best_idx])

    print(f"\nEvolution complete. Best correlation: {evo_fitness:.4f}")
    print(f"Improvement over SL: {evo_fitness - sl_fitness:.4f}")

    # ========================================
    # PHASE 3: RL BANDIT FINE-TUNING
    # ========================================
    print("\n" + "="*60)
    print("PHASE 3: RL BANDIT FINE-TUNING")
    print("="*60)

    best_agent.policy_network.train()
    rl_optimizer = AdamW(best_agent.policy_network.parameters(), lr=5e-6)
    rl_epochs = 2
    baseline = 0.0
    baseline_beta = 0.9
    entropy_coef = 0.01

    for epoch in range(rl_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = {k: v.to(best_agent.device) for k, v in batch.items()}
            sentiment_pred, alpha_pred = best_agent.forward(batch['input_ids'], batch['attention_mask'])

            # Reward: correlation-based
            sent_norm = (sentiment_pred - sentiment_pred.mean()) / (sentiment_pred.std() + 1e-8)
            alpha_norm = (batch['alpha'] - batch['alpha'].mean()) / (batch['alpha'].std() + 1e-8)
            reward = (sent_norm * alpha_norm)  # Element-wise correlation signal

            # Policy gradient with baseline
            baseline = baseline_beta * baseline + (1 - baseline_beta) * reward.mean().item()
            adv = reward - baseline

            # Loss (maximize correlation)
            loss = -(adv.detach() * sentiment_pred).mean()

            rl_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(best_agent.policy_network.parameters(), 1.0)
            rl_optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        val_corr = best_agent.evaluate_fitness(val_loader)
        print(f"RL Epoch {epoch+1}/{rl_epochs}: Loss={epoch_loss/n_batches:.4f}, Val Corr={val_corr:.4f}")

    # Save final model
    torch.save(best_agent.policy_network.state_dict(), f'{ticker_symbol}_hybrid_final.pt')
    print(f"Final model saved to {ticker_symbol}_hybrid_final.pt")

    # Test evaluation
    test_dataset = CorrelationDataset(test_df, best_agent.tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    test_corr = best_agent.evaluate_fitness(test_loader)

    print(f"\n{'='*60}")
    print(f"TEST SET CORRELATION: {test_corr:.4f}")
    print(f"{'='*60}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(best_fitness_history, label='Best Correlation', marker='o')
    plt.plot(avg_fitness_history, label='Avg Correlation', marker='s')
    plt.axhline(y=sl_fitness, color='r', linestyle='--', label='SL Baseline')
    plt.xlabel('Generation')
    plt.ylabel('Correlation (Fitness)')
    plt.title(f'{ticker_symbol}: Hybrid Training (SL + Evolution + RL)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{ticker_symbol}_training_progress.png', dpi=300)
    plt.show()

    return best_agent


# ================================================================
# USAGE
# ================================================================

# ================================================================
# USAGE
# ================================================================

def process_ticker_interactive():
    """Interactive function to process any ticker"""
    print("\n" + "="*60)
    print("SENTIMENT-ALPHA CORRELATION TRAINER")
    print("="*60)

    # Get ticker from user
    ticker = input("\nEnter ticker symbol (e.g., AAPL, MSFT, GOOGL, TSLA): ").strip().upper()

    if not ticker:
        print("Error: No ticker provided")
        return None

    # Get data path
    data_path = input("Enter data directory path (press Enter for current directory): ").strip()
    if not data_path:
        data_path = './'

    print(f"\nProcessing ticker: {ticker}")
    print(f"Data path: {data_path}")

    # Check if files exist
    required_files = [
        f'{ticker}.csv',
        f'{ticker}_article_scores.csv',
        f'{ticker}_CAPM.csv'
    ]

    print("\nChecking for required files...")
    missing_files = []
    for file in required_files:
        file_path = os.path.join(data_path, file)
        if os.path.exists(file_path):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (NOT FOUND)")
            missing_files.append(file)

    if missing_files:
        print(f"\nError: Missing files: {missing_files}")
        print("Please ensure all three files exist in the specified directory.")
        return None

    # Step 1: Merge files
    print("\n" + "="*60)
    print(f"STEP 1: MERGING CSV FILES FOR {ticker}")
    print("="*60)

    merged = merge_csv_files(ticker, base_path=data_path)

    if merged is None:
        print("Error: Failed to merge files")
        return None

    if len(merged) < 50:
        print(f"Warning: Only {len(merged)} samples found. Recommended minimum: 50")
        proceed = input("Continue anyway? (y/n): ").strip().lower()
        if proceed != 'y':
            return None

    # Step 2: Train
    print("\n" + "="*60)
    print(f"STEP 2: TRAINING HYBRID AGENT FOR {ticker}")
    print("="*60)
    print("\nThis will run three training phases:")
    print("  1. Supervised Learning (3 epochs)")
    print("  2. Evolutionary Fine-tuning (5 generations)")
    print("  3. RL Bandit Fine-tuning (2 epochs)")

    proceed = input("\nStart training? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Training cancelled")
        return None

    agent = train_hybrid_agent(ticker, base_path=data_path)

    # Step 3: Test predictions
    print("\n" + "="*60)
    print(f"STEP 3: TESTING PREDICTIONS FOR {ticker}")
    print("="*60)

    # Default test articles
    test_articles = [
        f"{ticker} reports record earnings beating all analyst expectations",
        f"{ticker} stock drops 5% amid supply chain disruptions and weak guidance",
        f"{ticker} announces new product line meeting market expectations"
    ]

    print("\nTesting on sample articles:")
    for i, text in enumerate(test_articles, 1):
        sent, alpha = agent.predict(text)
        print(f"\n{i}. {text}")
        print(f"   Predicted Sentiment: {sent:+.3f}")
        print(f"   Predicted Alpha: {alpha:+.3f}")

    # Allow custom predictions
    print("\n" + "-"*60)
    print("Try your own article text (or press Enter to finish)")
    print("-"*60)

    while True:
        custom_text = input("\nEnter article text (or press Enter to finish): ").strip()
        if not custom_text:
            break

        sent, alpha = agent.predict(custom_text)
        print(f"  Predicted Sentiment: {sent:+.3f}")
        print(f"  Predicted Alpha: {alpha:+.3f}")

    print(f"\n{'='*60}")
    print(f"Training complete for {ticker}!")
    print(f"Model saved as: {ticker}_hybrid_final.pt")
    print(f"Training plot saved as: {ticker}_training_progress.png")
    print(f"{'='*60}")

    return agent


def process_multiple_tickers():
    """Process multiple tickers in batch mode"""
    print("\n" + "="*60)
    print("BATCH PROCESSING MODE")
    print("="*60)

    # Get tickers
    tickers_input = input("\nEnter ticker symbols separated by commas (e.g., AAPL,MSFT,GOOGL): ").strip().upper()
    tickers = [t.strip() for t in tickers_input.split(',') if t.strip()]

    if not tickers:
        print("Error: No tickers provided")
        return None

    # Get data path
    data_path = input("Enter data directory path (press Enter for current directory): ").strip()
    if not data_path:
        data_path = './'

    print(f"\nWill process {len(tickers)} ticker(s): {', '.join(tickers)}")
    proceed = input("Continue? (y/n): ").strip().lower()

    if proceed != 'y':
        print("Cancelled")
        return None

    results = {}

    for i, ticker in enumerate(tickers, 1):
        print("\n" + "="*70)
        print(f"PROCESSING TICKER {i}/{len(tickers)}: {ticker}")
        print("="*70)

        try:
            # Merge
            merged = merge_csv_files(ticker, base_path=data_path)
            if merged is None or len(merged) < 50:
                print(f"Skipping {ticker}: Insufficient data")
                continue

            # Train
            agent = train_hybrid_agent(ticker, base_path=data_path)

            # Test
            test_text = f"{ticker} reports strong quarterly results"
            sent, alpha = agent.predict(test_text)

            results[ticker] = {
                'agent': agent,
                'test_sentiment': sent,
                'test_alpha': alpha
            }

            print(f"\n✓ {ticker} completed successfully")

        except Exception as e:
            print(f"\n✗ Error processing {ticker}: {e}")
            continue

    # Summary
    print("\n" + "="*70)
    print("BATCH PROCESSING SUMMARY")
    print("="*70)

    if results:
        print(f"\nSuccessfully processed {len(results)}/{len(tickers)} tickers:\n")
        for ticker in results:
            print(f"  ✓ {ticker}")
            print(f"      Model: {ticker}_hybrid_final.pt")
            print(f"      Plot: {ticker}_training_progress.png")
    else:
        print("\nNo tickers were successfully processed")

    return results


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  SENTIMENT-ALPHA CORRELATION TRAINER")
    print("  Hybrid Training: Supervised Learning + Evolution + RL")
    print("="*70)

    print("\nSelect mode:")
    print("  1. Single ticker (interactive)")
    print("  2. Multiple tickers (batch)")
    print("  3. Exit")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == '1':
        agent = process_ticker_interactive()

    elif choice == '2':
        results = process_multiple_tickers()

    elif choice == '3':
        print("Exiting...")

    else:
        print("Invalid choice")

        # Fallback: run with default AAPL
        print("\nFalling back to default ticker: AAPL")
        TICKER = 'AAPL'
        DATA_PATH = './'

        merged = merge_csv_files(TICKER, base_path=DATA_PATH)
        if merged is not None and len(merged) >= 50:
            agent = train_hybrid_agent(TICKER, base_path=DATA_PATH)

            test_articles = [
                "Apple reports record earnings beating all analyst expectations",
                "AAPL stock drops 5% amid supply chain disruptions",
                "Apple announces new product line meeting market expectations"
            ]

            print("\n" + "="*60)
            print("TESTING PREDICTIONS")
            print("="*60)

            for i, text in enumerate(test_articles, 1):
                sent, alpha = agent.predict(text)
                print(f"\n{i}. {text[:60]}...")
                print(f"   Predicted Sentiment: {sent:+.3f}")
                print(f"   Predicted Alpha: {alpha:+.3f}")