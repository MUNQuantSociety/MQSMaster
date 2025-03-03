# MQS Master Trading Bot Codebase

This repository provides the foundation for a multi-portfolio trading bot and its supporting data infrastructure. The code is split into two major components:

1. **Data & Infrastructure** – Manages market data ingestion (using the Financial Modeling Prep API), database connectivity (to be implemented in the future), authentication, and broker API interactions.
2. **Portfolios** – Contains the actual trading strategies, configuration, and portfolio management logic.

The codebase is organized to allow future extensibility (e.g., backtesting, live data updates, risk management) while also supporting concurrent and memory-efficient data backfilling.

---

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Data & Infrastructure](#data--infrastructure)
  - [Backfilling Data](#backfilling-data)
  - [Concurrent Backfill](#concurrent-backfill)
  - [Portfolios](#portfolios)
- [Extending the Codebase](#extending-the-codebase)
- [Notes and Best Practices](#notes-and-best-practices)
- [Important Environment Files Setup](#important-environment-files-setup)

---

### Directory Descriptions

- **data_infra/**: Contains modules related to data ingestion, storage, API communication, and orchestration.
  - **authentication/**: Contains modules for user and API authentication.
    - `apiAuth.py`: Loads API keys (e.g., the Financial Modeling Prep API key) from environment files.
    - `memberAuth.py`: Manages member authentication (controls access to the system).
  - **brokerAPI/**: Contains modules for broker API interactions (e.g., placing trades).
    - `brokerClient.py`: Connects to and interacts with broker APIs.
  - **data/**: Contains folders for storing local caches and temporary data.
    - `backfill_cache/`: Directory where temporary CSV files for backfilled market data are stored.
  - **database/**: Contains modules for connecting to and managing your database.
    - `MQSDBConnector.py`: Manages PostgreSQL (or another DB) connections (to be implemented fully later).
    - `backupManager.py`: Manages data backups.
    - `schemaDefinitions.py`: Defines and creates required database schemas.
  - **fmp.env**: An environment file containing the Financial Modeling Prep API key (only on the server).
  - **marketData/**: Contains modules to interact with market data providers.
    - `fmpMarketData.py`: Provides a thread-safe client to fetch market data from Financial Modeling Prep, with built-in rate limiting, retry logic, and error handling.
  - **orchestrator/**: Contains modules that orchestrate the data flow.
    - `backfill.py`: Implements the backfilling logic for downloading intraday data in batches and writing incrementally to disk.
    - `concurrent_backfill.py`: Provides a multi-threaded approach to backfill multiple tickers concurrently.
    - `realtimeDataIngestor.py`: (Placeholder) For live data ingestion.
    - `specific_backfill.py`: Example script to run backfilling sequentially.
- **portfolios/**: Contains trading strategy code.
  - `portfolioManager.py`: Coordinates multiple portfolio strategies.
  - `portfolio_1/`: Example portfolio with configuration (`config.txt`) and strategy logic (`strategy.py`).
- **requirements.txt**: Lists all required packages and their versions.

---

## Installation

### Clone the repository:

`git clone [<repository-url>](https://github.com/joshuakatt/MQSMaster/tree/main)`
`cd [<repository-folder>](https://github.com/joshuakatt/MQSMaster/tree/main)`

### Set up a Virtual Environment:

`python3 -m venv venv`
`source venv/bin/activate` # On Windows: venv\Scripts\activate

Install the Requirements:
Use the following command (which ensures a clean installation using only binary wheels):
`pip install --no-cache-dir --only-binary :all: -r requirements.txt`

The requirements.txt should include packages such as:
`requests==2.31.0
python-dotenv==1.0.0
pandas==1.5.3
psycopg2-binary==2.9.6
(Adjust versions as needed.)`

## Configuration

(Not Set Up Yet, Upcoming Functionality)
Important Environment Files Setup
Root .env File:
Create a .env file in the repository root (this is used for general configuration such as DB credentials and member authentication tokens).

For example:

# Example .env (for local development)

DB_HOST=your_postgres_host
DB_PORT=5432
DB_NAME=market_db
DB_USER=postgres
DB_PASSWORD=your_db_password

MEMBER_AUTH_TOKEN=your_member_token # Each user should set their own unique token

FMP API Key:
Create a file named fmp.env in the data_infra folder with your Financial Modeling Prep API key. For example:

# data_infra/fmp.env

FMP_API_KEY=taxFvdsV3ZQiBkff3fkxrAcatQV9C8wG

Note: The fmp.env file should only reside on the server, you don't have to specify this to your device.

Configuration
Important Environment Files Setup
Root .env File:
Create a .env file in the repository root (this is used for general configuration such as DB credentials and member authentication tokens). For example:

# Usage

## Data & Infrastructure

### Market Data Retrieval:

The class FMPMarketData (in data_infra/marketData/fmpMarketData.py) handles fetching historical and intraday market data from Financial Modeling Prep. It includes thread-safe rate limiting, retry logic, and error handling for internet outages.

Backfilling Data
Single-threaded Backfill:
Run specific_backfill.py (located in data_infra/orchestrator) to sequentially backfill data for a given set of tickers.

Concurrent (Multi-threaded) Backfill:
Run concurrent_backfill.py (located in data_infra/orchestrator) to backfill multiple tickers concurrently.

Each ticker's data will be written to its own CSV file (e.g., 2y_mkt_data_AAPL.csv).
An optional merging section at the end will combine these CSV files into a single file (all_tickers_combined.csv) in a memory-efficient, chunked manner.

To run the concurrent backfill:
python data_infra/orchestrator/concurrent_backfill.py

## Portfolios

### Trading Strategies:

Your trading strategies reside in the portfolios/ folder (for example, portfolios/portfolio_1/strategy.py).
The portfolio configuration is stored in portfolios/portfolio_1/config.txt.
portfolioManager.py coordinates multiple portfolios.
These modules can be extended as you develop your trading logic.
Extending the Codebase

Real-time Data Ingestion:
The file realtimeDataIngestor.py (in data_infra/orchestrator) is a placeholder. You can extend it to ingest live data and update your database or CSV files.

Database Integration:
Currently, backfilling writes data incrementally to CSV to avoid RAM overload. Once our database (PostgreSQL, SQLite, or DuckDB) is set up, you can modify the code in data_infra/database and update backfill.py to insert data directly into the database.

Concurrency Improvements:
The FMPMarketData class now supports multi-threading with thread locks to ensure rate limits are not exceeded. You can extend this to process multiple tickers concurrently using the provided concurrent_backfill.py script.

Notes and Best Practices
API Limits:
Financial Modeling Prep may enforce limits based on your plan. If you receive 429 errors ("Too Many Requests"), consider reducing the number of concurrent threads (MAX_WORKERS), adding additional sleeps, or upgrading your plan.

Memory Management:
For very large datasets (up to 10GB), the backfilling process writes data incrementally to CSV files. If merging is needed, use chunked processing to avoid loading all data into memory.

Directory Paths:
The code uses relative paths (e.g., for storing CSV files in data_infra/data/backfill_cache/), ensuring portability across systems.

Error Handling:
The API client in FMPMarketData includes robust error handling for timeouts, connection errors, and rate limits. Concurrency is managed with thread locks to prevent overlapping requests.

Environment Files:
Ensure that you create the necessary .env files (both root .env and data_infra/fmp.env) before running the scripts.

# Running the Code

Install Dependencies:

pip install --no-cache-dir --only-binary :all: -r requirements.txt
Set Up Environment Files:

Create a .env file in the repository root for general configuration.
Create a data_infra/fmp.env file with your FMP API key.
Run a Concurrent Backfill Example:

python data_infra/orchestrator/concurrent_backfill.py
This will backfill data for the tickers defined in the script over the specified date range, with each ticker’s data saved to its own CSV file in data_infra/data/backfill_cache/. Optionally, the script will merge these CSV files into a single file named all_tickers_combined.csv.

Run Portfolio Strategies:

Navigate to a portfolio folder (e.g., portfolios/portfolio_1/) and run the strategy script.
Ensure your portfolio’s configuration is set correctly in config.txt.
Conclusion
