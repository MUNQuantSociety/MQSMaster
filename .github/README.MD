# MQSMaster Trading Bot Codebase

This repository provides the foundation for a multi-portfolio trading bot and a backtesting framework.

---

## Table of Contents

- [Installation](#installation)

---

# Installation

This guide provides step-by-step instructions to clone, install, and run MQSMaster.

### Prerequisites

Before you begin, ensure you have the following software installed on your system:

1. `Git: For cloning the repository.`
1. `Python 3.12+: To run the application.(https://www.python.org/downloads/)`
1. `PgAdmin4 (Not mandatory, will work fine without).`

## Installation Steps

Follow these steps precisely to set up your environment.
### Install pip and uv
run the following command to install pip and uv:

1. `python3 -m pip install --upgrade pip`
1. `pip install uv`
1. `uv python list` # to check if uv is installed correctly
1. `uv python install python@3.12`
1. `uv python pin python@3.12`

#### Step 1: Clone the Repository

Open your terminal/cmd prompt, navigate to the directory where you want to store the project, and run the following commands one after the other:

1. `git clone https://github.com/MUNQuantSociety/MQSMaster.git`
1. `cd MQSMaster`

#### Step 2: Create and Activate the Virtual Environment

We will create a virtual environment named MQS to keep the project's dependencies isolated. Run this:

1. `uv venv MQS`

then run:

1. `source MQS/bin/activate` # (On Windows, use this command instead: MQS\Scripts\activate)

You will know the environment is active when you see (MQS) at the beginning of your terminal prompt. You must ALWAYS have your environment active whenever running any code in here.

#### Step 3: Set Up Environment Variables

The bot requires API keys and database credentials. We manage these using a .env file.
Ask the exec for the Data & Infra team for this file.

Copy the `.env.example` file or rename to `.env` in the root directory, then update it with your credentials:

1. `cp .env.example .env`(or just rename the file)
1. Open the `.env` file and fill in the required values (API keys, database credentials, etc.).
#### Step 4: Install Project and Dependencies

This project uses a pyproject.toml file for setup. The following commands will install the bot as a local package and then install all required third-party libraries.

Install the project in "editable" mode

1. `uv pip install -e .`
1. `uv pip install --no-cache-dir --only-binary :all: -r requirements.txt`

#### Step 5: Test your credentials

Run the following commands:

1. `python -m src.common.database.test`

If you see something like the following lines, congratulations, you have successfuly received a response from our database.

2025-09-20 16:54:37,793 INFO: Database connection pool created successfully.
✅ PostgreSQL connection successful! Current DB Time: 2025-09-20 15:24:37.975847-04:00

#### Step 6: Download the FinBERT Model (required for the NLP pipeline)

The fine-tuned FinBERT model is **not** in the repo. If you plan to run any of the NLP / sentiment code (`NLP/daemon.py`, `NLP/process_sentiment_pipeline.py`, etc.), download the model folder and drop it into `NLP/`:

1. Download from: <https://drive.google.com/drive/folders/1v7NjSuyFq4CTIctrw1bSv13JzkkMg1l8?usp=sharing>
2. Place the folder so the final path is:

   ```
   NLP/finbert-combined-final/
   ```

3. Confirm it contains the model weights, tokenizer, and `config.json`.

See [NLP/README.md](../NLP/README.md) and [docs/NLP/README.md](../docs/NLP/README.md) for details.
