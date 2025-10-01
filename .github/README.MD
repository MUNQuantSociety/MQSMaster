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

1. Git: For cloning the repository.
2. Python 3.12+: To run the application.
3. PgAdmin4 (Not mandatory, will work fine without).

## Installation Steps

Follow these steps precisely to set up your environment.

#### Step 1: Clone the Repository

Open your terminal/cmd prompt, navigate to the directory where you want to store the project, and run the following commands one after the other:

1. `git clone https://github.com/MUNQuantSociety/MQSMaster/tree/main`
2. `cd MQSMaster`

#### Step 2: Create and Activate the Virtual Environment

We will create a virtual environment named MQS to keep the project's dependencies isolated. Run this:

`python3 -m venv MQS`

then run:

`source MQS/bin/activate` # (On Windows, use this command instead: MQS\Scripts\activate)

You will know the environment is active when you see (MQS) at the beginning of your terminal prompt. You must ALWAYS have your environment active whenever running any code in here.

#### Step 3: Set Up Environment Variables

The bot requires API keys and database credentials. We manage these using a .env file.
Ask the exec for the Data & Infra team for this file.

add the .env file to the root directory.

#### Step 4: Install Project and Dependencies

This project uses a pyproject.toml file for setup. The following commands will install the bot as a local package and then install all required third-party libraries.

Install the project in "editable" mode

`pip install -e .`

`pip install --no-cache-dir --only-binary :all: -r requirements.txt`

#### Step 5: Test your credentials

Run the following commands:

`python -m src.common.database.test`

If you see something like the following lines, congratulations, you have successfuly received a response from our database.

2025-09-20 16:54:37,793 INFO: Database connection pool created successfully.
✅ PostgreSQL connection successful! Current DB Time: 2025-09-20 15:24:37.975847-04:00
