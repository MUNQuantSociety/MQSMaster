#!/bin/bash

# --- CONFIGURATION ---

# Find the absolute path of the directory where the script is located.
# This ensures that the script can be run from anywhere, including cron.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define the absolute path to the Python executable in your virtual environment.
PYTHON_VENV="/home/master/MQSMaster/mqs/bin/python"

# Load environment variables (like FMP_API_KEY) from the .env file.
# The .env file should be in the same directory as this script.
source "${SCRIPT_DIR}/.env"

# Set the exchange to monitor.
EXCHANGE="NASDAQ"

# --- FUNCTION DEFINITIONS ---

# Function to check if the market is open using the Financial Modeling Prep API.
is_market_open() {
  curl -s "https://financialmodelingprep.com/stable/exchange-market-hours?exchange=${EXCHANGE}&apikey=${FMP_API_KEY}" \
    | jq -e '.[0].isMarketOpen' > /dev/null
}

# --- SCRIPT START ---

# Change to the script's directory to ensure relative paths in Python scripts work correctly.
cd "$SCRIPT_DIR" || exit

# Array to hold the Process IDs (PIDs) of the background scripts.
pids=()

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting background processes using Python from: ${PYTHON_VENV}"

# Start the python scripts using the venv's python and add their PIDs to the array.
"$PYTHON_VENV" main.py &
pids+=($!)

"$PYTHON_VENV" data_infra/orchestrator/realTime/realtimeDataIngestor.py &
pids+=($!)

"$PYTHON_VENV" ./pnl_script.py &
pids+=($!)

echo "Started processes with PIDs: ${pids[@]}"

# --- MONITORING LOOP ---

while true; do
  # Check if the market is closed.
  if ! is_market_open; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Market is closed. Shutting down all processes."

    # Loop through all stored PIDs and send a termination signal to each.
    for pid in "${pids[@]}"; do
      echo "  -> Sending SIGTERM to process with PID: $pid"
      kill -SIGTERM "$pid"
    done

    # Wait for all background processes to actually terminate.
    wait "${pids[@]}" 2>/dev/null

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] All processes have been terminated. Exiting watchdog."
    break # Exit the while loop.
  fi

  # If the market is still open, wait for 3 minutes.
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Market is open. Checking again in 3 minutes."
  sleep 180 # 180 seconds = 3 minutes
done
