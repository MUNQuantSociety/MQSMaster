#!/bin/bash

# --- CONFIGURATION ---

# Find the absolute path of the directory where the script is located.
# This ensures that the script can be run from anywhere, including cron.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- !! ACTION REQUIRED !! ---
# Define the absolute path to the Python executable in your virtual environment.
# Based on your 'ls' command, this should be the correct path.
PYTHON_VENV="/home/master/MQSMaster/MQS/bin/python"

# Load environment variables (like FMP_API_KEY) from the .env file.
# The .env file should be in the same directory as this script.
if [ -f "${SCRIPT_DIR}/.env" ]; then
    source "${SCRIPT_DIR}/.env"
else
    echo "[ERROR] .env file not found at ${SCRIPT_DIR}/.env. Exiting."
    exit 1
fi

# Set the exchange to monitor.
EXCHANGE="NASDAQ"

# --- PRE-FLIGHT CHECKS ---

# Check if required commands are installed
for cmd in curl jq; do
  if ! command -v $cmd &> /dev/null; then
    echo "[ERROR] Required command '$cmd' is not installed. Please install it to continue. Exiting."
    exit 1
  fi
done

# Check if the Python virtual environment path is correct and executable
if [ ! -x "$PYTHON_VENV" ]; then
  echo "[ERROR] Python executable not found or not executable at: $PYTHON_VENV"
  echo "Please verify the path and permissions. Exiting."
  exit 1
fi

# --- FUNCTION DEFINITIONS ---

# Function to check if the market is open using the Financial Modeling Prep API.
is_market_open() {
  local response
  response=$(curl -s "https://financialmodelingprep.com/stable/exchange-market-hours?exchange=${EXCHANGE}&apikey=${FMP_API_KEY}")
  
  if [ -z "$response" ]; then
    echo "[WARNING] No response from API (check network or API key). Assuming market is closed."
    return 1 # Return "failure" (market closed)
  fi

  # Check if the response contains 'isMarketOpen' before passing to jq
  if ! echo "$response" | jq -e '.[0] | has("isMarketOpen")' > /dev/null; then
     echo "[WARNING] API response did not contain market status. Assuming market is closed."
     echo "API Response: $response"
     return 1 # Return "failure"
  fi

  echo "$response" | jq -e '.[0].isMarketOpen' > /dev/null
}

# --- SCRIPT START ---

# Change to the script's directory to ensure relative paths in Python scripts work correctly.
cd "$SCRIPT_DIR" || exit

# Array to hold the Process IDs (PIDs) of the background scripts.
pids=()
scripts_to_run=(
  "main.py"
  "data_infra/orchestrator/realTime/realtimeDataIngestor.py"
  "./pnl_script.py"
)

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting background processes using Python from: ${PYTHON_VENV}"

# Start the python scripts using the venv's python and add their PIDs to the array.
for script in "${scripts_to_run[@]}"; do
  "$PYTHON_VENV" "$script" &
  pid=$!
  # Small delay to see if the process crashes immediately
  sleep 1
  if ps -p $pid > /dev/null; then
    echo "  -> Started '$script' successfully with PID: $pid"
    pids+=($pid)
  else
    echo "[ERROR] Failed to start '$script'. Check the script for errors."
    # If one script fails, we should probably stop the others.
    echo "Shutting down other started processes."
    for p in "${pids[@]}"; do kill -SIGTERM "$p"; done
    exit 1
  fi
done

echo "All processes started successfully. PIDs: ${pids[@]}"

# --- MONITORING LOOP ---

while true; do
  # Check if the market is closed.
  if ! is_market_open; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Market is closed or API check failed. Shutting down all processes."

    # Loop through all stored PIDs and send a termination signal to each.
    for pid in "${pids[@]}"; do
      # Check if the process still exists before trying to kill it
      if ps -p "$pid" > /dev/null; then
        echo "  -> Sending SIGTERM to process with PID: $pid"
        kill -SIGTERM "$pid"
      else
        echo "  -> Process with PID $pid no longer exists."
      fi
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
