# --- CONFIGURATION ---

# Find the absolute path of the directory where the script is located.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load environment variables (like FMP_API_KEY) from the .env file.
source "${SCRIPT_DIR}/.env"

# Set the exchange to monitor.
EXCHANGE="NASDAQ"

# --- FUNCTION DEFINITIONS ---

# Function to check if the market is open using the Financial Modeling Prep API.
# It uses 'jq' to parse the JSON response. The function's success or failure
# (exit code) indicates the market status.
is_market_open() {
  curl -s "https://financialmodelingprep.com/stable/exchange-market-hours?exchange=${EXCHANGE}&apikey=${FMP_API_KEY}" \
    | jq -e '.[0].isMarketOpen' > /dev/null
}

# --- SCRIPT START ---

# Array to hold the Process IDs (PIDs) of the background scripts.
pids=()

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting background processes..."

# Start the first python script in the background and add its PID to the array.
python3 main.py &
pids+=($!)

# Start the second python script and add its PID to the array.
python3 data_infra/orchestrator/realTime/realtimeDataIngestor.py &
pids+=($!)

# Start the third python script and add its PID to the array.
python3 ./pnl_script.py &
pids+=($!)

echo "Started processes with PIDs: ${pids[@]}"

# --- MONITORING LOOP ---

# Loop indefinitely to check the market status.
while true; do
  # Check if the market is closed.
  if ! is_market_open; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Market is closed. Shutting down all processes."

    # Loop through all stored PIDs and send a termination signal to each.
    for pid in "${pids[@]}"; do
      echo "  -> Sending SIGTERM to process with PID: $pid"
      # The 'kill' command requests the process to terminate gracefully.
      kill -SIGTERM "$pid"
    done

    # Wait for all background processes to actually terminate.
    # This ensures the script doesn't exit before the Python scripts have cleaned up.
    wait "${pids[@]}" 2>/dev/null

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] All processes have been terminated. Exiting watchdog."
    break # Exit the while loop.
  fi

  # If the market is still open, print a status and wait for 3 minutes.
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Market is open. Checking again in 3 minutes."
  sleep 180 # 180 seconds = 3 minutes
done
