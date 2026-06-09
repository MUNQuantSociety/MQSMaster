#!/bin/bash

# --- CONFIGURATION ---
# Find the absolute path of the directory where the script is located.
# This ensures that the script can be run from anywhere, including cron.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- !! ACTION REQUIRED !! ---
# Define the absolute path to the Python executable in your virtual environment.
# Defaults to the Docker image venv path and can be overridden via PYTHON_VENV.
PYTHON_VENV="${PYTHON_VENV:-/app/MQS/bin/python}"

# Load environment variables (like FMP_API_KEY) from the .env file.
# The .env file should be in the same directory as this script.
if [ -f "${SCRIPT_DIR}/.env" ]; then
    source "${SCRIPT_DIR}/.env"
else
    echo "[ERROR] .env file not found at ${SCRIPT_DIR}/.env. trying .env.example."
    if [ -f "${SCRIPT_DIR}/.env.example" ]; then
        cp "${SCRIPT_DIR}/.env.example" "${SCRIPT_DIR}/.env"
        echo "[WARNING] Created .env from .env.example. Please verify credentials are correct."
        source "${SCRIPT_DIR}/.env"
    else
        echo "[ERROR] .env.example file not found at ${SCRIPT_DIR}/.env.example. Exiting."
        exit 1
    fi
fi

# Set the exchange to monitor.
EXCHANGE="NASDAQ"

# Where persistent (24/7) script watchers write their stdout/stderr.
PERSISTENT_LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$PERSISTENT_LOG_DIR"

# Delay (seconds) between a persistent script crashing and the watcher restarting it.
PERSISTENT_RESTART_DELAY=30

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

# Spawn a script under a detached auto-restart watcher.
# - Skips spawn if the script is already running (avoids duplicates on daily re-runs).
# - Survives termination of this start.sh (nohup + disown).
# - Restarts the script on any non-zero exit with a fixed backoff delay.
spawn_persistent() {
  local script="$1"
  local script_name
  script_name=$(basename "$script" .py)
  local logfile="${PERSISTENT_LOG_DIR}/${script_name}.watcher.log"

  if pgrep -f "$script" > /dev/null 2>&1; then
    echo "  -> '$script' is already running. Skipping persistent spawn."
    return 0
  fi

  # nohup detaches from the controlling terminal (ignores SIGHUP).
  # The inner bash -c runs an infinite supervisor loop that respawns the
  # Python process after each exit. Variables are passed positionally so
  # the body can stay in single-quotes (no host-side substitution).
  nohup bash -c '
    SCRIPT_PATH="$1"
    PYTHON="$2"
    LOG="$3"
    RESTART_DELAY="$4"
    while true; do
      ts=$(date "+%Y-%m-%d %H:%M:%S")
      echo "[$ts] [persistent] Starting $SCRIPT_PATH" >> "$LOG"
      "$PYTHON" "$SCRIPT_PATH" >> "$LOG" 2>&1
      ec=$?
      ts=$(date "+%Y-%m-%d %H:%M:%S")
      echo "[$ts] [persistent] $SCRIPT_PATH exited code=$ec. Restarting in ${RESTART_DELAY}s..." >> "$LOG"
      sleep "$RESTART_DELAY"
    done
  ' bash "$script" "$PYTHON_VENV" "$logfile" "$PERSISTENT_RESTART_DELAY" >/dev/null 2>&1 &

  local pid=$!
  disown "$pid" 2>/dev/null
  echo "  -> Started persistent watcher for '$script' (watcher PID: $pid, log: $logfile)"
}

# --- SCRIPT START ---

# Change to the script's directory to ensure relative paths in Python scripts work correctly.
cd "$SCRIPT_DIR" || exit

# Make repo root importable so `import RBP.*`, `import src.*`, etc. resolve.
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/src:${PYTHONPATH}"

# Scripts that run ONLY during market hours and get killed on close.
market_pids=()
market_scripts=(
  "./src/main.py"
  "./src/orchestrator/realTime/realtimeDataIngestor.py"
  "./src/orchestrator/realTime/pnl_script.py"
  "./src/orchestrator/rbp_runner.py"
)

# Scripts that run 24/7 with auto-restart, detached from this start.sh.
# Even after the market-hours watchdog exits, these keep running and will
# survive crashes via the spawn_persistent supervisor loop.
persistent_scripts=(
  "./NLP/main_NLP.py"
)

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting persistent (24/7) processes..."
for script in "${persistent_scripts[@]}"; do
  spawn_persistent "$script"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting market-hours processes using Python from: ${PYTHON_VENV}"

# Start the market-hours scripts using the venv's python and add their PIDs to the array.
for script in "${market_scripts[@]}"; do
  "$PYTHON_VENV" "$script" &
  pid=$!
  # Small delay to see if the process crashes immediately
  sleep 1
  if ps -p $pid > /dev/null; then
    echo "  -> Started '$script' successfully with PID: $pid"
    market_pids+=($pid)
  else
    echo "[ERROR] Failed to start '$script'. Check the script for errors."
    # If one market-hours script fails, shut the other market-hours ones down.
    # Persistent scripts keep running in the background.
    echo "Shutting down other started market-hours processes."
    for p in "${market_pids[@]}"; do kill -SIGTERM "$p"; done
    exit 1
  fi
done

echo "All processes started successfully. Market PIDs: ${market_pids[@]}"

# --- MONITORING LOOP ---

while true; do
  # Check if the market is closed.
  if ! is_market_open; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Market is closed or API check failed. Shutting down MARKET-HOURS processes only."
    echo "(Persistent processes such as ${persistent_scripts[*]} remain running in the background.)"

    # Loop through stored market PIDs and send a termination signal to each.
    for pid in "${market_pids[@]}"; do
      # Check if the process still exists before trying to kill it
      if ps -p "$pid" > /dev/null; then
        echo "  -> Sending SIGTERM to process with PID: $pid"
        kill -SIGTERM "$pid"
      else
        echo "  -> Process with PID $pid no longer exists."
      fi
    done

    # Wait for the market-hours background processes to actually terminate.
    wait "${market_pids[@]}" 2>/dev/null

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Market-hours processes terminated. Exiting watchdog."
    break # Exit the while loop.
  fi

  # If the market is still open, wait for 3 minutes.
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Market is open. Checking again in 3 minutes."
  sleep 180 # 180 seconds = 3 minutes
done
