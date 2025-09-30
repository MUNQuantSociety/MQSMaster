#!/bin/bash

# --- CONFIG ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_VENV="/home/master/MQSMaster/MQS/bin/python"

# Load environment variables
if [ -f "${SCRIPT_DIR}/.env" ]; then
    source "${SCRIPT_DIR}/.env"
else
    echo "[ERROR] .env file not found at ${SCRIPT_DIR}/.env. Exiting."
    exit 1
fi

# --- DEFAULTS (can be overridden via env) ---
TICKER="${TICKER:-AAPL}"
START_DATE="${START_DATE:-2023-01-01}"
END_DATE="${END_DATE:-$(date +%Y-%m-%d)}"

# --- SCRIPTS TO RUN ---
pids=()
scripts_to_run=(
  "NLP/fetch_articles.py $TICKER $START_DATE $END_DATE"
  "automatedarticles.py"
)

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching article ingestion processes..."

for script in "${scripts_to_run[@]}"; do
  $PYTHON_VENV $SCRIPT_DIR/$script &
  pid=$!
  sleep 1
  if ps -p $pid > /dev/null; then
    echo "  -> Started '$script' (PID: $pid)"
    pids+=($pid)
  else
    echo "[ERROR] Failed to start '$script'. Shutting down others."
    for p in "${pids[@]}"; do kill -SIGTERM "$p"; done
    exit 1
  fi
done

# --- SHUTDOWN HANDLER ---
trap "echo 'Stopping all processes...'; for p in \"${pids[@]}\"; do kill -SIGTERM \"$p\"; done; exit 0" SIGINT SIGTERM

wait