#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/.env"
EXCHANGE="NASDAQ"

# Function to check if the market is open
is_market_open() {
  curl -s "https://financialmodelingprep.com/stable/exchange-market-hours?exchange=${EXCHANGE}&apikey=${FMP_API_KEY}" \
    | jq -e '.[0].isMarketOpen' > /dev/null
}

# Start the engine in the background
python3 main.py &
engine_pid=$!

# check every 30 minutes is market is open
while true; do
sleep 20
  if ! is_market_open; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Market is closed."
    kill -SIGTERM "$engine_pid"
    wait "$engine_pid" 2>/dev/null
    echo "breaking"
    break
  fi

  sleep 1800  # 30 minutes
done