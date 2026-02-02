#!/bin/bash

# scraper_daemon.sh
# Always-restart daemon for NLP article scraping with batched ticker processing
# Ensures the scraping script runs every 5 minutes with 3 batches and 2-minute intervals

# --- CONFIGURATION ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/fetch_articles.py"
LOG_FILE="${SCRIPT_DIR}/daemon.log"
PID_FILE="${SCRIPT_DIR}/scraper_daemon.pid"

# Python executable - adjust path as needed
PYTHON_VENV="${PROJECT_ROOT}/bin/python"
# Fallback to system python if venv not found
if [ ! -x "$PYTHON_VENV" ]; then
    PYTHON_VENV="python3"
fi

# Ticker batches (split into 3 groups)
BATCH_1=("AAPL" "MSFT" "GOOGL")
BATCH_2=("AMZN" "TSLA")
BATCH_3=("NVDA" "AMD")

# Timing configuration
SCRAPE_INTERVAL=300  # 5 minutes in seconds
BATCH_INTERVAL=120   # 2 minutes between batches
RESTART_DELAY=30     # seconds
HEALTH_CHECK_INTERVAL=60  # seconds
MAX_RESTART_ATTEMPTS=10

# --- FUNCTIONS ---

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

cleanup() {
    log_message "Cleaning up daemon..."
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            log_message "Terminating scraper process (PID: $pid)"
            kill -TERM "$pid" 2>/dev/null
            sleep 5
            if ps -p "$pid" > /dev/null 2>&1; then
                log_message "Force killing scraper process (PID: $pid)"
                kill -KILL "$pid" 2>/dev/null
            fi
        fi
        rm -f "$PID_FILE"
    fi
    log_message "Daemon cleanup completed"
    exit 0
}

check_dependencies() {
    # Check if Python script exists
    if [ ! -f "$PYTHON_SCRIPT" ]; then
        log_message "ERROR: Python script not found at $PYTHON_SCRIPT"
        exit 1
    fi

    # Check if Python executable is available
    if ! command -v "$PYTHON_VENV" &> /dev/null; then
        log_message "ERROR: Python executable not found: $PYTHON_VENV"
        exit 1
    fi

    # Test Python script syntax
    if ! "$PYTHON_VENV" -m py_compile "$PYTHON_SCRIPT" 2>/dev/null; then
        log_message "ERROR: Python script has syntax errors"
        exit 1
    fi

    log_message "All dependencies check passed"
}

run_ticker_batch() {
    local batch_name=$1
    shift
    local tickers=("$@")
    
    log_message "Starting batch $batch_name with tickers: ${tickers[*]}"
    
    # Get date range (last 7 days)
    local end_date=$(date '+%Y-%m-%d')
    local start_date=$(date -d '7 days ago' '+%Y-%m-%d')
    
    for ticker in "${tickers[@]}"; do
        log_message "Processing ticker: $ticker"
        
        cd "$PROJECT_ROOT" || exit 1
        
        # Step 1: Run fetch_articles.py for this ticker
        log_message "Step 1: Fetching articles for $ticker"
        "$PYTHON_VENV" -m NLP.fetch_articles "$ticker" "$start_date" "$end_date" >> "$LOG_FILE" 2>&1
        local fetch_exit_code=$?
        
        if [ $fetch_exit_code -eq 0 ]; then
            log_message "Successfully fetched articles for $ticker"
            
            # Step 2: Process sentiment and update database
            log_message "Step 2: Processing sentiment and updating database for $ticker"
            "$PYTHON_VENV" -m NLP.process_sentiment_pipeline "$ticker" >> "$LOG_FILE" 2>&1
            local sentiment_exit_code=$?
            
            if [ $sentiment_exit_code -eq 0 ]; then
                log_message "Successfully processed sentiment and updated database for $ticker"
            else
                log_message "ERROR: Failed to process sentiment for $ticker (exit code: $sentiment_exit_code)"
            fi
        else
            log_message "ERROR: Failed to fetch articles for $ticker (exit code: $fetch_exit_code)"
        fi
        
        # Small delay between tickers to avoid rate limiting
        sleep 10
    done
    
    log_message "Completed batch $batch_name"
}

run_scraping_cycle() {
    log_message "=== Starting new scraping cycle ==="
    local cycle_start=$(date +%s)
    
    # Run batch 1
    run_ticker_batch "1" "${BATCH_1[@]}"
    
    # Wait 2 minutes before batch 2
    log_message "Waiting $BATCH_INTERVAL seconds before batch 2..."
    sleep $BATCH_INTERVAL
    
    # Run batch 2
    run_ticker_batch "2" "${BATCH_2[@]}"
    
    # Wait 2 minutes before batch 3
    log_message "Waiting $BATCH_INTERVAL seconds before batch 3..."
    sleep $BATCH_INTERVAL
    
    # Run batch 3
    run_ticker_batch "3" "${BATCH_3[@]}"
    
    local cycle_end=$(date +%s)
    local cycle_duration=$((cycle_end - cycle_start))
    log_message "=== Scraping cycle completed in ${cycle_duration} seconds ==="
    
    return 0
}

start_scraper() {
    log_message "Starting NLP scraper daemon with batched processing..."
    
    # Create PID file for this process
    echo $$ > "$PID_FILE"
    
    while true; do
        local cycle_start=$(date +%s)
        
        # Run the scraping cycle
        if run_scraping_cycle; then
            log_message "Scraping cycle completed successfully"
        else
            log_message "ERROR: Scraping cycle failed"
        fi
        
        # Calculate how long to sleep until next cycle
        local cycle_end=$(date +%s)
        local cycle_duration=$((cycle_end - cycle_start))
        local sleep_time=$((SCRAPE_INTERVAL - cycle_duration))
        
        if [ $sleep_time -gt 0 ]; then
            log_message "Sleeping for $sleep_time seconds until next cycle..."
            sleep $sleep_time
        else
            log_message "WARNING: Cycle took longer than interval ($cycle_duration > $SCRAPE_INTERVAL seconds)"
        fi
    done
}

is_scraper_running() {
    if [ ! -f "$PID_FILE" ]; then
        return 1
    fi
    
    local pid=$(cat "$PID_FILE")
    if ps -p "$pid" > /dev/null 2>&1; then
        return 0
    else
        rm -f "$PID_FILE"
        return 1
    fi
}

monitor_scraper() {
    local restart_count=0
    
    log_message "Starting scraper monitoring with always-restart policy"
    
    while true; do
        if is_scraper_running; then
            # Scraper is running, reset restart counter
            restart_count=0
            sleep "$HEALTH_CHECK_INTERVAL"
        else
            # Scraper is not running
            log_message "Scraper process not running. Attempting restart..."
            
            if [ "$restart_count" -ge "$MAX_RESTART_ATTEMPTS" ]; then
                log_message "ERROR: Maximum restart attempts ($MAX_RESTART_ATTEMPTS) reached. Exiting."
                exit 1
            fi
            
            restart_count=$((restart_count + 1))
            log_message "Restart attempt $restart_count/$MAX_RESTART_ATTEMPTS"
            
            if start_scraper &; then
                log_message "Scraper restarted successfully"
            else
                log_message "Failed to restart scraper. Waiting ${RESTART_DELAY}s before next attempt..."
                sleep "$RESTART_DELAY"
            fi
        fi
    done
}

show_status() {
    if is_scraper_running; then
        local pid=$(cat "$PID_FILE")
        log_message "Scraper is running (PID: $pid)"
        
        # Show recent log entries
        echo "Recent log entries:"
        tail -10 "$LOG_FILE"
    else
        log_message "Scraper is not running"
    fi
}

show_usage() {
    echo "Usage: $0 {start|stop|restart|status|monitor}"
    echo ""
    echo "Commands:"
    echo "  start    - Start the scraper daemon"
    echo "  stop     - Stop the scraper daemon"
    echo "  restart  - Restart the scraper daemon"
    echo "  status   - Show daemon status"
    echo "  monitor  - Start monitoring with always-restart policy"
    echo ""
    echo "Batched Processing:"
    echo "  - Batch 1: ${BATCH_1[*]}"
    echo "  - Batch 2: ${BATCH_2[*]}"
    echo "  - Batch 3: ${BATCH_3[*]}"
    echo "  - Interval: Every $SCRAPE_INTERVAL seconds"
    echo "  - Batch delay: $BATCH_INTERVAL seconds between batches"
}

# --- SIGNAL HANDLERS ---
trap cleanup SIGTERM SIGINT

# --- MAIN SCRIPT ---

# Create log file if it doesn't exist
touch "$LOG_FILE"

case "${1:-}" in
    start)
        log_message "=== Starting NLP Scraper Daemon ==="
        check_dependencies
        
        if is_scraper_running; then
            log_message "Scraper is already running"
            exit 1
        fi
        
        start_scraper &
        ;;
        
    stop)
        log_message "=== Stopping NLP Scraper Daemon ==="
        cleanup
        ;;
        
    restart)
        log_message "=== Restarting NLP Scraper Daemon ==="
        cleanup
        sleep 2
        check_dependencies
        start_scraper &
        ;;
        
    status)
        show_status
        ;;
        
    monitor)
        log_message "=== Starting NLP Scraper Monitor ==="
        check_dependencies
        
        # Start scraper if not running
        if ! is_scraper_running; then
            start_scraper &
        fi
        
        # Start monitoring loop
        monitor_scraper
        ;;
        
    *)
        show_usage
        exit 1
        ;;
esac