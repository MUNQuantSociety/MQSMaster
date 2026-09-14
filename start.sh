#!/bin/bash

# --- CONFIGURATION ---
# Find the absolute path of the directory where the script is located.
# This ensures that the script can be run from anywhere, including cron.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Absolute path to the venv Python. Override by exporting PYTHON_VENV (or
# setting it in .env, sourced below); otherwise it resolves to the venv beside
# this script -- which is /app/MQS/bin/python in the Docker image and MQS/bin/
# python locally, since SCRIPT_DIR is the repo root in both.
#
# Anchored to SCRIPT_DIR, not $(pwd): the executable check below runs before the
# cd into SCRIPT_DIR, so a cwd-relative path breaks every invocation from
# outside this directory -- cron included.
PYTHON_VENV="${PYTHON_VENV:-${SCRIPT_DIR}/MQS/bin/python}"

# Credentials the app expects, as both .env keys and process env var names
# (they're deliberately identical -- see MQS_AWS_INFRA ssm-parameters module).
ENV_KEYS=(FMP_API_KEY ALPHA_KEY APIFY_KEY db_user password host port database sslmode)

# start.sh sources .env (or falls back to empty .env.example which
# would clobber the ECS-injected env vars). Materialise a real .env
# from the secrets ECS already injected, so source preserves them.
#
# .env is for local/dev use only -- it is gitignored and dockerignored, so it
# never exists in the container; ECS instead sets these as real process
# environment variables (task definition `environment`/`secrets`) before this
# script runs. The old fallback copied .env.example (every value blank) to
# .env and sourced THAT -- since .env is always missing in ECS, that path
# always ran, silently overwriting the credentials ECS had already injected
# with empty strings. Writing .env from the current environment instead keeps
# the single `source` codepath below correct in both places: locally it picks
# up whatever .env already has, and in ECS it round-trips the same values
# that were already set, rather than erasing them.
if [ ! -f "${SCRIPT_DIR}/.env" ]; then
    echo "[INFO] No .env file at ${SCRIPT_DIR}/.env -- materialising one from the current environment (e.g. ECS task secrets)."
    : > "${SCRIPT_DIR}/.env"
    for var in "${ENV_KEYS[@]}"; do
        if [ -n "${!var:-}" ]; then
            printf '%s=%q\n' "$var" "${!var}" >> "${SCRIPT_DIR}/.env"
        fi
    done
fi

echo "[INFO] Loading environment from ${SCRIPT_DIR}/.env."
source "${SCRIPT_DIR}/.env"

# Fail fast and loud if required credentials are missing, rather than limping
# along with blank values -- from a stale .env, a misconfigured ECS task
# definition, or an SSM parameter that never got pushed.
required_vars=(FMP_API_KEY db_user password host port database)
missing_vars=()
for var in "${required_vars[@]}"; do
    if [ -z "${!var:-}" ]; then
        missing_vars+=("$var")
    fi
done
if [ ${#missing_vars[@]} -gt 0 ]; then
    echo "[ERROR] Missing required environment variable(s): ${missing_vars[*]}. Exiting."
    exit 1
fi

# Set the exchange to monitor.
EXCHANGE="NASDAQ"

# Where persistent (24/7) script watchers write their stdout/stderr.
PERSISTENT_LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$PERSISTENT_LOG_DIR"

# Delay (seconds) between a persistent script crashing and the watcher restarting it.
PERSISTENT_RESTART_DELAY=30

# Seconds to wait after launching a market script before deciding it started
# cleanly. Import errors and credential failures surface well inside this.
MARKET_START_GRACE=${MARKET_START_GRACE:-3}

# How many consecutive "market status unknown" checks to tolerate before
# shutting the session down anyway. At the 3-minute poll interval, 20 is about
# an hour of FMP being unreachable. Set high enough that a transient outage
# does not end the trading day, low enough that processes cannot run forever
# against a permanently broken check.
UNKNOWN_STATUS_MAX_STREAK=${UNKNOWN_STATUS_MAX_STREAK:-20}

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
# Report whether the exchange is currently open.
#
# Three distinct outcomes, because conflating them is dangerous:
#   0 -> open
#   1 -> confirmed closed        (caller shuts the session down)
#   2 -> could not determine     (caller keeps running and retries)
#
# The previous version returned "closed" for API errors too, so an outage, an
# expired key, or a changed response shape silently ended the trading day.
#
# Shape-agnostic on purpose: FMP has served this payload both as a bare object
# and as a single-element array. Indexing with .[0] against an object makes jq
# fail with "Cannot index object with number", which the old code counted as
# "closed" -- meaning the check could never return true.
is_market_open() {
  local response
  response=$(curl -s --max-time 15 "https://financialmodelingprep.com/stable/exchange-market-hours?exchange=${EXCHANGE}&apikey=${FMP_API_KEY}")

  if [ -z "$response" ]; then
    echo "[WARNING] No response from FMP (network or timeout). Market status UNKNOWN."
    return 2
  fi

  if ! echo "$response" | jq -e . > /dev/null 2>&1; then
    echo "[WARNING] FMP response is not valid JSON. Market status UNKNOWN."
    echo "  Response: ${response:0:200}"
    return 2
  fi

  # FMP signals auth and quota problems as an object with an error key, which
  # would otherwise look like a malformed status payload.
  if echo "$response" | jq -e 'type == "object" and (has("Error Message") or has("error") or has("message"))' > /dev/null 2>&1; then
    echo "[WARNING] FMP returned an error (check FMP_API_KEY / quota). Market status UNKNOWN."
    echo "  Response: ${response:0:200}"
    return 2
  fi

  # Normalise array-or-object to a single object.
  local normalized
  normalized=$(echo "$response" | jq -c 'if type == "array" then .[0] else . end' 2>/dev/null)

  # Presence must be tested with has(), NOT `.isMarketOpen // empty`: jq's `//`
  # treats `false` as empty, so a genuinely closed market would be misreported
  # as UNKNOWN and the session would be kept alive past the close.
  if ! echo "$normalized" | jq -e 'type == "object" and has("isMarketOpen")' > /dev/null 2>&1; then
    echo "[WARNING] FMP response has no isMarketOpen field. Market status UNKNOWN."
    echo "  Response: ${response:0:200}"
    return 2
  fi

  # Only a JSON boolean is a trustworthy answer. `jq -r` would render the
  # *string* "true" identically to boolean true, so the type is checked here
  # rather than in the shell -- a changed payload shape must read as UNKNOWN,
  # not as an open market.
  local status
  status=$(echo "$normalized" | jq -r 'if (.isMarketOpen | type) == "boolean" then (.isMarketOpen | tostring) else "non-boolean (" + (.isMarketOpen | type) + ")" end')
  case "$status" in
    true)  return 0 ;;
    false) return 1 ;;
    *)
      echo "[WARNING] FMP isMarketOpen has unexpected value '$status'. Market status UNKNOWN."
      echo "  Response: ${response:0:200}"
      return 2
      ;;
  esac
}

# Names of scripts that never made it into the run list, and why. Reported in
# the startup summary so a missing or broken script is visible in the logs
# rather than silently absent.
declare -a SKIPPED_SCRIPTS=()
declare -a FAILED_SCRIPTS=()

# Decide whether a single script is fit to run.
#
# Each script is judged on its own. A script that fails here is recorded and
# skipped; every other script still starts. This is deliberate: these are
# independent workloads (data ingestion, PnL, RBP forecasting) and losing one
# is not a reason to lose the rest.
#
# Checks, cheapest first:
#   1. the file exists           -- catches .dockerignore omissions and typos
#   2. it parses                 -- catches syntax errors without executing it
#
# Import errors are NOT caught here: resolving them would mean executing
# module-level code, which for these scripts means opening DB connections and
# API sessions. Those surface at launch instead, handled by launch_market_script.
validate_script() {
  local script="$1"

  if [ ! -f "$script" ]; then
    echo "  [SKIP] '$script' -- file not found."
    echo "         If this ran locally but not in a container, check .dockerignore:"
    echo "         excluded directories are absent from the image."
    SKIPPED_SCRIPTS+=("$script (not found)")
    return 1
  fi

  if [ ! -r "$script" ]; then
    echo "  [SKIP] '$script' -- not readable (permissions)."
    SKIPPED_SCRIPTS+=("$script (unreadable)")
    return 1
  fi

  # py_compile parses and byte-compiles without importing, so no module-level
  # side effects. -q keeps normal runs quiet.
  local compile_err
  if ! compile_err=$("$PYTHON_VENV" -m py_compile "$script" 2>&1); then
    echo "  [SKIP] '$script' -- does not compile:"
    echo "$compile_err" | sed 's/^/         /'
    SKIPPED_SCRIPTS+=("$script (syntax error)")
    return 1
  fi

  return 0
}

# Launch one market-hours script and confirm it survives startup.
#
# Records the PID on success. On early exit, logs the failure and returns
# non-zero WITHOUT touching any other process -- the caller keeps going. The
# script's own traceback has already gone to stdout (and so to CloudWatch),
# which is where the cause will be.
launch_market_script() {
  local script="$1"

  "$PYTHON_VENV" "$script" &
  local pid=$!

  # Grace period for the interpreter to reach steady state. Import errors and
  # bad credentials surface well inside this window.
  sleep "$MARKET_START_GRACE"

  if kill -0 "$pid" 2>/dev/null; then
    echo "  [OK]   '$script' running (PID: $pid)"
    market_pids+=("$pid")
    return 0
  fi

  # Reap it so the exit status is accurate rather than "no such process".
  local ec
  wait "$pid" 2>/dev/null
  ec=$?
  echo "  [FAIL] '$script' exited within ${MARKET_START_GRACE}s (code=$ec)."
  echo "         Traceback above. Other market scripts are unaffected."
  FAILED_SCRIPTS+=("$script (exit $ec)")
  return 1
}

# Check whether any process's command line contains $1, without relying on
# procps (pgrep/ps) -- minimal base images often lack that package entirely,
# which would otherwise make this check silently pass every time. /proc is
# part of the kernel, not a package, so this works anywhere Linux does.
#
# Prints the first matching PID on success (find_pid_by_cmdline) -- used both
# for the plain running/not-running check below and to locate a persistent
# script's current PID for memory logging, since spawn_persistent's watcher
# restarts it under a new PID on every crash and nothing else tracks that PID.
find_pid_by_cmdline() {
  local pattern="$1"
  local pid_dir
  for pid_dir in /proc/[0-9]*; do
    [ -r "${pid_dir}/cmdline" ] || continue
    if tr '\0' ' ' < "${pid_dir}/cmdline" 2>/dev/null | grep -qF -- "$pattern"; then
      basename "$pid_dir"
      return 0
    fi
  done
  return 1
}

is_process_running() {
  find_pid_by_cmdline "$1" >/dev/null
}

# Log one process's resident memory (VmRSS, from /proc -- no procps dependency)
# plus its full command line so the log line is self-identifying even if the
# caller only had a PID. Silently skips a PID that's gone by the time it's
# read (process exited between discovery and this call) rather than erroring.
log_process_memory() {
  local pid="$1"
  local rss_kb cmdline
  rss_kb=$(awk '/^VmRSS:/{print $2}' "/proc/${pid}/status" 2>/dev/null)
  [ -n "$rss_kb" ] || return 0
  cmdline=$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null)
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] [mem] ${cmdline:-PID $pid} (PID $pid): $(( rss_kb / 1024 )) MB"
}

# Purely observational: logs memory for every tracked market-hours PID plus
# each configured persistent script (looked up fresh each call, per
# find_pid_by_cmdline's docstring above). Informs whether market_task_memory
# needs raising now that persistent_scripts (NLP) can run folded into this
# same task -- see ecs-task-market's header comment in MQS_AWS_INFRA -- without
# committing to a number ahead of having real data.
log_memory_usage() {
  local pid script
  for pid in "${market_pids[@]}"; do
    log_process_memory "$pid"
  done
  for script in "${persistent_to_run[@]}"; do
    pid=$(find_pid_by_cmdline "$script") && log_process_memory "$pid"
  done
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

  if is_process_running "$script"; then
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
  "./src/orchestrator/retention/prune_market_data.py"
)

check_db=(
  "./src/common/database/test.py"
  "./src/common/database/create_all_tables.py"
)

# --- PREFLIGHT: build the run lists -------------------------------------------
#
# Validate every script up front so the summary reports everything wrong at
# once, rather than surfacing problems one launch at a time. Only scripts that
# pass are added to the *_to_run lists.

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Validating scripts..."

check_db_to_run=()
for script in "${check_db[@]}"; do
  if validate_script "$script"; then
    check_db_to_run+=("$script")
  fi
done

persistent_to_run=()
for script in "${persistent_scripts[@]}"; do
  if validate_script "$script"; then
    persistent_to_run+=("$script")
  fi
done

market_to_run=()
for script in "${market_scripts[@]}"; do
  if validate_script "$script"; then
    market_to_run+=("$script")
  fi
done

echo "  Validated: ${#market_to_run[@]}/${#market_scripts[@]} market, ${#persistent_to_run[@]}/${#persistent_scripts[@]} persistent., ${#check_db_to_run[@]}/${#check_db[@]} DB scripts."

# --- LAUNCH -------------------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running pre-flight DB checks..."
for script in "${check_db_to_run[@]}"; do
  echo "  -> Running '$script'..."
  if ! "$PYTHON_VENV" "$script"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] Pre-flight DB check '$script' failed. Exiting."
    exit 1
  fi
done


echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting persistent (24/7) processes..."
for script in "${persistent_to_run[@]}"; do
  spawn_persistent "$script"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting market-hours processes using Python from: ${PYTHON_VENV}"
for script in "${market_to_run[@]}"; do
  # Failure is recorded inside and deliberately not propagated -- one script
  # dying must not take the others with it.
  launch_market_script "$script" || true
done

# --- STARTUP SUMMARY ----------------------------------------------------------
#
# One place to look to see what is actually running today. A degraded start is
# loud but not fatal: trading continues with whatever came up.

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ----- startup summary -----"
echo "  Market processes running: ${#market_pids[@]}/${#market_scripts[@]}"

if [ ${#SKIPPED_SCRIPTS[@]} -gt 0 ]; then
  echo "  [WARNING] Skipped (failed validation):"
  for entry in "${SKIPPED_SCRIPTS[@]}"; do echo "    - $entry"; done
fi

if [ ${#FAILED_SCRIPTS[@]} -gt 0 ]; then
  echo "  [WARNING] Failed to start (exited during grace period):"
  for entry in "${FAILED_SCRIPTS[@]}"; do echo "    - $entry"; done
fi

if [ ${#SKIPPED_SCRIPTS[@]} -eq 0 ] && [ ${#FAILED_SCRIPTS[@]} -eq 0 ]; then
  echo "  All scripts started cleanly."
fi
echo "  ---------------------------"

# Nothing running means there is no market session to watch, so exit non-zero
# and let ECS surface a failed task rather than idling in the monitor loop.
if [ ${#market_pids[@]} -eq 0 ]; then
  echo "[ERROR] No market-hours processes started. Nothing to monitor. Exiting."
  echo "(Persistent processes, if any, keep running detached.)"
  exit 1
fi

echo "Market PIDs: ${market_pids[*]}"

# Baseline reading right after launch, before the recurring one in the
# monitoring loop below -- startup memory isn't steady-state, but it's a
# useful T+0 anchor to compare later readings against.
log_memory_usage

# --- MONITORING LOOP ---

unknown_streak=0

while true; do
  is_market_open
  market_status=$?

  # Status 2 = could not determine. Never shut the session down on an API
  # problem; keep trading and retry. Only give up after a sustained blackout,
  # so a genuinely stuck check cannot leave processes running indefinitely.
  if [ "$market_status" -eq 2 ]; then
    unknown_streak=$(( unknown_streak + 1 ))
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Market status unknown (${unknown_streak}/${UNKNOWN_STATUS_MAX_STREAK}). Leaving processes running."

    if [ "$unknown_streak" -ge "$UNKNOWN_STATUS_MAX_STREAK" ]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] Market status unknown for ${unknown_streak} consecutive checks (~$(( unknown_streak * 3 )) min). Shutting down market processes as a safety measure."
      market_status=1
    else
      sleep 180
      continue
    fi
  else
    unknown_streak=0
  fi

  # Check if the market is closed.
  if [ "$market_status" -ne 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Market is closed. Shutting down MARKET-HOURS processes only."
    echo "(Persistent processes such as ${persistent_to_run[*]} remain running in the background.)"

    # Loop through stored market PIDs and send a termination signal to each.
    for pid in "${market_pids[@]}"; do
      # Check if the process still exists before trying to kill it
      if kill -0 "$pid" 2>/dev/null; then
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

  # Liveness sweep. A market script that dies mid-session would otherwise be
  # invisible until close -- the watchdog only tracked the market as a whole.
  # Report each loss once and keep monitoring the survivors.
  still_alive=()
  for pid in "${market_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      still_alive+=("$pid")
    else
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARNING] Market process PID $pid exited mid-session."
      echo "  Traceback above if it crashed. Other processes continue."
    fi
  done
  market_pids=("${still_alive[@]}")

  # Everything died before the close -- nothing left to watch.
  if [ ${#market_pids[@]} -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] All market processes exited before market close. Exiting watchdog."
    exit 1
  fi

  log_memory_usage

  # If the market is still open, wait for 3 minutes.
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Market is open (${#market_pids[@]} process(es) alive). Checking again in 3 minutes."
  sleep 180 # 180 seconds = 3 minutes
done
