"""market_data retention pruner.

Persistent script (run via start.sh's persistent_scripts, alongside
NLP/main_NLP.py in the same container) that keeps market_data capped at a
rolling RETENTION_DAYS window. Touches no other table.

Runs forever, checking roughly once a day, but only actually prunes about
once a year: "last successful run" is tracked in an SSM Standard String
parameter (not a secret -- see MQS_AWS_INFRA modules/Livetrading/job-state),
not a local file. The market task's container is a fresh filesystem on
every single scheduled run (not just occasional redeploys), so any
local "last ran" state would be lost daily and defeat the once-a-year
gate entirely.

Note: this opens its own MQSDBConnector -- its own pooled connection,
held for the process lifetime -- alongside whatever pool NLPRunner holds
in the same container. That's deliberate, not an oversight: keeping the
two persistent scripts' DB access independent avoids coupling their
lifecycles together.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime, timedelta

import boto3
import pytz
from botocore.exceptions import ClientError

try:
    from common.database.MQSDBConnector import MQSDBConnector
except ImportError:
    from src.common.database.MQSDBConnector import MQSDBConnector

# --- Tunables -----------------------------------------------------------
# Module-level constants with env-var override, matching NLP/runner.py's
# SCRAPE_INTERVAL convention. All have sane defaults -- none are required --
# because CI clones whatever task definition is CURRENTLY LIVE and only
# swaps the image tag, so a Terraform-side env var addition does not reach
# a running container without a one-time manual bootstrap (see
# modules/Livetrading/ecs-task-market). The script must work standalone.

# Rolling window of market_data to keep. ~1.5 years by default.
RETENTION_DAYS = int(os.getenv("MARKET_DATA_RETENTION_DAYS", "545"))

# How often to wake up and check whether it's time to prune. Once a day is
# plenty -- the gate only cares about year-granularity elapsed time, and
# this keeps SSM GetParameter calls cheap and infrequent.
CHECK_INTERVAL_SECONDS = int(os.getenv("MARKET_DATA_PRUNE_CHECK_INTERVAL", str(24 * 3600)))

# Minimum days since the last successful prune before doing another.
MIN_INTERVAL_DAYS = int(os.getenv("MARKET_DATA_PRUNE_MIN_INTERVAL_DAYS", "365"))

# Rows deleted per batch (subquery-limited DELETE, looped). Keeps any single
# transaction short on a table that could hold years of minute bars.
BATCH_SIZE = int(os.getenv("MARKET_DATA_PRUNE_BATCH_SIZE", "5000"))

# SSM parameter holding the last successful run date (YYYY-MM-DD). Default
# matches the name Terraform's job-state module creates literally, so this
# works with zero env vars set.
SSM_PARAM_NAME = os.getenv(
    "MARKET_DATA_PRUNE_SSM_PARAM",
    "/mqsmaster-prod/jobs/market_data_prune_last_run",
)

# Fargate does not auto-populate AWS_REGION into the container (unlike
# Lambda), so this needs its own default.
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")

# All timestamps in this system are normalized to America/New_York (see
# CLAUDE.md) -- "today" here follows that convention rather than UTC or the
# container's system local time, so the retention window lines up with the
# rest of the codebase's notion of a trading day.
_NY_TZ = pytz.timezone("America/New_York")


def _today() -> date:
    return datetime.now(_NY_TZ).date()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("market_data_retention")


def should_prune(last_run: date | None, today: date, min_interval_days: int = MIN_INTERVAL_DAYS) -> bool:
    """Pure gate function -- no AWS/DB calls, fully unit-testable.

    True if never run before (last_run is None -- the seed value 1970-01-01
    reads back as this on first-ever deploy, so the first run prunes
    immediately rather than silently deferring a year) or if at least
    min_interval_days have elapsed since the last successful run.
    """
    if last_run is None:
        return True
    return (today - last_run).days >= min_interval_days


def _get_last_run_date(ssm_client) -> tuple[date | None, bool]:
    """Reads the last-run date from SSM.

    Returns (date_or_None, ok). ok is False for a ClientError other than
    ParameterNotFound (throttling, transient network, permissions) -- those
    must NOT be treated as "never ran", or a transient AWS hiccup would
    trigger a full prune attempt on every check until it clears. Only a
    genuine ParameterNotFound (first-ever run) maps to (None, True).
    """
    try:
        response = ssm_client.get_parameter(Name=SSM_PARAM_NAME)
        value = response["Parameter"]["Value"]
        return datetime.strptime(value, "%Y-%m-%d").date(), True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code == "ParameterNotFound":
            return None, True
        logger.warning("Could not read %s (%s); skipping this cycle.", SSM_PARAM_NAME, code)
        return None, False


def _set_last_run_date(ssm_client, today: date) -> None:
    ssm_client.put_parameter(
        Name=SSM_PARAM_NAME,
        Value=today.isoformat(),
        Type="String",
        Overwrite=True,
    )


def _prune_once(db: MQSDBConnector, today: date, retention_days: int, batch_size: int) -> int:
    """Deletes market_data rows older than retention_days, in batches.

    Batched via a subquery (Postgres DELETE has no LIMIT) so no single
    transaction holds a long-running lock on a table that could span years
    of minute bars. Stops when a batch deletes 0 rows or errors.
    """
    cutoff = today - timedelta(days=retention_days)
    delete_sql = """
        DELETE FROM market_data
        WHERE id IN (SELECT id FROM market_data WHERE date < %s LIMIT %s)
        RETURNING id
    """
    total_deleted = 0
    while True:
        result = db.execute_query(delete_sql, (cutoff, batch_size), fetch=True)
        if result.get("status") == "error":
            raise RuntimeError(f"Batch delete failed: {result.get('message')}")
        batch_deleted = len(result.get("data") or [])
        total_deleted += batch_deleted
        if batch_deleted < batch_size:
            break
    return total_deleted


def run_cycle(db: MQSDBConnector, ssm_client) -> None:
    today = _today()
    last_run, ok = _get_last_run_date(ssm_client)
    if not ok:
        return
    if not should_prune(last_run, today, MIN_INTERVAL_DAYS):
        return

    logger.info(
        "Retention window elapsed (last_run=%s) -- pruning market_data older than %d days.",
        last_run,
        RETENTION_DAYS,
    )
    deleted = _prune_once(db, today, RETENTION_DAYS, BATCH_SIZE)

    # Delete first, record state second: the delete is idempotent (re-running
    # it is harmless), but writing "done" before a successful prune risks
    # marking a failed/partial run as complete and losing a year of retention.
    _set_last_run_date(ssm_client, today)
    logger.info("Prune complete: deleted %d row(s), retention=%dd, last_run=%s", deleted, RETENTION_DAYS, today)


def main() -> None:
    logger.info(
        "market_data retention pruner started (retention=%dd, min_interval=%dd, check every %ds, param=%s)",
        RETENTION_DAYS,
        MIN_INTERVAL_DAYS,
        CHECK_INTERVAL_SECONDS,
        SSM_PARAM_NAME,
    )
    db = MQSDBConnector()
    ssm_client = boto3.client("ssm", region_name=AWS_REGION)
    while True:
        try:
            run_cycle(db, ssm_client)
        except KeyboardInterrupt:
            logger.info("Stopped by user.")
            break
        except Exception:
            logger.exception("Error in retention cycle; will retry next interval.")
        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
