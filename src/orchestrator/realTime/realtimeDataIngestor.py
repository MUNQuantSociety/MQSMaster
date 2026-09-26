import sys
import os
import time
import logging
import pandas as pd
import pytz
from datetime import datetime, time as dtime

try:
    from common.database.MQSDBConnector import MQSDBConnector
    from orchestrator.marketData.fmpMarketData import FMPMarketData
    from orchestrator.realTime.utils import load_tickers
except ImportError:
    from src.common.database.MQSDBConnector import MQSDBConnector
    from src.orchestrator.marketData.fmpMarketData import FMPMarketData
    from src.orchestrator.realTime.utils import load_tickers

# --- Configuration ---
LOG_FILE = os.environ.get('MARKET_DATA_INGESTOR_LOG', '/var/log/market_data_ingestor.log')
MARKET_OPEN = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)
FETCH_INTERVAL_SECONDS = 60
DB_TABLE_NAME = "market_data"
TIMEZONE = pytz.timezone("America/New_York")

# Every FMP quote feed we poll each cycle. Equity exchanges go through
# batch-exchange-quote; crypto and commodities have their own batch endpoints.
# The label doubles as the fallback `exchange` for rows FMP returns without one.
EQUITY_EXCHANGES = ("NASDAQ", "NYSE", "AMEX")
ASSET_CLASS_FEEDS = ("CRYPTO", "COMMODITY")
QUOTE_FEEDS = EQUITY_EXCHANGES + ASSET_CLASS_FEEDS

# Columns the upsert relies on; verified against the live schema at startup.
UPSERT_CONFLICT_COLUMNS = ("ticker", "timestamp")

# Uncovered tickers are listed by name only every N cycles to keep the log readable.
UNCOVERED_REPORT_EVERY_CYCLES = 30


def setup_logging():
    """
    Logs to stderr always (start.sh surfaces it) and to LOG_FILE when writable.
    A read-only /var/log on a dev box must not stop the ingestor from starting.
    """
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s')
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # MQSDBConnector calls logging.basicConfig at import time; drop that handler
    # or every line is emitted twice.
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    try:
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
    except OSError as e:
        root.warning(f"Log file {LOG_FILE} not writable ({e}); logging to stderr only.")


def verify_upsert_constraint(db: MQSDBConnector) -> bool:
    """
    Confirms a UNIQUE index on (ticker, timestamp) exists on the target table.

    `bulk_inject_to_db` uses ON CONFLICT (ticker, timestamp) DO NOTHING; without a
    matching unique index PostgreSQL rejects every insert, so we refuse to start
    rather than fail on every cycle. Returns False only when the catalog query
    succeeded and no such index exists; an inconclusive query returns True with a
    CRITICAL log. Column order inside the index does not matter for conflict
    inference, so we compare as sets.
    """
    sql = f"""
    SELECT i.relname AS index_name,
           array_agg(a.attname::text ORDER BY a.attnum) AS columns
    FROM pg_index ix
    JOIN pg_class t ON t.oid = ix.indrelid
    JOIN pg_class i ON i.oid = ix.indexrelid
    JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(ix.indkey)
    WHERE t.relname = '{DB_TABLE_NAME}' AND ix.indisunique
    GROUP BY i.relname;
    """
    result = db.read_db(sql=sql)
    if result['status'] != 'success':
        # Inconclusive, not a proven absence: refusing here would turn a transient
        # DB blip at open into a lost trading day (market scripts are not
        # auto-restarted). The first insert will fail loudly if the index is missing.
        logging.critical(
            f"Could not inspect indexes on {DB_TABLE_NAME} ({result.get('message')}); "
            f"proceeding without verifying the upsert constraint."
        )
        return True

    wanted = set(UPSERT_CONFLICT_COLUMNS)
    for row in result['data'] or []:
        if set(row['columns']) == wanted:
            logging.info(
                f"Upsert constraint OK: unique index '{row['index_name']}' on "
                f"{DB_TABLE_NAME}({', '.join(UPSERT_CONFLICT_COLUMNS)})."
            )
            return True

    logging.critical(
        f"No UNIQUE index on {DB_TABLE_NAME}({', '.join(UPSERT_CONFLICT_COLUMNS)}). "
        f"ON CONFLICT upserts would fail on every cycle. Create it with:\n"
        f"  CREATE UNIQUE INDEX market_data_ticker_timestamp_key "
        f"ON {DB_TABLE_NAME} (ticker, \"timestamp\");"
    )
    return False


# --- WARNING ---
# The function below is NOT robust to script crashes. It attempts to re-initialize
# state by reading the `volume` column, but that column stores the *interval* volume,
# not the *cumulative* volume from the API. For this to work correctly after a crash,
# the database schema would need a separate 'cumulative_volume' column to read from.

def initialize_volume_state(db: MQSDBConnector, tickers: set) -> dict:
    """
    Initializes the volume state from the DB for the current day to ensure
    continuity if the script restarts.
    """
    logging.info("Initializing volume state from database for today...")
    today_date = datetime.now(TIMEZONE).date()

    sql = f"""
    WITH LatestEntries AS (
        SELECT
            ticker,
            volume,
            RANK() OVER(PARTITION BY ticker ORDER BY timestamp DESC) as rnk
        FROM {DB_TABLE_NAME}
        WHERE date = '{today_date}' AND ticker = ANY(ARRAY{list(tickers)})
    )
    SELECT ticker, volume FROM LatestEntries WHERE rnk = 1;
    """

    result = db.read_db(sql=sql)

    if result['status'] == 'success' and result['data']:
        state = {row['ticker']: row['volume'] for row in result['data']}
        logging.warning(
            f"Resumed mid-day: seeded volume state for {len(state)} tickers from stored "
            f"*interval* volume, so each ticker's first bar this session will overstate "
            f"volume (see WARNING above initialize_volume_state)."
        )
        return state

    logging.info("No rows for today yet; starting with fresh volume state.")
    return {}


def process_market_data(api_data: list, tickers_to_track: set, last_known_volumes: dict,
                        feed_label: str = None) -> list:
    """
    Transforms raw API data into a format ready for database injection using
    vectorized pandas operations.

    `feed_label` is the feed the rows came from (e.g. "NYSE"); it is the fallback
    `exchange` for rows FMP returns with a null exchange, since the column is NOT NULL
    and one bad row would roll back the whole batch.
    """
    if not api_data:
        logging.info(f"[{feed_label}] API returned no data to process.")
        return []

    # 1. Load data and filter for tracked tickers
    df = pd.DataFrame(api_data)
    df = df[df['symbol'].isin(tickers_to_track)].copy()

    if df.empty:
        logging.info(f"[{feed_label}] No tracked tickers in this feed ({len(api_data)} rows returned).")
        return []

    # 2. Exchange: prefer FMP's per-row value, fall back to the feed label.
    if 'exchange' not in df.columns:
        df['exchange'] = None
    null_exchange = df['exchange'].isna()
    if null_exchange.any():
        if feed_label is None:
            logging.warning(
                f"Dropping {int(null_exchange.sum())} rows with null exchange and no feed label: "
                f"{sorted(df.loc[null_exchange, 'symbol'])}"
            )
            df = df[~null_exchange].copy()
        else:
            logging.warning(
                f"[{feed_label}] {int(null_exchange.sum())} rows had null exchange; "
                f"assigned '{feed_label}': {sorted(df.loc[null_exchange, 'symbol'])}"
            )
            df.loc[null_exchange, 'exchange'] = feed_label

    # 3. Rows without a price carry nothing useful; rows without volume still carry a price.
    if 'price' not in df.columns:
        df['price'] = None
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    no_price = df['price'].isna()
    if no_price.any():
        logging.warning(
            f"[{feed_label}] Dropping {int(no_price.sum())} rows with null price: "
            f"{sorted(df.loc[no_price, 'symbol'])}"
        )
        df = df[~no_price].copy()
    if df.empty:
        return []

    if 'volume' not in df.columns:
        df['volume'] = 0
    # Explicit float dtype: FMP returns None for some crypto volumes, which would
    # otherwise leave an object column whose fillna trips pandas' downcasting warning.
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').astype('float64')
    no_volume = df['volume'].isna()
    if no_volume.any():
        logging.warning(
            f"[{feed_label}] {int(no_volume.sum())} rows had null volume; recorded as 0: "
            f"{sorted(df.loc[no_volume, 'symbol'])}"
        )
        df['volume'] = df['volume'].fillna(0)

    # 4. Vectorized Interval Volume Calculation
    df_last_volumes = pd.DataFrame(list(last_known_volumes.items()), columns=['symbol', 'last_volume'])
    df_last_volumes['last_volume'] = pd.to_numeric(df_last_volumes['last_volume'], errors='coerce')
    df = pd.merge(df, df_last_volumes, on='symbol', how='left')
    # Avoid inplace=True by reassigning the column.
    df['last_volume'] = df['last_volume'].astype('float64').fillna(0)
    df['interval_volume'] = df['volume'] - df['last_volume']
    reset_mask = df['interval_volume'] < 0
    if reset_mask.any():
        logging.info(
            f"[{feed_label}] Cumulative volume went backwards for {int(reset_mask.sum())} tickers "
            f"(new session or rolling-window feed); using the full API volume as the interval."
        )
    df.loc[reset_mask, 'interval_volume'] = df['volume']

    # 5. Update the state for the next cycle (Optimized)
    # This vectorized approach is much faster than iterating.
    # It uses the cumulative volume from the API before it's overwritten.
    new_state = pd.Series(df.volume.values, index=df.symbol).to_dict()
    last_known_volumes.update(new_state)

    # 6. Format and align with DB schema
    df.rename(columns={
        'symbol': 'ticker',
        'price': 'close_price',
    }, inplace=True)
    df['volume'] = df['interval_volume'].round().astype('int64') # Overwrite cumulative with interval volume

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s').dt.tz_localize('UTC').dt.tz_convert(TIMEZONE).dt.round('min')
    df['date'] = df['timestamp'].dt.date

    # FMP's `open`/`dayHigh`/`dayLow` are session-level aggregates, not minute-bar
    # values; writing them here would fabricate OHL for every bar of the day.
    # Only backfilled bars (intraday endpoint) carry real per-interval OHL.
    df['open_price'] = None
    df['high_price'] = None
    df['low_price'] = None

    db_columns = [
        "ticker", "timestamp", "date", "exchange",
        "open_price", "high_price", "low_price", "close_price", "volume"
    ]

    before = len(df)
    df.dropna(subset=['timestamp', 'ticker'], inplace=True)
    if len(df) < before:
        logging.warning(f"[{feed_label}] Dropped {before - len(df)} rows with null timestamp.")

    logging.info(
        f"[{feed_label}] feed={len(api_data)} rows, tracked={len(df)}, "
        f"exchanges={df['exchange'].value_counts().to_dict()}"
    )
    return df[db_columns].to_dict('records')


def fetch_feed(fmp: FMPMarketData, feed_label: str):
    """Routes a feed label to the matching FMP batch endpoint."""
    if feed_label in EQUITY_EXCHANGES:
        return fmp.get_realtime_data(feed_label)
    if feed_label == "CRYPTO":
        return fmp.get_batch_crypto_quotes()
    if feed_label == "COMMODITY":
        return fmp.get_batch_commodity_quotes()
    raise ValueError(f"Unknown quote feed: {feed_label}")


def run_ingestion_cycle(fmp: FMPMarketData, db: MQSDBConnector, tickers_to_track: set,
                        volume_state: dict, cycle: int = 0) -> set:
    """
    Executes a single fetch-process-inject cycle across every quote feed.

    Returns the set of tracked tickers no feed reported this cycle. A ticker that
    appears in one feed is removed from the lookup for later feeds so a symbol
    listed twice is never processed (and its volume state advanced) twice.
    """
    logging.info(f"--- Ingestion cycle {cycle} ---")
    remaining = set(tickers_to_track)
    rows_to_insert = []
    failed_feeds = []

    for feed_label in QUOTE_FEEDS:
        if not remaining:
            break
        started = time.time()
        api_data = fetch_feed(fmp, feed_label)
        elapsed = time.time() - started

        if api_data is None:
            logging.warning(f"[{feed_label}] Fetch failed after {elapsed:.2f}s; feed skipped this cycle.")
            failed_feeds.append(feed_label)
            continue

        rows = process_market_data(api_data, remaining, volume_state, feed_label=feed_label)
        remaining -= {row['ticker'] for row in rows}
        rows_to_insert.extend(rows)
        logging.info(f"[{feed_label}] fetched in {elapsed:.2f}s, {len(rows)} rows prepared.")

    if not rows_to_insert:
        logging.warning(
            f"Cycle {cycle}: no rows to insert "
            f"(failed feeds: {failed_feeds or 'none'}, uncovered tickers: {len(remaining)})."
        )
        return remaining

    result = db.bulk_inject_to_db(DB_TABLE_NAME, rows_to_insert,
                                  conflict_columns=list(UPSERT_CONFLICT_COLUMNS))
    if result["status"] == "success":
        logging.info(
            f"Cycle {cycle}: prepared={len(rows_to_insert)} "
            f"inserted={result.get('inserted_count', '?')} "
            f"ignored={result.get('ignored_count', '?')} (unchanged quote timestamp) "
            f"failed_feeds={failed_feeds or 'none'} uncovered={len(remaining)}"
        )
    else:
        logging.error(
            f"Cycle {cycle}: database injection of {len(rows_to_insert)} rows failed, "
            f"whole batch rolled back: {result['message']}"
        )
    return remaining


def main():
    """
    Main data-collection loop. Fetches bulk exchange data, processes it,
    and performs an efficient bulk insert into the database.
    """
    setup_logging()
    logging.info("======= Starting Real-Time Data Ingestor =======")
    logging.info(
        f"Config: feeds={list(QUOTE_FEEDS)} interval={FETCH_INTERVAL_SECONDS}s "
        f"table={DB_TABLE_NAME} tz={TIMEZONE.zone} log={LOG_FILE}"
    )

    db = MQSDBConnector()
    fmp = FMPMarketData()

    try:
        if not verify_upsert_constraint(db):
            logging.critical("Refusing to start without the upsert constraint. Exiting.")
            sys.exit(1)

        tickers_to_track = set(load_tickers())
        if not tickers_to_track:
            logging.error("No tickers loaded from backfill/tickers.json. Exiting.")
            sys.exit(1)
        logging.info(f"Tracking {len(tickers_to_track)} tickers from backfill/tickers.json.")

        volume_state = initialize_volume_state(db, tickers_to_track)

        cycle = 0
        while True:
            cycle += 1
            start_time = time.time()
            uncovered = run_ingestion_cycle(fmp, db, tickers_to_track, volume_state, cycle=cycle)

            # First cycle and then periodically: name the tickers no feed serves so a
            # stale tickers.json entry is visible without grepping the DB.
            if uncovered and (cycle == 1 or cycle % UNCOVERED_REPORT_EVERY_CYCLES == 0):
                logging.warning(
                    f"{len(uncovered)} tracked tickers not present in any feed: {sorted(uncovered)}"
                )

            elapsed_time = time.time() - start_time
            sleep_time = max(0, FETCH_INTERVAL_SECONDS - elapsed_time)
            if sleep_time == 0:
                logging.warning(
                    f"Cycle {cycle} took {elapsed_time:.2f}s, longer than the "
                    f"{FETCH_INTERVAL_SECONDS}s interval; running behind."
                )
            else:
                logging.info(f"Cycle {cycle} finished in {elapsed_time:.2f}s. Sleeping for {sleep_time:.2f}s.")
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        logging.info("Script manually interrupted.")
    except SystemExit:
        # Deliberate exit (missing constraint, no tickers): keep the non-zero code
        # so start.sh reports [FAIL] instead of a clean stop.
        raise
    except Exception as e:
        logging.critical(f"A critical error occurred in the main loop: {e}", exc_info=True)
    finally:
        db.close_all_connections()
        logging.info("======= Real-Time Data Ingestor Stopped =======")

if __name__ == '__main__':
    main()
