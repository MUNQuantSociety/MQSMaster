Backfill CLI Documentation
==========================

This directory provides a unified command‑line interface for market data backfilling and related ingestion workflows.

Primary entrypoint: `python -m src.orchestrator.backfill.backfill_cli <command> [options]`

Commands Overview
-----------------
1. `specific`   – Backfill a continuous date range for one or more tickers.
2. `concurrent` – Backfill multiple tickers in parallel (threaded workers).
3. `inject-csv` – Load previously downloaded CSV bar data from disk into the database.

Date Format
-----------
All `--start` / `--end` arguments use `DDMMYY` (e.g., `040325` for 4 March 2025). They are parsed into `date` objects internally.

Supported Intervals
-------------------
Bar interval (minutes): `1, 5, 15, 30, 60` (validated). If the upstream data provider limits availability at certain intervals you may see empty responses; those will be reported per ticker.

Common Arguments
----------------
These are shared across (most) subcommands:

* `--start DDMMYY` – Start date (inclusive)
* `--end DDMMYY` – End date (inclusive)

* `--tickers T1 T2 ...` – Explicit list (otherwise falls back to a `tickers.json` near the CLI if present)

* `--exchange EXCH` – Exchange code (default: `NASDAQ`)

* `--interval N` – Interval in minutes (default: `1`)

* `--dry-run` – Fetch & parse but skip DB insertion (still logs counts)

* `--output-filename FILE` – If supported by downstream fetch, writes combined CSV (current implementation: only honored by the underlying `backfill_data` path; concurrent mode may ignore)

* `--log-level LEVEL` – One of `DEBUG|INFO|WARNING|ERROR` (default: `INFO`)
* `--on-conflict MODE` – `fail` (raise / rely on unique constraint) or `ignore` (append `ON CONFLICT DO NOTHING`). Default: `fail`.

Command-Specific Arguments
--------------------------
`specific`:
* (inherits all common arguments)
* Intended for a modest number of tickers; processes sequentially.

`concurrent`:
* `--threads INT` – Max worker threads (default: 6). Do not exceed the database connection pool size (or available outbound API quota) to avoid throttling.

`inject-csv`:
* `--csv-dir DIR` – Folder containing CSV dumps matching the loader’s expected schema.
* `--threads INT` – Worker threads for parallel file ingestion (default: 5).

Exit Codes
----------
* `0` – Successful run (even if some tickers returned no data).
* Non‑zero – CLI argument errors or unrecoverable ingestion exceptions.

Examples
--------
Backfill two tickers over January 2025 (1‑minute bars) without inserting (dry run):
```
python -m src.orchestrator.backfill.backfill_cli specific \
	--start 010125 --end 310125 --tickers AAPL MSFT \
	--interval 1 --dry-run --on-conflict ignore --log-level INFO
```

Insert data for one ticker and also produce a CSV file:
```
python -m src.orchestrator.backfill.backfill_cli specific \
	--start 010125 --end 070125 --tickers NVDA \
	--interval 5 --on-conflict ignore --output-filename nvda_0101_0701.csv
```

Concurrent multi‑ticker ingest (8 threads):
```
python -m src.orchestrator.backfill.backfill_cli concurrent \
	--start 010125 --end 310125 --tickers AAPL MSFT AMD INTC META \
	--interval 1 --threads 8 --on-conflict ignore
```

Dry run for all tickers in `tickers.json` (no explicit tickers passed):
```
python -m src.orchestrator.backfill.backfill_cli specific \
	--start 010125 --end 050125 --interval 15 --dry-run
```

CSV Injection:
```
python -m src.orchestrator.backfill.backfill_cli inject-csv \
	--csv-dir ./historical_csv --threads 8
```

Conflict Handling (`--on-conflict`)
----------------------------------
* `fail` (default): Relies on a unique constraint (e.g. `(ticker, timestamp)`). Duplicate attempts raise an error (or the database rejects the batch). Best for detecting unexpected duplication early.
* `ignore`: Appends `ON CONFLICT (ticker, timestamp) DO NOTHING` to inserts. Duplicate rows are silently skipped. Row counts in logs under this mode reflect “prepared” rows (not necessarily committed if duplicates existed).

Dry Run Mode (`--dry-run`)
-------------------------
* Fetches data and builds insertion batches but does NOT write to the database.
* Useful for verifying availability, API quotas, and approximate row volumes.
* Gap fill (if/when integrated as a flag) should be skipped in dry run to avoid extraneous API calls.

Performance & Concurrency
-------------------------
* Set `--threads` conservatively based on:
	- DB connection pool size.
	- Remote API rate limits.
	- Local CPU/network characteristics.
* Too many threads can increase contention, add retries, or lead to throttling.

Output CSV (`--output-filename`)
--------------------------------
* The underlying fetch routine may write *one* CSV per invocation (not per ticker) depending on implementation.
* In concurrent mode, a single aggregated CSV output is usually not produced unless explicitly implemented; treat this flag as best-effort.

Environment / Prerequisites
---------------------------
* Python environment with project dependencies installed (see project root `requirements.txt` / `pyproject.toml`).
* Database configured (PostgreSQL assumed) with `market_data` table and UNIQUE constraint on `(ticker, timestamp)` if using `--on-conflict ignore` reliably.
* API key(s) loaded via environment variables (e.g., `FMP_API_KEY`) or a `.env` file if the underlying data provider requires authentication.
* A `tickers.json` file located alongside the CLI, structured as a JSON list, e.g.: `['AAPL', 'MSFT', 'GOOGL']`.

Logging
-------
Use `--log-level DEBUG` during troubleshooting to surface raw responses, SQL preparation counts, and exception traces. In production, keep `INFO` or higher.

Typical Log Lines
-----------------
* Per ticker completion: `[AAPL] Inserted in 3.27s (ins=12345 skip=0)`
* Final summary: `Summary: tickers=5 inserted=56789 skipped=12 elapsed=0:01:42`

Return / Exit Strategy
----------------------
The CLI presently logs a summary; it does not (yet) emit machine‑parsable JSON. If you need structured output for orchestration, consider wrapping the module or extending it with a `--summary-json` flag (planned enhancement).

Troubleshooting
---------------
| Symptom | Possible Cause | Action |
|---------|----------------|--------|
| Zero rows inserted | Empty provider response | Use `--dry-run` + `--log-level DEBUG` to inspect; check interval availability |
| Duplicate key errors | Missing `--on-conflict ignore` or no unique index | Add unique index or switch conflict mode |
| Very slow run | Large date span + 1m bars sequentially | Use `concurrent` with moderate `--threads` |
| Connection errors | Thread count > pool size | Lower `--threads` or increase pool size |
| Mixed date ordering error | `--start` after `--end` | Swap or correct arguments |

Planned / Optional Enhancements
--------------------------------
* Conflict statistics (count actual duplicates skipped under `ignore`).

FAQ
---
**Q:** Does `--dry-run` still write CSV?  
**A:** Yes, if the lower layer honors `output_filename`; it only suppresses database writes.

**Q:** Why do inserted counts seem unchanged on repeat runs with `--on-conflict ignore`?  
**A:** The current log shows “prepared” count; duplicates are silently ignored. A future enhancement may add a “committed” vs “skipped_conflict” breakdown.

**Q:** Can I backfill thousands of tickers at once?  
**A:** Technically yes with `concurrent`, but watch API quotas and DB pool limits; consider batching logically.



