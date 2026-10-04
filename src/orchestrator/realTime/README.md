# Real-time prices and PnL

Use the repository [Windows/macOS setup](../../../README.md) to install the environment, configure FMP/PostgreSQL, and create tables. Run these commands from the repository root in separate activated terminals:

```bash
python -m src.orchestrator.realTime.realtimeDataIngestor
python -m src.orchestrator.realTime.pnl_script
```

The price ingestor reads [../backfill/tickers.json](../backfill/tickers.json), polls up to five FMP feeds, and stores quote snapshots in `market_data`. The PnL worker reads books and prices, calculates realized/unrealized PnL and portfolio equity, and writes `pnl_book`. Both target a 60-second cycle.

For the complete, separate ingestion diagram, see [Real-time ingestor workflow](../../../docs/workflows/realtime-ingestor-workflow.md). For how these workers fit into startup and shutdown, see the [live stack](../../../docs/workflows/live_trading_workflow.md).

To keep an ingestor file log locally:

Windows PowerShell:
```powershell
New-Item -ItemType Directory -Force logs | Out-Null
$env:MARKET_DATA_INGESTOR_LOG = Join-Path (Get-Location) 'logs/market_data_ingestor.log'
python -m src.orchestrator.realTime.realtimeDataIngestor
```

macOS/Linux:
```bash
mkdir -p logs
export MARKET_DATA_INGESTOR_LOG="$PWD/logs/market_data_ingestor.log"
python -m src.orchestrator.realTime.realtimeDataIngestor
```

Stop foreground workers with Ctrl+C. Standalone invocation does not impose market hours. The ingestor requires a unique index on `market_data(ticker, timestamp)`, inserts duplicates with `DO NOTHING`, and leaves open/high/low NULL because these feeds provide quotes rather than minute candles. Review the detailed workflow's volume-state limitations before interpreting volume after a restart.
