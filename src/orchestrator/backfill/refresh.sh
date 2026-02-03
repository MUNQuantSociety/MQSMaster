0 0 * * 0 cd /MQSMaster && python src/orchestrator/backfill/refresh.py --threads 8 --interval 1 --start 010120 --end 010226 
#? Script to refresh the tickers.json file with the latest S&P 500 tickers from FMPMarket Data API
#?      This runs at midnight (00:00) every Sunday. Breaking it down:
#?          0 0 - at 00:00 (midnight)
#?          * * 0 - every day of month, every month, on Sunday (0)
#? The cd changes to your project directory before running the script