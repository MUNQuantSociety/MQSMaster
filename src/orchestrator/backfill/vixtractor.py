# # import pandas as pd
# # from datetime import datetime
# # from data_infra.authentication.apiAuth import APIAuth
# # from data_infra.marketData.fmpMarketData import FMPMarketData

# # # Define date range
# # # start_date = "2023-01-01" 
# # # end_date = "2024-01-01"

# # start_date = "2024-01-01" 
# # end_date = "2024-01-02"

# # # Initialize data client
# # fmp_client = FMPMarketData()
# # print(fmp_client)

# # # Fetch VIX 1-minute intraday data
# # vix_data = fmp_client.get_intraday_data(
# #     # tickers="^VIX",
# #     tickers="AAPL",
# #     from_date=start_date,
# #     to_date=end_date,
# #     interval=1
# # )

# # # Check and store as CSV
# # if vix_data:
# #     df = pd.DataFrame(vix_data)
# #     df['datetime'] = pd.to_datetime(df['date'])
# #     df = df.sort_values('datetime')
# #     df.to_csv("vix_intraday_1min.csv", index=False)
# #     print(f"✅ Saved {len(df)} rows of 1-minute VIX data to vix_intraday_1min.csv")
# # else:
# #     print("❌ Failed to retrieve VIX intraday data.")

# import requests, pandas as pd, datetime as dt

# API   = "IPSiO49jnQdx4Y1XqyghXojVFzJnwtKb"
# SYM   = "^VIX"                        # URL-encoded ^VIX
# START = dt.date(2024, 1, 1)
# STOP  = dt.date(2023, 1, 1)
# step  = dt.timedelta(days=5)           # 5-day slices ≈ 5×390 = 1 950 rows

# dfs = []
# t0  = START
# while t0 < STOP:
#     t1 = min(t0 + step - dt.timedelta(days=1), STOP)
#     # url = (f"https://financialmodelingprep.com/api/v3/historical-chart/1min/{SYM}"
#     #        f"?from={t0}&to={t1}&apikey={API}")
#     url = (f"https://financialmodelingprep.com/stable/historical-chart/1min?symbol=^VIX&apikey={API}")
#     chunk = requests.get(url, timeout=15).json()
#     dfs.append(pd.DataFrame(chunk))
#     t0 = t1 + dt.timedelta(days=1)

# df = pd.concat(dfs).sort_values("date")
# df.to_csv("vix_1min.csv", index=False)
# print(f"Saved {len(df):,} rows to CSV")

# import requests
# import pandas as pd
# from urllib.parse import quote

          # URL-encodes the caret → “%5EVIX”
# ENDPOINT = (
#     f"https://financialmodelingprep.com/stable/historical-chart/1min"
#     f"?symbol={SYMBOL}&apikey={API_KEY}"
# )

# ENDPOINT = (
#     f"https://financialmodelingprep.com/stable/historical-chart/1min?symbol=^VIX&from=2023-03-01&to=2023-03-04&apikey={API_KEY}"
# )



# # 1) Hit the API ..................................................................
# response = requests.get(ENDPOINT, timeout=10)
# response.raise_for_status()          # raises if the call failed
# data = response.json()               # list[dict] – each dict is one bar

# # 2) Convert to DataFrame and save ................................................
# df = pd.DataFrame(data)
# df.to_csv("vix_1min_latest.csv", index=False)

# print(f"Saved {len(df):,} rows to vix_1min_latest.csv")

import requests
import pandas as pd
from datetime import datetime, timedelta
from urllib.parse import quote

API_KEY = "IPSiO49jnQdx4Y1XqyghXojVFzJnwtKb"        #  ← put your paid key here
SYMBOL  = quote("^VIX")    

START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 1, 1)
STEP_DAYS = 1  # because 1-minute bars only work for ~2 days at a time

all_data = []

print("Starting VIX data download...")

current_start = START_DATE
while current_start < END_DATE:
    current_end = min(current_start + timedelta(days=STEP_DAYS), END_DATE)

    from_str = current_start.strftime("%Y-%m-%d")
    to_str = current_end.strftime("%Y-%m-%d")

    url = (
        f"https://financialmodelingprep.com/stable/historical-chart/1min"
        f"?symbol={SYMBOL}&from={from_str}&to={to_str}&apikey={API_KEY}"
    )

    try:
        print(f"Fetching: {from_str} → {to_str}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, list) and data:
            df = pd.DataFrame(data)
            df["datetime"] = pd.to_datetime(df["date"])
            all_data.append(df)
        else:
            print(f"⚠️ No data returned for {from_str} → {to_str}")

    except Exception as e:
        print(f"❌ Failed for {from_str} → {to_str}: {e}")

    current_start = current_end  # advance window

# Combine and save
if all_data:
    full_df = pd.concat(all_data)
    full_df.sort_values("datetime", inplace=True)
    full_df.to_csv("vix_1min_full_2023.csv", index=False)
    print(f"✅ Saved full VIX dataset with {len(full_df):,} rows to vix_1min_full_2023.csv")
else:
    print("❌ No data retrieved.")
