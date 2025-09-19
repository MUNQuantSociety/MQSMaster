import os
import pandas as pd

# Define full paths to your folders
folders = {
    "Portfolio_1": "/Users/mpunduchikoya/Documents/MUN/MQSMaster/data_infra/data/20250717_121233_backtest_1",
    "Portfolio_2": "/Users/mpunduchikoya/Documents/MUN/MQSMaster/data_infra/data/20250717_120905_backtest_2"
}

# Dictionary to hold combined DataFrames
portfolios_data = {}

for port_name, folder_path in folders.items():
    dataframes = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            filepath = os.path.join(folder_path, filename)
            df = pd.read_csv(filepath)
            df['ticker'] = filename.replace('.csv', '')
            dataframes.append(df)

    # Combine into one DataFrame per portfolio
    combined_df = pd.concat(dataframes, ignore_index=True)
    portfolios_data[port_name] = combined_df

# Show samples
print("\n📈 Portfolio 1 Sample:")
print(portfolios_data["Portfolio_1"].head())

print("\n📉 Portfolio 2 Sample:")
print(portfolios_data["Portfolio_2"].head())
