import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tabulate import tabulate

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


# Load historical data from CSV
historical_filename = 'historical_candlestick_data.csv'
df_historical = pd.read_csv(historical_filename)

# Convert timestamp columns to datetime
df_historical['open Time'] = pd.to_datetime(df_historical['open Time'], unit='ms')
df_historical['Close Time'] = pd.to_datetime(df_historical['Close Time'], unit='ms')

# Set the 'Open Time' as the DataFrame index
df_historical.set_index('open Time', inplace=True)

# Resample the data to the desired timeframe (e.g., daily)
desired_timeframe = '1D'
df_resampled = df_historical.resample(desired_timeframe).agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum',
    'Close Time': 'last',  # You might want to keep the last Close Time
    'Quote Asset Volume': 'sum',
    'Number of Trades': 'sum',
    'Taker Buy Base Asset Volume': 'sum',
    'Taker Buy Quote Asset Volume': 'sum'
})



# Calculate additional features and indicators (e.g., moving averages, RSI, MACD)
window = 14  # Adjust the window size as needed for indicators like RSI and moving averages
df_resampled['SMA'] = df_resampled['Close'].rolling(window=window).mean()
df_resampled['EMA'] = df_resampled['Close'].ewm(span=window, adjust=False).mean()

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(df_resampled.index, df_resampled['Close'], label='Close Price', color='blue')
plt.plot(df_resampled.index, df_resampled['SMA'], label='SMA ({0} days)'.format(window), color='orange')
plt.plot(df_resampled.index, df_resampled['EMA'], label='EMA ({0} days)'.format(window), color='green')
plt.title('Price with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Handle missing data (forward fill for simplicity)
df_resampled.fillna(method='ffill', inplace=True)

# Handle outliers and anomalies (using z-score for simplicity)
z_scores = np.abs((df_resampled['Close'] - df_resampled['Close'].mean()) / df_resampled['Close'].std())
outliers_threshold = 3  # Adjust the threshold as needed
df_resampled = df_resampled[z_scores < outliers_threshold]

# Now df_resampled contains the preprocessed and cleaned data

# Selecting required columns
selected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
df_selected = df_resampled[selected_columns]

# Display the resampled data as a table in the terminal
print(tabulate(df_selected, headers='keys', tablefmt='grid'))

# Save the preprocessed data to a new CSV file
preprocessed_filename = 'pre_processed_data.csv'
df_selected.to_csv(preprocessed_filename, index=True)
print('Preprocessed data saved to {}'.format(preprocessed_filename))


# ... (continue with your trading strategy development)

