import requests
import websocket
import json
import csv

# Replace these with your own API key and secret key
API_KEY = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
SECRET_KEY = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'

# Define the symbol and interval for the trading pair
symbol = 'BTCUSDT'  # Replace with the desired trading pair
interval = '5m'     # Replace with the desired interval (e.g., '1m', '5m', '1h')

def get_historical_candlestick_data(symbol, interval, limit):
    url = f'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }

    response = requests.get(url, params=params)
    data = response.json()
    return data

def save_data_to_csv(data, filename):
    with open(filename, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Open', 'High', 'Low', 'Close', 'Volume', 'open Time', 'Close Time', 'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume'])
        for candlestick in data:
            csv_writer.writerow(candlestick)

def on_message(ws, message):
    data = json.loads(message)
    if 'data' in data:
        candlestick = data['data']['k']

        with open('live_candlestick_data.csv', 'a', newline='') as file:
            csv_writer = csv.writer(file)
            if file.tell() == 0:
                # Write header only if the file is empty
                csv_writer.writerow(['Symbol', 'Open', 'High', 'Low', 'Close', 'Volume', 'Open Time',  'Close Time', 'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume'])
            csv_writer.writerow([
                symbol,  # Symbol
                candlestick['o'],  # Open
                candlestick['h'],  # High
                candlestick['l'],  # Low
                candlestick['c'],  # Close
                candlestick['v'],  # Volume
                candlestick['T'],  # Open Time
                candlestick['T'],  # Close Time
                candlestick['q'],  # Quote Asset Volume
                candlestick['n'],  # Number of Trades
                candlestick['V'],  # Taker Buy Base Asset Volume
                candlestick['Q'],  # Taker Buy Quote Asset Volume
            ])

def on_error(ws, error):
    print(f"WebSocket Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("WebSocket Closed")
    # Save live data to CSV before closing the WebSocket
    live_filename = 'live_candlestick_data.csv'
    save_data_to_csv(live_data, live_filename)
    print(f'Live data saved to {live_filename}')

def on_open(ws):
    payload = {
        "method": "SUBSCRIBE",
        "params": [
            f"{symbol}@kline_{interval}",
        ],
        "id": 1
    }
    ws.send(json.dumps(payload))

if __name__ == "__main__":
    # Fetch historical data
    historical_data = get_historical_candlestick_data(symbol, interval, limit=1000)
    historical_filename = 'historical_candlestick_data.csv'
    save_data_to_csv(historical_data, historical_filename)
    print(f'Historical data saved to {historical_filename}')

    live_data = []  # Initialize list to store live data
    # WebSocket for real-time data
    url = f"wss://stream.binance.com:9443/ws/{symbol}@kline_{interval}"
    ws = websocket.WebSocketApp(url, on_message=on_message, on_error=on_error, on_close=on_close)
    ws.on_open = on_open
    print("WebSocket for real-time data is running.")
    ws.run_forever()
