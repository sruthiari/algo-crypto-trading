import websocket
import requests
import time
import random
import json
import pprint
from variables import streams5m, cryptos, streams1m
import numpy as np
import json
import requests
from typing import Optional
#
# import warnings
# from requests.packages.urllib3.exceptions import RequestsDependencyWarning


def calculate_average(data, n):
    """
    Calculate the average of the last 'n' samples in the data series.

    Args:
        data (list of float): The data series to calculate the average from.
        n (int): The number of last samples to consider for the average.

    Returns:
        float: The calculated average.
    """
    if not isinstance(data, list) or not all(isinstance(i, (int, float)) for i in data):
        raise ValueError("Data should be a list of integers or floats.")

    if not isinstance(n, int) or n <= 0:
        raise ValueError("n should be a positive integer.")

    data_length = len(data)

    if n > data_length:
        raise ValueError("n should be less than or equal to the length of the data series.")

    return sum(data[-n:]) / n


def moving_average(data, window_size):
    """
    Calculate a moving average of the data series.

    Args:
        data (list of float): The data series to calculate the moving average for.
        window_size (int): The size of the moving window used to calculate the average.

    Returns:
        list of float: A list of moving average values.
    """

    if not isinstance(data, list) or not all(isinstance(i, (int, float)) for i in data):
        raise ValueError("Data should be a list of integers or floats.")

    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("Window size should be a positive integer.")

    data_length = len(data)

    return [sum(data[max(0, i - window_size + 1):i + 1]) / min(i + 1, window_size) for i in range(data_length)]


def EMA(x, n, smoothing=2):
    """
    Calculate the exponential moving average (EMA) of a data series.

    The initial value is derived from a simple moving average (SMA), and subsequent
    values are calculated using the exponential moving average formula.

    Args:
        x (list of float): The data series to calculate the EMA for.
        n (int): The period over which the EMA and initial SMA are calculated.
        smoothing (int, optional): The smoothing factor used in the EMA formula. Defaults to 2.

    Returns:
        list of float: The calculated EMA series.
    """

    if not isinstance(x, list) or not all(isinstance(i, (int, float)) for i in x):
        raise ValueError("x should be a list of integers or floats.")

    if not isinstance(n, int) or n <= 0:
        raise ValueError("n should be a positive integer.")

    if not isinstance(smoothing, (int, float)) or smoothing <= 0:
        raise ValueError("Smoothing factor should be a positive number.")

    # Calculate the simple moving average for the initial value
    sma_initial = sum(x[:n]) / n

    ema = [sma_initial]

    # Define the weighting factor
    p = smoothing / (n + 1)

    # Calculate the EMA for each data point after the initial SMA
    for price in x[n:]:
        ema.append(price * p + ema[-1] * (1 - p))

    return ema


def updateEMA(actual, previous, n, smoothing=2):
    """
    Return new value of EMA using the previous EMA sample.

    Args:
        actual (float): The current data point.
        previous (float): The previous data point or EMA value.
        n (int): The period over which the EMA is calculated.
        smoothing (int, optional): The smoothing factor, usually set to 2. Defaults to 2.

    Returns:
        float: The updated EMA value.
    """

    # Parameter validations
    if not isinstance(actual, (int, float)) or not isinstance(previous, (int, float)):
        raise ValueError("Actual and previous values should be integers or floats.")

    if not isinstance(n, int) or n <= 0:
        raise ValueError("Period (n) should be a positive integer.")

    if not isinstance(smoothing, (int, float)) or smoothing <= 0:
        raise ValueError("Smoothing factor should be a positive number.")

    # EMA calculation
    p = smoothing / (n + 1)  # Rho, weight factor
    ema = actual * p + previous * (1 - p)  # EMA formula

    return ema


def relativeStrengthIndicator(data, period, n=1):
    """
    Calculates the RSI-period of the last n samples and prints the results
    to the console.

    Args:
        data (list): The input data array to calculate RSI for.
        period (int): The period over which to calculate average gains and losses.
        n (int, optional): Specifies the number of last samples to return. Defaults to 1.

    Returns:
        numpy.ndarray: A numpy array of the last 'n' RSI values.
    """
    delta = np.diff(data)

    # Calculate the average gains and losses over the specified period
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)

    avg_gains = np.convolve(gains, np.ones((period,)) / period, mode='valid')
    avg_losses = np.convolve(losses, np.ones((period,)) / period, mode='valid')

    # Calculate relative strength and RSI
    rs = avg_gains / (avg_losses + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    return rsi[-n:]


def diff(arr, N=1):
    """Computes the difference between adjacent elements in the array.

    Args:
        arr (list): The input array to differentiate.
        N (int, optional): The number of steps over which to differentiate. Defaults to 1.

    Returns:
        list: A list of differentiated values based on 'N' step differences.
    """
    return [arr[i] - arr[i - N] for i in range(N, len(arr))]

def wait():
    """Ensures there is adequate time to perform operations before a 5-minute candle closes.

    The function waits until the current time is in the 1st or 6th minute of a 5-minute interval
    before proceeding with further operations.
    """
    i = 0
    while time.gmtime()[4] % 5 in {0, 4}:
        print('Waiting until safe to proceed with requests' + '.' * (i % 4) + ' ' * (4 - i % 4), end='\r')
        i += 1
        time.sleep(1)


def init_wallet():
    """Initialize the wallet with default values for each cryptocurrency and set USDT and BNBFORFEE values."""
    global wallet

    wallet = {crypto: 0 for crypto in cryptos}
    wallet['USDT'] = INIT_USDT
    wallet['BNBFORFEE'] = INIT_BNB

def printThresholds():
    """Print the calculated thresholds in a formatted table."""
    print('CALCULATED THRESHOLDS:')
    header = "{:<20} {:<20} {:<20}".format("CRYPTO", "SELL LIMIT", "STOP LOSS")
    print(header)
    print('-' * len(header))

    for crypto in cryptos:
        sellLimit, stopLoss = (round(x, 5) if x else "None" for x in thresholds[crypto])
        row = "{:<20} {:<20} {:<20}".format(crypto, str(sellLimit), str(stopLoss))
        print(row)

    print('! Calculated Thresholds are not used in this version.')

# working till here from down
def calculateThresholds(crypto, verbose=False):
    """ Calculate sell limit and stop loss based on the previous 24h.
    returns:
        (sell_limit, stop_loss) all in percentage (1 -> x1 -> 100%)
        sell_limit is None if sell limit is less than x1.005
    """
    if verbose:
        print(f'\n\nCalculate Thresholds of {crypto}')
        for ma_key in [50, 14, 6]:
            print(f'Length of ma[{crypto}][{ma_key}]: {len(ma[crypto][ma_key])}')

    relaxThresholds = 0.8
    i = 90  # First point to have all MAs without crashing

    wins, losses = [], []  # Array of percentages up and down

    while i < 498:
        if isGoingToRise(crypto, usingGlobals=False, lastIndex=i):
            if verbose:
                print(f'\n\n Is going to rise {crypto}')

            j = find_next_cross_down(crypto, i)

            if j is None:
                break  # No more crosses, exit the loop

            if verbose:
                print(
                    f"{crypto} rose on {i} ({time.gmtime(candles[crypto][i]['t'] / 1000)}) and fell on {j} ({time.gmtime(candles[crypto][j]['t'] / 1000)})")

            actual = closes[crypto][i]
            maximum = max(candles[crypto][k]['h'] for k in range(i + 1, j + 1))
            minimum = min(closes[crypto][k] for k in range(i + 1, j + 1))

            if maximum != actual:
                wins.append(maximum / actual - 1)
            if minimum != actual:
                losses.append(1 - minimum / actual)

            i = j
        else:
            i += 1

    if verbose:
        print(f'len wins: {len(wins)}')
        print(f'len losses: {len(losses)}')

    avgWin, avgLoss = calculate_avg_wins_losses(wins, losses, relaxThresholds)

    return (avgWin, avgLoss) if avgWin > 0.005 else (None, avgLoss)


def find_next_cross_down(crypto, start_index):
    j = start_index + 1
    while j < 498:
        if (ma[crypto][6][j] >= ma[crypto][14][j] and ma[crypto][6][j + 1] < ma[crypto][14][j + 1]):
            return j + 1  # cross is made on j+1
        j += 1
    return None


def calculate_avg_wins_losses(wins, losses, relaxThresholds):
    avgWin = sum(wins) / len(wins) if wins else 0
    avgWin *= relaxThresholds

    avgLoss = sum(losses) / len(losses) if losses else 0
    avgLoss *= relaxThresholds

    return avgWin, avgLoss


def get_data(catchUp=False):
    """Fetch and update data for a list of cryptocurrencies"""
    global candles, closes, ma, ema, rsi, thresholds, lastTradeTime, cryptos

    try:
        for crypto in cryptos:
            print('Catching up' if catchUp else f'Getting request of {crypto}', end='\r')

            limit = determine_limit(catchUp, crypto) if catchUp else 1000
            if limit:
                update_crypto_data(crypto, limit, catchUp)
        print('\n\n\n')
    except Exception as e:
        print('Programming error on get_data:', e)


def determine_limit(catchUp, crypto):
    """Determine the limit for the number of data points to fetch"""
    global candles

    if not catchUp:
        return None

    # Find the last correct closed candle
    lastCloseTime = None
    i = 1
    while not lastCloseTime:
        # If the candle -i is a close candle
        if candles[crypto][-i]['x']:
            lastCloseTime = candles[crypto][-i]['T'] / 1000
        else:
            i += 1

    # Get last close real time
    timeNow = time.gmtime()
    extraMin = timeNow.tm_min % 5
    lastCloseRealTime = time.time() - extraMin * 60 - timeNow.tm_sec

    # Difference in seconds
    diffTime = int(lastCloseRealTime - lastCloseTime)

    # Difference in 5 min candles
    diffCandles = int(round(diffTime / (60 * 5), 1))

    assert diffTime % (60 * 5) < 100, 'Misaligned time calculus'

    # +1 because it requests the last non-closed too, which is
    # not needed
    limit = diffCandles + 1 if diffCandles != 0 else None

    return limit


def update_crypto_data(crypto, limit, catchUp):
    """Fetch new data for a single cryptocurrency and update the global variables accordingly"""
    url = f'https://api.binance.com/api/v3/klines?symbol={crypto}&interval=5m&limit={limit}'
    try:
        newCandles = format_candles(requests.get(url).json()[:-1], crypto)

        if not catchUp:
            reset_crypto_data(crypto)

        candles[crypto].extend(newCandles)
        closes[crypto].extend([float(x['c']) for x in newCandles])
        calculate_moving_averages(crypto, limit)
        calculate_exponential_moving_averages(crypto, limit)
        rsi[crypto].extend(relativeStrengthIndicator(closes[crypto], RSI_PERIOD, limit - 1))

        # thresholds[crypto] = calculateThresholds(crypto, verbose=True)
        print_status(crypto, 'OK', catchUp)
    except Exception as e:
        print_status(crypto, f'ERROR {e}', catchUp)


def format_candles(newCandles, crypto):
    """Format candle data to match the desired structure"""
    return [{'t': x[0], 'T': x[6], 's': crypto, 'i': '5m', 'f': None, 'L': None, 'o': x[1],
             'c': x[4], 'h': float(x[2]), 'l': x[3], 'v': x[5], 'n': x[8], 'x': True,
             'q': x[7], 'V': x[9], 'Q': x[10], 'B': x[11]} for x in newCandles]


def reset_crypto_data(crypto):
    """Reset data variables for a single cryptocurrency"""
    candles[crypto] = []
    closes[crypto] = []
    ma[crypto] = {50: [None] * 50, 14: [None] * 14, 6: [None] * 6}
    ema[crypto] = {50: [None] * 50, 20: [None] * 20}
    rsi[crypto] = []
    lastTradeTime[crypto] = None


def calculate_moving_averages(crypto, limit):
    """Calculate moving averages for a single cryptocurrency"""
    for n in [6, 14, 50]:
        ma[crypto][n].extend(moving_average(closes[crypto][-limit + 1:], n))


def calculate_exponential_moving_averages(crypto, limit):
    """Calculate exponential moving averages for a single cryptocurrency"""
    for n in [20, 50]:
        ema[crypto][n].extend(EMA(closes[crypto][-limit + 1:], n))


def print_status(crypto, status, catchUp):
    """Print the status of data fetching for a single cryptocurrency"""
    print('Catching up' if catchUp else f'Getting request of {crypto} {" " * (10 - len(crypto))} {status}')

def update_crypto_data(crypto, limit, catchUp):
    """Fetch new data for a single cryptocurrency and update the global variables accordingly"""
    url = f'https://api.binance.com/api/v3/klines?symbol={crypto}&interval=5m&limit={limit}'
    try:
        newCandles = format_candles(requests.get(url).json()[:-1], crypto)

        if not catchUp:
            reset_crypto_data(crypto)

        candles[crypto].extend(newCandles)
        closes[crypto].extend([float(x['c']) for x in newCandles])
        calculate_moving_averages(crypto, limit)
        calculate_exponential_moving_averages(crypto, limit)
        rsi[crypto].extend(relativeStrengthIndicator(closes[crypto], RSI_PERIOD, limit - 1))

        # thresholds[crypto] = calculateThresholds(crypto, verbose=True)
        print_status(crypto, 'OK', catchUp)
    except Exception as e:
        print_status(crypto, f'ERROR {e}', catchUp)


def format_candles(newCandles, crypto):
    """Format candle data to match the desired structure"""
    return [{'t': x[0], 'T': x[6], 's': crypto, 'i': '5m', 'f': None, 'L': None, 'o': x[1],
             'c': x[4], 'h': float(x[2]), 'l': x[3], 'v': x[5], 'n': x[8], 'x': True,
             'q': x[7], 'V': x[9], 'Q': x[10], 'B': x[11]} for x in newCandles]


def reset_crypto_data(crypto):
    """Reset data variables for a single cryptocurrency"""
    candles[crypto] = []
    closes[crypto] = []
    ma[crypto] = {50: [None] * 50, 14: [None] * 14, 6: [None] * 6}
    ema[crypto] = {50: [None] * 50, 20: [None] * 20}
    rsi[crypto] = []
    lastTradeTime[crypto] = None


def calculate_moving_averages(crypto, limit):
    """Calculate moving averages for a single cryptocurrency"""
    for n in [6, 14, 50]:
        ma[crypto][n].extend(moving_average(closes[crypto][-limit + 1:], n))


def calculate_exponential_moving_averages(crypto, limit):
    """Calculate exponential moving averages for a single cryptocurrency"""
    for n in [20, 50]:
        ema[crypto][n].extend(EMA(closes[crypto][-limit + 1:], n))


def print_status(crypto, status, catchUp):
    """Print the status of data fetching for a single cryptocurrency"""
    print('Catching up' if catchUp else f'Getting request of {crypto} {" " * (10 - len(crypto))} {status}')


def get_bnb_price() -> float:
    """Fetches and returns the latest BNB price from the Binance API.

    Returns:
        float: The current BNB price.
    """
    url = 'https://api.binance.com/api/v3/avgPrice?symbol=BNBUSDT'
    try:
        response = requests.get(url)
        response.raise_for_status()
        return float(response.json()['price'])
    except requests.exceptions.RequestException as e:
        print(f"Could not fetch BNB price: {e}")
        return 0.0


def update_thresholds(thresholds: dict) -> None:
    """Writes the current thresholds data to a JSON file.

    Args:
        thresholds (dict): The thresholds data to write to file.
    """
    with open('thresholds.json', 'w+') as textFile:
        json.dump(thresholds, textFile, indent=4)


def update_candles(candles: dict) -> None:
    """Writes the current candles data to a JSON file.

    Args:
        candles (dict): The candles data to write to file.
    """
    with open('candles.json', 'w+') as textFile:
        json.dump(candles, textFile, indent=4)

def updateTradeHistory(
        crypto: str,
        action: bool,
        long: bool,
        price: float,
        nCoins: float,
        total: float,
        fee: float,
        time: float,
        target: Optional[float] = None,
        stop: Optional[float] = None) -> None:
    """
    Updates the trade history with details of a new trade.

    Args:
    - crypto (str): The cryptocurrency involved in the trade.
    - action (bool): True for a buy action, False for a sell action.
    - long (bool): True for a long position, False for a short position.
    - price (float): The price in USDT.
    - nCoins (float): The number of coins involved in the trade.
    - total (float): The total value in USDT (calculated as nCoins * price).
    - fee (float): The fee incurred for the trade, in BNB.
    - time (float): The timestamp of the trade.
    - target (float, optional): The target price for the trade. Defaults to None.
    - stop (float, optional): The stop price for the trade. Defaults to None.

    Returns:
    - None
    """
    LEVERAGE = 10  # replace with the actual value of your LEVERAGE
    trade = {
        'crypto': crypto,
        'time': time,
        'action': 'Buy' if action else 'Sell',
        'position': f"{('Long' if long else 'Short')} x{LEVERAGE}",
        'stop': stop,
        'target': target,
        'price': price,
        'filled': nCoins,
        'fee': fee,
        'total': round(total, 10)  # Avoid issues with floating-point precision
    }

    with open('tradeHistory.txt', 'a') as textFile:
        textFile.write(',' + json.dumps(trade))


def isGoingToRise(crypto, verbose=False):
    """Determines if a cryptocurrency is about to rise based on its EMA and fractal values."""
    try:
        ema20 = ema[crypto][20][-3]
        ema50 = ema[crypto][50][-3]
        fractalMin = float(candles[crypto][-3]['l'])

        if verbose:
            comparisons = [
                (ema20, '>', fractalMin, '>', ema50),
                (fractalMin, '<', candles[crypto][-5]['l']),
                (fractalMin, '<', candles[crypto][-4]['l']),
                (fractalMin, '<', candles[crypto][-2]['l']),
                (fractalMin, '<', candles[crypto][-1]['l']),
            ]

            for comp in comparisons:
                print(f"{comp[0]} {comp[1]} {comp[2]}{' ' + comp[3] if len(comp) > 3 else ''} ?")

        # Check if it's a minimum fractal and if the minimum is below ema20 and above ema50.
        if (ema20 > fractalMin > ema50 and
                all(fractalMin < float(candles[crypto][i]['l']) for i in [-5, -4, -2, -1])):
            if verbose:
                print('Yes')
            return True

    except Exception as e:
        print('Programming error on isGoingToRise:', e)

    return False


def isGoingToFall(crypto, verbose=False):
    """Determines if a cryptocurrency is about to fall based on its EMA and fractal values."""
    try:
        ema20 = ema[crypto][20][-3]
        ema50 = ema[crypto][50][-3]
        fractalMax = float(candles[crypto][-3]['h'])

        if verbose:
            comparisons = [
                (ema20, '<', fractalMax),
                (fractalMax, '>', candles[crypto][-5]['h']),
                (fractalMax, '>', candles[crypto][-4]['h']),
                (fractalMax, '>', candles[crypto][-2]['h']),
                (fractalMax, '>', candles[crypto][-1]['h'])
            ]

            for comp in comparisons:
                print(f"{comp[0]} {comp[1]} {comp[2]} ?")

        # Check if it's a maximum fractal and if the maximum is above ema20 and below ema50.
        if (ema20 < fractalMax < ema50 and
                all(fractalMax > float(candles[crypto][i]['h']) for i in [-5, -4, -2, -1])):
            if verbose:
                print('Yes')
            return True

    except Exception as e:
        print('Programming error on isGoingToFall:', e)

    return False


def buy(crypto, is_long):
    """Buys a cryptocurrency at the current closing price with specified leverage."""
    try:
        global wallet, closes, currentCryptos

        # Get the price to spend
        if (wallet['USDT'] > 200 and currentCryptos < MAX_CURRENT_CRYPTOS and
                (not lastTradeTime[crypto] or
                 time.time() - lastTradeTime[crypto] > 3600)):
            price = closes[crypto][-1]
            print('Bought', 'long ' if is_long else 'short', 'x' + str(LEVERAGE),
                  crypto, ' ' * (9 - len(crypto)), 'at', f"{price:.5f} USDT")

        # Ensuring necessary conditions to proceed with the buy
        if wallet['USDT'] > 200 and currentCryptos < MAX_CURRENT_CRYPTOS and (not lastTradeTime.get(crypto) or time.time() - lastTradeTime[crypto] > 3600):
            # print(f'Bought {"long" if is_long else "short"} x{LEVERAGE} {crypto} {" " * (9 - len(crypto))} at {closes[crypto][-1]}')
            print('Bought', 'long ' if is_long else 'short', 'x' + str(LEVERAGE),
                  crypto, ' ' * (9 - len(crypto)), 'at', f"{closes[crypto][-1]:.5f} USDT")

            # Determine the USDT amount to spend
            usdtSpent = min(INIT_USDT / MAX_CURRENT_CRYPTOS, wallet['USDT'] / MAX_CURRENT_CRYPTOS)
            wallet['USDT'] -= usdtSpent

            # Determine the buying price and number of coins to buy
            buyPrice = closes[crypto][-1]
            nCoins = usdtSpent * LEVERAGE / buyPrice
            wallet[crypto] = [nCoins, buyPrice, is_long]

            # Calculate and deduct the fee
            bnbPrice = get_bnb_price()
            fee_rate = MAKERFEERATE if is_long else TAKERFEERATE
            fee = nCoins * buyPrice * fee_rate / bnbPrice
            wallet['BNBFORFEE'] -= fee

            # Update trade history with transaction details
            updateTradeHistory(
                crypto, True, is_long, buyPrice, nCoins, usdtSpent, fee,
                time.time(), stop=targetStop[crypto][0], target=targetStop[crypto][1]
            )

            # Increment the count of current cryptos
            currentCryptos += 1

    except Exception as e:
        print('Programming error on buy:', e)


def sell(crypto, price):
    """Sell a cryptocurrency at a specified price."""
    try:
        global wallet, currentCryptos, lastTradeTime

        nCoins = wallet[crypto][0]
        buyPrice = wallet[crypto][1]
        position_type = wallet[crypto][2]

        profit_percent = round((price / buyPrice - 1) * 100, 3) if position_type else round((1 - price / buyPrice) * 100, 3)

        action_type = 'target' if (price >= buyPrice if position_type else price <= buyPrice) else 'stop'

        # Long or short position-specific variables
        usdtEarnt = nCoins * (price - buyPrice + buyPrice / LEVERAGE)
        fee_rate = TAKERFEERATE if position_type else MAKERFEERATE

        # Print transaction details
        position_str = 'long ' if position_type else 'short'
        print(f'Sold {position_str} x{LEVERAGE} {crypto} {" " * (9 - len(crypto))} at {round(price, 5)} ({buyPrice}) {action_type} ({profit_percent}%)')

        # Reset the crypto wallet entry
        wallet[crypto] = 0

        # Update the USDT wallet entry with the earned amount
        wallet['USDT'] += usdtEarnt

        # Calculate and deduct the fee
        bnbPrice = get_bnb_price()
        fee = nCoins * price * fee_rate / bnbPrice
        wallet['BNBFORFEE'] -= fee

        # Update the trade history
        updateTradeHistory(crypto, False, position_type, price, nCoins, usdtEarnt, fee, time.time())

        # Update the last trade time for the crypto
        lastTradeTime[crypto] = time.time()

        # Decrease the count of current cryptos
        currentCryptos -= 1

    except Exception as e:
        print('Programming error on sell:', e)


def init_socket():
    """Initialize the WebSocket connection to the Binance stream."""
    global ws  # Global WebSocket app instance

    # Ensure there is sufficient time interval between requests and the generation of new candlesticks
    wait()

    # Reset the WebSocket instance before creating a new connection
    ws = None

    # Define the WebSocket endpoint
    SOCKET = 'wss://stream.binance.com:9443/stream?streams=' + streams1m

    # Set a timeout interval to detect unresponsive connections and avoid hangs
    websocket.setdefaulttimeout(5)

    # Initialize the WebSocketApp with appropriate event handlers
    ws = websocket.WebSocketApp(
        SOCKET,
        on_open=on_open,
        on_close=on_close,
        on_message=on_message,
        on_error=on_error
    )

    # Start the WebSocket connection
    ws.run_forever()


def on_open(ws):
    """Handles the actions to perform when the websocket is opened."""
    global connectionFailed

    print('Opened connection.')

    # Fetch initial data based on the connection status
    get_data(connectionFailed)

    # Reset the connectionFailed flag to indicate a successful connection
    connectionFailed = False


def on_close(ws):
    print('Closed connection.')


def on_error(ws, err):
    global connectionFailed

    log_error(err)
    handle_reconnection(ws)
    attempt_data_recovery()


def log_error(err):
    print(f'Socket disconnected due to: {err}')
    print('Attempting to reconnect...')


def handle_reconnection(ws):
    connectionFailed = True
    time.sleep(5)
    ws.close(status=1002)


def attempt_data_recovery():
    print('Attempting to recover lost data...')
    init_socket()

def on_message(ws, message):
    try:
        kline = json.loads(message)['data']['k']
        crypto = kline['s']

        print(f"{crypto}     ", end='\r')

        if not kline['x']:
            return

        global candles, closes, ma, sma, rsi, wallet, targetStop

        update_candle_data(crypto, kline)
        update_indicators(crypto)
        check_trading_conditions(crypto, kline)

    except Exception as e:
        print('Programming error at on_message:', e)


def update_candle_data(crypto, kline):
    closes[crypto].append(float(kline['c']))
    candles[crypto].append(kline)

    size = len(closes[crypto])
    if sum(len(close) == size for close in closes.values()) == 71:
        update_candles()


def update_indicators(crypto):
    for n in [6, 14, 50]:
        ma[crypto][n].append(calculate_average(closes[crypto], n))

    for n in [20, 50]:
        ema[crypto][n].append(updateEMA(closes[crypto][-1], ema[crypto][n][-1], n))

    rsi[crypto].append(relativeStrengthIndicator(closes[crypto], RSI_PERIOD)[0])


def check_trading_conditions(crypto, kline):
    if wallet[crypto]:
        check_sell_conditions(crypto, kline)
    else:
        check_buy_conditions(crypto)


def check_sell_conditions(crypto, kline):
    if wallet[crypto][2]:
        check_long_position_sell(crypto, kline)
    else:
        check_short_position_sell(crypto, kline)


def check_long_position_sell(crypto, kline):
    if float(kline['h']) >= targetStop[crypto][1]:
        sell(crypto, targetStop[crypto][1])
    elif float(kline['l']) <= targetStop[crypto][0]:
        sell(crypto, targetStop[crypto][0])


def check_short_position_sell(crypto, kline):
    if float(kline['l']) <= targetStop[crypto][1]:
        sell(crypto, targetStop[crypto][1])
    elif float(kline['h']) >= targetStop[crypto][0]:
        sell(crypto, targetStop[crypto][0])


def check_buy_conditions(crypto):
    price = closes[crypto][-1]
    stop = ema[crypto][50][-3]

    if isGoingToRise(crypto, verbose=False):
        target = price + (price - stop) * RRRATIO
        if target / price > 1.001:
            targetStop[crypto] = [stop, target]
            buy(crypto, True)
    elif isGoingToFall(crypto, verbose=False):
        target = price - (stop - price) * RRRATIO
        if target / price <= 0.999:
            targetStop[crypto] = [stop, target]
            buy(crypto, False)


# Constants
RSI_PERIOD = 14
INIT_USDT = 1000
INIT_BNB = 10  # Needed to pay fees

MAX_CURRENT_CRYPTOS = 20  # Stop buying if this number of positions are open
MIN_USDT = 200  # Stop buying if this number is reached

RRRATIO = 1.5  # Risk Reward Ratio
LEVERAGE = 5  # Futures leverage
MAKERFEERATE = 0.00018
TAKERFEERATE = 0.00036

"""
A little explanation on how leverage positions are coded.
Opening a position is buying from now on, even it is a short position, it is
still considered buying.
Closing a position is selling from now on, even it is a short position, it is
still considered selling.

When buying:
    - The total USDT spent recorded on the code is the initial margin price.
    - The number of coins owned are the equivalent of margin*LEVERAGE.
    - Example: Buy 50€ of VET at 0.08 with LEVERAGE=5
        - usdt spent: 50 USDT
        - actual position price (not logged on tradeHistory): 50*5 = 250 USDT
        - coins owned: 250 USDT/0.08(USDT/VET) = 3125 VET

When selling:
    - The total USDT earned is the initial margin + profit.


"""

# Globals
candles = {}
closes = {}

# tresholds[crypto] = [sell_limit, stop_loss] (percentage x1)
thresholds = {}

# targetStop[crypto] = [stop_price, target_price] (price, not percentage)
targetStop = {}

lastTradeTime = {}

ma = {}
ema = {}
rsi = {}
# tradeHistory = []
currentCryptos = 0

""" Indicates how many coins are in the wallet
    wallet[crypto] = (amount, priceOfBuy, long)
        long = True if long position
               False if short position 
"""
wallet = {}
# State representing if the websocket connection is lost to request lost data.
connectionFailed = False

init_wallet()

# Initialize websocket connection
ws = None  # socket
init_socket()
