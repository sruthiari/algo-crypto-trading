import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate


def load_and_preprocess_data(filename):
    df = pd.read_csv(filename)
    df['open Time'] = pd.to_datetime(df['open Time'], unit='ms')
    df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms')
    df.set_index('open Time', inplace=True)
    return df.last('2D')


def calculate_rsi(df, window):
    delta = df['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rsi = 100 - (100 / (1 + (avg_gain / (avg_loss + 1e-10))))
    return rsi


def define_trading_signals(df, rsi, oversold_threshold, overbought_threshold):
    df['Buy Signal'] = np.where(rsi < oversold_threshold, 1, 0)
    df['Sell Signal'] = np.where(rsi > overbought_threshold, -1, 0)


def trading_simulation(df, initial_balance, risk_percentage):
    df['Position'] = 0
    df['Entry Price'] = 0
    df['Exit Price'] = 0
    df['Position Size'] = 0
    df['Risk Amount'] = 0

    current_balance = initial_balance
    position_size = 0
    entry_price = 0

    for index, row in df.iterrows():
        if row['Buy Signal'] == 1:
            position_size = (current_balance * risk_percentage) / abs(row['Close'] - entry_price)
            entry_price = row['Close']
            current_balance -= (position_size * entry_price)

            df.at[index, 'Position'] = 1
            df.at[index, 'Entry Price'] = entry_price
            df.at[index, 'Position Size'] = position_size
            df.at[index, 'Risk Amount'] = current_balance

        elif row['Sell Signal'] == -1 and position_size > 0:
            exit_price = row['Close']
            current_balance += (position_size * exit_price)
            position_size = 0

            df.at[index, 'Position'] = -1
            df.at[index, 'Exit Price'] = exit_price
            df.at[index, 'Risk Amount'] = current_balance

    df['Portfolio Value'] = initial_balance + df['Risk Amount']
    df['Portfolio Return'] = df['Portfolio Value'].pct_change()
    return df


def evaluate_performance(df):
    returns = df['Portfolio Return'].dropna()

    annualized_return = ((1 + returns.mean()) ** 252) - 1
    volatility = returns.std() * np.sqrt(252)

    sharpe_ratio = annualized_return / volatility
    sortino_ratio = annualized_return / returns[returns < 0].std() * np.sqrt(252)

    cumulative_return = (1 + returns).cumprod()
    high_water_mark = cumulative_return.cummax()
    drawdown = (high_water_mark - cumulative_return) / high_water_mark
    max_drawdown = drawdown.max()

    winning_trades = returns[returns > 0]
    losing_trades = returns[returns <= 0]
    win_loss_ratio = len(winning_trades) / len(losing_trades)

    profit_factor = winning_trades.sum() / abs(losing_trades.sum())

    metrics = {
        "Annualized Return": annualized_return,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Maximum Drawdown": max_drawdown,
        "Win/Loss Ratio": win_loss_ratio,
        "Profit Factor": profit_factor,
    }

    return metrics


def main():
    filename = 'historical_candlestick_data.csv'
    df = load_and_preprocess_data(filename)

    window = 14
    rsi = calculate_rsi(df, window)
    define_trading_signals(df, rsi, 30, 70)

    initial_balance = 100000
    risk_percentage = 0.02
    df = trading_simulation(df, initial_balance, risk_percentage)

    metrics = evaluate_performance(df)
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print(tabulate(df[['Close', 'Buy Signal', 'Sell Signal', 'Position', 'Entry Price', 'Exit Price', 'Position Size',
                       'Risk Amount', 'Portfolio Value', 'Portfolio Return']], headers='keys', tablefmt='grid'))

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Portfolio Value'], label='Portfolio Value', color='blue')
    plt.title('Portfolio Performance')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
