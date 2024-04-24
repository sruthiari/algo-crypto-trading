# algo-crypto-trading

Step 1: once you install python and all related libraries.

Step 2: delete all the .csv files

           historical_candlestick_data.csv
           live_candlestick_data.csv
           pre_processed_data.csv

Step 3: running the project

            1.get the data from binance : right click on get_data.py and run the file. it will get the data and create new historical_candlestick_data.csv
              and live_candlestick_data.csv file.
            2.processes the data: right click on the data_clensing.py and run. it will create a new pre_processed_data.csv.
                Calculate additional features and indicators (e.g., moving averages, RSI, MACD)
                Handle missing data (forward fill for simplicity)
                Handle outliers and anomalies (using z-score for simplicity)
            3. run the algorithm: right click on the Algorithm_trading.py and run. It will collect all the data and do long, shor and buy/sell trading and display the on console.
                and also it will update tradeHistory.text file.
            4. backtesting, metrics to check efficiency of the algorithm: right click on the backtesting_trend_strategy.py and run. it will display metrics and graphs etc.

 NOTE: Once you run the Algorithm_trading.py file, please leave it for at least 10mins to perform automatic trading and show the results on the console.

 the algorithm class is very big as instructed by the professor for the bot correction.