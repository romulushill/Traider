import MetaTrader5 as mt5
import numpy as np
import threading
import json
import time
from modules.utilities import Console
from modules.training import *
from modules.display import *

# Establish connection to MetaTrader 5 terminal


class Main:
    def __init__(self, login, password, server, symbol, timeframe, analysis_volume, starting_point, maximum_pattern_length):
        self.login = login
        self.password = password
        self.server = server
        self.symbol = symbol
        self.timeframe = timeframe
        self.analysis_volume = analysis_volume
        self.starting_point = starting_point
        self.maximum_pattern_length = maximum_pattern_length
        self.connection = self.connect_to_mt5()
        self.console = Console()
        self.trainer = Training(database_reference=f"./data/{self.symbol}/datastore.json",result_reference="outcome")

    def connect_to_mt5(self):
        conn = mt5.initialize(login=self.login, password=self.password, server=self.server)
        if not conn:
            self.console.log("initialize() failed, error code =", mt5.last_error(),3)
            quit()
            return False
        else:
            return conn

    def fetch_bar_data(self, symbol, timeframe, num_candles, start_pos=0):
        rates = mt5.copy_rates_from_pos(symbol, timeframe, start_pos, num_candles)
        return rates

    # Define a function to compare two sequences of prices (patterns)
    def compare_patterns(self, pattern1, pattern2):
        # Example: Using Pearson correlation coefficient as a similarity metric
        correlation = np.corrcoef(pattern1, pattern2)[0, 1]
        return correlation

    def backset(self):
        with open(f"./data/{self.symbol}/datastore.json", "r") as fp:
            contents = json.load(fp)

        # Fetch historical data
        data = self.fetch_bar_data(self.symbol, self.timeframe, self.analysis_volume, start_pos=0)
        close_prices = np.array([x["close"] for x in data])

        data_obj = []
        self.console.printProgressBar(0, self.analysis_volume, prefix = 'Progress:', suffix = 'Complete')
        for index, bar in enumerate(data):
            # self.console.log(f"Working on bar: {index}",2)
            zero_level = bar["close"]

            temp_obj = {}
            for position in range(self.starting_point, self.maximum_pattern_length+1):
                # self.console.log(position,2)
                bar_data = self.fetch_bar_data(self.symbol, self.timeframe, 1, index + position)[0]
                close_val = bar_data["close"]
                high_val = bar_data["high"]
                low_val = bar_data["low"]
                # self.console.log(f"Iterating through bars previous to: {index}\nCurrently on previous bar number: {position}\nThe close of this bar is: {close_val}\n",1)

                # DETERMINE THE OFFSET FROM THE CURRENT BAR CLOSE LEVEL AND SET THAT AS THE DATAPOINT
                if close_val > zero_level:
                    temp_obj[position] = close_val - zero_level
                elif close_val < zero_level:
                    temp_obj[position] = close_val - zero_level
                else:
                    temp_obj[position] = 0

            if index <= 0:
                pos = 0
            else:
                pos = index-1

            # Get the resulting closing price
            resulting_close = self.fetch_bar_data(self.symbol, self.timeframe, 1, pos)[0]["close"]
            # Get the previous closing price
            previous_close = self.fetch_bar_data(self.symbol, self.timeframe, 1, pos + 1)[0]["close"]
            # Calculate the outcome
            # outcome = resulting_close - previous_close
            # self.console.log(f"Analysis of Bar: {index}\Resulting Close: {resulting_close}\nPrevious Close: {previous_close}\nDirection: {outcome}",2)

            # Lets calculate the average direction for the next bars.
            if resulting_close > previous_close:
                outcome = 1
            elif resulting_close < previous_close:
                outcome = 0
            else:
                outcome = 0.5

            temp_obj["outcome"] = (outcome)  # Represent negative outcome as -1

            data_obj.append(temp_obj)
            self.console.printProgressBar(index + 1, self.analysis_volume, prefix="Progress:", suffix="Complete")

        contents["Dataset"] = data_obj
        with open(f"./data/{self.symbol}/datastore.json", "w") as fp:
            json.dump(contents, fp)
        return contents

    def train(self):
        resp = self.trainer.train(width=self.maximum_pattern_length, volume=self.analysis_volume)
        self.model = resp
        return resp

    def predict(self, data_properties):

        trade_features = []
        # Make dict into numpy array
        trade_features.append(np.array(data_properties))
        # Reshaping Procedure for the array to fit the model
        trade_features = np.reshape(trade_features, (1, self.maximum_pattern_length))
        # Predict the probability of the car being criminal
        probability = self.model.predict(trade_features)[0][0]
        # If the probability is greater than 0.5, classify as criminal
        return probability

    def trade(self):
        previous_close = None
        while True:
            time.sleep(0.01)
            current_close = self.fetch_bar_data(
                self.symbol, self.timeframe, 0, self.starting_point
            )[0]["close"]
            if previous_close is None or previous_close != current_close:
                nodes = self.fetch_bar_data(
                    self.symbol,
                    self.timeframe,
                    self.maximum_pattern_length,
                    self.starting_point,
                )
                closes = [node["close"] for node in nodes]
                print("New bar formed. Running trade logic...")
                resp = self.predict(closes)
                self.console.log(resp, 1)
                pos = 0
                display = {}
                for close in closes:
                    display[pos] = close
                    pos += 1
                display["outcome"] = float(resp)
                previous_close = current_close  # Update previous close
                self.train()
                # plot_forex_chart(data=display)


# Main function


# Specify symbol, timeframe, and number of candles
symbol = "GBPUSD"
timeframe = mt5.TIMEFRAME_M1
analysis_volume = 500 # Amount of candles to run based on. The volume of candles which are analysed for patterns. Cant include all time because that would be massive.
starting_point = 0 # Minimum amount of bars to constitute a pattern
maximum_pattern_length = 25 # Maximum amount of bars to constitute a pattern. This effects the performance of the model.

runner = Main(login=80780203, password="6pYjS@Vi", server="MetaQuotes-Demo", symbol=symbol, timeframe=timeframe, analysis_volume=analysis_volume, starting_point=starting_point, maximum_pattern_length=maximum_pattern_length)

runner.backset()
runner.train()
runner.trade()

# # Record patterns and their outcomes
# patterns = {}
# for i in range(len(close_prices)):
#     self.console.log("Identifying patterns.")
#     for j in range(i + 1, len(close_prices)):
#         self.console.log("Comparing closing price.")
#         pattern = close_prices[i:j]
#         outcome = "Buy" if close_prices[j] > close_prices[j - 1] else "Sell"
#         if tuple(pattern) not in patterns:
#             self.console.log("Pattern detected. Appending to patterns.")
#             patterns[tuple(pattern)] = outcome

# # Identify patterns that consistently lead to the same outcome
# consistent_patterns = {}
# for pattern, outcome in patterns.items():
#     self.console.log("Identifying replicating patterns.")
#     if outcome not in consistent_patterns:
#         consistent_patterns[outcome] = [pattern]
#     else:
#         consistent_patterns[outcome].append(pattern)

# # Print identified patterns and outcomes
# for outcome, patterns in consistent_patterns.items():
#     self.console.log(f"Outcome: {outcome}")
#     for pattern in patterns:
#         self.console.log(f"Pattern: {pattern}")

# # Disconnect from MetaTrader 5 terminal
# mt5.shutdown()
