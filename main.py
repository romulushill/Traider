import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
import time
from keras.utils import to_categorical
import threading

# Global variables to count buy and sell trades
BuyTrades = 0
SellTrades = 0


class ForexPredictor:
    def __init__(
        self,
        symbol,
        timeframe,
        pattern_length,
        balancerisk=0.03,
        risklevel=1 / 3,
        magic=1,
        num_bars=1000,
        epochs=20,
        batch_size=32,
        prediction_interval=0.001,
    ):
        self.balancerisk = balancerisk
        self.risklevel = risklevel
        self.magic = magic
        self.connected = False
        self.symbol = symbol
        self.timeframe = timeframe
        self.pattern_length = pattern_length
        self.model = None
        self.scaler = MinMaxScaler()
        self.connect_to_mt5()
        self.train_and_predict_continuous(
            num_bars, epochs, batch_size, prediction_interval
        )

    def connect_to_mt5(self):
        resp = mt5.initialize(
            login=80780203, password="6pYjS@Vi", server="MetaQuotes-Demo"
        )
        if resp:
            self.connected = True
        return self.connected

    def collect_data(self, num_bars=1, start_pos=0):
        bars = mt5.copy_rates_from_pos(self.symbol, self.timeframe, start_pos, num_bars)
        df = pd.DataFrame(bars)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        return df

    def preprocess_data(self, data):
        scaled_data = self.scaler.fit_transform(data[["close"]])
        sequences = []
        labels = []
        for i in range(len(scaled_data) - self.pattern_length):
            sequences.append(scaled_data[i : i + self.pattern_length])
            # Determine the trend for the subsequent movement
            next_price = scaled_data[i + self.pattern_length]
            current_price = scaled_data[i + self.pattern_length - 1]
            label = (
                1 if next_price > current_price else 0
            )  # 1 for upwards trend, 0 for downwards trend
            labels.append(label)
        X = np.array(sequences)
        y = to_categorical(labels, num_classes=2)  # One-hot encode labels
        return X, y

    def build_model(self):
        self.model = Sequential()
        self.model.add(Dense(64, input_shape=(self.pattern_length,), activation="relu"))
        self.model.add(
            Dense(2, activation="sigmoid")
        )  # Output layer for binary classification
        self.model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

    def train_model(self, X_train, y_train, epochs, batch_size):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def evaluate_model(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print("Test Loss:", loss)
        print("Test Accuracy:", accuracy)

    def predict_next_movement(self, data):
        scaled_data = self.scaler.transform(data[["close"]])
        last_sequence = scaled_data[-self.pattern_length :]
        prediction = self.model.predict(np.array([last_sequence]))[0]
        return prediction  # Probabilities of upwards and downwards trends

    def SymbolType(self):
        currency = self.symbol
        result = ((currency[3:]).split("."))[0]
        if currency == "XAGUSD":
            return "SILVER"
        elif result == "JPY":
            return "JPY"
        else:
            return "STANDARD"

    def CalculatePoundRate(self):
        currency = self.symbol
        firstpair = currency[0:3]
        Rate = 0
        if firstpair == "GBP":
            Rate = 1
        elif firstpair == "EUR":
            val = "EURGBP"
            val2 = mt5.symbol_info_tick(self.symbol).bid
            Rate = 1 / val2
        else:
            CurrencyPair = f"GBP{firstpair}"
            Rate = mt5.symbol_info_tick(CurrencyPair).bid
        return Rate

    def CalculateVolume(self, entry, takeprofit, stoploss):
        balance = int(mt5.account_info().balance)
        RiskedCash = (balance / 100) * int(self.balancerisk)
        if entry > takeprofit:
            pips = entry - takeprofit
        elif entry < takeprofit:
            pips = takeprofit - entry
        else:
            pips = 0
        if self.SymbolType() == "STANDARD":
            pips = pips * 10000
        else:
            pips = pips * 100
        PoundsPerPip = RiskedCash / pips

        PoundRate = self.CalculatePoundRate()

        if (self.SymbolType()) == "JPY":
            entry = entry / 100

        volume = (PoundsPerPip * 0.1) * (entry * PoundRate)
        return volume

    def safe_to_trade(self, tradetype):
        positions = mt5.positions_get()
        if positions is None or len(positions) == 0:
            print("No open positions.")
            return True

        relevant_positions = [
            p for p in positions if p.symbol == self.symbol and p.magic == self.magic
        ]
        if len(relevant_positions) == 0:
            return True

        for position in relevant_positions:
            if position.type == tradetype:
                print(
                    "There is already a position on this chart in the same direction."
                )
                return False

        return True

    def place_trade(self, predicted_movement, current_price):
        global BuyTrades, SellTrades  # Define global variables

        # The larger the predicted movement, the larger trade we take and the larger risk we implement.
        if predicted_movement[0] > 0.5 or predicted_movement[1] > 0.5:
            last_bar_length = (
                self.collect_data(1, 1).iloc[0]["high"]
                - self.collect_data(1, 1).iloc[0]["low"]
            )
            if last_bar_length > 0:
                if predicted_movement[0] > 0.5:
                    print("Predicted Upwards Trend")
                    upstrength = predicted_movement[0] - 0.5
                    distance = upstrength * last_bar_length * 100
                    tradetype = mt5.ORDER_TYPE_BUY
                    price = mt5.symbol_info_tick(self.symbol).ask
                    takeprofit = price + distance
                    stoploss = price - (self.risklevel * distance)
                    comment = f"""Str: {upstrength}"""
                    BuyTrades += 1  # Increment BuyTrades counter
                elif predicted_movement[1] > 0.5:
                    print("Predicted Downwards Trend")
                    downstrength = predicted_movement[1] - 0.5
                    distance = downstrength * last_bar_length * 100
                    tradetype = mt5.ORDER_TYPE_SELL
                    price = mt5.symbol_info_tick(self.symbol).bid
                    takeprofit = price - distance
                    stoploss = price + (self.risklevel * distance)
                    comment = f"""Str: {downstrength}"""
                    SellTrades += 1  # Increment SellTrades counter
                else:
                    return False

                safe = self.safe_to_trade(tradetype)
                if safe:
                    volume = self.CalculateVolume(price, takeprofit, stoploss)
                    print(
                        f"Entering trade of type: {tradetype}\nPrice: {price}\nTake Profit: {takeprofit}\nStop Loss: {stoploss}\nVolume: {volume}\nComment: {comment}"
                    )

                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": self.symbol,
                        "volume": 0.5,
                        "type": tradetype,
                        "price": price,
                        "tp": takeprofit,
                        "sl": stoploss,
                        "deviation": 20,
                        "magic": self.magic,
                        "comment": comment,
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    try:
                        try:
                            response = mt5.order_send(request)
                            if (response.retcode == 10009) or (
                                response.retcode == 10008
                            ):
                                print("Success")
                                return True

                            elif response.retcode == 10027:
                                print("Enable Algo Trading On Platform.")
                            else:
                                print(f"Failed: {response.retcode}")
                                return False
                        except Exception as e:
                            print(f"Failed with an error: {e}")
                            print(mt5.last_error())
                            return False
                    except Exception as e:
                        print(e)
                        return False
                else:
                    print("Unsafe to trade.")
                    return False
            else:
                print("Doji candle - skip.")
                return False
        else:
            print("Insignificant trend.")
            return False

    def train_and_predict_continuous(
        self, num_bars, epochs, batch_size, prediction_interval
    ):
        data = self.collect_data(num_bars)
        X, y = self.preprocess_data(data)
        split_ratio = 0.8
        split_index = int(len(X) * split_ratio)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        self.build_model()
        self.train_model(X_train, y_train, epochs, batch_size)
        self.evaluate_model(X_test, y_test)
        self.continuous_prediction(prediction_interval, num_bars)

    def continuous_prediction(self, interval, num_bars):
        previous_time = None
        while True:
            time.sleep(interval)
            current_data = self.collect_data(1, 0)
            current_time = current_data.index[
                -1
            ]  # Get the timestamp of the current bar

            # Wait for a new bar to be formed
            while previous_time and previous_time == current_time:
                time.sleep(interval)
                current_data = self.collect_data(1, 0)
                current_time = current_data.index[-1]

            # Collect new data and make predictions only when a new bar is formed
            new_data = self.collect_data(num_bars)
            predicted_movement = self.predict_next_movement(new_data)
            print("Predicted Movement:", predicted_movement)

            # Place trade based on prediction
            if previous_time:  # Only place trade if this is not the first iteration
                self.place_trade(predicted_movement, current_data["close"].iloc[-1])
                print(f"BuyTrades | SellTrades\n{BuyTrades} | {SellTrades}")

            previous_time = current_time


targets = [
    {
        "symbol": "GBPUSD",
        "timeframe": mt5.TIMEFRAME_M1,
        "pattern_length": 25,
        "balancerisk": 0.03,
        "risklevel": 1 / 2,
        "magic": 1,
        "num_bars": 90000,
        "epochs": 20,
        "batch_size": 32,
        "prediction_interval": 0.001,
    },
    {
        "symbol": "EURUSD",
        "timeframe": mt5.TIMEFRAME_M1,
        "pattern_length": 25,
        "balancerisk": 0.03,
        "risklevel": 1 / 2,
        "magic": 1,
        "num_bars": 90000,
        "epochs": 20,
        "batch_size": 32,
        "prediction_interval": 0.001,
    },
    {
        "symbol": "USDJPY",
        "timeframe": mt5.TIMEFRAME_M1,
        "pattern_length": 25,
        "balancerisk": 0.03,
        "risklevel": 1 / 2,
        "magic": 1,
        "num_bars": 90000,
        "epochs": 20,
        "batch_size": 32,
        "prediction_interval": 0.001,
    },
    {
        "symbol": "USDCHF",
        "timeframe": mt5.TIMEFRAME_M1,
        "pattern_length": 25,
        "balancerisk": 0.03,
        "risklevel": 1 / 2,
        "magic": 1,
        "num_bars": 90000,
        "epochs": 20,
        "batch_size": 32,
        "prediction_interval": 0.001,
    },
]

# Example usage:
if __name__ == "__main__":
    for pair in targets:
        threading.Thread(
            target=ForexPredictor,
            args=(
                pair["symbol"],
                pair["timeframe"],
                pair["pattern_length"],
                pair["balancerisk"],
                pair["risklevel"],
                pair["magic"],
                pair["num_bars"],
                pair["epochs"],
                pair["batch_size"],
                pair["prediction_interval"],
            ),
        ).start()
