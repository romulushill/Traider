import MetaTrader5 as mt5
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from modules.utilities import Console

class Main:
    def __init__(self, symbols=[]):
        self.console = Console()
        self.console.log("Hi")

    # Establish connection to MetaTrader 5 terminal
    def connect_to_mt5(self):
        if not mt5.initialize(
            login=80780203, password="6pYjS@Vi", server="MetaQuotes-Demo"
        ):
            print("initialize() failed, error code =", mt5.last_error())
            quit()

    # Fetch live data from MetaTrader 5
    def fetch_live_data(self, symbol, timeframe, num_bars):
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
        close_prices = [x['close'] for x in rates]
        return np.array(close_prices)

    # Generate labels based on patterns (Replace with your own logic)
    def generate_labels(self, data):
        # Dummy labels for demonstration
        labels = np.random.randint(0, 2, len(data))  # 0: Sell, 1: Buy
        return labels

    # Create the neural network model
    def create_model(self, input_shape):
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(64, activation="relu", input_shape=input_shape),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model

    # Train the neural network
    def train_model(self, model, X_train, y_train):
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)


robot = Main(sets=[{"Symbol":"GBPUSD","Timeframe":mt5.TIMEFRAME_M1}])

# Main function
def main():
    # Connect to MetaTrader 5 terminal
    connect_to_mt5()

    # Fetch live data from MetaTrader 5
    symbol = "GBPUSD"
    timeframe = mt5.TIMEFRAME_M1
    num_bars = 1000
    data = fetch_live_data(symbol, timeframe, num_bars)

    # Generate labels based on patterns
    labels = generate_labels(data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Create the neural network model
    model = create_model(input_shape=(1,))

    # Train the model
    train_model(model, X_train, y_train)

    # Make trading decisions based on patterns
    current_data = fetch_live_data(symbol, timeframe, 1)
    prediction = model.predict(current_data.reshape(1, -1))
    if prediction >= 0.5:
        print("Pattern indicates Buy")
        # Place Buy order using MetaTrader 5
        # Replace this with your own function to place orders
    else:
        print("Pattern indicates Sell")
        # Place Sell order using MetaTrader 5
        # Replace this with your own function to place orders

if __name__ == "__main__":
    main()
