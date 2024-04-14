import json
import numpy as np
import tensorflow as tf
from datetime import datetime


class Training:
    def __init__(self, database_reference=None, width=25, result_reference=None):
        self.model = None
        self.width = width
        self.result_reference = result_reference
        try:
            with open(database_reference, "r") as fp:
                self.database = json.load(fp)
            self.data = self.database["Dataset"]
            print("Loaded dataset.")
            return
        except Exception as error:
            self.database = None
            self.data = None
            print(f"Failed to load dataset:\n{error}")
            return None

    def train(self, width, volume):

        self.width = width  # Assuming the last element is the outcome
        self.volume = volume
        self.result_reference = "outcome"  # Assuming

        features_array = []
        labels_array = []

        for element in self.data:
            internal = []
            for aspect in range(0, self.width):
                internal.append(element[f"{aspect}"])
            features_array.append(internal)
            labels_array.append([element[self.result_reference]])

        features = np.array(features_array)  # THE DATAROW
        labels = np.array(labels_array)  # THE ANSWERS

        if self.model is None:
            # Define the neural network architecture only if model is not yet initialized
            self.model = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(self.width, activation="relu", input_shape=(self.width,)),
                    tf.keras.layers.Dense(4, activation="relu"),
                    tf.keras.layers.Dense(1, activation="sigmoid"),  # No activation function in the output layer
                ]
            )

            # Compile the model
            self.model.compile(
                optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
            )

        # Train the model on the new data
        self.model.train_on_batch(features, labels)

        # Save the model weights
        current_datetime = datetime.now()
        formatted_date = current_datetime.strftime("%Y-%m-%d")
        formatted_time = str(current_datetime.strftime("%H.%M.%S"))
        self.path = f"./resources/models/weightings-{formatted_date}-{formatted_time}.weights.h5"
        self.model.save_weights(self.path)

        return self.model

