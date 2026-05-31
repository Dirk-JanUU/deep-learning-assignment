from sklearn.preprocessing import StandardScaler
import numpy as np

class DataScaler:
    def __init__(self, scaler=StandardScaler()):
        self.scaler = scaler

    def fit(self, data):
        self.scaler.fit(data)

    def transform(self, data):
        return self.scaler.transform(data)

    def fit_transform(self, data):
        return self.scaler.fit_transform(data)
    
class DataSequenceGenerator:

    def create_sequences(x_data, y_data, sequence_length=256, step_size=128):

        sequences = []
        labels = []

        for matrix, label in zip(x_data, y_data):

            # Original: (features, time) convert to (time, features)
            matrix = matrix.T

            total_time_steps = matrix.shape[0]

            # Sliding window
            for start in range(
                0,
                total_time_steps - sequence_length,
                step_size
            ):

                end = start + sequence_length

                sequence = matrix[start:end]

                sequences.append(sequence)
                labels.append(label)

        return np.array(sequences), np.array(labels)