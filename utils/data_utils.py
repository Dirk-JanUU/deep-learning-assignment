import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class data_converter:
    def __init__(self, scaler=MinMaxScaler()):
        self.scaler = scaler

    def load_scaled_data(self,file_path):
        dataframe = pd.read_csv(file_path)

        values = dataframe.values

        scaled = self.scaler.fit_transform(values)

        return scaled.flatten(), self.scaler
    
    def reverse_scaled_data(self, data):
        return self.scaler.inverse_transform(data)
    
def create_sequences( data, lookback):
        X, y = [], []

        for i in range(len(data) - lookback):
            X.append(data[i:i + lookback])
            y.append(data[i + lookback])

        # To test wether the model actually learns order, I try to shuffle: COMMENT DO DISABLE IT
        #for i in range(len(data) - lookback):
        #     np.random.seed(2)
        #     np.random.shuffle(X[i])
        #

        return np.array(X), np.array(y)