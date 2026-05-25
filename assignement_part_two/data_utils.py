from sklearn.preprocessing import MinMaxScaler

class DataScaler:
    def __init__(self, scaler=MinMaxScaler()):
        self.scaler = scaler

    def fit(self, data):
        self.scaler.fit(data)

    def transform(self, data):
        return self.scaler.transform(data)

    def fit_transform(self, data):
        return self.scaler.fit_transform(data)
    
class DataBatchGenerator:

    def create_sequences(self, data, sequence_length):
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i + sequence_length])
        return sequences
    