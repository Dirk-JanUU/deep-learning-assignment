import read_data
import data_utils
import torch
import torch.nn as nn
import numpy as np

class LongShortTermMemoryNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size = 4, layers = 2, dropout=0.2):
        super(LongShortTermMemoryNetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = layers
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    # x contains [batch_size, sequence_length, input_size]
    def forward(self, x):

        lstm_out, (hidden, cell) = self.lstm(x)

        final_hidden = hidden[-1]

        logits = self.fc(final_hidden)

        return logits


def train_model(model, data_x, data_y, loss_function, optimizer, num_epochs):
    for epoch in range(num_epochs):
        
        model.train()
        
        optimizer.zero_grad()

        # TODO: Ensure data_x has the correct shape [batch_size, sequence_length, input_size]
        outputs = model(data_x)

        loss = loss_function(outputs, data_y)

        loss.backward()

        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

if __name__ == "__main__":

    persons, labels, x_data, y_data = read_data.load_data_from_h5_files()
    context = read_data.DataSet(persons, x_data, y_data, labels)

    scaler = data_utils.DataScaler()

    all_sequences = np.concatenate(context.x_data, axis=0)

    scaler.fit(all_sequences)

    for i in range(len(context.x_data)):
        context.x_data[i][:] = scaler.transform(context.x_data[i])

    model = LongShortTermMemoryNetwork(input_size=context.x_data[0].shape[1], hidden_size=128, output_size=len(labels))
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # TODO: Find out how to downsample and
    #  how to split the data into batches like only split the files or also parts of the data within the fill.
    train_model(model, context.x_data, context.y_data, loss_function, optimizer, num_epochs=100)




