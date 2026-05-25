import read_data
import data_utils
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

class LongShortTermMemoryNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size = 4, layers = 2, dropout=0.2):
        super(LongShortTermMemoryNetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = layers
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=layers, batch_first=True, dropout=dropout)

        self.norm = nn.LayerNorm(hidden_size)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, output_size)
        )

    # x shape: [batch_size, sequence_length, input_size]
    def forward(self, x):

        lstm_out, (hidden, cell) = self.lstm(x)

        out = lstm_out[:, -1, :]

        out = self.norm(out)

        logits = self.classifier(out)

        return logits


def train_model(model: LongShortTermMemoryNetwork, train_x, train_y, loss_function, optimizer: torch.optim.Optimizer, num_epochs: int, batch_size: int = 32):
    dataset = torch.utils.data.TensorDataset(train_x, train_y)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    for epoch in range(num_epochs):

        model.train()

        total_loss = 0.0

        for batch_x, batch_y in dataloader:

            optimizer.zero_grad()

            outputs = model(batch_x)

            loss = loss_function(outputs, batch_y)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Loss: {avg_loss:.4f}"
        )

if __name__ == "__main__":

    persons, x_data, y_data = read_data.load_data_from_h5_files()
    context = read_data.DataSet(persons, x_data, y_data)

    scaler = data_utils.DataScaler()

    all_sequences = np.concatenate(context.x_data, axis=0)

    scaler.fit(all_sequences)

    for i in range(len(context.x_data)):
        context.x_data[i][:] = scaler.transform(context.x_data[i])

    sequence_length = 256
    step_size = 128

    X, y = data_utils.DataSequenceGenerator.create_sequences(
        context.x_data,
        context.y_data,
        sequence_length=sequence_length,
        step_size=step_size
    )

    X_tensor = torch.tensor(X, dtype=torch.float32)

    y_tensor = torch.tensor(y, dtype=torch.long)

    input_size = X_tensor.shape[2]

    model = LongShortTermMemoryNetwork(input_size=input_size, hidden_size=128, output_size=len(set(context.labels.values())))
    loss_function = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(),lr=0.001)

    # TODO: still have to Find out how to downsample
    train_model(model, X_tensor, y_tensor, loss_function, optimizer, num_epochs=10)




