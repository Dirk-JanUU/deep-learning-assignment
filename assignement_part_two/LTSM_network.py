import read_data
import data_utils
import torch
import torch.nn as nn
import numpy as np
from scipy.signal import decimate

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

def test_model(model: LongShortTermMemoryNetwork, test_x, test_y, loss_function,batch_size: int = 32):

    dataset = torch.utils.data.TensorDataset(test_x, test_y)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for batch_x, batch_y in dataloader:

            outputs = model(batch_x)

            loss = loss_function(outputs, batch_y)

            total_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == batch_y).sum().item()

            total += batch_y.size(0)

    avg_loss = total_loss / len(dataloader)

    accuracy = correct / total

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    with torch.no_grad():

        outputs = model(test_x)

        loss = loss_function(outputs, test_y)

        _, predicted = torch.max(outputs, 1)

        accuracy = (predicted == test_y).float().mean().item()

        print(f"Test Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

# currently 2034 samples per second which times 17.5 for one scane is 35645 samples per scane,
# which divided by 20 is approximatly 102 samples per second which times 17.5 is 1787 samples per scane
def down_sample(context, factor=20):
    context.x_data = [
        decimate(x, factor, axis=1)
        for x in context.x_data
    ]

def min_max_scaling(context):
    for i in range(len(context.x_data)):
        scaler = data_utils.DataScaler()

        scaler.fit(context.x_data[i])

        context.x_data[i][:] = scaler.transform(context.x_data[i])

# with downsampling factor of 20, we have 1787 samples per scane,
# with sequence length of 256 and step size of 128, we get approximately 13 sequences per scane
def create_sequences(context: read_data.DataSet, sequence_length: int = 256, step_size: int = 128):
    X, y = data_utils.DataSequenceGenerator.create_sequences(
        context.x_data,
        context.y_data,
        sequence_length=sequence_length,
        step_size=step_size
    )
    
    return X,y

def convert_data(down_sample, min_max_scaling, create_sequences, context):
    down_sample(context)

    min_max_scaling(context)

    X, y = create_sequences(context)

    X_tensor = torch.tensor(X, dtype=torch.float32)

    y_tensor = torch.tensor(y, dtype=torch.long)

    return X_tensor,y_tensor

def retrieve_context(parent_folder, subdirectory="Intra", type_of_data="train"):
    persons = read_data.load_data_from_h5_files(parent_folder, subdirectory, type_of_data)

    raw_scans = [person.get_scans()[i] for person in persons for i in range(len(person.get_scans()))]
    x_data_train = np.array([scan.matrix for scan in raw_scans])
    y_data_train = np.array([scan.task for scan in raw_scans])

    context_train = read_data.DataSet(persons, x_data_train, y_data_train)
    return context_train
