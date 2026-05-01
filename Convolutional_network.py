import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils.data_utils import data_converter, create_sequences

class ConvolutionalNetwork(nn.Module):
    def __init__(self, kernel_size=3, padding=1):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 64, kernel_size=kernel_size, padding=padding)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=kernel_size, padding=padding)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # Not fully sure how exactly scaling the convolutional output to the original range works, 
        # but adaptive pooling should do the trick in priventing wrong input sizes to the fully connected layer.
        self.adaptive_pool = nn.AdaptiveMaxPool1d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128, 50)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(50, 1) 

    # Conv1d expects input in the shape: [batch_size, channels, sequence_length] 
    # thus in our case it should be x = [batch_size, 1 (laser measurement), lookback]
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        x = self.adaptive_pool(x)
        x = self.flatten(x)

        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(X_train, y_train, X_val, y_val, lr=0.001, epochs=50, batch_size=32):

    model = ConvolutionalNetwork()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=False
    )

    for epoch in range(epochs):
        model.train()

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()

            preds = model(x_batch)
            loss = loss_fn(preds, y_batch)

            loss.backward()
            optimizer.step()

    model.eval()

    with torch.no_grad():
        val_preds = model(X_val)
        val_loss = loss_fn(val_preds, y_val)

    return val_loss.item(), val_preds.numpy(), y_val.numpy()

if __name__ == "__main__":

    converter = data_converter()
    scaled_values, scaler = converter.load_scaled_data("Xtrain.csv")

    results = []

    for lookback in [8, 16, 32, 64, 128]:

        X, y = create_sequences(scaled_values, lookback)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.2,
            shuffle=False
        )

        X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
        X_val   = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)

        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        y_val   = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

        if lookback == 128:
            y_train_np = y_train.squeeze().numpy()
            y_val_np = y_val.squeeze().numpy()

            train_idx = range(len(y_train_np))
            val_idx = range(len(y_train_np), len(y_train_np) + len(y_val_np))

            plt.plot(train_idx, y_train_np, label="Train")
            plt.plot(val_idx, y_val_np, label="Validation")

            plt.title("Train vs Validation Targets")
            plt.xlabel("Time step")
            plt.ylabel("Value")
            plt.legend()
            plt.show()

        val_loss, predictions, targets = train_model(
            X_train, y_train,
            X_val, y_val
        )

        preds_original = converter.reverse_scaled_data(predictions)
        targets_original = converter.reverse_scaled_data(targets)

        results.append({
            "lookback": lookback,
            "values_loss": val_loss,
            "predictions_original": preds_original,
            "targets_original": targets_original
        })

    print("\nSummary:")
    for r in results:
        print(f"\nLookback: {r['lookback']}")
        print(f"Validation loss: {r['values_loss']:.6f}")
        print("Sample predictions:")
        print(r["predictions_original"][:10])
        print("Sample targets:")
        print(r["targets_original"][:10])