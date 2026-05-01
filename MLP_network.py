import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from utils.data_utils import data_converter, create_sequences
import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

def train_model(lookback, X_train, y_train, X_val, y_val, hidden_size=16, lr=0.001, epochs=50):

    model = MLP(input_size=lookback, hidden_size=hidden_size, output_size=1)
    loss_fn = nn.MSELoss()

    # adam sounded efficient and good to use in the lecture
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()

        preds = model(X_train)
        loss = loss_fn(preds, y_train)

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

    # I guess lookback is enough to use as a hyperparameter to test, learning rate and hidden size could maybe also be tested
    for lookback in [2, 4, 8, 16, 32, 64, 128]:

        X, y = create_sequences(scaled_values, lookback)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.2,
            shuffle=False
        )

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

        if lookback == 128:  # Just plot for one lookback value to visualize the data
            y_train_np = y_train.squeeze().numpy()
            y_val_np = y_val.squeeze().numpy()

            # create x-axis indices
            train_idx = range(len(y_train_np))
            val_idx = range(len(y_train_np), len(y_train_np) + len(y_val_np))

            plt.plot(train_idx, y_train_np, label="Train")
            plt.plot(val_idx, y_val_np, label="Validation")

            plt.title("Train vs Validation Targets")
            plt.xlabel("Time step")
            plt.ylabel("Value")
            plt.legend()
            plt.show()

        val_loss, predictions, targets = train_model(lookback, X_train, y_train, X_val, y_val)

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
        print(f"Values loss: {r['values_loss']:.6f}")
        print("Sample predictions:")
        print(r["predictions_original"][:10])
        print("Sample targets:")
        print(r["targets_original"][:10])