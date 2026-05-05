import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from utils.data_utils import data_converter, create_sequences
import matplotlib.pyplot as plt

# input_size = lookback steps
# hidden_size = number of neurons in the hidden layer
# output_size = 1 predicting laser measurements at the next time step
# dropout_prob = 0.2 for regularization because of low amount of neurons is 0.2 maybe a good value
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.2):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

# x_train, y_train, x_val, y_val = training and validation datasets
# lr =  learning rate for the gradient descent optimization
# epochs = number of times the entire training dataset is passed through the model during training
# batch_size = number of samples processed in one batch
def train_model(lookback, X_train, y_train, X_val, y_val, hidden_size=16, lr=0.001, epochs=50, batch_size=32):

    model = MLP(input_size=lookback, hidden_size=hidden_size, output_size=1)
    loss_fn = nn.MSELoss()

    # adam sounded efficient and good to use in the lecture
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

def visualize_targets(y_train, y_val):
    y_train_np = y_train.squeeze().numpy()
    y_val_np = y_val.squeeze().numpy()

    # create x-axis indices
    train_idx = range(len(y_train_np))
    val_idx = range(len(y_train_np), len(y_train_np) + len(y_val_np))

    plt.plot(train_idx, y_train_np, label="Train_labels")
    plt.plot(val_idx, y_val_np, label="Validation_labels")

    plt.title("Train vs Validation Targets")
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

if __name__ == "__main__":

    converter = data_converter()
    scaled_values, scaler = converter.load_scaled_data("Xtrain.csv")

    results = []

    # I guess lookback is enough to use as a hyperparameter to test, compared to many other hyperparameters
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

        if lookback == 2:
            visualize_targets(y_train, y_val)

        val_loss, predictions, targets = train_model(lookback, X_train, y_train, X_val, y_val)

        preds_original = converter.reverse_scaled_data(predictions)
        targets_original = converter.reverse_scaled_data(targets)

        plt.figure(figsize=(10, 5))
        plt.plot(preds_original, label="Predictions")
        plt.plot(targets_original, label="Targets")
        plt.title(f"Predictions vs Targets (Lookback={lookback})")
        plt.xlabel("Time step")
        plt.ylabel("Value")
        plt.legend()
        plt.show()

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