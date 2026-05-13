from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from assignment_part_one.data_utils import data_converter, create_sequences


class RecurrentNetwork(nn.Module):
    """
    Recurrent model for univariate time series forecasting.
    """
    def __init__(self, hidden_size=64, num_layers=2, dropout=0.2, cell='LSTM'):
        super().__init__()
        rnn_cls = nn.LSTM
        self.rnn = rnn_cls(
            input_size=1,                                         
            # univariate => 1 feature
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,            
            # dropout only applies between stacked layers
            batch_first=True,                                      
            # input shape: [batch, seq_len, 1]
        )
        self.fc = nn.Linear(hidden_size, 1)

    # Recurrent layers expect input shape [batch_size, seq_len, n_features].
    # For us: [batch_size, lookback, 1]
    def forward(self, x):
        out, _ = self.rnn(x)          # out: [batch, seq_len, hidden]
        last = out[:, -1, :]          # take the output at the final time step
        return self.fc(last)          # [batch, 1]


def train_model(X_train, y_train, X_val, y_val,
                hidden_size=64, num_layers=2, dropout=0.2, cell='LSTM',
                lr=1e-3, epochs=200, batch_size=32, patience=15):
    
    model = RecurrentNetwork(hidden_size, num_layers, dropout, cell)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )

    best_val_mse = float('inf')
    best_val_mae = float('inf')
    best_state = None
    epochs_since_improvement = 0
    history = {"train": [], "val": [], "val_mae": []}

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = functional.mse_loss(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_mse = functional.mse_loss(val_preds, y_val).item()
            val_mae = functional.l1_loss(val_preds, y_val).item()

        history["train"].append(float(np.mean(train_losses)))
        history["val"].append(val_mse)
        history["val_mae"].append(val_mae)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_val_mae = val_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_val_mse, best_val_mae, history

def test_model(X_test, y_test, model):
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test)
        test_mse = functional.mse_loss(test_preds, y_test).item()
        test_mae = functional.l1_loss(test_preds, y_test).item()
    return model, test_mse, test_mae, test_preds

def recursive_forecast(model, seed_window, n_steps):
    """
    Roll the model forward "n_steps" times, feeding each prediction back as the
    next input. This is what the assignment asks for in part (c).

    seed_window: 1-D tensor of length "lookback", in SCALED space
    n_steps: how many steps to forecast (200 for this assignment)
    returns: 1-D numpy array of length n_steps, in SCALED space
    """
    model.eval()
    current = seed_window.clone().reshape(1, -1, 1).float()   
    # [1, lookback, 1]
    predictions = []

    with torch.no_grad():
        for _ in range(n_steps):
            pred = model(current)                             
            # [1, 1]
            predictions.append(pred.item())
            next_step = pred.unsqueeze(-1) 
            # [1, 1, 1]
            # slide the window: drop oldest, append the new prediction
            current = torch.cat([current[:, 1:, :], next_step], dim=1)

    return np.array(predictions)

def evaluate_forecast(predictions, targets):
    """Both arrays in the ORIGINAL (unscaled) space."""
    predictions = np.asarray(predictions).flatten()
    targets = np.asarray(targets).flatten()
    mse = float(np.mean((predictions - targets) ** 2))
    mae = float(np.mean(np.abs(predictions - targets)))
    return mse, mae

if __name__ == "__main__":

    converter = data_converter()
    scaled_values, scaler = converter.load_scaled_data("Xtrain.csv")

    results = []

    for lookback in [16, 32, 64, 128]:

        X, y = create_sequences(scaled_values, lookback)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.2,
            shuffle=False,
        )

        # recurrent layers want [batch, seq_len, n_features], so unravel the feature dim
        X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
        X_val   = torch.tensor(X_val,   dtype=torch.float32).unsqueeze(-1)
        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        y_val   = torch.tensor(y_val,   dtype=torch.float32).unsqueeze(1)

        model, val_mse, val_mae, history = train_model(
            X_train, y_train, X_val, y_val,
            hidden_size=64, num_layers=2, dropout=0.2, cell='LSTM',
            lr=1e-3, epochs=200, batch_size=32, patience=15,
        )

        #print(history)

        # one-step-ahead predictions on validation
        model.eval()
        with torch.no_grad():
            val_preds_scaled = model(X_val).numpy()
            train_preds = model(X_train)
            per_sample_loss = functional.mse_loss(train_preds, y_train, reduction='none')
            per_sample_loss = per_sample_loss.squeeze().numpy()  # shape: (num_samples,)

        fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(10, 7),sharex=True)
        ax1.plot(y_train.squeeze().numpy(), label="Training Values")
        ax1.set_title(f"Training Values vs Per-Timestep Loss (Lookback={lookback})")
        ax1.set_ylabel("Value")
        ax1.legend()
        ax2.plot(per_sample_loss, label="Loss", color = "red")
        ax2.set_xlabel("Timestep")
        ax2.set_ylabel("Squared Error")
        ax2.legend()
        plt.show()

        val_targets_scaled = y_val.numpy()

        preds_original   = converter.reverse_scaled_data(val_preds_scaled)
        targets_original = converter.reverse_scaled_data(val_targets_scaled)

        # part (c): recursive 200-step forecast.
        # seed with the last "lookback" points of the full training series
        # this is what predicts the next 200 when we get test set
        seed = torch.tensor(scaled_values[-lookback:], dtype=torch.float32)
        recursive_scaled = recursive_forecast(model, seed, n_steps=200)
        recursive_original = converter.reverse_scaled_data(
            recursive_scaled.reshape(-1, 1)
        )

        results.append({
            "lookback": lookback,
            "val_mse": val_mse,
            "val_mae": val_mae,
            "history": history,
            "predictions_original": preds_original,
            "targets_original": targets_original,
            "recursive_forecast": recursive_original.flatten(),
        })

        # quick sanity plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))

        axes[0].plot(history["train"], label="train")
        axes[0].plot(history["val"],   label="val")
        axes[0].set_title(f"Loss curves (lookback={lookback})")
        axes[0].set_xlabel("epoch")
        axes[0].legend()

        axes[1].plot(targets_original, label="true")
        axes[1].plot(preds_original,   label="1-step pred", alpha=0.8)
        axes[1].set_title("Validation: 1-step-ahead")
        axes[1].legend()

        axes[2].plot(recursive_original, label="recursive forecast")
        axes[2].set_title("200-step recursive forecast (from end of train)")
        axes[2].set_xlabel("step ahead")
        axes[2].legend()

        plt.tight_layout()
        plt.show()

    print("\nSummary validation:")

    scaled_values_test, scaler_test = converter.load_scaled_data("Xtest.csv")
    
    X_test_data, y_test_data = create_sequences(scaled_values_test, lookback=16)

    X_test = torch.tensor(X_test_data, dtype=torch.float32).unsqueeze(-1)
    y_test = torch.tensor(y_test_data,   dtype=torch.float32).unsqueeze(1)
    
    model, val_mse, val_mae, test_preds = test_model(X_test, y_test, model)

    print("\nSummary test:")
    print(f"\nLookback: {16}  |  Test MSE: {val_mse:.6f}  |  Test MAE: {val_mae:.6f}")

    test_preds_original = converter.reverse_scaled_data(
        test_preds.numpy()
    )

    y_test_original = converter.reverse_scaled_data(
        y_test.numpy()
    )


    plt.plot(results[0]['targets_original'], label="validation 1-step targets")
    plt.plot(results[0]['predictions_original'],   label="validation 1-step predictions", alpha=0.8)
    plt.title(f"Validation: 1-step-ahead(lookback = 16)")
    plt.xlabel("Time")
    plt.ylabel("Laser measurement")
    plt.legend()
    plt.show()

    plt.plot(y_test_original, label="test 1-step targets")
    plt.plot(test_preds_original,   label="test 1-step predictions", alpha=0.8)
    plt.title(f"Test: 1-step-ahead(lookback = 16)")
    plt.xlabel("Time")
    plt.ylabel("Laser measurement")
    plt.legend()
    plt.show()

    
    for r in results:
        print(f"\nLookback: {r['lookback']}  |  val MSE: {r['val_mse']:.6f}  |  val MAE: {r['val_mae']:.6f}")
    
        forecast = np.array(r['recursive_forecast']).flatten()
        true_values = y_test_original.flatten()

        n = min(len(forecast), len(true_values))

        forecast = forecast[:n]
        true_values = true_values[:n]

        squared_error = (forecast - true_values) ** 2
        mse = mean_squared_error(true_values, forecast)

        min_err = squared_error.min()
        max_err = squared_error.max()

        normalized_error = (squared_error - min_err) / (max_err - min_err + 1e-8)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        ax1.plot(forecast, label="recursive forecast")
        ax1.plot(true_values, label="true values", alpha=0.7)

        ax1.set_title(
            f"200-step recursive forecast Test set (lookback={r['lookback']})"
        )
        ax1.set_ylabel("Laser measurement")
        ax1.set_xlabel("Time")
        ax1.legend()

        ax2.plot(
            normalized_error,
            color='red',
            label="normalized squared error"
        )

        normalized_mse = (mse - min_err) / (max_err - min_err + 1e-8)

        ax2.axhline(
            y=normalized_mse,
            color='black',
            linestyle='--',
            label=f'normalized MSE = {normalized_mse:.4f}'
        )

        ax2.set_title("Normalized forecast error (0 to 1)")
        ax2.set_xlabel("Step ahead")
        ax2.set_ylabel("Normalized Error")
        ax2.set_ylim(0, 1)

        ax2.legend()

        plt.tight_layout()
        plt.show()