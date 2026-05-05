import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils.data_utils import data_converter, create_sequences


# kernel_size = size of the convolutional kernel, looks for one feature with given size
# padding = 1 to keep the output size the same as the input size after convolution, common prctice is (F-1)/2 where F is the kernel size
# stride = 1 to move the kernel one step at a time
# dropout_prob = 0.2 for regularization because of low amount of neurons is 0.2 maybe a good value
class ConvolutionalNetwork(nn.Module):
    def __init__(self, kernel_size=3, padding=1, stride=1, dropout_prob=0.2):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 64, kernel_size=kernel_size, padding=padding, stride=stride)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=kernel_size, padding=padding, stride=stride)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # adaptive pooling should do the trick in priventing wrong input sizes to the fully connected layer.
        self.adaptive_pool = nn.AdaptiveMaxPool1d(1)

        # go from tensor shape to vector shape
        self.flatten = nn.Flatten()

        # go from 128 input features to 50 hidden neurons, then to 1 output feature for the prediction
        self.fc1 = nn.Linear(128, 50)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(50, 1) 

    # Conv1d expects input in the shape: [batch_size, channels (features), sequence_length] 
    # thus in our case it should be x = [batch_size, 1 (laser measurement), lookback]
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        x = self.adaptive_pool(x)
        x = self.flatten(x)

        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# x_train, y_train, x_val, y_val = training and validation datasets
# lr =  learning rate for the gradient descent optimization
# epochs = number of times the entire training dataset is passed through the model during training
# batch_size = number of samples processed in one batch
def train_model(lookback, X_train, y_train, X_val, y_val,
                lr=0.001, epochs=50, batch_size=32):

    model = ConvolutionalNetwork()

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=False
    )

    loss_on_train = []
    loss_on_val = []

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

            train_preds = model(X_train)
            val_preds = model(X_val)

            train_loss = loss_fn(train_preds, y_train)
            val_loss = loss_fn(val_preds, y_val)

            loss_on_train.append(train_loss.item())
            loss_on_val.append(val_loss.item())

    model.eval()

    with torch.no_grad():
        exo_val_preds = model(X_val)
        exo_val_loss = loss_fn(exo_val_preds, y_val)

        intro_val_preds = []

        current_window = X_train[-1].clone()
        for _ in range(len(y_val)):
            pred = model(current_window.unsqueeze(0))
            pred_value = pred.item()
            intro_val_preds.append(pred_value)
            current_window = torch.roll(current_window,shifts=-1, dims=1)
            current_window[0, -1] = pred_value

        intro_val_preds = torch.tensor(intro_val_preds,dtype=torch.float32).unsqueeze(1)
        intro_val_loss = loss_fn(intro_val_preds,y_val)

        # I want also to study where the model is wrong the most (expecting at reset moments)
        sample_loss_fn = nn.MSELoss(reduction='none')
        per_timestep_loss = sample_loss_fn(train_preds, y_train)
        per_timestep_loss = per_timestep_loss.squeeze().numpy()

    return exo_val_loss.item(), exo_val_preds.numpy(), intro_val_loss.item(), intro_val_preds.numpy(), loss_on_train, loss_on_val, per_timestep_loss

def visualize_targets(y_train, y_val):
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

if __name__ == "__main__":

    converter = data_converter()
    scaled_values, scaler = converter.load_scaled_data("Xtrain.csv")

    results = []

    # not to small lookback with kernel size of 3.
    for lookback in [8, 16, 32, 64, 128]:

        X, y = create_sequences(scaled_values, lookback)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.2,
            shuffle=False
        )

        X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

        X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
        y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

        exo_val_loss, exo_preds, intro_val_loss, intro_preds, loss_on_train, loss_on_val, per_timestep_loss = train_model(lookback, X_train, y_train, X_val, y_val)

        plt.figure(figsize=(10, 7))
        plt.plot(loss_on_train, label="Training")
        plt.plot(loss_on_val, label="Validation")
        plt.title(f"Loss evolution through epochs (Lookback={lookback})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss Value")
        plt.legend()
        plt.show()

        fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(10, 7),sharex=True)
        ax1.plot(y_train.squeeze().numpy(), label="Training Values")
        ax1.set_title(f"Training Values vs Per-Timestep Loss (Lookback={lookback})")
        ax1.set_ylabel("Value")
        ax1.legend()
        ax2.plot(per_timestep_loss, label="Loss", color = "red")
        ax2.set_xlabel("Timestep")
        ax2.set_ylabel("Squared Error")
        ax2.legend()
        plt.show()


        #if lookback == 128:
        #    visualize_dataset(y_train, y_val)

        intro_preds_original = converter.reverse_scaled_data(intro_preds)
        exo_preds_original = converter.reverse_scaled_data(exo_preds)
        targets_original = converter.reverse_scaled_data(y_val)

        plt.figure(figsize=(10, 5))
        plt.plot(exo_preds_original, label="Predictions")
        plt.plot(intro_preds_original, label="Self-Predictions")
        plt.plot(targets_original, label="Targets")
        plt.title(f"Predictions vs Targets (Lookback={lookback})")
        plt.xlabel("Time step")
        plt.ylabel("Value")
        plt.legend()
        plt.show()

        results.append({
            "lookback": lookback,
            "normal values_loss": intro_val_loss,
            "self propagating loss": exo_val_loss,
            "predictions_original": intro_preds_original,
            "targets_original": targets_original
        })

    print("\nSummary:")
    for r in results:
        print(f"\nLookback: {r['lookback']}")
        print(f"Values loss: {r['intro_val_loss']:.6f}")
        print("Sample predictions:")
        print(r["predictions_original"][:10])
        print("Sample targets:")
        print(r["targets_original"][:10])