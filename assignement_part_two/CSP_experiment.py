import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from mne.decoding import CSP
from sklearn.model_selection import train_test_split
import read_data
from LTSM_network import retrieve_context, convert_data, train_model, test_model, create_sequences, down_sample, min_max_scaling
from CNN_network import ConvolutionalNeuralNetwork
#from transformer import train, test

from matplotlib import pyplot as plt
import time

class CSP_CNN1(nn.Module):
    def __init__(self, electrodes, n_classes, n_csp_components, csp_filters, dropout=0.3):
        super().__init__()
        self.csp_layer = nn.Linear(electrodes, n_csp_components, bias=False)
        self.csp_layer.weight = nn.Parameter(
            torch.tensor(csp_filters, dtype=torch.float32),
            requires_grad=True
        )
        self.temporal_convs = nn.Sequential(
            nn.Conv2d(n_csp_components, 30, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(30),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(30, 10, kernel_size=(1, 10), padding=(0, 2), dilation=(1, 10)),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),               # (batch, 10, 1, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                # (batch, 10)
            nn.Linear(10, 40),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(40, n_classes)
        )

    def forward(self, x):
        # x: (batch, electrodes, time)
        x = x.permute(0, 2, 1)         # (batch, time, electrodes)
        x = self.csp_layer(x)           # (batch, time, n_csp)
        x = x.permute(0, 2, 1)         # (batch, n_csp, time)
        x = x.unsqueeze(2)             # (batch, n_csp, 1, time)
        x = self.temporal_convs(x)     # (batch, 10, 1, 1)
        return self.classifier(x)


class CSP_CNN2(nn.Module):
    def __init__(self, electrodes, n_classes, n_csp_components, csp_filters, dropout=0.3):
        super().__init__()
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(electrodes, electrodes, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(electrodes),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d((1, 2)),
        )
        self.csp_layer = nn.Linear(electrodes, n_csp_components, bias=False)
        self.csp_layer.weight = nn.Parameter(
            torch.tensor(csp_filters, dtype=torch.float32),
            requires_grad=True
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(n_csp_components, n_csp_components, kernel_size=(1, 9),
                      padding=(0, 4), groups=n_csp_components),
            nn.BatchNorm2d(n_csp_components),
            nn.ReLU(),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(n_csp_components, 10, kernel_size=1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),               # (batch, 10, 1, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                # (batch, 10)
            nn.Linear(10, 40),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(40, n_classes)
        )

    def forward(self, x):
        # x: (batch, electrodes, time)
        x = x.unsqueeze(2)             # (batch, electrodes, 1, time)
        x = self.temporal_conv(x)       # (batch, electrodes, 1, time/2)
        x = x.squeeze(2)               # (batch, electrodes, time/2)
        x = x.permute(0, 2, 1)         # (batch, time/2, electrodes)
        x = self.csp_layer(x)          # (batch, time/2, n_csp)
        x = x.permute(0, 2, 1)         # (batch, n_csp, time/2)
        x = x.unsqueeze(2)             # (batch, n_csp, 1, time/2)
        x = self.depthwise(x)          # (batch, n_csp, 1, time/2)
        x = self.pointwise(x)          # (batch, 10, 1, 1)
        return self.classifier(x)
    
    
def run_and_plot(model = "all", N_CSP = 4, task_type = "intra"):
    # This function runs experiments, plot epoch loss, training time, and test accuracy (3 plots)
    if model not in ["all", "standard", "csp1", "csp2"]:
        raise ValueError("model must be one of 'all', 'standard', 'csp1', 'csp2'")
    
    N_CSP = 4 # number of CSP components to keep; typically 4-8 is good for 2-class problems, more for multi-class

    if task_type == "intra":
        context_train = retrieve_context(parent_folder="Final_project_data", subdirectory="Intra", type_of_data="train")
        X_tensor_train, y_tensor_train = convert_data(down_sample, min_max_scaling, create_sequences, context_train)
        X_tensor_train = X_tensor_train.permute(0, 2, 1)  # (trials, electrodes, timepoints)
        print(f"Training data shape after permute: {X_tensor_train.shape}")
        print(f"Training labels shape: {y_tensor_train.shape}")
        print(f"Training labels: {list(context_train.labels.values())}")

        context_test = retrieve_context(parent_folder="Final_project_data", subdirectory="Intra", type_of_data="test")
        X_tensor_test, y_tensor_test = convert_data(down_sample, min_max_scaling, create_sequences, context_test)
        X_tensor_test = X_tensor_test.permute(0, 2, 1)  # (trials, electrodes, timepoints)
        #print(f"Test data shape after permute: {X_tensor_test.shape}")
        #print(f"Test labels shape: {y_tensor_test.shape}")

        n_classes = len(set(context_train.labels.values()))

        # CSP(n_trials, n_channels, n_timepoints) 
        csp = CSP(n_components=N_CSP, reg=None, log=False, norm_trace=False)
        csp.fit(X_tensor_train, y_tensor_train)
        # csp.filters_ shape: (n_channels, n_channels); take first N_CSP rows
        filters = csp.filters_[:N_CSP]  # (n_csp, n_channels)
        #X_train_csp = np.array([filters @ trial for trial in X_train])  # (trials, n_csp, timepoints)
        #X_test_csp = np.array([filters @ trial for trial in X_test])    # (trials, n_csp, timepoints)

        X_train_t = torch.as_tensor(X_tensor_train, dtype=torch.float32)
        X_test_t = torch.as_tensor(X_tensor_test, dtype=torch.float32)
        y_train_t = torch.as_tensor(y_tensor_train, dtype=torch.long)
        y_test_t = torch.as_tensor(y_tensor_test, dtype=torch.long)

        print(f"Data shapes: X_train {X_train_t.shape}, y_train {y_train_t.shape}, X_test {X_test_t.shape}, y_test {y_test_t.shape}")

        n_timepoints = X_train_t.shape[2]

        if model == "all":
            # Train standard CNN
            std_model = ConvolutionalNeuralNetwork(input_size=X_train_t.shape[1], output_size=n_classes)
            time_0 = time.time()
            std_losses = train_model(std_model, X_train_t, y_train_t, nn.CrossEntropyLoss(), optim.Adam(std_model.parameters(), lr=0.001), num_epochs=20)
            std_training_time = time.time() - time_0
            std_model.eval()
            with torch.no_grad():
                std_preds = std_model(X_test_t).argmax(1).numpy()
            std_acc = (std_preds == y_test_t.numpy()).mean()
            print(f"\nStandard CNN - Training time: {std_training_time:.2f}s, Test accuracy: {std_acc:.4f}")

            # Train CSP-CNN1
            csp1_model = CSP_CNN1(X_train_t.shape[1], n_classes, N_CSP, filters)
            time_0 = time.time()
            csp1_losses = train_model(csp1_model, X_train_t, y_train_t, nn.CrossEntropyLoss(), optim.Adam(csp1_model.parameters(), lr=0.001), num_epochs=20)
            csp1_training_time = time.time() - time_0
            csp1_model.eval()
            with torch.no_grad():
                csp1_preds = csp1_model(X_test_t).argmax(1).numpy()
            csp1_acc = (csp1_preds == y_test_t.numpy()).mean()
            print(f"CSP-CNN1 - Training time: {csp1_training_time:.2f}s, Test accuracy: {csp1_acc:.4f}")

            # Train CSP-CNN2
            csp2_model = CSP_CNN2(X_train_t.shape[1], n_classes, N_CSP, filters)
            time_0 = time.time()
            csp2_losses = train_model(csp2_model, X_train_t, y_train_t, nn.CrossEntropyLoss(), optim.Adam(csp2_model.parameters(), lr=0.001), num_epochs=20)
            csp2_training_time = time.time() - time_0
            csp2_model.eval()
            with torch.no_grad():
                csp2_preds = csp2_model(X_test_t).argmax(1).numpy()
            csp2_acc = (csp2_preds == y_test_t.numpy()).mean()
            print(f"CSP-CNN2 - Training time: {csp2_training_time:.2f}s, Test accuracy: {csp2_acc:.4f}")

            # Plot results
            plt.figure(figsize=(15, 4))
            plt.suptitle(task_type)
            
            plt.subplot(1, 3, 1)
            plt.plot(std_losses, label="Standard CNN")
            plt.plot(csp1_losses, label="CSP-CNN1")
            plt.plot(csp2_losses, label="CSP-CNN2")
            plt.title("Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            
            plt.subplot(1, 3, 2)
            plt.bar(["Standard CNN", "CSP-CNN1", "CSP-CNN2"], [std_acc, csp1_acc, csp2_acc])
            plt.ylim(0, 1)
            plt.title("Test Accuracy")
            plt.ylabel("Accuracy")
            
            plt.subplot(1, 3, 3)
            plt.bar(["Standard CNN", "CSP-CNN1", "CSP-CNN2"], [std_training_time, csp1_training_time, csp2_training_time])
            plt.title("Training Time")
            plt.ylabel("Time (s)")
            
            plt.tight_layout()
            plt.show()

    elif model == "standard":
        std_model = ConvolutionalNeuralNetwork(input_size=X_train_t.shape[1], output_size=n_classes)
        time_0 = time.time()
        losses = train_model(std_model, X_train_t, y_train_t, nn.CrossEntropyLoss(), optim.Adam(std_model.parameters(), lr=0.001), num_epochs=20)
        training_time = time.time() - time_0
        print(f"\nStandard CNN Training time: {training_time:.2f} seconds")
        std_model.eval()
        with torch.no_grad():
            preds = std_model(X_test_t).argmax(1).numpy()
        acc = (preds == y_test_t.numpy()).mean()
        print(f"\nStandard CNN Test accuracy: {acc:.4f}")

        plt.figure(figsize=(12, 4))
        plt.suptitle(task_type)
        
        plt.subplot(1, 3, 1)
        plt.plot(losses)
        plt.title("Standard CNN Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
    
        plt.subplot(1, 3, 2)
        plt.bar(["Standard CNN"], [acc])
        plt.ylim(0, 1)
        plt.title("Standard CNN Test Accuracy")
        plt.ylabel("Accuracy")
    
        plt.subplot(1, 3, 3)
        plt.bar(["Standard CNN"], [training_time])
        plt.title("Standard CNN Training Time")
        plt.ylabel("Time (s)")
        plt.tight_layout()
        plt.show()

    
    elif model == "csp1":
        csp1_model = CSP_CNN1(X_train_t.shape[1], n_classes, N_CSP, filters)
        time_0 = time.time()
        losses = train_model(csp1_model, X_train_t, y_train_t, nn.CrossEntropyLoss(), optim.Adam(csp1_model.parameters(), lr=0.001), num_epochs=20)
        training_time = time.time() - time_0
        print(f"\nCSP as input layer Training time: {training_time:.2f} seconds")
        csp1_model.eval()
        with torch.no_grad():
            preds = csp1_model(X_test_t).argmax(1).numpy()
        acc = (preds == y_test_t.numpy()).mean()
        print(f"\nCSP as input layer Test accuracy: {acc:.4f}")

        plt.figure(figsize=(12, 4))
        plt.suptitle(task_type)

        plt.subplot(1, 3, 1)
        plt.plot(losses)
        plt.title("CSP-CNN1 Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
    
        plt.subplot(1, 3, 2)
        plt.bar(["CSP-CNN1"], [acc])
        plt.ylim(0, 1)
        plt.title("CSP-CNN1 Test Accuracy")
        plt.ylabel("Accuracy")
    
        plt.subplot(1, 3, 3)
        plt.bar(["CSP-CNN1"], [training_time])
        plt.title("CSP-CNN1 Training Time")
        plt.ylabel("Time (s)")
        plt.tight_layout()
        plt.show()

            
    elif model == "csp2":
        csp2_model = CSP_CNN2(X_train_t.shape[1], n_classes, N_CSP, filters)
        train_model(csp2_model, X_train_t, y_train_t, nn.CrossEntropyLoss(),
                    optim.Adam(csp2_model.parameters(), lr=0.001), num_epochs=20)
        csp2_model.eval()
        with torch.no_grad():
            preds = csp2_model(X_test_t).argmax(1).numpy()
        acc = (preds == y_test_t.numpy()).mean()
        print(f"\nCSP as middle layer Test accuracy: {acc:.4f}")

        plt.figure(figsize=(12, 4))
        plt.suptitle(task_type)

        plt.subplot(1, 3, 1)
        plt.plot(losses)
        plt.title("CSP-CNN2 Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")      

        plt.subplot(1, 3, 2)
        plt.bar(["CSP-CNN2"], [acc])
        plt.ylim(0, 1)
        plt.title("CSP-CNN2 Test Accuracy")
        plt.ylabel("Accuracy")
        
        plt.subplot(1, 3, 3)
        plt.bar(["CSP-CNN2"], [training_time])
        plt.title("CSP-CNN2 Training Time")
        plt.ylabel("Time (s)")
        plt.tight_layout()
        plt.show()

       
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run CSP-CNN experiments")
    parser.add_argument("--model", type=str, default="all", choices=["all", "standard", "csp1", "csp2"], help="Which model to run")
    parser.add_argument("--task", type=str, default = "intra", choices=["intra", "contra"], help = "Which kind of neuroimaging task to learn")
    args = parser.parse_args()
    run_and_plot(model=args.model, task_type=args.task)