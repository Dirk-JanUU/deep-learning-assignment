import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from mne.decoding import CSP
from experiments.Wavelet_experiment import Wavelet_CNN1, Wavelet_CNN2
from LTSM_network import retrieve_context, convert_data, train_model, test_model, create_sequences, down_sample, min_max_scaling, LongShortTermMemoryNetwork
from CNN_network import ConvolutionalNeuralNetwork
from matplotlib import pyplot as plt
import time

N_BANDS = 5
DOWNSAMPLE_FACTOR = 80
SFREQ = 2034 / DOWNSAMPLE_FACTOR

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
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
    
    
def run_and_plot(model = "all", task_type = "intra",  N_CSP = 4, downsample_factor = DOWNSAMPLE_FACTOR):
    # This function runs experiments, plot epoch loss, training time, and test accuracy (3 plots)
    if model not in ["all", "standard", "csp1", "csp2", "lstm", "wave", "wave2"]:
        raise ValueError("model must be one of 'all', 'standard', 'csp1', 'csp2', 'lstm', 'wave', 'wave2'")
    
    N_CSP = 4 # number of CSP components to keep; typically 4-8 is good for 2-class problems, more for multi-class

    if task_type == "intra":
        context_train = retrieve_context(parent_folder="Final_project_data", subdirectory="Intra", type_of_data="train")
        X_tensor_train, y_tensor_train = convert_data(down_sample, min_max_scaling, create_sequences, context_train, down_fact=downsample_factor)
        X_tensor_train = X_tensor_train.permute(0, 2, 1)  # (trials, electrodes, timepoints)
        print(f"Training data shape after permute: {X_tensor_train.shape}")
        print(f"Training labels shape: {y_tensor_train.shape}")
        print(f"Training labels: {list(context_train.labels.values())}")

        context_test = retrieve_context(parent_folder="Final_project_data", subdirectory="Intra", type_of_data="test")
        X_tensor_test, y_tensor_test = convert_data(down_sample, min_max_scaling, create_sequences, context_test, down_fact=downsample_factor)
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
            # standard multi perceptron network
            mlp_model = MultiLayerPerceptron(input_size=X_train_t.shape[1] * X_train_t.shape[2], hidden_size=64, output_size=n_classes)
            time_0 = time.time()
            mlp_losses = train_model(mlp_model, X_train_t, y_train_t, nn.CrossEntropyLoss(), optim.Adam(mlp_model.parameters(), lr=0.001), num_epochs=20)
            mlp_training_time = time.time() - time_0
            mlp_model.eval()
            with torch.no_grad():
                mlp_preds = mlp_model(X_test_t).argmax(1).numpy()
            mlp_acc = (mlp_preds == y_test_t.numpy()).mean()
            print(f"\nMLP - Training time: {mlp_training_time:.2f}s, Test accuracy: {mlp_acc:.4f}")

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

            # Train LSTM
            X_train_t_ltsm = X_train_t.permute(0, 2, 1)  # (batch_size, sequence_length, electrodes)
            X_test_t_ltsm = X_test_t.permute(0, 2, 1)
            lstm_model = LongShortTermMemoryNetwork(input_size=X_train_t_ltsm.shape[2], hidden_size=64, layers=2, output_size=n_classes,dropout=0.5)
            time_0 = time.time()
            lstm_losses = train_model(lstm_model, X_train_t_ltsm, y_train_t, nn.CrossEntropyLoss(), optim.Adam(lstm_model.parameters(), lr=0.001), num_epochs=20)
            lstm_training_time = time.time() - time_0
            lstm_model.eval()
            with torch.no_grad():
                lstm_preds = lstm_model(X_test_t_ltsm).argmax(1).numpy()
            lstm_acc = (lstm_preds == y_test_t.numpy()).mean()

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

            # Train Wavelet CNN1
            w1 = Wavelet_CNN1(X_train_t.shape[1], SFREQ, n_classes)
            time_0 = time.time()
            w1_losses = train_model(w1, X_train_t, y_train_t, nn.CrossEntropyLoss(), optim.Adam(w1.parameters(), lr=0.001), num_epochs=20)
            w1_training_time = time.time() - time_0
            w1.eval()
            with torch.no_grad():
                w1_preds = w1(X_test_t).argmax(1).numpy()
            w1_acc = (w1_preds == y_test_t.numpy()).mean()
            print(f"Wavelet CNN1 - Training time: {w1_training_time:.2f}s, Test accuracy: {w1_acc:.4f}")

            # Train Wavelet CNN2
            w2 = Wavelet_CNN2(X_train_t.shape[1], SFREQ, n_classes)
            time_0 = time.time()
            w2_losses = train_model(w2, X_train_t, y_train_t, nn.CrossEntropyLoss(), optim.Adam(w2.parameters(), lr=0.001), num_epochs=20)
            w2_training_time = time.time() - time_0
            w2.eval()
            with torch.no_grad():
                w2_preds = w2(X_test_t).argmax(1).numpy()
            w2_acc = (w2_preds == y_test_t.numpy()).mean()
            print(f"Wavelet CNN2 - Training time: {w2_training_time:.2f}s, Test accuracy: {w2_acc:.4f}")

            # Plot results
            plt.figure(figsize=(15, 4))
            plt.suptitle(task_type)
            
            plt.subplot(1, 3, 1)
            plt.plot(mlp_losses, label="MLP")
            plt.plot(std_losses, label="CNN")
            plt.plot(lstm_losses, label="LSTM")
            plt.plot(csp1_losses, label="CSP-1")
            plt.plot(csp2_losses, label="CSP-2")
            plt.plot(w1_losses, label="Wave-1")
            plt.plot(w2_losses, label="Wave-2")
            plt.title("Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            
            plt.subplot(1, 3, 2)
            plt.bar(["MLP", "CNN","LSTM", "CSP-1", "CSP-2", "Wave-1", "Wave-2"], [mlp_acc, std_acc, lstm_acc, csp1_acc, csp2_acc, w1_acc, w2_acc])
            plt.ylim(0, 1)
            plt.title("Test Accuracy")
            plt.ylabel("Accuracy")
            
            plt.subplot(1, 3, 3)
            plt.bar(["MLP", "CNN", "LSTM", "CSP-1", "CSP-2", "Wave-1", "Wave-2"], [mlp_training_time, std_training_time, lstm_training_time, csp1_training_time, csp2_training_time, w1_training_time, w2_training_time])
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
            plt.bar(["CNN"], [acc])
            plt.ylim(0, 1)
            plt.title("Standard CNN Test Accuracy")
            plt.ylabel("Accuracy")
        
            plt.subplot(1, 3, 3)
            plt.bar(["CNN"], [training_time])
            plt.title("CNN Training Time")
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

        # X_train_t:[384, 248, 256]
        # model expects [batch_size, sequence_length, input_size]
        elif model == "lstm":
            X_train_t = X_train_t.permute(0, 2, 1)  # (batch_size, sequence_length, electrodes)
            X_test_t = X_test_t.permute(0, 2, 1)
            lstm_model = LongShortTermMemoryNetwork(input_size=X_train_t.shape[2], hidden_size=64, layers=2, output_size=n_classes)
            time_0 = time.time()
            losses = train_model(lstm_model, X_train_t, y_train_t, nn.CrossEntropyLoss(), optim.Adam(lstm_model.parameters(), lr=0.001), num_epochs=20)
            training_time = time.time() - time_0
            lstm_model.eval()
            with torch.no_grad():
                preds = lstm_model(X_test_t).argmax(1).numpy()
            acc = (preds == y_test_t.numpy()).mean()

            plt.figure(figsize=(12, 4))
            plt.suptitle(task_type)
            plt.subplot(1, 3, 1)
            plt.plot(losses)
            plt.title("LSTM Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.subplot(1, 3, 2)
            plt.bar(["LSTM"], [acc])
            plt.ylim(0, 1)
            plt.title("LSTM Test Accuracy")
            plt.ylabel("Accuracy")
            plt.subplot(1, 3, 3)
            plt.bar(["LSTM"], [training_time])
            plt.title("LSTM Training Time")
            plt.ylabel("Time (s)")
            plt.tight_layout()
            plt.show()

        if (model == "wave"):
            w1 = Wavelet_CNN1(X_train_t.shape[1], SFREQ, n_classes)
            time_0 = time.time()
            w1_losses = train_model(w1, X_train_t, y_train_t, nn.CrossEntropyLoss(), optim.Adam(w1.parameters(), lr=0.001), num_epochs=20)
            w1_training_time = time.time() - time_0
            w1.eval()
            with torch.no_grad():
                w1_preds = w1(X_test_t).argmax(1).numpy()
            w1_acc = (w1_preds == y_test_t.numpy()).mean()

            plt.figure(figsize=(12, 4))
            plt.suptitle(task_type)
            plt.subplot(1, 3, 1)
            plt.plot(w1_losses)
            plt.title("Wavelet CNN1 Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.subplot(1, 3, 2)
            plt.bar(["Wavelet CNN1"], [w1_acc])
            plt.ylim(0, 1)
            plt.title("Wavelet CNN1 Test Accuracy")
            plt.ylabel("Accuracy")
            plt.subplot(1, 3, 3)
            plt.bar(["Wavelet CNN1"], [w1_training_time])
            plt.title("Wavelet CNN1 Training Time")
            plt.ylabel("Time (s)")
            plt.tight_layout()
            plt.show()

        if (model == "wave2"):
            w2 = Wavelet_CNN2(X_train_t.shape[1], SFREQ, n_classes)
            time_0 = time.time()
            w2_losses = train_model(w2, X_train_t, y_train_t, nn.CrossEntropyLoss(), optim.Adam(w2.parameters(), lr=0.001), num_epochs=20)
            w2_training_time = time.time() - time_0
            w2.eval()
            with torch.no_grad():
                w2_preds = w2(X_test_t).argmax(1).numpy()
            w2_acc = (w2_preds == y_test_t.numpy()).mean()

            plt.figure(figsize=(12, 4))
            plt.suptitle(task_type)
            plt.subplot(1, 3, 1)
            plt.plot(w2_losses)
            plt.title("Wavelet CNN2 Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.subplot(1, 3, 2)
            plt.bar(["Wavelet CNN2"], [w2_acc])
            plt.ylim(0, 1)
            plt.title("Wavelet CNN2 Test Accuracy")
            plt.ylabel("Accuracy")
            plt.subplot(1, 3, 3)
            plt.bar(["Wavelet CNN2"], [w2_training_time])
            plt.title("Wavelet CNN2 Training Time")
            plt.ylabel("Time (s)")
            plt.tight_layout()
            plt.show()

    if task_type == "cross":
         
        tests = ["test1", "test2", "test3"]
        context_train = retrieve_context(parent_folder="Final_project_data", subdirectory="Cross", type_of_data="train")
        X_tensor_train, y_tensor_train = convert_data(down_sample, min_max_scaling, create_sequences, context_train, down_fact=downsample_factor)
        X_tensor_train = X_tensor_train.permute(0, 2, 1)  # (trials, electrodes, timepoints)
        print(f"Training data shape after permute: {X_tensor_train.shape}")
        print(f"Training labels shape: {y_tensor_train.shape}")
        print(f"Training labels: {list(context_train.labels.values())}")

        n_classes = len(set(context_train.labels.values()))
        csp = CSP(n_components=N_CSP, reg=None, log=False, norm_trace=False)
        csp.fit(X_tensor_train, y_tensor_train)
        filters = csp.filters_[:N_CSP]  # (n_csp, n_channels)

        X_tensor_tests = []
        Y_tensor_tests = []

        for test in tests:
            context_test = retrieve_context(parent_folder="Final_project_data", subdirectory="Cross", type_of_data=test)
            X_tensor_test, y_tensor_test = convert_data(down_sample, min_max_scaling, create_sequences, context_test, down_fact=downsample_factor)
            X_tensor_test = X_tensor_test.permute(0, 2, 1)  # (trials, electrodes, timepoints)
            X_tensor_tests.append(X_tensor_test)
            Y_tensor_tests.append(y_tensor_test)

        if model == "all":

            # MLP
            mlp_model = MultiLayerPerceptron(input_size=X_tensor_train.shape[1] * X_tensor_train.shape[2], hidden_size=128, output_size=n_classes, dropout=0.5)
            time_0 = time.time()
            mlp_losses = train_model(mlp_model, X_tensor_train, y_tensor_train, nn.CrossEntropyLoss(), optim.Adam(mlp_model.parameters(), lr=0.001), num_epochs=20)
            mlp_training_time = time.time() - time_0
            mlp_model.eval()
            mlp_test_accuracies = []
            with torch.no_grad():
                for X_test, y_test in zip(X_tensor_tests, Y_tensor_tests):
                    mlp_preds = mlp_model(X_test).argmax(dim=1).numpy()
                    mlp_acc = (mlp_preds == y_test.numpy()).mean()
                    mlp_test_accuracies.append(mlp_acc)

            # STANDARD CNN
            std_model = ConvolutionalNeuralNetwork(input_size=X_tensor_train.shape[1], output_size=n_classes)
            time_0 = time.time()
            std_losses = train_model(std_model, X_tensor_train, y_tensor_train, nn.CrossEntropyLoss(), optim.Adam(std_model.parameters(), lr=0.001), num_epochs=20)
            std_training_time = time.time() - time_0
            std_model.eval()
            std_test_accuracies = []

            with torch.no_grad():
                for X_test, y_test in zip(X_tensor_tests, Y_tensor_tests):
                    std_preds = std_model(X_test).argmax(1).numpy()
                    std_acc = (std_preds == y_test.numpy()).mean()
                    std_test_accuracies.append(std_acc)

            # LSTM
            X_tensor_train = X_tensor_train.permute(0, 2, 1)  # (batch_size, sequence_length, electrodes)
            X_tensor_tests = [x.permute(0, 2, 1) for x in X_tensor_tests]
            lstm_model = LongShortTermMemoryNetwork(input_size=X_tensor_train.shape[2], hidden_size=64, layers=2, output_size=n_classes,dropout=0.5)
            time_0 = time.time()
            lstm_losses = train_model(lstm_model, X_tensor_train, y_tensor_train, nn.CrossEntropyLoss(), optim.Adam(lstm_model.parameters(), lr=0.001), num_epochs=20)
            lstm_training_time = time.time() - time_0
            lstm_model.eval()
            lstm_test_accuracies = []
            
            with torch.no_grad():
                for X_test, y_test in zip(X_tensor_tests, Y_tensor_tests):
                    lstm_preds = lstm_model(X_test).argmax(1).numpy()
                    lstm_acc = (lstm_preds == y_test.numpy()).mean()
                    lstm_test_accuracies.append(lstm_acc)

            # CSP-CNN1
            X_tensor_train = X_tensor_train.permute(0, 2, 1)  # (batch_size, electrodes, sequence_length)
            csp_cnn1_filters = filters[:N_CSP]
            csp1_model = CSP_CNN1(X_tensor_train.shape[1], n_classes, N_CSP, csp_cnn1_filters)
            time_0 = time.time()
            csp1_losses = train_model(csp1_model, X_tensor_train, y_tensor_train, nn.CrossEntropyLoss(), optim.Adam(csp1_model.parameters(), lr=0.001), num_epochs=20)
            csp1_training_time = time.time() - time_0
            csp1_model.eval()
            csp1_test_accuracies = []
            X_tensor_tests = [x.permute(0, 2, 1) for x in X_tensor_tests]
            with torch.no_grad(): 
                for X_test, y_test in zip(X_tensor_tests, Y_tensor_tests):
                    csp1_preds = csp1_model(X_test).argmax(1).numpy()
                    csp1_acc = (csp1_preds == y_test.numpy()).mean()
                    csp1_test_accuracies.append(csp1_acc)

            # CSP-CNN2
            csp_2_filters = filters[:N_CSP]
            csp2_model = CSP_CNN2(X_tensor_train.shape[1], n_classes, N_CSP, csp_2_filters)
            csp2_time_0 = time.time()
            csp2_losses = train_model(csp2_model, X_tensor_train, y_tensor_train, nn.CrossEntropyLoss(), optim.Adam(csp2_model.parameters(), lr=0.001), num_epochs=20)
            csp2_training_time = time.time() - csp2_time_0
            print(f"\nCSP as middle layer Training time: {csp2_training_time:.2f} seconds")
            csp2_model.eval()
            csp2_test_accuracies = []
            
            with torch.no_grad():
                for X_test, y_test in zip(X_tensor_tests, Y_tensor_tests):
                    csp2_preds = csp2_model(X_test).argmax(1).numpy()
                    csp2_acc = (csp2_preds == y_test.numpy()).mean()
                    csp2_test_accuracies.append(csp2_acc)

            # Wavelet CNN1
            w1 = Wavelet_CNN1(X_tensor_train.shape[1], SFREQ, n_classes)
            time_0 = time.time()
            w1_losses = train_model(w1, X_tensor_train, y_tensor_train, nn.CrossEntropyLoss(), optim.Adam(w1.parameters(), lr=0.001), num_epochs=20)
            w1_training_time = time.time() - time_0
            w1.eval()
            w1_test_accuracies = []
            with torch.no_grad():
                for X_test, y_test in zip(X_tensor_tests, Y_tensor_tests):
                    w1_preds = w1(X_test).argmax(1).numpy()
                    w1_acc = (w1_preds == y_test.numpy()).mean()
                    w1_test_accuracies.append(w1_acc)

            # Wavelet CNN2
            w2 = Wavelet_CNN2(X_tensor_train.shape[1], SFREQ, n_classes)
            time_0 = time.time()
            w2_losses = train_model(w2, X_tensor_train, y_tensor_train, nn.CrossEntropyLoss(), optim.Adam(w2.parameters(), lr=0.001), num_epochs=20)
            w2_training_time = time.time() - time_0
            w2.eval()
            w2_test_accuracies = []

            with torch.no_grad():
                for X_test, y_test in zip(X_tensor_tests, Y_tensor_tests):
                    w2_preds = w2(X_test).argmax(1).numpy()
                    w2_acc = (w2_preds == y_test.numpy()).mean()
                    w2_test_accuracies.append(w2_acc)

            plt.figure(figsize=(15, 4))
            plt.suptitle(task_type)
            plt.subplot(1, 3, 1)
            plt.plot(mlp_losses, label="MLP")
            plt.plot(std_losses, label="CNN")
            plt.plot(lstm_losses, label="LSTM")
            plt.plot(csp1_losses, label="CSP-1")
            plt.plot(csp2_losses, label="CSP-2")
            plt.plot(w1_losses, label="Wave-1")
            plt.plot(w2_losses, label="Wave-2")
            plt.title("Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.subplot(1, 3, 2)
            models = ["MLP", "CNN", "LSTM", "CSP-1", "CSP-2", "Wave-1", "Wave-2"]
            x = np.arange(len(models))
            width = 0.25

            num_runs = len(std_test_accuracies)

            for i in range(num_runs):
                plt.bar(
                    x + (i - (num_runs - 1) / 2) * width,
                    [
                        mlp_test_accuracies[i],
                        std_test_accuracies[i],
                        lstm_test_accuracies[i],
                        csp1_test_accuracies[i],
                        csp2_test_accuracies[i],
                        w1_test_accuracies[i],
                        w2_test_accuracies[i]
                    ],
                    width,
                    label=f"Test {i+1}",
                )

            plt.xticks(x, models)
            plt.ylim(0, 1)
            plt.ylabel("Accuracy")
            plt.title("Test Accuracy")
            plt.legend()
            plt.subplot(1, 3, 3)
            plt.bar(["MLP", "CNN", "LSTM", "CSP-1", "CSP-2", "Wave-1", "Wave-2"], [mlp_training_time, std_training_time, lstm_training_time, csp1_training_time, csp2_training_time, w1_training_time, w2_training_time])
            plt.title("Training Time")
            plt.ylabel("Time (s)")
            plt.tight_layout()
            plt.show()

        if model == "csp1":
            # CSP1
            csp1_model = CSP_CNN1(X_tensor_train.shape[1], n_classes, N_CSP, filters)
            csp1_time_0 = time.time()
            csp1_losses = train_model(csp1_model, X_tensor_train, y_tensor_train, nn.CrossEntropyLoss(), optim.Adam(csp1_model.parameters(), lr=0.001), num_epochs=20)
            csp1_training_time = time.time() - csp1_time_0
            print(f"\nCSP as input layer Training time: {csp1_training_time:.2f} seconds")
            csp1_model.eval()
            csp1_test_accuracies = []
            
            with torch.no_grad():
                for X_test, y_test in zip(X_tensor_tests, Y_tensor_tests):
                    csp1_preds = csp1_model(X_test).argmax(1).numpy()
                    csp1_acc = (csp1_preds == y_test.numpy()).mean()
                    csp1_test_accuracies.append(csp1_acc)

            plt.figure(figsize=(15, 4))
            plt.suptitle(task_type)
            plt.subplot(1, 3, 1)
            plt.plot(csp1_losses, label="CSP-CNN1")
            plt.title("CSP-CNN1 Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.subplot(1, 3, 2)
            models = ["CSP-CNN1"]
            x = np.arange(len(models))
            width = 0.25
            num_runs = len(csp1_test_accuracies)
            for i in range(num_runs):
                plt.bar(
                    x + (i - (num_runs - 1) / 2) * width,
                    [csp1_test_accuracies[i]],
                    width,
                    label=f"Test {i+1}",
                )
            plt.xticks(x, models)
            plt.ylim(0, 1)
            plt.ylabel("Accuracy")
            plt.title("Test Accuracy")
            plt.legend()
            plt.subplot(1, 3, 3)
            plt.bar(["CSP-CNN1"], [csp1_training_time])
            plt.title("CSP-CNN1 Training Time")
            plt.ylabel("Time (s)")
            plt.tight_layout()
            plt.show()

        if model == "csp2":
            # CSP2
            csp2_model = CSP_CNN2(X_tensor_train.shape[1], n_classes, N_CSP, filters)
            csp2_time_0 = time.time()
            csp2_losses = train_model(csp2_model, X_tensor_train, y_tensor_train, nn.CrossEntropyLoss(), optim.Adam(csp2_model.parameters(), lr=0.001), num_epochs=20)
            csp2_training_time = time.time() - csp2_time_0
            print(f"\nCSP as middle layer Training time: {csp2_training_time:.2f} seconds")
            csp2_model.eval()
            csp2_test_accuracies = []
            
            with torch.no_grad():
                for X_test, y_test in zip(X_tensor_tests, Y_tensor_tests):
                    csp2_preds = csp2_model(X_test).argmax(1).numpy()
                    csp2_acc = (csp2_preds == y_test.numpy()).mean()
                    csp2_test_accuracies.append(csp2_acc)

            plt.figure(figsize=(15, 4))
            plt.suptitle(task_type)
            plt.subplot(1, 3, 1)
            plt.plot(csp2_losses, label="CSP-CNN2")
            plt.title("CSP-CNN2 Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.subplot(1, 3, 2)
            models = ["CSP-CNN2"]
            x = np.arange(len(models))
            width = 0.25
            num_runs = len(csp2_test_accuracies)
            for i in range(num_runs):
                plt.bar(
                    x + (i - (num_runs - 1) / 2) * width,
                    [csp2_test_accuracies[i]],
                    width,
                    label=f"Test {i+1}",
                )
            plt.xticks(x, models)
            plt.ylim(0, 1)
            plt.ylabel("Accuracy")
            plt.title("Test Accuracy")
            plt.legend()
            plt.subplot(1, 3, 3)
            plt.bar(["CSP-CNN2"], [csp2_training_time])
            plt.title("CSP-CNN2 Training Time")
            plt.ylabel("Time (s)")
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run CSP-CNN experiments")
    parser.add_argument("--model", type=str, default="all", choices=["all", "standard", "csp1", "csp2", "lstm", "wave", "wave2", "mlp"], help="Which model to run")
    parser.add_argument("--task", type=str, default = "intra", choices=["intra", "contra"], help = "Which kind of neuroimaging task to learn")
    args = parser.parse_args()
    run_and_plot(model="all", task_type="cross")



# i was here :D