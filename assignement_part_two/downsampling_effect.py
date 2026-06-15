import matplotlib.pyplot as plt
from experiments.LTSM_experiment import convert_data, retrieve_context, min_max_scaling, down_sample, create_sequences, train_model
import torch.optim as optim
import torch
import numpy as np
from CNN_network import ConvolutionalNeuralNetwork
import torch.nn as nn

down_sample_range = [10,100]
down_sample_factors = np.arange(10, 101, 10)

accuracies = []

for factor in down_sample_factors:
    print(f"Testing for Downsampling Factor: {factor}")

    context_train = retrieve_context(parent_folder="Final_project_data", subdirectory="Intra", type_of_data="train")
    context_test = retrieve_context(parent_folder="Final_project_data", subdirectory="Intra", type_of_data="test")
    
    X_tensor_train, y_tensor_train = convert_data(down_sample, min_max_scaling, create_sequences, context_train, down_fact = factor)
    X_tensor_train = X_tensor_train.permute(0, 2, 1)  # (trials, electrodes, timepoints)
    print(f"Training data shape after permute: {X_tensor_train.shape}")
    print(f"Training labels shape: {y_tensor_train.shape}")
    print(f"Training labels: {list(context_train.labels.values())}")

    X_tensor_test, y_tensor_test = convert_data(down_sample, min_max_scaling, create_sequences, context_test, down_fact = factor)
    X_tensor_test = X_tensor_test.permute(0, 2, 1)  # (trials, electrodes, timepoints)
    #print(f"Test data shape after permute: {X_tensor_test.shape}")
    #print(f"Test labels shape: {y_tensor_test.shape}")

    n_classes = len(set(context_train.labels.values()))

    X_train_t = torch.as_tensor(X_tensor_train, dtype=torch.float32)
    X_test_t = torch.as_tensor(X_tensor_test, dtype=torch.float32)
    y_train_t = torch.as_tensor(y_tensor_train, dtype=torch.long)
    y_test_t = torch.as_tensor(y_tensor_test, dtype=torch.long)

    X_train_t = X_train_t.permute(0, 2, 1)  # (batch_size, sequence_length, electrodes)
    X_test_t = X_test_t.permute(0, 2, 1)

    std_model = ConvolutionalNeuralNetwork(input_size=X_train_t.shape[1], output_size=n_classes)
    train_model(std_model, X_train_t, y_train_t, nn.CrossEntropyLoss(),optim.Adam(std_model.parameters(), lr=0.001), num_epochs=20)
    std_model.eval()
    with torch.no_grad():
        preds = std_model(X_test_t).argmax(1).numpy()
    acc = (preds == y_test_t.numpy()).mean()
    print(f"\nStandard CNN Test accuracy: {acc:.4f}")


    print(f"Accuracy : {acc}")

    accuracies.append(acc)

plt.figure()
plt.plot(down_sample_factors, accuracies, marker='o')
plt.title('Accuracy vs Downsampling Factor')
plt.xlabel('Downsampling Factor')
plt.ylabel('Accuracy')
plt.show()

