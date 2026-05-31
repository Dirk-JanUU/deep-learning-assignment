import torch
import torch.nn as nn
from LTSM_network import LongShortTermMemoryNetwork, retrieve_context, test_model, train_model
import pre_process
import numpy as np
import torch.optim as optim


if __name__ == "__main__":

    # ONE SUBJECT TRAINING AND TESTING

    context_train = retrieve_context(parent_folder = "Final_project_data")
    raw_scans = [person.get_scans()[i] for person in context_train.persons for i in range(len(person.get_scans()))]
    
    # Pre Processing
    pre_processed_scans = []
    for idx, scan in enumerate(raw_scans):
        pre_processed_scans.append(pre_process(scan.matrix, sfreq=4034, feature_extraction="wavelets", downsample_factor=2, normalization_technique="minmax"))
        print(f"Pre-Processed: {idx + 1} / {len(raw_scans)}", end='\r', flush=True)

    X_tensor_train = torch.tensor(np.transpose(pre_processed_scans, (3, 0, 1, 2)), dtype=torch.float32) # Reshaping to (time_steps, num_samples, num_electrodes, bands_intensity)
    num_samples, time_steps, num_electrodes, bands_intensity = X_tensor_train.shape
    total_features = num_samples * num_electrodes * bands_intensity
    X_tensor_flattened_train = X_tensor_train.permute(1, 0, 2, 3).reshape(time_steps, total_features) # flattening to (time_steps, total_features) for LSTM input

    y_tensor_train = torch.tensor(context_train.y_data, dtype=torch.long)

    model = LongShortTermMemoryNetwork(input_size=X_tensor_flattened_train.shape[1], hidden_size=128, output_size=len(set(context_train.labels.values())))

    train_model(model, X_tensor_flattened_train, y_tensor_train, nn.CrossEntropyLoss(), optim.Adam(model.parameters(),lr=0.001), num_epochs=20)

    context_test = retrieve_context(parent_folder = "Final_project_data", subdirectory = "Intra", type_of_data = "test")

    raw_scans = [person.get_scans()[i] for person in context_test.persons for i in range(len(person.get_scans()))]
    
    # Pre Processing
    pre_processed_scans = []
    for idx, scan in enumerate(raw_scans):
        pre_processed_scans.append(pre_process(scan.matrix, sfreq=4034, feature_extraction="wavelets", downsample_factor=2, normalization_technique="minmax"))
        print(f"Pre-Processed: {idx + 1} / {len(raw_scans)}", end='\r', flush=True)

    X_tensor_test = torch.tensor(np.transpose(pre_processed_scans, (3, 0, 1, 2)), dtype=torch.float32) # Reshaping to (time_steps, num_samples, num_electrodes, bands_intensity)
    num_samples, time_steps, num_electrodes, bands_intensity = X_tensor_test.shape
    total_features = num_samples * num_electrodes * bands_intensity
    X_tensor_flattened_test = X_tensor_test.permute(1, 0, 2, 3).reshape(time_steps, total_features) # flattening to (time_steps, total_features) for LSTM input

    y_tensor_test = torch.tensor(context_test.y_data, dtype=torch.long)

    test_model(model, X_tensor_flattened_test, y_tensor_test, nn.CrossEntropyLoss())

    # ONE SUBJECT TRAINING AND TESTING