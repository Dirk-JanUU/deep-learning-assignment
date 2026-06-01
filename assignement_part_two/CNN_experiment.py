import torch
import torch.nn as nn
import torch.optim as optim

from CNN_network import ConvolutionalNeuralNetwork
# Same data pipeline + train/test loops as the LSTM ->
# comparison between models is fair.
from LTSM_network import create_sequences, retrieve_context, convert_data, down_sample, min_max_scaling
from LTSM_network import train_model, test_model


if __name__ == "__main__":

    # ONE SUBJECT TRAINING AND TESTING (INTRA)

    context_train = retrieve_context(parent_folder="Final_project_data")
    X_tensor_train, y_tensor_train = convert_data(down_sample, min_max_scaling, create_sequences, context_train)
    model = ConvolutionalNeuralNetwork(input_size=X_tensor_train.shape[2], output_size=len(set(context_train.labels.values())))
    train_model(model, X_tensor_train, y_tensor_train, nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=0.001), num_epochs=20)

    # Train accuracy (for the train-vs-test gap analysis in task (d)):
    print("--- INTRA train accuracy ---")
    test_model(model, X_tensor_train, y_tensor_train, nn.CrossEntropyLoss())

    context_test = retrieve_context(parent_folder="Final_project_data", subdirectory="Intra", type_of_data="test")
    X_tensor_test, y_tensor_test = convert_data(down_sample, min_max_scaling, create_sequences, context_test)
    print("--- INTRA test accuracy ---")
    test_model(model, X_tensor_test, y_tensor_test, nn.CrossEntropyLoss())

    # ONE SUBJECT TRAINING AND TESTING (INTRA)
    # MULTI SUBJECT TRAINING AND TESTING (CROSS)

    context_train = retrieve_context(parent_folder="Final_project_data", subdirectory="Cross", type_of_data="train")
    X_tensor_train, y_tensor_train = convert_data(down_sample, min_max_scaling, create_sequences, context_train)
    model = ConvolutionalNeuralNetwork(input_size=X_tensor_train.shape[2], output_size=len(set(context_train.labels.values())))
    train_model(model, X_tensor_train, y_tensor_train, nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=0.001), num_epochs=20)

    print("--- CROSS train accuracy ---")
    test_model(model, X_tensor_train, y_tensor_train, nn.CrossEntropyLoss())

    for test_split in ["test1", "test2", "test3"]:
        context_test = retrieve_context(parent_folder="Final_project_data", subdirectory="Cross", type_of_data=test_split)
        X_tensor_test, y_tensor_test = convert_data(down_sample, min_max_scaling, create_sequences, context_test)
        print(f"--- CROSS test accuracy ({test_split}) ---")
        test_model(model, X_tensor_test, y_tensor_test, nn.CrossEntropyLoss())

    # MULTI SUBJECT TRAINING AND TESTING (CROSS)