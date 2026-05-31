import torch
import torch.nn as nn
import torch.optim as optim

from LTSM_network import LongShortTermMemoryNetwork, create_sequences, retrieve_context, convert_data, down_sample, min_max_scaling
from LTSM_network import LongShortTermMemoryNetwork, train_model, test_model


if __name__ == "__main__":

    # ONE SUBJECT TRAINING AND TESTING

    context_train = retrieve_context(parent_folder = "Final_project_data")
    X_tensor_train, y_tensor_train = convert_data(down_sample, min_max_scaling, create_sequences, context_train)
    model = LongShortTermMemoryNetwork(input_size=X_tensor_train.shape[2], hidden_size=128, output_size=len(set(context_train.labels.values())))
    train_model(model, X_tensor_train, y_tensor_train, nn.CrossEntropyLoss(), optim.Adam(model.parameters(),lr=0.001), num_epochs=20)

    context_test = retrieve_context(parent_folder = "Final_project_data", subdirectory = "Intra", type_of_data = "test")
    X_tensor_test, y_tensor_test = convert_data(down_sample, min_max_scaling, create_sequences, context_test)
    test_model(model, X_tensor_test, y_tensor_test, nn.CrossEntropyLoss())

    # ONE SUBJECT TRAINING AND TESTING
    # TWO SUBJECT TRAINING AND TESTING

    context_train = retrieve_context(parent_folder = "Final_project_data", subdirectory = "Cross", type_of_data = "train")
    X_tensor_train, y_tensor_train = convert_data(down_sample, min_max_scaling, create_sequences, context_train)
    model = LongShortTermMemoryNetwork(input_size=X_tensor_train.shape[2], hidden_size=128, output_size=len(set(context_train.labels.values())))
    train_model(model, X_tensor_train, y_tensor_train, nn.CrossEntropyLoss(), optim.Adam(model.parameters(),lr=0.001), num_epochs=20)

    context_test1 = retrieve_context(parent_folder = "Final_project_data", subdirectory = "Cross", type_of_data = "test")
    X_tensor_test, y_tensor_test = convert_data(down_sample, min_max_scaling, create_sequences, context_test1)
    test_model(model, X_tensor_test, y_tensor_test, nn.CrossEntropyLoss())

    context_test2 = retrieve_context(parent_folder = "Final_project_data", subdirectory = "Cross", type_of_data = "test2")
    X_tensor_test, y_tensor_test = convert_data(down_sample, min_max_scaling, create_sequences, context_test2)
    test_model(model, X_tensor_test, y_tensor_test, nn.CrossEntropyLoss())

    context_test3 = retrieve_context(parent_folder = "Final_project_data", subdirectory = "Cross", type_of_data = "test3")
    X_tensor_test, y_tensor_test = convert_data(down_sample, min_max_scaling, create_sequences, context_test3)
    test_model(model, X_tensor_test, y_tensor_test, nn.CrossEntropyLoss())

    # TWO SUBJECT TRAINING AND TESTING