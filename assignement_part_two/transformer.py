import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from read_data import load_data_from_h5_files, ScanData
from visualize_data import plot_electrode_activation_through_time, plot_electrodes_activations__over_single_timestep

def transformer_block(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    attention = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    attention = layers.Dropout(dropout)(attention)
    attention = layers.LayerNormalization(epsilon=1e-6)(inputs + attention)

    ff = layers.Dense(ff_dim, activation='relu')(attention)
    ff = layers.Dense(inputs.shape[-1])(ff)
    ff = layers.Dropout(dropout)(ff)
    return layers.LayerNormalization(epsilon=1e-6)(attention + ff)

def build_model(input_shape, head_size=64, num_heads=2, ff_dim=128, num_layers=2, num_classes=10):
    inputs = layers.Input(shape=input_shape)  # shape=(10, 1)
    x = inputs
    for _ in range(num_layers):
        x = transformer_block(x, head_size, num_heads, ff_dim)
    x = layers.GlobalAveragePooling1D()(x)  # Now it will work fine
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    return model

def normalize(scan, technique = "minmax"):
    if technique == "minmax":
        for electrode in range(scan.shape[1]):
            scan[:, electrode] = (scan[:, electrode] - np.min(scan[:, electrode])) / (np.max(scan[:, electrode]) - np.min(scan[:, electrode]))
        return scan
    elif technique == "zscore":
        for electrode in range(scan.shape[1]):
            scan[:, electrode] = (scan[:, electrode] - np.mean(scan[:, electrode])) / np.std(scan[:, electrode])
        return scan
    else:
        raise ValueError("Unsupported pre-processing function")

def downsample(scan, factor=5): # Downsample scan by averaging every 'factor' time steps
    num_time_steps = scan.shape[1]
    num_electrodes = scan.shape[0]
    new_time_steps = num_time_steps // factor
    downsampled_scan = np.zeros((num_electrodes, new_time_steps))   
    for i in range(new_time_steps):
        start = i * factor
        end = start + factor
        downsampled_scan[:, i] = np.mean(scan[:, start:end], axis=1)
    return downsampled_scan

def pre_process(raw_scans, downsample_factor=5, normalization_technique="minmax"):
    pre_processed_scans = [normalize(scan.matrix, technique=normalization_technique) for scan in raw_scans] # Normalizing
    downsampled_scans = np.array([downsample(scan, factor=downsample_factor) for scan in pre_processed_scans]) # Downsampling
    return downsampled_scans

def train(classification_type, downsample = 5, normalization = "minmax", head_size = 64,  num_heads = 2, ff_layers = 128, num_transformer_layers = 5 , epochs=20, batch_size=1):
    persons = load_data_from_h5_files(parent_directory="Final_project_data", subdirectory=classification_type, type_of_data="train")
    raw_scans = [person.get_scans()[i] for person in persons for i in range(len(person.get_scans()))] # Extracting all scans 
    pre_processed_scans = pre_process(raw_scans, downsample_factor=downsample, normalization_technique=normalization) # Pre-processing
    x_data = np.transpose(pre_processed_scans, (0, 2, 1)) # Reshaping to (num_samples, time_steps, num_electrodes) -> attention mechanism expects the time dimension to be the first dimension of the input
    y_data = np.array([scan.task for scan in raw_scans]) # Extracting labels

    input_shape = x_data[0].shape 
    number_of_classes = len(set(y_data))

    # Visualizing wether the pre-processing and downsampling steps are working correctly
    #plot_electrodes_activations__over_single_timestep(raw_scans[0], timestep=100)
    #plot_electrodes_activations__over_single_timestep(ScanData("1", "0", downsampled_scans[0]), timestep=100) 
    #plot_electrode_activation_through_time(raw_scans[0], electrode_idxs=[0, 50, 100, 200])
    #plot_electrode_activation_through_time(ScanData("1", "0", downsampled_scans[0]), electrode_idxs=[0, 50, 100, 200]) 
    # ==================================================================================

    print("============= Data Summary =============")
    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {number_of_classes}")
    print(f"Number of samples: {len(x_data)}")
    print("========================================")
    
    model = build_model(input_shape, head_size = head_size, num_heads = num_heads, ff_layers = ff_layers, num_layers = num_transformer_layers,  num_classes=number_of_classes)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_data, y_data, epochs=epochs, batch_size=batch_size)
    return model

def test(model, classification_type):
    if classification_type == "Intra":
        persons = load_data_from_h5_files(parent_directory="Final_project_data", subdirectory=classification_type, type_of_data="test")
        raw_scans = [person.get_scans()[i] for person in persons for i in range(len(person.get_scans()))] 
        x_test = np.transpose(pre_process(raw_scans, downsample_factor=5, normalization_technique="minmax"), (0, 2, 1)) 
        y_test = np.array([scan.task for scan in raw_scans]) 
        loss, accuracy = model.evaluate(x_test, y_test)
        print(f"Test accuracy: {accuracy}")
    elif classification_type == "Cross":
        persons1 = load_data_from_h5_files(parent_directory="Final_project_data", subdirectory=classification_type, type_of_data="test1")
        persons2 = load_data_from_h5_files(parent_directory="Final_project_data", subdirectory=classification_type, type_of_data="test2")
        persons3 = load_data_from_h5_files(parent_directory="Final_project_data", subdirectory=classification_type, type_of_data="test3")

        raw_scans1 = [person.get_scans()[i] for person in persons1 for i in range(len(person.get_scans()))]
        raw_scans2 = [person.get_scans()[i] for person in persons2 for i in range(len(person.get_scans()))]
        raw_scans3 = [person.get_scans()[i] for person in persons3 for i in range(len(person.get_scans()))]
        x_test1 = np.transpose(pre_process(raw_scans1), (0, 2, 1))
        x_test2 = np.transpose(pre_process(raw_scans2), (0, 2, 1))
        x_test3 = np.transpose(pre_process(raw_scans3), (0, 2, 1))
        y_test1 = np.array([scan.task for scan in raw_scans1])
        y_test2 = np.array([scan.task for scan in raw_scans2])
        y_test3 = np.array([scan.task for scan in raw_scans3])      
        loss1, accuracy1 = model.evaluate(x_test1, y_test1)
        loss2, accuracy2 = model.evaluate(x_test2, y_test2)
        loss3, accuracy3 = model.evaluate(x_test3, y_test3)
        print(f"Test accuracy for test1: {accuracy1}")
        print(f"Test accuracy for test2: {accuracy2}")
        print(f"Test accuracy for test3: {accuracy3}")
    else:
        raise ValueError("Unsupported classification type")



if __name__=="__main__":
    model = train(classification_type= "Intra", downsample=5, normalization = "minmax", head_size = 64, num_heads=2, ff_layers = 128, num_transformer_layers = 2)
    test(model, classification_type= "Intra")
