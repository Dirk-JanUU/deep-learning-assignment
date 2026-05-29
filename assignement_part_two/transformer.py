import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from read_data import load_data_from_h5_files, ScanData
from visualize_data import plot_electrode_activation_through_time, plot_electrodes_activations__over_single_timestep
from pre_process import pre_process

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

def train(classification_type, feature_extraction = False, downsample_factor = 2, num_transformer_layers = 2, head_size = 64, num_heads = 2, ff_layers = 128):
    persons = load_data_from_h5_files(parent_directory="Final_project_data", subdirectory=classification_type, type_of_data="train")
    raw_scans = [person.get_scans()[i] for person in persons for i in range(len(person.get_scans()))] # Extracting all scans 

    # Pre Processing
    pre_processed_scans = []
    for idx, scan in enumerate(raw_scans):
        pre_processed_scans.append(pre_process(scan.matrix, sfreq=2034, feature_extraction=feature_extraction, downsample_factor=downsample_factor))
        print(f"Pre-Processed: {idx + 1} / {len(raw_scans)}", end='\r', flush=True)
    print()
    
    x_data = None
    if feature_extraction == "wavelets":
        x_data = np.transpose(pre_processed_scans, (0, 3, 1, 2)) # Reshaping to (num_samples, time_steps, num_electrodes, bands_intensity) -> attention mechanism expects the time dimension to be the first dimension of the input
    elif feature_extraction == "fourier":
        raise ValueError("Fourier feature extraction is not compatible with the transformer architecture because it does not preserve the temporal structure needed for attention mechanisms. Please use 'wavelets' or set feature_extraction to False.")
    else:
        x_data = np.transpose(pre_processed_scans, (0, 2, 1)) # Reshaping to (num_samples, time_steps, num_electrodes) -> attention mechanism expects the time dimension to be the first dimension of the input
    
    y_data = np.array([scan.task for scan in raw_scans]) # Extracting labels

    input_shape = x_data[0].shape 
    number_of_classes = len(set(y_data))

    print("============= Data Summary =============")
    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {number_of_classes}")
    print("========================================")

    model = build_model(input_shape=input_shape, head_size=head_size, num_heads=num_heads, ff_dim=ff_layers, num_layers=num_transformer_layers, num_classes=number_of_classes)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_data, y_data, epochs=10, batch_size = 1 )
    return model

def test(model, classification_type, downsample_factor=10):
    if classification_type == "Intra":
        persons = load_data_from_h5_files(parent_directory="Final_project_data", subdirectory=classification_type, type_of_data="test")
        raw_scans = [person.get_scans()[i] for person in persons for i in range(len(person.get_scans()))] 
        x_test = np.transpose([pre_process(scan.matrix, downsample_factor=downsample_factor, normalization_technique="minmax") for scan in raw_scans], (0, 2, 1)) 
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
        x_test1 = np.transpose([pre_process(scan.matrix, downsample_factor=downsample_factor, normalization_technique="minmax") for scan in raw_scans1], (0, 2, 1))
        x_test2 = np.transpose([pre_process(scan.matrix, downsample_factor=downsample_factor, normalization_technique="minmax") for scan in raw_scans2], (0, 2, 1))
        x_test3 = np.transpose([pre_process(scan.matrix, downsample_factor=downsample_factor, normalization_technique="minmax") for scan in raw_scans3], (0, 2, 1))
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
    model = train(classification_type= "Intra", downsample_factor=10, num_transformer_layers=2, head_size=64, num_heads=2, ff_layers=10)
    test(model, classification_type= "Intra", downsample_factor=10)
