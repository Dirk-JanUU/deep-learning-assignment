from sklearn.model_selection import GroupKFold
import numpy as np
from read_data import load_data_from_h5_files
from pre_process import pre_process

from transformer import build_model
from CNN_network import ConvolutionalNeuralNetwork


from read_data import DataSet

from LTSM_network import create_sequences, retrieve_context, convert_data, down_sample, min_max_scaling
from LTSM_network import train_model, test_model, LongShortTermMemoryNetwork

import torch
import torch.nn as nn
import torch.optim as optim

def crossvalidation(model_type = '',classification_type= 'Cross', downsample_factor=10): 
    
    persons = load_data_from_h5_files(parent_directory="Final_project_data", subdirectory=classification_type, type_of_data = "train")
    print(len(persons))
    print([person.id for person in persons])
    
    group=[]
    scans =[]
    labels = []
    
    f_acc = []
    f_acc_label =[]
    
    for person in persons:
        for scan in person.get_scans():
            group.append(person.id)
            scans.append(scan)
            labels.append(scan.task)
    
    group = np.array(group)
    scans = np.array(scans)
    labels = np.array(labels)
    
    kfold = GroupKFold (n_splits=2)
    splits = kfold.split(scans,labels,group)
    
    for fold, (train_id,val_id) in enumerate(splits,start=1):
        
        train_scans = scans[train_id]
        validation_scans = scans[val_id]
        
        if model_type == "transformer":
            
            x_train = np.transpose([pre_process(scan.matrix,
                                            downsample_factor=downsample_factor,
                                            normalization_technique="minmax") 
                                for scan in train_scans],
                                (0, 2, 1)
                                )
        
            y_train = np.array([scan.task for scan in train_scans])
        
            x_val = np.transpose([pre_process(scan.matrix,
                                            downsample_factor=downsample_factor,
                                            normalization_technique="minmax") 
                                for scan in validation_scans],
                                (0, 2, 1)
                                )
        
        
            y_val = np.array([scan.task for scan in validation_scans])
        
            print(f"Fold {fold}")
            print("Train persons:", set(group[train_id]))
            print("Validation persons:", set(group[val_id]))
            
            model = build_model_transformer(x_train = x_train,num_of_classes= len(set(labels)))
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model.fit(x_train, y_train, epochs=10, batch_size = 1 )
            
            loss, accuracy = model.evaluate(x_val, y_val)
            f_acc.append(accuracy)
            print(f"Accuracy:: {accuracy}")
            print(f"Loss: {loss}")
            
            predictions = model.predict(x_val)
            y_pred = np.argmax(predictions,axis=1)
            
            label_accuracy = calculated_acc_Labels(y_val,y_pred)
            f_acc_label.append(label_accuracy)
            print("accuracy labels",label_accuracy)
            
        elif model_type ==  "CNN":
            
            context_train = DataSet(
                persons=[],
                matrixes_data=np.array([scan.matrix for scan in train_scans]),
                matrixes_label=np.array([scan.task for scan in train_scans])
            )
            
            context_val = DataSet(
                persons=[],
                matrixes_data=np.array([scan.matrix for scan in validation_scans]),
                matrixes_label=np.array([scan.task for scan in validation_scans])
            ) 
            
            X_tensor_train, y_tensor_train = convert_data(down_sample, min_max_scaling, create_sequences, context_train)
            X_tensor_val, y_tensor_val = convert_data(down_sample, min_max_scaling, create_sequences, context_val)
            
            model = ConvolutionalNeuralNetwork(input_size=X_tensor_train.shape[2], output_size=len(set(context_train.labels.values())))
            train_model(model, X_tensor_train, y_tensor_train, nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=0.001), num_epochs=20)
            test_model(model, X_tensor_val, y_tensor_val, nn.CrossEntropyLoss())
            
            y_pred = predict_labels(model, X_tensor_val)
            y_true = y_tensor_val.numpy()
            
            accuracy = np.mean(y_pred == y_true)
            f_acc.append(accuracy)
            
            
            acc = calculated_acc_Labels(y_true,y_pred)
            f_acc_label.append(acc)
            print("accuracy labels",acc)
            
        
        elif model_type ==  "LSTM":
        
            context_train = DataSet(
                persons=[],
                matrixes_data=np.array([scan.matrix for scan in train_scans]),
                matrixes_label=np.array([scan.task for scan in train_scans])
            )
            
            context_val = DataSet(
                persons=[],
                matrixes_data=np.array([scan.matrix for scan in validation_scans]),
                matrixes_label=np.array([scan.task for scan in validation_scans])
            ) 
            
            X_tensor_train, y_tensor_train = convert_data(down_sample, min_max_scaling, create_sequences, context_train)
            X_tensor_val, y_tensor_val = convert_data(down_sample, min_max_scaling, create_sequences, context_val)
            
            model = LongShortTermMemoryNetwork(input_size=X_tensor_train.shape[2], hidden_size=128, output_size=len(set(context_train.labels.values())))
            train_model(model, X_tensor_train, y_tensor_train, nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=0.001), num_epochs=20)
            test_model(model, X_tensor_val, y_tensor_val, nn.CrossEntropyLoss())
            
            y_pred = predict_labels(model, X_tensor_val)
            y_true = y_tensor_val.numpy()
            
            accuracy = np.mean(y_pred == y_true)
            f_acc.append(accuracy)
            
            
            acc = calculated_acc_Labels(y_true,y_pred)
            f_acc_label.append(acc)
            print("accuracy labels",acc)
        
    print("Mean accuracy",np.mean(f_acc))
    print("Mean labels accuracy",mean_labels(f_acc_label))
    return

def build_model_transformer(x_train,num_of_classes):
    num_transformer_layers = 2
    head_size = 64
    num_heads = 2
    ff_layers = 128
    return build_model(input_shape=x_train[0].shape, head_size=head_size, num_heads=num_heads, ff_dim=ff_layers, num_layers=num_transformer_layers, num_classes=num_of_classes)

def calculated_acc_Labels(true , pred):
    accuracy = {}
    for label in np.unique(true):
        m = (true == label)
        correct = pred[m]==true[m]
        accuracy[int(label)]= np.mean(correct)
    return accuracy

def predict_labels (model, x):
    model.eval()
    with torch.no_grad():
        output = model(x)
        predict = torch.argmax(output,dim=1)

    return predict.numpy()

def mean_labels(acc_labels):
    
    mean_labels = {}
    
    labels = acc_labels[0].keys()
    
    for label in labels:
        values = [fold_acc[label] for fold_acc in acc_labels]
        mean_labels[label] = np.mean(values)
        
    return mean_labels

if __name__=="__main__":
    crossvalidation(model_type = "CNN")

