import read_data
import data_utils
import torch
import torch.nn as nn
import numpy as np
from scipy.signal import decimate
 
 
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size=4, dropout=0.3):
        super(ConvolutionalNeuralNetwork, self).__init__()
 
        # input_size = number of electrodes (treated as input channels like 248)
        self.input_size = input_size
        self.output_size = output_size
 
        # 1D convolutions slide over TIME. Each electrode is an input channel,
        # so every filter learns a temporal pattern jointly across electrodes.
        # Pooling over time gives translation invariance: the model reacts to
        # "which areas are active" rather than to the exact moment they fire.
        self.features = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
 
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
 
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # collapse the time axis -> (batch, 128, 1)
        )
 
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
 
            nn.Linear(128, output_size)
        )
 
    # x shape: [batch_size, sequence_length, input_size]  (same layout as the LSTM)
    def forward(self, x):
 
        # Conv1d expects [batch, channels, length] -> [batch, electrodes, time]
        #x = x.permute(0, 2, 1)
 
        out = self.features(x)
 
        logits = self.classifier(out)
 
        return logits
