import torch.nn as nn
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import torch
import torch.nn as nn
import pywt
import numpy as np
from CNN_network import ConvolutionalNeuralNetwork
from LTSM_network import train_model, test_model
from LTSM_network import convert_data, create_sequences, down_sample, min_max_scaling, retrieve_context

class WaveletConv1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        wavelet = pywt.Wavelet('db4')
        filt = np.array(wavelet.dec_lo)

        kernel_size = len(filt)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )

        with torch.no_grad():
            for i in range(out_channels):
                for j in range(in_channels):
                    self.conv.weight[i, j] = torch.tensor(
                        filt,
                        dtype=torch.float32
                    )

    def forward(self, x):
        return self.conv(x)
    
class WaveletCNN(nn.Module):
    def __init__(self, input_size, output_size=128):
        super().__init__()

        self.wavelet = WaveletConv1D(
            in_channels=input_size,
            out_channels=output_size
        )

    def forward(self, x):
        x = self.wavelet(x)
        return x

class GuidedCNN(nn.Module):
    def __init__(self, guided, output_size=4, dropout=0.3):
        super(GuidedCNN, self).__init__()
 
        self.features = nn.Sequential(
            guided,
            # nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
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
        x = x.permute(0, 2, 1)  # Reshape to [batch_size, input_size, sequence_length] for Conv1d
        features = self.features(x)
        output = self.classifier(features)
        return output

def rsa(matrix1, matrix2, distance_metric='correlation'):
    
    matrix1 = np.asarray(matrix1)
    matrix2 = np.asarray(matrix2)

    if matrix1.shape[0] != matrix2.shape[0]:
        raise ValueError(
            "Both matrices must have the same number of conditions (rows)."
        )

    rdm_vec1 = pdist(matrix1, metric=distance_metric)
    rdm_vec2 = pdist(matrix2, metric=distance_metric)

    rsa_corr, p_value = spearmanr(rdm_vec1, rdm_vec2)

    rdm1 = squareform(rdm_vec1)
    rdm2 = squareform(rdm_vec2)

    return rsa_corr, p_value, rdm1, rdm2

if __name__ == "__main__":
    context_train = retrieve_context(parent_folder="Final_project_data", subdirectory="Intra", type_of_data="train", filename="rest_105923_1.h5")
    X_tensor_train, y_tensor_train = convert_data(down_sample, min_max_scaling, create_sequences, context_train)

    guided_cnn_model = GuidedCNN(WaveletConv1D(in_channels=X_tensor_train.shape[2], out_channels=128), output_size=len(set(context_train.labels.values())))
    train_model(guided_cnn_model, X_tensor_train, y_tensor_train, nn.CrossEntropyLoss(), torch.optim.Adam(guided_cnn_model.parameters(), lr=0.001), num_epochs=20)

    cnn_model = ConvolutionalNeuralNetwork(input_size=X_tensor_train.shape[2], output_size=len(set(context_train.labels.values())))
    train_model(cnn_model, X_tensor_train, y_tensor_train, nn.CrossEntropyLoss(), torch.optim.Adam(cnn_model.parameters(), lr=0.001), num_epochs=20)

    print("Evaluating Guided CNN layer:")
    print(guided_cnn_model.features)

    print("Evaluating CNN layer:")
    print(cnn_model.features)

    with torch.no_grad():
        guided_cnn_features = guided_cnn_model.features(X_tensor_train.permute(0, 2, 1)).squeeze(-1).cpu().numpy()
        cnn_features = cnn_model.features(X_tensor_train.permute(0, 2, 1)).squeeze(-1).cpu().numpy()
        rsa_corr, p_value, rdm_guided_cnn, rdm_cnn = rsa(guided_cnn_features, cnn_features)
        print(f"RSA Correlation: {rsa_corr:.4f}, p-value: {p_value:.4e}")


