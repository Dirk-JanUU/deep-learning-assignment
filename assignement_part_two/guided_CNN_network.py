from sympy import im
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
from pre_process import pre_process

class WaveletConv1D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        wavelet = pywt.Wavelet('db4')
        filt = np.array(wavelet.dec_lo)

        self.in_channels = in_channels

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels * 5,
            kernel_size=len(filt),
            padding="same",
            bias=False
        )

        with torch.no_grad():

            self.conv.weight.zero_()

            for ch in range(in_channels):
                for band in range(5):

                    out_idx = ch * 5 + band

                    self.conv.weight[
                        out_idx,
                        ch,
                        :
                    ] = torch.tensor(filt)

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

    def __init__(self, guided, input_size, output_size=4):
        super().__init__()

        self.features = nn.Sequential(
            guided,
            nn.BatchNorm1d(input_size * 5),
            nn.ReLU()
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size * 5, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):

        x = x.permute(0, 2, 1)

        features = self.features(x)

        pooled = self.pool(features)

        return self.classifier(pooled)
 
class StandardCNN(nn.Module):

    def __init__(self, input_size, output_size=4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(
                input_size,
                input_size * 5,
                kernel_size=7,
                padding="same"
            ),
            nn.BatchNorm1d(input_size * 5),
            nn.ReLU()
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size * 5, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):

        x = x.permute(0, 2, 1)

        features = self.features(x)

        pooled = self.pool(features)

        return self.classifier(pooled)

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

    feature_technique = "wavelets"
    window = X_tensor_train[0].numpy()
    window = window.T
    wavelet_full = pre_process(
        context_train.persons[0].scans[0].matrix,
        sfreq=2034,
        feature_extraction="wavelets"
    )
    sequence_length = 256

    wavelet_windows = []

    # both CNN use batches of 256 time points as input, so we need to create the same windows from the wavelet transform to compare them in the RSA analysis.
    # # so from the full wavelet transform we create the same windows as the CNNs to compare them in the RSA analysis. 
    # We create 12 windows of 256 time points each, which corresponds to the same windowing technique used in the CNNs.
    for i in range(12):

        start = i * sequence_length
        end = start + sequence_length

        wavelet_window = wavelet_full[:, :, start:end]

        wavelet_windows.append(wavelet_window)

    wavelet_features = np.stack(wavelet_windows)
        
    guided_cnn_model = GuidedCNN(guided=WaveletConv1D(X_tensor_train.shape[2]),input_size=X_tensor_train.shape[2], output_size=len(set(context_train.labels.values())))
    train_model(guided_cnn_model, X_tensor_train, y_tensor_train, nn.CrossEntropyLoss(), torch.optim.Adam(guided_cnn_model.parameters(), lr=0.001), num_epochs=20)

    cnn_model = StandardCNN(input_size=X_tensor_train.shape[2], output_size=len(set(context_train.labels.values())))
    train_model(cnn_model, X_tensor_train, y_tensor_train, nn.CrossEntropyLoss(), torch.optim.Adam(cnn_model.parameters(), lr=0.001), num_epochs=20)

    with torch.no_grad():

        print("Evaluating Guided CNN layer:")
        print(guided_cnn_model.features)

        print("Evaluating CNN layer:")
        print(cnn_model.features)

        x = X_tensor_train.permute(0, 2, 1)

        guided_features = guided_cnn_model.features(x)

        cnn_features = cnn_model.features(x)

        batch_size = guided_features.shape[0]

        guided_features = guided_features.view(
            batch_size,
            248,
            5,
            256
        )

        cnn_features = cnn_features.view(
            batch_size,
            248,
            5,
            256
        )

        wavelet_rsa = wavelet_features.reshape(12, -1)

        guided_rsa = guided_features.reshape(12, -1).cpu().numpy()

        cnn_rsa = cnn_features.reshape(12, -1).cpu().numpy()

        rsa_corr, p_value, _, _ = rsa(
            guided_rsa,
            cnn_rsa
        )

        rsa_corr_guided, p_value, _, _ = rsa(wavelet_rsa, guided_rsa)
        rsa_corr_cnn, p_value, _, _ = rsa(wavelet_rsa, cnn_rsa)

        print(
            f"RSA Correlation guided vs normal CNN: {rsa_corr:.4f}, "
            f"p-value guided vs normal CNN: {p_value:.4e}"
        )

        print(
            f"RSA Correlation guided vs ground truth: {rsa_corr_guided:.4f}, "
            f"p-value guided vs ground truth: {p_value:.4e}"
        )

        print(
            f"RSA Correlation normal CNN vs ground truth: {rsa_corr_cnn:.4f}, "
            f"p-value normal CNN vs ground truth: {p_value:.4e}"
        )