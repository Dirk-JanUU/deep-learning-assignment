import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from LTSM_experiment import retrieve_context, convert_data, min_max_scaling, down_sample, create_sequences, train_model, test_model
import time

N_BANDS = 5
DOWNSAMPLE_FACTOR = 20
SFREQ = 2034 / DOWNSAMPLE_FACTOR


def morlet_kernels(freqs, sfreq, n_cycles, kernel_len=None):
    """Build initial real/imag Morlet kernels. Used to seed the learnable params."""
    freqs = np.asarray(freqs, dtype=np.float64)
    n_cycles = np.asarray(n_cycles, dtype=np.float64)
    sigma_t = n_cycles / (2.0 * np.pi * freqs)              # std in seconds
    if kernel_len is None:
        kernel_len = int(2 * round(5 * sigma_t.max() * sfreq) + 1)
    half = kernel_len // 2
    t = np.arange(-half, half + 1) / sfreq
    real = np.zeros((len(freqs), kernel_len))
    imag = np.zeros((len(freqs), kernel_len))
    for i, (f, s) in enumerate(zip(freqs, sigma_t)):
        gauss = np.exp(-(t ** 2) / (2 * s ** 2))
        norm = 1.0 / (s * np.sqrt(np.pi))
        real[i] = norm * gauss * np.cos(2 * np.pi * f * t)
        imag[i] = norm * gauss * np.sin(2 * np.pi * f * t)
    return (torch.tensor(real, dtype=torch.float32),
            torch.tensor(imag, dtype=torch.float32),
            kernel_len)


class MorletLayer(nn.Module):
    """
    Learnable Morlet front end.
    Input:  (B, E, T)            raw signal
    Output: (B, E, n_bands, T)   band power over time

    The real/imag filter banks are initialized as Morlet wavelets but are
    nn.Parameters, so they adapt during training. Band aggregation is fixed.
    """
    def __init__(self, sfreq, learnable=True):
        super().__init__()
        delta = np.arange(1, 4, 1)
        theta = np.arange(4, 8, 1)
        alpha = np.arange(8, 12, 2)
        beta = np.arange(12, 30, 6)
        gamma = np.arange(30, 101, 30)
        self.freqs = np.concatenate([delta, theta, alpha, beta, gamma])
        n_cycles = np.logspace(np.log10(8.0), np.log10(3.0), len(self.freqs))

        real, imag, klen = morlet_kernels(self.freqs, sfreq, n_cycles)
        self.pad = klen // 2
        real = real.unsqueeze(1)        # (n_freqs, 1, klen)  -> grouped conv per electrode
        imag = imag.unsqueeze(1)

        if learnable:
            self.real = nn.Parameter(real)
            self.imag = nn.Parameter(imag)
        else:
            self.register_buffer('real', real)
            self.register_buffer('imag', imag)

        bands = [(0.5, 4), (4, 8), (8, 12), (12, 30), (30, 100)]
        self.band_idx = [np.where((self.freqs >= lo) & (self.freqs < hi))[0]
                         for lo, hi in bands]
        self.n_bands = len(bands)

    def forward(self, x):
        # x: (B, E, T)
        B, E, T = x.shape
        xf = x.reshape(B * E, 1, T)                          # each electrode independently
        rr = F.conv1d(xf, self.real, padding=self.pad)       # (B*E, n_freqs, T)
        ii = F.conv1d(xf, self.imag, padding=self.pad)
        power = rr ** 2 + ii ** 2                            # (B*E, n_freqs, T)
        power = power.reshape(B, E, len(self.freqs), T)
        out = torch.stack([power[:, :, idx, :].mean(2) for idx in self.band_idx],
                          dim=2)                              # (B, E, n_bands, T)
        return out

class Wavelet_CNN1(nn.Module):
    def __init__(self, input_size, sfreq, output_size=4, dropout=0.3,
                 n_bands=N_BANDS, learnable=True):
        """
        input_size = number of electrodes (E).
        Channels into the conv stack = E * n_bands.
        """
        super().__init__()
        self.morlet = MorletLayer(sfreq, learnable=learnable)
        in_ch = input_size * n_bands

        self.features = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=7, padding=3),
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
            nn.AdaptiveAvgPool1d(1),          # collapse time -> (B, 128, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_size),
        )

    def forward(self, x):
        # x: (B, electrodes, timesteps) -- raw signal
        x = self.morlet(x)                   # (B, E, n_bands, T)
        B, E, Bands, T = x.shape
        x = x.reshape(B, E * Bands, T)       # stack bands+electrodes into channels
        x = self.features(x)
        return self.classifier(x)
    
class Wavelet_CNN2(nn.Module):
    def __init__(self, input_size, sfreq, output_size=4, dropout=0.3,
                 n_bands=N_BANDS, learnable=True):
        """
        input_size = number of electrodes (E) -> Conv2d in_channels.
        """
        super().__init__()
        self.morlet = MorletLayer(sfreq, learnable=learnable)
        pad_b = (n_bands - 1) // 2          # keep band axis = n_bands when odd

        self.tmpband = nn.Sequential(
            nn.Conv2d(input_size, 64, kernel_size=(n_bands, 5), padding=(pad_b, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),                                  # keep bands
            nn.Conv2d(64, 128, kernel_size=(n_bands, 5), padding=(pad_b, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),                                  # keep bands
            nn.Conv2d(128, 128, kernel_size=(n_bands, 1)),         # NO pad -> bands collapse to 1
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.tmp = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(1, 10), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),          # collapse time -> (B, 128, 1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_size),
        )

    def forward(self, x):
        # x: (B, electrodes, timesteps) -- raw signal
        x = self.morlet(x)        # (B, E, n_bands, T)
        x = self.tmpband(x)
        x = self.tmp(x)
        return self.classifier(x)


if __name__ == "__main__":
    context_train = retrieve_context(parent_folder="Final_project_data", subdirectory="Intra", type_of_data="train")
    X_tensor_train, y_tensor_train = convert_data(down_sample, min_max_scaling, create_sequences, context_train)
    X_tensor_train = X_tensor_train.permute(0, 2, 1)  # (trials, electrodes, timepoints)
    print(f"Training data shape after permute: {X_tensor_train.shape}")
    print(f"Training labels shape: {y_tensor_train.shape}")
    print(f"Training labels: {list(context_train.labels.values())}")

    context_test = retrieve_context(parent_folder="Final_project_data", subdirectory="Intra", type_of_data="test")
    X_tensor_test, y_tensor_test = convert_data(down_sample, min_max_scaling, create_sequences, context_test)
    X_tensor_test = X_tensor_test.permute(0, 2, 1)  # (trials, electrodes, timepoints)

    n_classes = len(set(context_train.labels.values()))

    X_train_t = torch.as_tensor(X_tensor_train, dtype=torch.float32)
    X_test_t = torch.as_tensor(X_tensor_test, dtype=torch.float32)
    y_train_t = torch.as_tensor(y_tensor_train, dtype=torch.long)
    y_test_t = torch.as_tensor(y_tensor_test, dtype=torch.long)

    w1 = Wavelet_CNN2(X_train_t.shape[1], SFREQ, n_classes)
    time_0 = time.time()
    w1_losses = train_model(w1, X_train_t, y_train_t, nn.CrossEntropyLoss(), optim.Adam(w1.parameters(), lr=0.001), num_epochs=20)
    w1_training_time = time.time() - time_0
    w1.eval()
    with torch.no_grad():
        w1_preds = w1(X_test_t).argmax(1).numpy()
    w1_acc = (w1_preds == y_test_t.numpy()).mean()
    print(f"\nStandard CNN - Training time: {w1_training_time:.2f}s, Test accuracy: {w1_acc:.4f}")


