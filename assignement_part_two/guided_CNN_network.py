import torch
import torch.nn as nn
import numpy as np
import pywt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
from LTSM_network import (
    train_model, test_model,
    convert_data, create_sequences,
    down_sample, min_max_scaling, retrieve_context
)
from pre_process import pre_process

def normalize(x: np.ndarray):
    x = x - x.mean(axis=1, keepdims=True)
    x = x / (x.std(axis=1, keepdims=True) + 1e-8)
    return x


class WaveletConv1D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        wavelet = pywt.Wavelet('db4')
        lo = np.array(wavelet.dec_lo)
        hi = np.array(wavelet.dec_hi)

        self.conv_low = nn.Conv1d(in_channels, in_channels * 5, len(lo), padding=3, bias=False)
        self.conv_high = nn.Conv1d(in_channels, in_channels * 5, len(hi), padding=3, bias=False)

        with torch.no_grad():
            self.conv_low.weight.zero_()
            self.conv_high.weight.zero_()

            for ch in range(in_channels):
                for b in range(5):
                    out = ch * 5 + b
                    self.conv_low.weight[out, ch, :] = torch.tensor(lo)
                    self.conv_high.weight[out, ch, :] = torch.tensor(hi)

    def forward(self, x):
        return self.conv_low(x) + self.conv_high(x)

class GuidedCNN(nn.Module):
    def __init__(self, guided, input_size, output_size=4):
        super().__init__()

        self.first_layer = nn.Sequential(
            guided,
            nn.BatchNorm1d(input_size * 5),
            nn.ReLU()
        )

        self.backbone = nn.Sequential(
            nn.Conv1d(input_size * 5, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        x = self.first_layer(x)
        x = self.backbone(x)
        return self.classifier(x)

class StandardCNN(nn.Module):
    def __init__(self, input_size, output_size=4):
        super().__init__()

        self.first_layer = nn.Sequential(
            nn.Conv1d(input_size, input_size * 5, 7, padding=3),
            nn.BatchNorm1d(input_size * 5),
            nn.ReLU()
        )

        self.backbone = nn.Sequential(
            nn.Conv1d(input_size * 5, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        x = self.first_layer(x)
        x = self.backbone(x)
        return self.classifier(x)

def rsa(m1, m2):
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    if m1.shape[0] != m2.shape[0]:
        raise ValueError("Rows must match")

    r1 = pdist(m1, metric='correlation')
    r2 = pdist(m2, metric='correlation')

    corr, p = spearmanr(r1, r2)

    return corr, p, squareform(r1), squareform(r2)

def extract_activations(model, X, device="cpu"):
    model = model.to(device)
    model.eval()

    loader = DataLoader(TensorDataset(X), batch_size=32)

    acts = {"features": [], "pool": [], "hidden": [], "logits": []}

    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)

            f = model.first_layer(xb)
            acts["features"].append(f.mean(-1).cpu())

            p = model.backbone(f)
            acts["pool"].append(p.mean(-1).cpu())

            h = model.classifier[0:3](p)
            acts["hidden"].append(h.cpu())

            l = model.classifier[3:](h)
            acts["logits"].append(l.cpu())

    return {k: torch.cat(v, 0) for k, v in acts.items()}

def Wavelet_RSA_similarity(model,
                           wavelet_features,
                           X_data,
                           model_name="Model",
                           device="cpu",
                           labels=None):

    n = min(len(wavelet_features), len(X_data))
    wavelet_features = wavelet_features[:n]
    X_data = X_data[:n]

    wavelet_ref = wavelet_features.reshape(n, -1)
    wavelet_ref = normalize(wavelet_ref)

    X_t = torch.tensor(X_data, dtype=torch.float32)

    acts = extract_activations(model, X_t, device)

    layer_order = ["features", "pool", "hidden", "logits"]

    scores = {}

    for layer in layer_order:
        A = acts[layer].numpy()
        A = normalize(A)
        scores[layer], _, _, _ = rsa(wavelet_ref, A)

    fig, ax = plt.subplots(figsize=(9, 5))

    values = [scores[l] for l in layer_order]
    colors = ["#3776AB", "#9B6DD8", "#E8924A", "#3DB37C"]

    bars = ax.bar(labels if labels else layer_order, values, color=colors)

    ax.set_ylim(0, 1)
    ax.set_title(f"{model_name}: RSA similarity to Wavelets per layer")
    ax.set_ylabel("RSA correlation")
    ax.grid(axis="y", alpha=0.3)

    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, v + 0.03, f"{v:.3f}", ha="center")

    plt.tight_layout()
    plt.savefig(f"{model_name}_rsa.png", dpi=120)
    plt.close()

    return scores

def build_wavelets(scans, target_len=256):

    features = []

    for scan in scans:
        w = pre_process(scan.matrix, downsample_factor=100)
        w = np.asarray(w)

        C = w.shape[0]
        out = np.zeros((C, target_len))

        old = np.linspace(0, 1, w.shape[1])
        new = np.linspace(0, 1, target_len)

        for c in range(C):
            out[c] = np.interp(new, old, w[c])

        out = normalize(out)
        features.append(out)

    return np.stack(features)

if __name__ == "__main__":

    ctx_train = retrieve_context("Final_project_data", "Intra", "train")
    ctx_test = retrieve_context("Final_project_data", "Intra", "test")

    X_train, y_train = convert_data(down_sample, min_max_scaling, create_sequences, ctx_train)
    X_test, y_test = convert_data(down_sample, min_max_scaling, create_sequences, ctx_test)

    X_train = X_train.permute(0, 2, 1)
    X_test = X_test.permute(0, 2, 1)

    guided = GuidedCNN(
        WaveletConv1D(X_train.shape[1]),
        input_size=X_train.shape[1],
        output_size=len(set(ctx_train.labels.values()))
    )

    cnn = StandardCNN(
        input_size=X_train.shape[1],
        output_size=len(set(ctx_train.labels.values()))
    )

    train_model(guided, X_train, y_train,
                nn.CrossEntropyLoss(),
                torch.optim.Adam(guided.parameters(), lr=0.001),
                num_epochs=20)

    train_model(cnn, X_train, y_train,
                nn.CrossEntropyLoss(),
                torch.optim.Adam(cnn.parameters(), lr=0.001),
                num_epochs=20)

    test_model(guided, X_test, y_test, nn.CrossEntropyLoss())
    test_model(cnn, X_test, y_test, nn.CrossEntropyLoss())

    train_scans = [s for p in ctx_train.persons for s in p.get_scans()]
    test_scans = [s for p in ctx_test.persons for s in p.get_scans()]

    wave_test = build_wavelets(test_scans)

    guided_scores = Wavelet_RSA_similarity(
        guided, wave_test, X_test.numpy(),
        "Guided CNN",
        labels=["Conv1d k = 7 + BN + ReLU", "AdaptiveAvgPool", "linear + ReLU (128 hidden)", "Linear (logits)"]
    )

    cnn_scores = Wavelet_RSA_similarity(
        cnn, wave_test, X_test.numpy(),
        "Standard CNN",
        labels=["Conv1d k = 7 + BN + ReLU", "AdaptiveAvgPool", "linear + ReLU (128 hidden)", "Linear (logits)"]
    )

    print("\nGUIDED:", guided_scores)
    print("\nSTANDARD:", cnn_scores)