"""
Architecture relationships:
StandardCNN      = pure CNN baseline (user-provided)
CSP_CNN          = frozen CSP feature extractor + StandardCNN.classifier head
CSPGuidedCNN     = architectural mimicry of CSP operations + StandardCNN-style classifier

CSP_CKA_similarity(model)
    Compares CSP-derived representations against each layer of `model`
    and saves a bar plot of linear CKA scores.

Dependencies
    pip install git+https://github.com/RistoAle97/centered-kernel-alignment
"""

from __future__ import annotations
import data_utils
from read_data import load_data_from_h5_files
from CNN_network import ConvolutionalNeuralNetwork

import matplotlib
matplotlib.use("Agg")  # safe for headless environments

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import eigh
from scipy.signal import decimate
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

def _normalized_cov(X: np.ndarray) -> np.ndarray:
    """Σ = X Xᵀ / trace(X Xᵀ).  X: (C, T) → (C, C)"""
    cov = X @ X.T
    return cov / np.trace(cov)


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Linear CKA between feature matrices (n, d1) and (n, d2).
    Formula from https://github.com/RistoAle97/centered-kernel-alignment

        CKA(X, Y) = ‖Yᵀ X‖_F² / (‖Xᵀ X‖_F · ‖Yᵀ Y‖_F)

    Invariant to orthogonal transformations and isotropic scaling.
    """
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    num = (Y.T @ X).norm() ** 2
    den = (X.T @ X).norm() * (Y.T @ Y).norm()
    # Clamp to [0, 1] — Cauchy-Schwarz guarantees this analytically,
    # but float32 can drift slightly above when correlation is near-perfect.
    return torch.clamp(num / (den + 1e-10), 0.0, 1.0)

def down_sample(scans: list[np.ndarray], factor: int = 20) -> list[np.ndarray]:
    """Downsample each scan along the time axis (axis=1)."""
    return [decimate(scan, factor, axis=1) for scan in scans]


def min_max_scale(scans: list[np.ndarray]) -> list[np.ndarray]:
    """Min-max scale each scan independently across electrodes × timesteps."""
    scaled = []
    for scan in scans:
        scaler = data_utils.DataScaler()
        scaler.fit(scan)
        scaled.append(scaler.transform(scan))
    return scaled

# ─────────────────────────────────────────────────────────────────────────────
# Multiclass CSP  (One-vs-Rest)
# ─────────────────────────────────────────────────────────────────────────────

class MulticlassCSP:
    """One-vs-Rest CSP. Produces N · 2m spatial filters total."""

    def __init__(self, n_filters: int = 2):
        self.n_filters = n_filters
        self.classes_: np.ndarray | None = None
        self.W_: dict[int, np.ndarray] = {}

    def _fit_binary(self, X_pos, X_neg) -> np.ndarray:
        Sp = np.mean([_normalized_cov(x) for x in X_pos], axis=0)
        Sn = np.mean([_normalized_cov(x) for x in X_neg], axis=0)
        Sc = Sp + Sn
        Sc += np.eye(Sc.shape[0]) * 1e-6  # regularize
        eigvals, eigvecs = eigh(Sp, Sc)            # ascending
        m = self.n_filters
        idx = np.concatenate([np.arange(m), np.arange(-m, 0)])
        return eigvecs[:, idx]                     # (C, 2m)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MulticlassCSP":
        self.classes_ = np.unique(y)
        for cls in self.classes_:
            mask = y == cls
            self.W_[cls] = self._fit_binary(list(X[mask]), list(X[~mask]))
        return self

    def get_stacked_filters(self) -> np.ndarray:
        """All OVR filters stacked as (N · 2m, C)."""
        return np.concatenate([self.W_[c].T for c in self.classes_], axis=0)

    def get_filters_tensor(self) -> torch.Tensor:
        return torch.tensor(self.get_stacked_filters(), dtype=torch.float32)

class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = nn.Sequential(
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

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 4)
        )

class StandardCNN(nn.Module):
    def __init__(self, input_size, output_size=4, dropout=0.3):
        super().__init__()

        self.first_layer = nn.Conv1d(
            input_size,
            64,
            kernel_size=7,
            padding=3
        )

        self.backbone = nn.Sequential(
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

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        #x = x.permute(0, 2, 1)
        x = self.first_layer(x)
        x = self.backbone(x)
        return self.classifier(x)


# ─────────────────────────────────────────────────────────────────────────────
# CSP + CNN  —  CSP as feature extractor + StandardCNN-style classifier head
# ─────────────────────────────────────────────────────────────────────────────

class CSP_CNN(nn.Module):
    """
    1. Frozen CSP spatial filtering (1×1 Conv1d with CSP weights).
    2. Log-variance pooling over time  → (B, n_csp) feature vector.
    3. Classifier head with identical structure to StandardCNN.classifier,
       input dim adapted from `input_size*5` → `n_csp` (CSP outputs n_csp feats).

    Only the classifier head is trainable.
    """

    def __init__(
        self,
        csp_filters: torch.Tensor,    # (n_csp, n_channels)
        output_size: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        n_csp, n_channels = csp_filters.shape

        # ── Frozen CSP layer  (1×1 conv = pure spatial filter) ─────────────
        self.csp = nn.Conv1d(n_channels, n_csp, kernel_size=1, bias=False)
        with torch.no_grad():
            self.csp.weight.data = csp_filters.unsqueeze(-1)   # (n_csp, C, 1)
        for p in self.csp.parameters():
            p.requires_grad = False

        # ── Classifier head (StandardCNN.classifier structure) ─────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_csp, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )

    def forward(self, x):
        x = self.csp(x)                              # (B, n_csp, T)
        x = torch.log(x.var(dim=-1) + 1e-8)          # (B, n_csp)   log-variance
        return self.classifier(x)


# ─────────────────────────────────────────────────────────────────────────────
# CSP-Guided CNN
#   Architecture mimics CSP operations. 
#   The hypothesis: does this structural inductive bias alone push the
#   network to learn CSP-like representations under gradient descent?
# ─────────────────────────────────────────────────────────────────────────────

class _LogVarPool(nn.Module):
    """
    Square → AvgPool over time → log.   Encodes CSP's log-variance step.
        z_k(t)           — virtual channel from spatial filter
        var(z_k)         = mean_t z_k(t)²
        f_k              = log(var(z_k))
    Replaces the plain AdaptiveAvgPool of StandardCNN.
    """
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):                              # (B, F, T)
        return torch.log(self.avgpool(x ** 2) + 1e-8)  # (B, F, 1)


class CSPGuidedCNN(nn.Module):
    def __init__(self, input_size, output_size=4, dropout=0.3):
        super().__init__()

        self.first_layer = nn.Conv1d(
            input_size,
            64,
            kernel_size=1,
            bias=False
        )

        self.backbone = nn.Sequential(
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

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        x = self.first_layer(x)
        x = self.backbone(x)
        return self.classifier(x)

# ─────────────────────────────────────────────────────────────────────────────
# Training & evaluation
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
):
    """Cross-entropy training. Only parameters with requires_grad=True are optimised."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)

    model = model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    criterion = nn.CrossEntropyLoss()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_enc,   dtype=torch.long)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item()
        avg_loss = running / len(loader)
        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {avg_loss:.4f}"
        )
    return model, le


def evaluate_model(model, X_test, y_test, le: LabelEncoder, device: str = "cpu") -> float:
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds_enc = model(X_t).argmax(dim=-1).cpu().numpy()
    return accuracy_score(y_test, le.inverse_transform(preds_enc))


# ─────────────────────────────────────────────────────────────────────────────
# CKA per layer
# ─────────────────────────────────────────────────────────────────────────────

def _extract_layer_activations(
    model: nn.Module,
    X_tensor: torch.Tensor,
    device: str = "cpu",
    batch_size: int = 32,
) -> dict[str, torch.Tensor]:
    """
    Run `model` on `X_tensor` and extract activations from every layer,
    flattened to (B, Features) for CKA compatibility.
    Works with any architecture (CNN, MLP, Transformer, etc.).
    """
    model = model.to(device).eval()
    loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=False)

    # Collect all leaf modules (the actual compute layers, not containers)
    named_leaves = [
        (name, module)
        for name, module in model.named_modules()
        if len(list(module.children())) == 0 and name != ""
    ]

    # Register forward hooks
    activations: dict[str, list[torch.Tensor]] = {name: [] for name, _ in named_leaves}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            # Handle modules that return tuples (e.g. transformer layers, LSTM)
            out = output[0] if isinstance(output, (tuple, list)) else output
            activations[name].append(out.detach().flatten(1).cpu())
        return hook_fn

    for name, module in named_leaves:
        hooks.append(module.register_forward_hook(make_hook(name)))

    # Forward pass
    with torch.no_grad():
        for (xb,) in loader:
            model(xb.to(device))

    # Cleanup hooks
    for h in hooks:
        h.remove()

    return {name: torch.cat(tensors, dim=0) for name, tensors in activations.items()}


def CSP_CKA_similarity(
    model: nn.Module,
    csp_filters: torch.Tensor,
    X_data: np.ndarray,
    model_name: str = "Model",
    device: str = "cpu",
    save_dir: str = ".",
    layer_labels: dict[str, str] | None = None,
) -> dict[str, float]:
    """
    Compare CSP-derived representations against each layer of `model`
    using linear CKA. Works with any architecture.

    Parameters
    ----------
    model        : any nn.Module
    csp_filters  : (n_csp, n_channels) tensor of CSP spatial filters
    X_data       : (n_trials, n_channels, n_times)
    layer_labels : optional {layer_name: display_label} override for x-axis
    """
    device = torch.device(device)

    # ── 1. CSP reference representation ───────────────────────────────────
    X_t = torch.tensor(X_data, dtype=torch.float32).to(device)
    csp_w = csp_filters.to(device).unsqueeze(-1)
    with torch.no_grad():
        csp_signals = F.conv1d(X_t, csp_w)
    csp_ref = csp_signals.flatten(1).cpu()

    # ── 2. Extract activations from every leaf layer ─────────────────────
    acts = _extract_layer_activations(model, X_t.cpu(), device=str(device))

    # ── 3. CKA scores ───────────────────────────────────────────────────
    layer_names = list(acts.keys())
    scores = {
        name: float(linear_cka(acts[name], csp_ref).item())
        for name in layer_names
    }

    # ── 4. Plot ──────────────────────────────────────────────────────────
    labels_map = layer_labels or {}
    labels = [labels_map.get(n, n) for n in layer_names]
    values = [scores[n] for n in layer_names]

    fig, ax = plt.subplots(figsize=(max(9, len(layer_names) * 1.2), 5))
    colours = plt.cm.viridis(np.linspace(0.2, 0.85, len(layer_names)))
    bars = ax.bar(range(len(values)), values, color=colours)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Linear CKA  (CSP vs. layer)", fontsize=12)
    ax.set_title(f"{model_name}: CKA similarity to CSP per layer", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                f"{v:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    fname = f"{save_dir}/cka_{model_name.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"  saved {fname}")
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--CKA", type=bool, default=False, help="Run CKA analysis and plotting (takes a while)")
    args = parser.parse_args()
    
    EPOCHS     = 20
    BATCH_SIZE = 1
    DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}\n")

    #train_persons = load_data_from_h5_files(subdirectory="Intra", type_of_data="train")
    #test_persons  = load_data_from_h5_files(subdirectory="Intra", type_of_data="test")

    train_persons = load_data_from_h5_files(subdirectory="Cross", type_of_data="train")
    test_persons  = load_data_from_h5_files(subdirectory="Cross", type_of_data="test1")
    #test_persons  = load_data_from_h5_files(subdirectory="Cross", type_of_data="test2")
    #test_persons  = load_data_from_h5_files(subdirectory="Cross", type_of_data="test3")

    train_scans = [scan for p in train_persons for scan in p.get_scans()]
    test_scans  = [scan for p in test_persons  for scan in p.get_scans()]

    # Expected shape: (n_trials, n_channels, n_times)
    #X_train, y_train = np.array([down_sample(min_max_scaling(scan.matrix) , downsample_factor = 20) for scan in train_scans]), np.array([scan.task for scan in train_scans])
    #X_test,  y_test  = np.array([down_sample(min_max_scaling(scan.matrix), downsample_factor = 20) for scan in test_scans]), np.array([scan.task for scan in test_scans])
    #X_train, y_train = np.array([pre_process(scan.matrix, downsample_factor = 100) for scan in train_scans]), np.array([scan.task for scan in train_scans])
    #X_test,  y_test  = np.array([pre_process(scan.matrix, downsample_factor = 100) for scan in test_scans]), np.array([scan.task for scan in test_scans])

    #X_train, y_train = np.array(normalization([downsample(scan.matrix, factor = 20) for scan in train_scans])), np.array([scan.task for scan in train_scans])
    #X_test,  y_test  = np.array(normalization([downsample(scan.matrix, factor = 20) for scan in test_scans])), np.array([scan.task for scan in test_scans])

    X_train, y_train = np.array(min_max_scale(down_sample([scan.matrix for scan in train_scans], factor=20))), np.array([scan.task for scan in train_scans])
    X_test,  y_test  = np.array(min_max_scale(down_sample([scan.matrix for scan in test_scans], factor=20))), np.array([scan.task for scan in test_scans])

    N_CHANNELS = X_train[0].shape[0]
    N_CLASSES  = len(set(y_train))

    print(f"Training data shape: {X_train.shape[0]}  Classes: {N_CLASSES}")

    # ── Fit CSP on training data ──────────────────────────────────────────
    print("Fitting CSP on training data...")
    csp = MulticlassCSP(n_filters=2)
    csp.fit(X_train, y_train)
    csp_filters = csp.get_filters_tensor()
    print(f"  CSP filters shape : {tuple(csp_filters.shape)}\n")

    # ── Train all three models ────────────────────────────────────────────
    print("=" * 60); print("Training StandardCNN"); print("=" * 60)
    std_model = ConvolutionalNeuralNetwork(input_size=N_CHANNELS, output_size=N_CLASSES)
    std_model, le_std = train_model(std_model, X_train, y_train,
                                    EPOCHS, BATCH_SIZE, device=DEVICE)
    acc_std = evaluate_model(std_model, X_test, y_test, le_std, DEVICE)
    print(f"StandardCNN test accuracy: {acc_std:.4f}")

    print("\n" + "=" * 60); print("Training CSP + CNN"); print("=" * 60)
    csp_cnn = CSP_CNN(csp_filters=csp_filters, output_size=N_CLASSES)
    csp_cnn, le_csp = train_model(csp_cnn, X_train, y_train,
                                  EPOCHS, BATCH_SIZE, device=DEVICE)
    acc_csp_cnn = evaluate_model(csp_cnn, X_test, y_test, le_csp, DEVICE)
    print(f"CSP + CNN test accuracy: {acc_csp_cnn:.4f}")

    print("\n" + "=" * 60); print("Training CSP-guided CNN"); print("=" * 60)
    csp_guided = CSPGuidedCNN(input_size=N_CHANNELS, output_size=N_CLASSES)
    csp_guided, le_g = train_model(csp_guided, X_train, y_train,
                                   EPOCHS, BATCH_SIZE, device=DEVICE)
    acc_csp_guided = evaluate_model(csp_guided, X_test, y_test, le_g, DEVICE)
    print(f"CSP-guided CNN test accuracy: {acc_csp_guided:.4f}")