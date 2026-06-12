import os
import copy
import itertools

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from mne.decoding import CSP
import matplotlib.pyplot as plt

from ckatorch.core import cka_base
from ckatorch.plot import plot_cka

from CNN_network import ConvolutionalNeuralNetwork
from CSP_experiment import CSP_CNN1, CSP_CNN2
from Wavelet_experiment import Wavelet_CNN1, Wavelet_CNN2
from LTSM_network import (retrieve_context, convert_data, create_sequences,
                          down_sample, min_max_scaling, train_model)

N_EPOCHS = 20
DOWNSAMPLE_FACTOR = 40
N_CSP = 4
SFREQ = 2034 / DOWNSAMPLE_FACTOR
OUT_DIR = "cka_plots"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

MODEL_LAYERS = {
    "CNN": ["features.0", "features.4", "features.8", "classifier.1", "classifier.4"],
    "CSP1": ["csp_layer", "temporal_convs.0", "temporal_convs.5", "classifier.1", "classifier.4"],
    "CSP2": ["temporal_conv.0", "csp_layer", "depthwise.0", "pointwise.0", "classifier.1", "classifier.4"],
    "Wav1": ["features.0", "features.4", "features.8", "classifier.1", "classifier.4"],
    "Wav2": ["tmpband.0", "tmpband.4", "tmpband.8", "tmp.0", "classifier.1", "classifier.4"],
}


def train_models():
    context_train = retrieve_context(parent_folder="Final_project_data", subdirectory="Intra", type_of_data="train")
    X_train, y_train = convert_data(down_sample, min_max_scaling, create_sequences, context_train, down_fact=DOWNSAMPLE_FACTOR)
    X_train = X_train.permute(0, 2, 1)

    n_classes = len(set(context_train.labels.values()))

    csp = CSP(n_components=N_CSP, reg=None, log=False, norm_trace=False)
    csp.fit(X_train, y_train)
    filters = csp.filters_[:N_CSP]

    cnn = ConvolutionalNeuralNetwork(input_size=X_train.shape[1], output_size=n_classes)
    csp1 = CSP_CNN1(X_train.shape[1], n_classes, N_CSP, filters)
    csp2 = CSP_CNN2(X_train.shape[1], n_classes, N_CSP, filters)
    wav1 = Wavelet_CNN1(X_train.shape[1], SFREQ, n_classes)
    wav2 = Wavelet_CNN2(X_train.shape[1], SFREQ, n_classes)

    untrained_models = [copy.deepcopy(cnn), copy.deepcopy(csp1), copy.deepcopy(csp2), copy.deepcopy(wav1), copy.deepcopy(wav2)]

    print("=================== Training CNN =======================")
    train_model(cnn, X_train, y_train, nn.CrossEntropyLoss(), optim.Adam(cnn.parameters(), lr=0.001), num_epochs=N_EPOCHS)
    print("=================== Training CSP1 =======================")
    train_model(csp1, X_train, y_train, nn.CrossEntropyLoss(), optim.Adam(csp1.parameters(), lr=0.001), num_epochs=N_EPOCHS)
    print("=================== Training CSP2 =======================")
    train_model(csp2, X_train, y_train, nn.CrossEntropyLoss(), optim.Adam(csp2.parameters(), lr=0.001), num_epochs=N_EPOCHS)
    print("=================== Training Wavelet1 =======================")
    train_model(wav1, X_train, y_train, nn.CrossEntropyLoss(), optim.Adam(wav1.parameters(), lr=0.001), num_epochs=N_EPOCHS)
    print("=================== Training Wavelet2 =======================")
    train_model(wav2, X_train, y_train, nn.CrossEntropyLoss(), optim.Adam(wav2.parameters(), lr=0.001), num_epochs=N_EPOCHS)

    return untrained_models, [cnn, csp1, csp2, wav1, wav2]


def get_activations(model, layer_names, x):
    acts = {}
    handles = []
    for name, module in model.named_modules():
        if name in layer_names:
            def make_hook(nm):
                def hook(mod, inp, out):
                    acts[nm] = out.reshape(out.shape[0], -1)
                return hook
            handles.append(module.register_forward_hook(make_hook(name)))
    model.eval()
    with torch.no_grad():
        model(x.to(DEVICE))
    for h in handles:
        h.remove()
    return acts


def compute_cka(model_a, layers_a, model_b, layers_b, X):
    acts_a = get_activations(model_a, layers_a, X)
    acts_b = get_activations(model_b, layers_b, X)
    n_a, n_b = len(layers_a), len(layers_b)
    matrix = torch.zeros(n_a, n_b)
    for i, la in enumerate(layers_a):
        for j, lb in enumerate(layers_b):
            matrix[i, j] = cka_base(acts_a[la].cpu(), acts_b[lb].cpu(),
                                    kernel="linear", method="fro_norm")
    return matrix


def plot_into_axis(ax, matrix, layers_a, layers_b, name_a, name_b, subtitle):
    plt.sca(ax)
    plot_cka(
        cka_matrix=matrix,
        first_layers=layers_a,
        second_layers=layers_b,
        first_name=name_a,
        second_name=name_b,
        title=subtitle,
        show_ticks_labels=True,
        short_tick_labels_splits=2,
        use_tight_layout=False,
        show_annotations=True,
        show_half_heatmap=False,
        show_img=False,
        save_path=None,
        vmin=0.0,
        vmax=1.0,
    )


def compare_pair(untrained, trained, name_a, name_b, X):
    layers_a = MODEL_LAYERS[name_a]
    layers_b = MODEL_LAYERS[name_b]

    init_a, init_b = untrained[name_a].to(DEVICE), untrained[name_b].to(DEVICE)
    trn_a, trn_b = trained[name_a].to(DEVICE), trained[name_b].to(DEVICE)

    matrix_init = compute_cka(init_a, layers_a, init_b, layers_b, X)
    matrix_trained = compute_cka(trn_a, layers_a, trn_b, layers_b, X)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    plot_into_axis(axes[0], matrix_init, layers_a, layers_b, name_a, name_b, "Initialized")
    plot_into_axis(axes[1], matrix_trained, layers_a, layers_b, name_a, name_b, "Trained")

    fig.suptitle(f"{name_a} vs {name_b}", fontsize=16)
    fig.tight_layout()
    title = f"{name_a}_vs_{name_b}".replace(" ", "_").replace("/", "-")
    fig.savefig(f"{OUT_DIR}/{title}.png", dpi=400, bbox_inches="tight")
    plt.close("all")


def run_cka_analysis(untrained, trained, X):
    os.makedirs(OUT_DIR, exist_ok=True)
    for name in trained:
        compare_pair(untrained, trained, name, name, X)
    for na, nb in itertools.combinations(trained, 2):
        compare_pair(untrained, trained, na, nb, X)
    print(f"Saved CKA plots to '{OUT_DIR}/'")


if __name__ == "__main__":
    untrained_models, trained_models = train_models()
    u_cnn, u_csp1, u_csp2, u_wav1, u_wav2 = untrained_models
    cnn, csp1, csp2, wav1, wav2 = trained_models

    context_test = retrieve_context(parent_folder="Final_project_data",
                                    subdirectory="Intra", type_of_data="test")
    X_test, _ = convert_data(down_sample, min_max_scaling, create_sequences,
                             context_test, down_fact=DOWNSAMPLE_FACTOR)
    X_test = X_test.permute(0, 2, 1)

    untrained = {"CNN": u_cnn, "CSP1": u_csp1, "CSP2": u_csp2, "Wav1" : u_wav1, "Wav2": u_wav2}
    trained = {"CNN": cnn, "CSP1": csp1, "CSP2": csp2, "Wav1": wav1, "Wav2": wav2}
    run_cka_analysis(untrained, trained, X_test)