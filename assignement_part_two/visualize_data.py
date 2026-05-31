import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from read_data import ScanData, load_data_from_h5_files
from pre_process import pre_process

TASK_MARKERS = {
    "rest": "o",
    "motor": "s",
    "math": "^",
    "memory": "D"
}

def num_to_range(num, inMin, inMax, outMin, outMax):
  return outMin + (float(num - inMin) / float(inMax - inMin) * (outMax
                  - outMin))

def mean_informative_electrode(brain_scan):
    elec_activ_sum = np.abs(brain_scan).sum(axis=0)
    mean_electrodes = []

    for i, col in enumerate(brain_scan.T):
        temp = elec_activ_sum[i] / 2
        mean_electrode = 0
        for c in np.abs(col):
            temp -= c
            if temp < 0:
                temp = abs(temp)
                mean_electrode += temp / c
                break
            mean_electrode += 1
        mean_electrodes.append(mean_electrode)

    return mean_electrodes

def max_informative_electrode(brain_scan):

    max_electrodes = []

    for i, col in enumerate(brain_scan.T):
        max_electrodes.append(np.abs(col).argmax())

    return max_electrodes

def plot_informative_electrodes(persons, operation, downsample=10):

    fig, ax = plt.subplots(figsize=(14, 8))
    cmap = plt.get_cmap("tab10")

    # person_ids = [p.id for p in persons]
    id_change_index_index = 25
    for p in persons:
        p.id += f"_{id_change_index_index}"
        id_change_index_index += 50
    person_colors = {
        person.id: cmap(i % cmap.N)
        for i, person in enumerate(persons)
    }

    grouped = {}

    for person in persons:
        grouped[person.id] = {}

        for scan in person.get_scans():
            task = scan.get_task_name()

            curve = np.asarray(operation(scan.matrix))
            curve = curve[::downsample]

            grouped[person.id].setdefault(task, [])
            grouped[person.id][task].append(curve)

    for person in persons:

        color = person_colors[person.id]

        all_task_means = []

        for task, scans in grouped[person.id].items():

            marker = TASK_MARKERS.get(task, "o")

            # single scans (high opacity)
            
            for curve in scans:
                x = np.arange(len(curve))

                ax.scatter(
                    x, curve,
                    color=color,
                    marker=marker,
                    alpha=0.75,
                    s=22
                )

            # task mean (low opacity)

            task_mean = np.mean(scans, axis=0)
            x = np.arange(len(task_mean))

            ax.scatter(
                x,
                task_mean,
                color=color,
                marker=marker,
                alpha=0.25,
                s=35
            )

            all_task_means.append(task_mean)

        # person mean across tasks 

        if len(all_task_means) > 0:
            person_mean = np.mean(all_task_means, axis=0)
            x = np.arange(len(person_mean))

            ax.plot(
                x,
                person_mean,
                color=color,
                linewidth=3
            )

            ax.scatter(
                x,
                person_mean,
                color=color,
                marker="*",
                s=120,    
                alpha=1.0
            )

    ax.set_xlabel("Electrode (downsampled)")
    ax.set_ylabel("Mean Informative Electrode")
    ax.set_title("Mean Informative Electrode (Person × Task structure)")
    ax.grid(True)

    person_handles = [
        Line2D(
            [0], [0],
            color=person_colors[p.id],
            lw=3,
            label=f"Person {p.id}"
        )
        for p in persons
    ]

    person_legend = ax.legend(
        handles=person_handles,
        title="Persons",
        loc="upper left"
    )

    ax.add_artist(person_legend)

    task_handles = [
        Line2D(
            [0], [0],
            marker=m,
            linestyle="None",
            color="black",
            markersize=8,
            label=task
        )
        for task, m in TASK_MARKERS.items()
    ]

    ax.legend(
        handles=task_handles,
        title="Tasks",
        loc="upper right"
    )

    plt.tight_layout()
    plt.show()

def plot_electrode_activation_through_time(scan, electrode_idxs, pre_processn=False):
    if pre_processn:
        scan = ScanData(scan.id, scan.task, pre_process(scan.matrix))

    plt.figure(figsize=(14, 8))
    plt.title("Electrodes Activation through time")
    plt.xlabel("Time")
    plt.ylabel("Electrode")

    for idx in electrode_idxs:
        plt.plot(scan.matrix[idx], label=f"Electrode {idx}", alpha= num_to_range(idx, min(electrode_idxs), max(electrode_idxs), 0.2, 1.0))
    plt.legend()
    plt.show()

def plot_electrodes_activations__over_single_timestep(scan, timestep=0, pre_processn=False):
    if pre_processn:
        scan = ScanData(scan.id, scan.task, pre_process(scan.matrix))

    activations = scan.matrix.T[timestep]
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(activations)), activations)
    plt.xlabel("Electrode Index")
    plt.ylabel("Activation")
    plt.title(f"Map of Electrodes Activations at Timestep {timestep}")
    plt.grid(True)
    plt.show()

def plot_band_powers_over_electrodes(scan):
    pre_processed_scan = pre_process(scan.matrix, feature_extraction="fourier")
    band_names = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
    plt.figure(figsize=(14, 8))
    plt.title("Band Powers across Electrodes")
    plt.xlabel("Electrode Index")
    plt.ylabel("Average Power")
    for i in range(pre_processed_scan.shape[1]):
        plt.plot(pre_processed_scan[:, i], label=f"{band_names[i]} Band")
    plt.legend()
    plt.show()

def plot_band_powers_over_time(scan, electrode_idx=0):
    pre_processed_scan = pre_process(scan.matrix, sfreq=2034, feature_extraction="wavelets")
    band_names = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
    
    electrode_data = pre_processed_scan[electrode_idx, :, :]
    
    plt.figure(figsize=(14, 8))
    plt.title(f"Time-Frequency Power for Electrode {electrode_idx}")
    plt.xlabel("Time Samples")
    plt.ylabel("Power (fT²)")
    
    for i in range(electrode_data.shape[0]):
        plt.plot(electrode_data[i, :], label=f"{band_names[i]} Band", alpha=0.7)
        
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    parent_folder = "Final_project_data"
    pers1 = load_data_from_h5_files(parent_folder, "Cross", "train")
    pers2 = load_data_from_h5_files(parent_folder, "Cross", "test1")
    pers3 = load_data_from_h5_files(parent_folder, "Cross", "test2")
    pers4 = load_data_from_h5_files(parent_folder, "Cross", "test3")
    pers5 = load_data_from_h5_files(parent_folder, "Intra", "train")
    pers6 = load_data_from_h5_files(parent_folder, "Intra", "test")

    # plot_informative_electrodes(pers1, mean_informative_electrode, downsample=1000)
    # plot_electrodes_activations__over_single_timestep(pers1[0].get_scans()[0], timestep=100, pre_processn=False)
    # plot_electrodes_activations__over_single_timestep(pers1[0].get_scans()[0], timestep=100, pre_processn=True)
    # plot_electrode_activation_through_time(pers1[0].get_scans()[0], [0, 50, 200], pre_processn=False)
    # plot_electrode_activation_through_time(pers1[0].get_scans()[0], [0, 50, 200], pre_processn=True)
    # plot_band_powers_over_electrodes(pers1[0].get_scans()[0])
    plot_band_powers_over_time(pers1[0].get_scans()[0], electrode_idx=0)
