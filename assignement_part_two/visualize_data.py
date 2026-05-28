import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from read_data import load_data_from_h5_files
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import gaussian_filter1d


TASK_MARKERS = {
    "rest": "o",
    "motor": "s",
    "math": "^",
    "memory": "D"
}


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

def plot_electrode(persons, operation, downsample=10):

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

def plot_electrodes_activation():
    plt.figure(figsize=(14, 8))
    plt.title("One Electrode through time for a single scan")
    plt.xlabel("Time")
    plt.ylabel("Electrode")
    plt.plot(pers1[0].get_scans()[0].matrix[0], alpha=0.5)
    plt.plot(pers1[0].get_scans()[0].matrix[100], alpha=0.8)
    plt.plot(pers1[0].get_scans()[0].matrix[200], alpha=0.2)
    plt.show()


if __name__ == "__main__":
    parent_folder = "Final_project_data"
    pers1 = load_data_from_h5_files(parent_folder, "Cross", "train")
    pers2 = load_data_from_h5_files(parent_folder, "Cross", "test1")
    pers3 = load_data_from_h5_files(parent_folder, "Cross", "test2")
    pers4 = load_data_from_h5_files(parent_folder, "Cross", "test3")
    pers5 = load_data_from_h5_files(parent_folder, "Intra", "train")
    pers6 = load_data_from_h5_files(parent_folder, "Intra", "test")


    plot_electrode(pers1, mean_informative_electrode, downsample=1000)

