from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":

    mat = loadmat("Xtest.mat")

    mat = {
        k: v
        for k, v in mat.items()
        if not k.startswith("__")
    }

    data_dict = {}

    for key, value in mat.items():

        data_dict[key] = value.flatten()

    data = pd.DataFrame(data_dict)

    data.to_csv("Xtest.csv", index=False)

    plt.plot(data['Xtest'])
    plt.xlabel("Time")
    plt.ylabel("Laser measurement")
    plt.show()