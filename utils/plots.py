import os
from matplotlib import pyplot as plt
import pandas as pd

RESULTS_ROOT = "results/"
FILE_NAME = "evaluation_run_2022-06-22_11-25-51_data_monthly_3"
SAVE_ROOT = "plots/"


def create_yearly_plot(
    root: str = RESULTS_ROOT,
    filename: str = FILE_NAME,
    save_root: str = SAVE_ROOT
):

    path = os.path.join(root, filename + ".csv")
    data = pd.read_csv(path)
    fig, ax = plt.subplots()

    ax.plot(data["evaluation_year"], data["OSS"]*100, label="prediction")
    ax.plot(data["evaluation_year"], data["OSS_theory"]*100, label="theory", linestyle="dotted")
    ax.legend(loc="lower right", fancybox=True)

    plt.axhline(y=0, color="black", linestyle="--")
    plt.grid(axis="x", linestyle="--")
    plt.xticks(rotation=45)
    plt.title("Theory vs Neuronal Network")
    # plt.show()

    save_path = os.path.join(save_root, filename + ".png")
    plt.savefig(save_path)


if __name__ == "__main__":
    create_yearly_plot()
