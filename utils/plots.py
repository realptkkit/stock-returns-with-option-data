import os
from matplotlib import pyplot as plt
import pandas as pd

RESULTS_ROOT = "results/"
FILE_NAME = "evaluation_run_2022-06-28_11-50-59_data_monthly_3"
SAVE_ROOT = "plots/"


def create_yearly_plot(
    root: str = RESULTS_ROOT,
    filename: str = FILE_NAME,
    save_root: str = SAVE_ROOT
):
    """Creates the Plots of the given run."""

    path = os.path.join(root, filename + ".csv")
    data = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(data["evaluation_year"], data["OSS"]
            * 100, label="Prediction", marker="s")
    ax.plot(data["evaluation_year"], data["OSS_theory"]*100,
            label="Benchmark", linestyle="dotted", marker="^")
    ax.legend(loc="lower right", fancybox=True)

    plt.axhline(y=0, color="black", linestyle="--")
    plt.grid(axis="x", linestyle="--")
    plt.xticks(data["evaluation_year"], rotation=45)
    plt.title("Explained variance at one year horizon")
    plt.ylabel('R²ₒₒₛ [%]')
    plt.xlabel('Years')

    save_path = os.path.join(save_root, filename + ".png")
    plt.savefig(save_path)


if __name__ == "__main__":
    create_yearly_plot()
