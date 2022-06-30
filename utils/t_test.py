import pandas as pd
from scipy.stats import ttest_1samp, ttest_ind_from_stats, ttest_ind
import os

RESULTS_ROOT = "results/"
FILE_NAME = ["ensamble_2022-06-28_16-29-20_data_yearly_SW",
             "ensamble_2022-06-28_16-29-20_data_yearly_SW"]
SAVE_ROOT = "plots/"


def ttest(
    root: str = RESULTS_ROOT,
    filename: str = FILE_NAME
):

    path = os.path.join(root, filename + ".csv")

    data = pd.read_csv(path)

    oss = data["OSS"]
    oss_theory = data["OSS_theory"]

    theory = ttest_1samp(
        oss_theory, 0
    )

    against_zero = ttest_1samp(
        oss, 0
    )

    model_vs_theory = ttest_ind(
        oss_theory,
        oss
    )

    stats = {
        "theory": theory,
        "zero": against_zero,
        "model_vs_theory": model_vs_theory
    }

    return stats
