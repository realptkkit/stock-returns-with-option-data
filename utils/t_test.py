import pandas as pd
from scipy.stats import ttest_1samp, ttest_ind_from_stats, ttest_ind
import os

RESULTS_ROOT = "results/"
FILE_NAME = ["evaluation_run_2022-06-28_17-11-51_data_yearly_3",
             "ensamble_2022-06-28_16-29-20_data_yearly_SW"]
SAVE_ROOT = "plots/"


def ttest(
    root: str = RESULTS_ROOT,
    filename: str = FILE_NAME
):
    """Performs the one sided t-test for the given experiment"""
    path = os.path.join(root, filename + ".csv")

    data = pd.read_csv(path)

    oss = data["OSS"]
    oss_theory = data["OSS_theory"]

    theory = ttest_1samp(
        oss_theory, 0, alternative="greater"
    )

    against_zero = ttest_1samp(
        oss, 0, alternative="greater"
    )

    model_vs_theory = ttest_ind(
        oss, oss_theory, alternative="greater"
    )

    stats = {
        "theory": theory.pvalue,
        "zero": against_zero.pvalue,
        "model_vs_theory": model_vs_theory.pvalue,
        "theory_std": oss_theory.std(),
        "std": oss.std()
    }

    return stats
