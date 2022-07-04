from typing import List, Tuple
import pandas as pd
import numpy as np


def average_yearly_logs(
    model_log_list: List[Tuple[int, pd.DataFrame]],
    log_columns: List[str],
    evaluation_range: List[int],
    eval_index_str: str = "evaluation_year"
):
    ensamble_averaged_yearly = pd.DataFrame(columns=log_columns)
    for ind, year in enumerate(evaluation_range):
        row = {eval_index_str: year}
        for col in log_columns:
            col_values = np.empty([len(model_log_list), 1])
            for tuple in model_log_list:
                ind = data = tuple[0]
                data = tuple[1]
                value = data.loc[year, col]
                col_values[ind] = value
            row[col] = col_values.mean()
        temp_df = pd.DataFrame([row])
        ensamble_averaged_yearly = pd.concat(
            [ensamble_averaged_yearly, temp_df])

    ensamble_averaged_yearly[eval_index_str] = ensamble_averaged_yearly[eval_index_str].astype(
        int)
    ensamble_averaged_yearly.set_index([eval_index_str], inplace=True)
    return ensamble_averaged_yearly
