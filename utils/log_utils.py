from typing import List, Tuple
import pandas as pd


def average_yearly_logs(
    model_log_list: List[Tuple[str, pd.DataFrame]],
    log_columns: List[str],
    evaluation_range: List[str]
):
    ensamble_averaged_yearly = pd.DataFrame(columns=log_columns)
    for ind, year in enumerate(evaluation_range):
        row = {"evaluation_year": year}
        for col in log_columns:
            col_values = []
            for index, tuple in enumerate(model_log_list):
                model_num = tuple[0]
                data = tuple[1]
                value = data.loc[year, col]
                col_values.append(value)
            row[col] = col_values.mean()
        temp_df = pd.DataFrame([row])
        ensamble_averaged_yearly = pd.concat([ensamble_averaged_yearly, temp_df])

    return ensamble_averaged_yearly
       
            


