from typing import List
import pandas as pd
from pandas.tseries.offsets import BMonthEnd, MonthEnd, MonthBegin
import tensorflow as tf


class WindowGenerator():
    def __init__(
            self, data: pd.DataFrame,
            labels: List[str],
            features: List[str],
            training_width: int = 10,
            validation_width: int = 5,
            testing_width: int = 1):

        # Data, Labels and Features
        self.data = data
        self.labels = labels
        self.features = features
        # Start Date, End Date and amount of years in the data set
        self.start_date = min(data.index)
        self.end_date = max(data.index)

        # Window sizes
        self.training_width = training_width
        self.validation_width = validation_width
        self.testing_width = testing_width

        # Start date of evaluation and testing
        self.evaluation_start = self.start_date + \
            MonthBegin(self.training_width*12)
        self.testing_start = self.evaluation_start + \
            MonthBegin(self.validation_width*12)
        self.testing_end = self.testing_start + MonthEnd(self.testing_width*12)

        # Calculate overall evaluation range
        self.evaluation_range = [
            y
            for y
            in range(self.testing_start.year, self.end_date.year + 1)
        ]

    def get_evaluation_range(self):
        return self.evaluation_range

    def get_current_evaluation_year(self):
        return self.testing_start.year

    def update_dates(self, n: int = 1):
        self.training_width += n
        self.evaluation_start = self.start_date + \
            MonthBegin(self.training_width*12)
        self.testing_start = self.evaluation_start + \
            MonthBegin(self.validation_width*12)
        self.testing_end = self.testing_start + MonthEnd(self.testing_width*12)

    def data_end_reached(self):
        if self.end_date <= self.testing_start:
            return True
        else:
            return False

    def split(self):
        """Splits Data Accordingly

        Returns:
            pd.DataFrame: X_train, y_train, X_validation, y_validation, X_test, y_test
        """
        X_train = self.data[self.data.index <
                            self.evaluation_start][self.features].to_numpy()
        y_train = self.data[self.data.index <
                            self.evaluation_start][self.labels].to_numpy()

        X_validation = self.data[(self.data.index >= self.evaluation_start) & (
            self.data.index < self.testing_start)][self.features].to_numpy()
        y_validation = self.data[(self.data.index >= self.evaluation_start) & (
            self.data.index < self.testing_start)][self.labels].to_numpy()

        X_test = self.data[(self.data.index >= self.testing_start) & (
            self.data.index < self.testing_end)][self.features].to_numpy()
        y_test = self.data[(self.data.index >= self.testing_start) & (
            self.data.index < self.testing_end)][self.labels].to_numpy()
        # X_train = tf.data.Dataset.from_generator(
        #     X_train, 
        #     output_signature=tf.TensorSpec(shape=(), dtype=tf.float32))

        return X_train, y_train, X_validation, y_validation, X_test, y_test

    def __repr__(self) -> str:
        return f"""
        evaluation_start: {self.evaluation_start}
        testing_start: {self.testing_start}
        testing_end: {self.testing_end}
        """
