from datetime import datetime
from typing import List, Tuple
import pandas as pd
from pandas.tseries.offsets import BMonthEnd, MonthEnd, MonthBegin
import tensorflow as tf


class WindowGenerator():
    """This is a helper class that operates as the dataloader over a given dataset. 
    The WindowGenerator iterates over the set with a given window width and splits data
    accordingly in train, validation and evaluation set.
    """

    def __init__(
            self, data: pd.DataFrame,
            labels: List[str],
            features: List[str],
            training_width: int = 10,
            validation_width: int = 1,
            testing_width: int = 1):
        """Initates the window generator.add()

        Args:
            data (pd.DataFrame): The complete dataset.add()
            labels (List[str]): A list of string with the label column names.
            features (List[str]): A list of strings with the feature column names.
            training_width (int, optional): The width of the training window. Defaults to 10.
            validation_width (int, optional): The width of the validation window. Defaults to 1.
            testing_width (int, optional): The width of the testing window. Defaults to 1.
        """
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
            in range(self.testing_start.year, self.end_date.year)  # Add plus 1
        ]

    def get_feature_shape(self) -> Tuple[int, int]:
        """Returns the shape of the featur.

        Returns:
            Tuple[int, int]: Tuple widh height and widht.
        """
        return self.data[self.features].shape

    def get_evaluation_range(self) -> List[int]:
        """Returns the validation range.

        Returns:
            List[int]: Contains each year on which the evaluation should run for.
        """
        return self.evaluation_range

    def get_current_evaluation_year(self) -> int:
        """Get the current evaluation year of the window

        Returns:
            int: Returns an int specifing the year.
        """
        return self.testing_start.year

    def update_dates(self, n: int = 1) -> None:
        """Updates the training width. Adds n years to the training range and adjusts
        validation and evaluation size.add()

        Args:
            n (int, optional): Determines the amount of years which should be added to the
            training range. Defaults to 1.
        """
        self.training_width += n
        self.evaluation_start = self.start_date + \
            MonthBegin(self.training_width*12)
        self.testing_start = self.evaluation_start + \
            MonthBegin(self.validation_width*12)
        self.testing_end = self.testing_start + MonthEnd(self.testing_width*12)

    def data_end_reached(self) -> bool:
        """Tests if the evaluation range lays beyond the last data point.

        Returns:
            bool: Returns True if the end is reached.
        """
        if self.end_date <= self.testing_end:
            return True
        else:
            return False

    def split(self):
        """Splits Data Accordingly.

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

        return X_train, y_train, X_validation, y_validation, X_test, y_test

    def __repr__(self) -> str:
        return f"""
        evaluation_start: {self.evaluation_start}
        testing_start: {self.testing_start}
        testing_end: {self.testing_end}
        """
