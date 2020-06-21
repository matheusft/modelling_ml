from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import numpy as np
import os


class MlModel:

    def __init__(self):
        self.dataset = pd.DataFrame()

    def read_dataset(self, address):
        """Read a dataset into a pandas dataframe from a file address.

        Args:
            address (str): file address.

        Returns:
            int: The return value. 1 for Invalid file extension
                                   2 for reading exception
        """
        filename, file_extension = os.path.splitext(address)
        try:
            if file_extension == '.csv':
                self.dataset = pd.read_csv(address)
                self.column_types_pd_series = self.dataset.dtypes
                return 0

            elif file_extension == '.xls' or file_extension == '.xlsx':
                self.dataset = pd.read_excel(address)
                self.column_types_pd_series = self.dataset.dtypes
                return 0
            else:
                return 1  # Invalid file extension
        except:
            return 2  # Exception

    def generate_histogram(self, column):
        fig, ax = plt.subplots()
        ax = self.dataset[column].hist()
        return ax

    def generate_boxplot(self, column):
        return self.dataset[column].boxplot()

    def generate_plot(self, column):
        fig, ax = plt.subplots()
        self.dataset[column].plot(ax=ax)
        return ax

    def pre_process_data(self, scaling, rm_duplicate, rm_outliers, replace, filter):

        scaling, rm_duplicate, rm_outliers, replace, filter
        self.pre_processed_dataset = self.dataset.copy()

        # Scaling the numeric values in the pre_processed_dataset
        if scaling:
            numeric_columns_to_not_scale = []
            numeric_input_columns = self.pre_processed_dataset.select_dtypes(include=['float64', 'int']).columns.drop(
                labels=numeric_columns_to_not_scale).to_list()
            input_scaler = MinMaxScaler(feature_range=(-1, 1))
            standardised_numeric_input = input_scaler.fit_transform(self.pre_processed_dataset[numeric_input_columns])

            # Updating the scaled values in the pre_processed_dataset
            self.pre_processed_dataset[numeric_input_columns] = standardised_numeric_input

        if rm_duplicate:
            subset_columns_to_drop = []
            self.pre_processed_dataset.drop_duplicates(subset=subset_columns_to_drop, inplace=True)

        if rm_outliers[0]:
            # Computes the Z-score of each value in the column, relative to the column mean and standard deviation
            # Remove Outliers by removing rows that are not within 'standard_deviation_threshold' standard deviations from mean
            # 1std comprises 68% of the data, 2std comprises 95% and 3std comprises 99.7%
            standard_deviation_threshold = rm_outliers[1]
            numeric_columns = self.pre_processed_dataset.select_dtypes(include=['float64', 'int']).columns.to_list()
            self.pre_processed_dataset = self.pre_processed_dataset[
                (np.abs(stats.zscore(self.pre_processed_dataset[numeric_columns])) < standard_deviation_threshold).all(
                    axis=1)]

        if replace[0]:
            for rule in replace[1]:
                value_to_replace = rule[0] # Replace from the ComboBox
                target_column = rule[1]
                new_value = rule[2]
                self.pre_processed_dataset[target_column].replace(to_replace=value_to_replace, value=new_value,
                                                                  inplace=True)

        if filter[0]:
            pass
            # DataFrame = DataFrame[DataFrame['column'] == 1]

        return self.pre_processed_dataset
