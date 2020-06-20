from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
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

    def generate_histogram(self,column):
        fig, ax = plt.subplots()
        ax = self.dataset[column].hist()
        return ax

    def generate_boxplot(self,column):
        return self.dataset[column].boxplot()

    def generate_plot(self,column):
        fig, ax = plt.subplots()
        self.dataset[column].plot(ax=ax)
        return ax

    def pre_process_data(self,scaling,rm_duplicate,rm_outliers,replace,filter):

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
            subset_columns_to_drop = self.pre_processed_dataset.columns  # Todo add this
            self.pre_processed_dataset.drop_duplicates(subset = subset_columns_to_drop, inplace = True)


        # # https://stackoverflow.com/questions/23199796/detect-and-exclude-outliers-in-pandas-data-frame
        # # http://statisticshelper.com/wp-content/uploads/2018/08/empirical-rule-with-z-scores.png
        # # Computes the Z-score of each value in the column, relative to the column mean and standard deviation
        # # Remove Outliers by removing rows thatr are not within 'standard_deviation_threshold' standard deviations from mean
        # # 1std comprises 68% of the data, 2std comprises 95% and 3std comprises 99.7%
        # standard_deviation_threshold = 6
        # dataset_filtered = dataset_no_constant_columns[(np.abs(stats.zscore(
        #     dataset_no_constant_columns.iloc[:, len(not_numeric_columns_indexes):])) <
        #                                                 standard_deviation_threshold).all(axis=1)]
        #
        # DataFrame.replace(self, to_replace=None, value=None, inplace=False, limit=None, regex=False, method='pad')
        #
        #
        #
        # ###filtering
        #
        # DataFrame = DataFrame[DataFrame['column'] == 1]
        #
        #


        return self.pre_processed_dataset