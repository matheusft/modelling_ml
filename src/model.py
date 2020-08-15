from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, recall_score, f1_score, precision_score, \
    accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import numpy as np
import operator
import os


class MlModel:

    def __init__(self):
        self.dataset = pd.DataFrame()
        self.column_types_pd_series = []
        self.categorical_variables = []
        self.integer_variables = []
        self.numeric_variables = []

        self.pre_processed_dataset = pd.DataFrame()
        self.pre_processed_column_types_pd_series = []
        self.pre_processed_categorical_variables = []
        self.pre_processed_integer_variables = []
        self.pre_processed_numeric_variables = []

        self.is_data_loaded = False
        self.input_scaler = []

    def read_dataset(self, address):
        """Read a dataset into a pandas dataframe from a file address.

        Args:
            address (str): file address.

        Returns:
            str: The return value. sucess
                                   invalid_file_extension
                                   exception_in_the_file
        """
        filename, file_extension = os.path.splitext(address)

        try:
            #Todo: Check also for .data files, etc
            if file_extension == '.csv':
                self.dataset = pd.read_csv(address)
            elif file_extension == '.xls' or file_extension == '.xlsx':
                self.dataset = pd.read_excel(address)
            else:
                return 'invalid_file_extension'  # Invalid file extension
            self.is_data_loaded = True
            self.dataset.dropna(inplace = True)
            self.pre_processed_dataset = self.dataset.copy()
            #This contains the type of all columns
            self.update_datasets_info()
            return 'sucess'
        except:
            return 'exception_in_the_file'  # Exception

    def update_datasets_info(self):

        dataset = self.dataset
        dataset.reset_index(inplace=True, drop=True)
        self.column_types_pd_series = dataset.dtypes
        self.categorical_variables = dataset.select_dtypes(include=['object']).columns.to_list()
        self.integer_variables = dataset.select_dtypes(include=['int64']).columns.to_list()
        self.numeric_variables = dataset.select_dtypes(include=['int64', 'float64']).columns.to_list()

        dataset = self.pre_processed_dataset
        dataset.reset_index(inplace=True, drop=True)
        self.pre_processed_column_types_pd_series = dataset.dtypes
        self.pre_processed_categorical_variables = dataset.select_dtypes(include=['object']).columns.to_list()
        self.pre_processed_integer_variables = dataset.select_dtypes(include=['int64']).columns.to_list()
        self.pre_processed_numeric_variables = dataset.select_dtypes(include=['int64', 'float64']).columns.to_list()

    def remove_outliers(self, cut_off):
        # Remove Outliers by removing rows that are not within cut_off standard deviations from mean
        numeric_columns = self.pre_processed_dataset.select_dtypes(include=['float64', 'int']).columns.to_list()
        z_score = stats.zscore(self.pre_processed_dataset[numeric_columns])
        self.pre_processed_dataset = self.pre_processed_dataset[(np.abs(z_score) < cut_off).all(axis=1)]
        self.update_datasets_info()

    def scale_numeric_values(self):

        dataset = self.pre_processed_dataset
        self.input_scaler = MinMaxScaler(feature_range=(-1, 1))
        standardised_numeric_input = self.input_scaler.fit_transform(dataset[self.pre_processed_numeric_variables])
        # Updating the scaled values in the pre_processed_dataset
        dataset[self.pre_processed_numeric_variables] = standardised_numeric_input
        self.update_datasets_info()

    def remove_duplicate_rows(self):
        self.pre_processed_dataset.drop_duplicates(inplace=True)
        self.update_datasets_info()

    def remove_constant_variables(self):
        dataset = self.pre_processed_dataset
        self.pre_processed_dataset = dataset.loc[:, (dataset != dataset.iloc[0]).any()]
        self.update_datasets_info()

    def replace_values(self, target_variable, is_numeric_variable, new_value, old_values):

        variable_data_type = self.pre_processed_column_types_pd_series[target_variable]
        if is_numeric_variable:
            value_to_replace = float(old_values)
            new_value = float(new_value) if '.' in new_value or 'e' in new_value.lower() else int(new_value)
        else:
            value_to_replace = old_values
        # Making sure the value to be replaced mataches with the dtype of the dataset
        value_to_replace = pd.Series(value_to_replace).astype(variable_data_type).values[0]
        # Converting to either float or int, depending if . or e is in the string
        self.pre_processed_dataset[target_variable].replace(to_replace=value_to_replace, value=new_value,inplace=True)

        self.update_datasets_info()

    def filter_out_values(self, filtering_variable, filtering_value, filtering_operator):

        column_of_filtering_variable = self.pre_processed_dataset[filtering_variable]
        dataset = self.pre_processed_dataset
        if filtering_operator == 'Equal to':
            self.pre_processed_dataset = dataset[~operator.eq(column_of_filtering_variable, filtering_value)]
        elif filtering_operator == 'Not equal to':
            self.pre_processed_dataset = dataset[~operator.ne(column_of_filtering_variable, filtering_value)]
        elif filtering_operator == 'Less than':
            self.pre_processed_dataset = dataset[~operator.lt(column_of_filtering_variable, filtering_value)]
        elif filtering_operator == 'Less than or equal to':
            self.pre_processed_dataset =dataset[~operator.le(column_of_filtering_variable, filtering_value)]
        elif filtering_operator == 'Greater than':
            self.pre_processed_dataset = dataset[~operator.gt(column_of_filtering_variable, filtering_value)]
        elif filtering_operator == 'Greater than or equal to':
            self.pre_processed_dataset =dataset[~operator.ge(column_of_filtering_variable, filtering_value)]

        self.update_datasets_info()

    def split_data_train_test(self, model_parameters):

        # Making a copy of the pre_processed_dataset using only input/output columns
        input_dataset = self.pre_processed_dataset[
            model_parameters['input_variables'] + model_parameters['output_variables']].copy()
        input_dataset.reset_index(inplace=True)
        # Selecting the categorical variables that are in the training set
        categorical_variables_in_training = list(set(self.pre_processed_categorical_variables) & set(
            model_parameters['input_variables']))

        self.categorical_encoders = {}
        encoded_categorical_columns = pd.DataFrame()
        for column in categorical_variables_in_training:
            # Creating an encoder for each non-nueric column and appending to a list of encoders
            self.categorical_encoders[column] = LabelEncoder()
            values_to_fit_transform = input_dataset[column].values
            self.categorical_encoders[column].fit(values_to_fit_transform)
            # Creating a dataframe with the encoded columns
            encoded_categorical_columns[column] = self.categorical_encoders[column].transform(values_to_fit_transform)

        data_indexes = np.array(input_dataset.index)
        if model_parameters['shuffle_samples']:
            np.random.shuffle(data_indexes)

        # Splitting the indexes of the Dtaframe into train_indexes and test_indexes
        train_indexes = data_indexes[0:round(len(data_indexes) * model_parameters['train_percentage'])]
        test_indexes = data_indexes[round(len(data_indexes) * model_parameters['train_percentage']):]

        # Replacing the categorical values with the encoded values
        input_dataset[categorical_variables_in_training] = encoded_categorical_columns

        train_dataset = input_dataset.loc[train_indexes]
        test_dataset = input_dataset.loc[test_indexes]

        # Dataframes to numpy arrays
        x_train = train_dataset[model_parameters['input_variables']].values
        x_test = test_dataset[model_parameters['input_variables']].values
        y_train = train_dataset[model_parameters['output_variables']].values
        y_test = test_dataset[model_parameters['output_variables']].values

        if len(model_parameters['output_variables']) == 1:
            # Make sure the y array is in the format (n_samples,)
            y_train = y_train.ravel()
            y_test = y_test.ravel()

        # if the target class is an integer which was scaled between 0 and 1
        if not model_parameters['is_regression'] and self.pre_processed_column_types_pd_series[
            model_parameters['output_variables'][0]].kind == 'i': #Todo add condition to check if it was scaled as well
            original_target_categories = self.dataset[model_parameters['output_variables']].values
            y_train = original_target_categories[train_indexes]
            y_test = original_target_categories[test_indexes]

        return {'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test}

    def train(self, model_parameters, algorithm_parameters):

        split_dataset = self.split_data_train_test(model_parameters)
        x_train = split_dataset['x_train']
        x_test = split_dataset['x_test']
        y_train = split_dataset['y_train']
        y_test = split_dataset['y_test']

        if model_parameters['is_regression']:

            if model_parameters['algorithm'] == 'nn':
                ml_model = MLPRegressor(hidden_layer_sizes=tuple(algorithm_parameters['n_of_neurons_each_layer']),
                                        max_iter=algorithm_parameters['max_iter'],
                                        solver=algorithm_parameters['solver'],
                                        activation=algorithm_parameters['activation_func'],
                                        alpha=algorithm_parameters['alpha'],
                                        learning_rate=algorithm_parameters['learning_rate'],
                                        validation_fraction=algorithm_parameters['validation_percentage'])

                ml_model.fit(x_train, y_train)
                y_pred = ml_model.predict(x_test)
            elif model_parameters['algorithm'] == 'svm':
                algorithm_parameters = []
            elif model_parameters['algorithm'] == 'random_forest':
                algorithm_parameters = []
            elif model_parameters['algorithm'] == 'grad_boosting':
                algorithm_parameters = []

            r2_score_result = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)

            if len(model_parameters['output_variables']) == 1:
                np_y_test = np.array(y_test).flatten()
                valid_indexes = [i for i, x in enumerate(np_y_test) if x != 0]
                np_y_pred = np.array(y_pred).flatten()
                percentage_errors = abs(np_y_test[valid_indexes] - np_y_pred[valid_indexes] / np_y_test[valid_indexes])
                array_zero_errors = np.zeros(abs(len(np_y_test) - len(valid_indexes)))
                percentage_errors_with_zeros = np.concatenate((percentage_errors, array_zero_errors))
                data_to_plot = percentage_errors_with_zeros
            else:
                data_to_plot = {'values': [], 'labels': model_parameters['output_variables']}
                for i in range(len(model_parameters['output_variables'])):
                    # Todo: organizar isso melhor, renomear np_y_test
                    y_test_column_i = np.array(y_test[:, i])
                    valid_indexes = [j for j, x in enumerate(y_test_column_i) if x != 0]
                    y_pred_column_i = y_pred[:, i]
                    percentage_errors = abs(
                        (y_test_column_i[valid_indexes] - y_pred_column_i[valid_indexes]) / y_test_column_i[
                            valid_indexes])
                    array_zero_errors = np.zeros(abs(len(y_test_column_i) - len(valid_indexes)))
                    percentage_errors_with_zeros = np.concatenate((percentage_errors, array_zero_errors))
                    data_to_plot['values'].append(percentage_errors_with_zeros.mean())

            training_output = {'r2_score': r2_score_result, 'mse': mse, 'mae': mae, 'rmse': rmse,
                               'data_to_plot': data_to_plot}

            return training_output

        else:

            # Todo : classification y values can be either objects or ints - check this when updating the input/output tab
            self.output_class_label_encoder = LabelEncoder()
            self.output_class_label_encoder.fit(np.concatenate((y_train, y_test)).ravel())
            encoded_y_train = self.output_class_label_encoder.transform(y_train.ravel())
            encoded_y_test = self.output_class_label_encoder.transform(y_test.ravel())

            if model_parameters['algorithm'] == 'nn':
                ml_model = MLPClassifier(hidden_layer_sizes=tuple(algorithm_parameters['n_of_neurons_each_layer']),
                                         max_iter=algorithm_parameters['max_iter'],
                                         solver=algorithm_parameters['solver'],
                                         activation=algorithm_parameters['activation_func'],
                                         alpha=algorithm_parameters['alpha'],
                                         learning_rate=algorithm_parameters['learning_rate'],
                                         validation_fraction=algorithm_parameters['validation_percentage'])
                ml_model.fit(x_train, encoded_y_train)
                encoded_y_pred = ml_model.predict(x_test)
            elif model_parameters['algorithm'] == 'svm':
                algorithm_parameters = []
            elif model_parameters['algorithm'] == 'random_forest':
                algorithm_parameters = []
            elif model_parameters['algorithm'] == 'grad_boosting':
                algorithm_parameters = []
            elif model_parameters['algorithm'] == 'knn':
                algorithm_parameters = []

            number_of_classes = len(np.unique(np.concatenate((y_train, y_test))))
            if number_of_classes > 2:
                average_value = 'macro'
                # Todo Understand the difference between macro and micro
            else:
                average_value = 'binary'

            recall = recall_score(encoded_y_test, encoded_y_pred, average=average_value, zero_division=0)
            f1 = f1_score(encoded_y_test, encoded_y_pred, average=average_value, zero_division=0)
            accuracy = accuracy_score(encoded_y_test, encoded_y_pred)
            precision = precision_score(encoded_y_test, encoded_y_pred, average=average_value, zero_division=0)

            df_conf = pd.DataFrame(confusion_matrix(encoded_y_test, encoded_y_pred))
            df_conf.set_index(self.output_class_label_encoder.inverse_transform(df_conf.index), inplace=True)
            df_conf.columns = self.output_class_label_encoder.inverse_transform(df_conf.columns)

            training_output = {'recall_score': recall, 'f1_score': f1, 'precision_score': precision,
                               'accuracy': accuracy,
                               'data_to_plot': df_conf}

            return training_output
