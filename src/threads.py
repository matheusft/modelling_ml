from view import QtCore, QtWidgets
import pandas as pd
import seaborn as sns

class Train_Model_WorkerSignals(QtCore.QObject):
    finished = QtCore.pyqtSignal(object, object)
class Train_Model_Thread(QtCore.QRunnable):
    """
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    """

    def __init__(self, *args, **kwargs):
        super(Train_Model_Thread, self).__init__()

        self.ml_model = args[0]
        self.model_parameters = args[1]
        self.algorithm_parameters = args[2]
        self.ui = args[3]

        self.args = args
        self.kwargs = kwargs
        self.signals = Train_Model_WorkerSignals()

    @QtCore.pyqtSlot()
    def run(self):
        # training the model
        result = self.ml_model.train(self.model_parameters, self.algorithm_parameters)
        # sending the output of the thread to the assigned function
        self.signals.finished.emit(result, self.model_parameters)

class Load_Dataset_WorkerSignals(QtCore.QObject):
    display_message = QtCore.pyqtSignal(object, object, object, object)
    populate_tablewidget_with_dataframe = QtCore.pyqtSignal(object, object)
    update_train_test_shape_label = QtCore.pyqtSignal()
    stop_spinner = QtCore.pyqtSignal(object, object, object)
class Load_Dataset_Thread(QtCore.QRunnable):
    """
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    """

    def __init__(self, *args, **kwargs):
        super(Load_Dataset_Thread, self).__init__()

        self.ui = args[0]
        self.ml_model = args[1]
        self.file_path = args[2]

        self.args = args
        self.kwargs = kwargs
        self.signals = Load_Dataset_WorkerSignals()

    @QtCore.pyqtSlot()
    def run(self):

        ui = self.ui
        ml_model = self.ml_model

        return_code = self.ml_model.read_dataset(self.file_path)

        if return_code == 'sucess':
            self.signals.populate_tablewidget_with_dataframe.emit(self.ui.dataset_tableWidget, self.ml_model.dataset)
            self.signals.populate_tablewidget_with_dataframe.emit(self.ui.pre_process_dataset_tableWidget,
                                                                  self.ml_model.dataset)
            widgets_to_enable = [ui.plot_radioButton, ui.boxplot_radioButton, ui.histogram_radioButton,
                                 ui.remove_duplicates_pushButton, ui.remove_constant_variables_pushButton,
                                 ui.numeric_scaling_pushButton, ui.remove_outliers_pushButton,
                                 ui.addrule_filter_value_pushButton, ui.addrule_replace_value_pushButton,
                                 ui.add_input_columns_pushButton, ui.add_output_columns_pushButton,
                                 ui.remove_preprocessing_rule_pushButton,
                                 ui.clear_preprocessing_rule_pushButton]

            [item.setEnabled(True) for item in widgets_to_enable]

            # Here we update the variable_to_plot_comboBox
            if ui.variable_to_plot_comboBox.count() > 0:  # If the comboBox is not empty
                # Disconnecting
                ui.variable_to_plot_comboBox.setUpdatesEnabled(False)  # Disconnect the signal first, then clear
                ui.replace_columnSelection_comboBox.setUpdatesEnabled(False)  # Disconnect the signal first, then clear
                ui.filter_columnSelection_comboBox.setUpdatesEnabled(False)  # Disconnect the signal first, then clear
                # Clearing
                ui.variable_to_plot_comboBox.clear()  # Delete all values from comboBox, then re-connect the signal
                ui.replace_columnSelection_comboBox.clear()  # Delete all values from comboBox, then re-connect the signal
                ui.filter_columnSelection_comboBox.clear()  # Delete all values from comboBox, then re-connect the signal

                # Re-connecting (This will avoid triggering unwanted signals now!)
                ui.variable_to_plot_comboBox.setUpdatesEnabled(True)
                ui.replace_columnSelection_comboBox.setUpdatesEnabled(True)
                ui.filter_columnSelection_comboBox.setUpdatesEnabled(True)

            self.signals.update_train_test_shape_label.emit()

            if ui.available_columns_listWidget.count() != 0:
                ui.available_columns_listWidget.clear()

            if ui.input_columns_listWidget.count() != 0:
                ui.input_columns_listWidget.clear()

            if ui.output_columns_listWidget.count() != 0:
                ui.output_columns_listWidget.clear()

            ui.preprocess_sequence_listWidget.clear()
            ui.clas_output_colum_comboBox.clear()
            ui.train_model_pushButton.setDisabled(True)

            # Filling the comboBoxes
            for each_column in ml_model.dataset.columns:
                ui.variable_to_plot_comboBox.addItem(each_column)  # Fill variable_to_plot_comboBox from the Visualise Tab
                ui.replace_columnSelection_comboBox.addItem(each_column)  # from the Pre-process Tab
                ui.filter_columnSelection_comboBox.addItem(each_column)  # from the Pre-process Tab
                ui.available_columns_listWidget.addItem(each_column)

                if ml_model.column_types_pd_series[each_column].kind in 'iO':  # i = Integer , O = Object
                    ui.clas_output_colum_comboBox.addItem(each_column)
                    # Setting the index so the widget does not show empty
                    ui.clas_output_colum_comboBox.setCurrentIndex(0)

            # Setting the index so the widget does not show empty
            if len(ml_model.dataset.columns) > 0 :
                ui.variable_to_plot_comboBox.setCurrentIndex(0)
                ui.replace_columnSelection_comboBox.setCurrentIndex(0)
                ui.filter_columnSelection_comboBox.setCurrentIndex(0)

            return

        elif return_code == 'invalid_file_extension':
            self.signals.display_message.emit(QtWidgets.QMessageBox.Warning, 'Error',
                                              'Invalid Input format. \nTry Excel or .csv format',
                                              'Error')

        elif return_code == 'exception_in_the_file':
            self.signals.display_message.emit(QtWidgets.QMessageBox.Warning, 'Error', 'Invalid Input File', 'Error')

        self.signals.stop_spinner.emit(self.ui.dataset_tableWidget, 'stop_spinner' , [])
        self.signals.stop_spinner.emit(self.ui.pre_process_dataset_tableWidget, 'stop_spinner' , [])

class Pre_Process_Dataset_WorkerSignals(QtCore.QObject):
    update_pre_process_tableWidget = QtCore.pyqtSignal(object, object)
    display_message = QtCore.pyqtSignal(object, object, object, object)
class Pre_Process_Dataset_Thread(QtCore.QRunnable):
    """
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    """

    def __init__(self, *args, **kwargs):
        super(Pre_Process_Dataset_Thread, self).__init__()

        self.ui = args[0]
        self.ml_model = args[1]

        self.args = args
        self.kwargs = kwargs
        self.signals = Pre_Process_Dataset_WorkerSignals()

    @QtCore.pyqtSlot()
    def run(self):

        ui = self.ui
        ml_model = self.ml_model

        old_pre_processed_dataset = ml_model.pre_processed_dataset.copy()
        ml_model.pre_processed_dataset = ml_model.dataset.copy()

        listwidget = ui.preprocess_sequence_listWidget
        for i in range(listwidget.count()):
            # Getting the data embedded in each item from the listWidget
            item_data = listwidget.item(i).data(QtCore.Qt.UserRole)

            if item_data['pre_processing_action'] == 'rm_duplicate_rows':
                ml_model.remove_duplicate_rows()
            elif item_data['pre_processing_action'] == 'rm_constant_var':
                ml_model.remove_constant_variables()
            elif item_data['pre_processing_action'] == 'apply_num_scaling':
                ml_model.scale_numeric_values()
            elif item_data['pre_processing_action'] == 'rm_outliers':
                ml_model.remove_outliers(item_data['cut_off'])
            elif item_data['pre_processing_action'] == 'replace_values':
                target_variable = item_data['variable']
                new_value = item_data['new_values']
                old_values = item_data['old_values']
                ml_model.replace_values(target_variable,new_value,old_values)
            elif item_data['pre_processing_action'] == 'apply_filtering':
                filtering_variable = item_data['variable']
                filtering_value = item_data['filtering_value']
                filtering_operator = item_data['filtering_operator']
                ml_model.filter_out_values(filtering_variable,filtering_value,filtering_operator)

        table_widget = ui.pre_process_dataset_tableWidget
        filling_dataframe = ml_model.pre_processed_dataset

        if filling_dataframe.empty:
            self.signals.display_message.emit(QtWidgets.QMessageBox.Critical, 'Invalid Pre-processing',
                            'This pre-processing rule is too restrictive and would return an empty dataset', 'Error')
            # Undo the processing
            self.ml_model.pre_processed_dataset = old_pre_processed_dataset
            filling_dataframe = old_pre_processed_dataset
            # Drop the inavlid rule
            listwidget.takeItem(listwidget.count() - 1)

        self.signals.update_pre_process_tableWidget.emit(table_widget,filling_dataframe)


class Plotting_in_MplWidget_Thread(QtCore.QRunnable):
    """
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    """

    def __init__(self, *args, **kwargs):
        super(Plotting_in_MplWidget_Thread, self).__init__()

        self.ui = args[0]
        self.target_widget = args[1]
        self.content = args[2]

        self.args = args
        self.kwargs = kwargs

    @QtCore.pyqtSlot()
    def run(self):

        ui = self.ui
        target_widget = self.target_widget
        content = self.content

        target_widget.canvas.axes.clear()
        # If the figure has multiple axes - This happens when the Confusion Matrix is plotted
        if len(target_widget.canvas.figure.axes) > 1:
            target_widget.canvas.figure.clf()  # Clear the figure
            target_widget.canvas.axes = target_widget.canvas.figure.add_subplot()  # Add an Axes to the figure
        target_widget.canvas.axes.axis('on')

        if target_widget.objectName() == 'dataVisualisePlot_widget':
            if content['is_categorical']:
                content['data'].value_counts().plot(kind='bar', ax=target_widget.canvas.axes, grid=False,
                                                    title='Sample Count')
                target_widget.canvas.axes.tick_params(axis='x', labelrotation=60)
            elif ui.plot_radioButton.isChecked() and ui.plot_radioButton.isEnabled():
                content['data'].plot(ax=target_widget.canvas.axes, grid=False, title='Linear Plot')
            elif ui.boxplot_radioButton.isChecked() and ui.boxplot_radioButton.isEnabled():
                pd.DataFrame(content['data']).boxplot(ax=target_widget.canvas.axes, grid=False)
                target_widget.canvas.axes.axes.set_title('Boxplot')
            elif ui.histogram_radioButton.isChecked() and ui.histogram_radioButton.isEnabled():
                pd.DataFrame(content['data']).hist(ax=target_widget.canvas.axes, grid=False)
                target_widget.canvas.axes.axes.set_title('Histogram')
            else:
                target_widget.canvas.axes.axis('off')

        elif target_widget.objectName() == 'model_train_widget':
            if content['is_regression']:
                if len(content['output_variables']) == 1:
                    pd.DataFrame(content['data']).hist(ax=target_widget.canvas.axes, grid=False)
                    target_widget.canvas.axes.axes.set_title('Histogram of Percentage Errors')
                    target_widget.canvas.axes.axes.set_ylabel('Number of Occurrences')
                    target_widget.canvas.axes.axes.set_xlabel('Percentage Error (%)')
                else:
                    target_widget.canvas.axes.bar(content['data']['labels'], content['data']['values'])
                    target_widget.canvas.axes.set_xticklabels(content['data']['labels'], rotation='vertical')
                    target_widget.canvas.axes.axes.set_ylabel('Mean Percentage Error (%)')
                    target_widget.canvas.axes.axes.set_title('Individual Mean Percentage Error (%)')
            else:
                # Print the number labels in the cells if n_of_classes < 10, otherwise just colours
                is_annot = False if len(content['data']) > 10 else True
                _ = sns.heatmap(content['data'], cmap="YlGnBu", annot=is_annot, ax=target_widget.canvas.axes)
                target_widget.canvas.axes.tick_params(axis='y', labelrotation=0)
                target_widget.canvas.axes.tick_params(axis='x', labelrotation=60)
                target_widget.canvas.axes.set_xlabel('Actual')
                target_widget.canvas.axes.set_ylabel('Predicted')
                target_widget.canvas.axes.axes.set_title('Confusion Matrix')

        target_widget.canvas.draw()
