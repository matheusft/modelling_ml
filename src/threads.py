from view import QtCore, QtWidgets

class Train_Model_WorkerSignals(QtCore.QObject):
    finished = QtCore.pyqtSignal(object, object, object)


class Train_Model_Thread(QtCore.QRunnable):
    """
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

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
        self.signals.finished.emit(self.ui, result, self.model_parameters)


class Load_Dataset_WorkerSignals(QtCore.QObject):

    update_train_test_shape_label = QtCore.pyqtSignal(object, object)
    display_message = QtCore.pyqtSignal(object, object, object, object)
    update_table_widget = QtCore.pyqtSignal(object, object, object, object)


class Load_Dataset_Thread(QtCore.QRunnable):
    """
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
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
        if self.file_path == 'populate_tablewidget_only':
            self.populate_tablewidget_with_dataframe(self, self.ui, self.ui.pre_process_dataset_tableWidget,
                                                     self.ml_model.pre_processed_dataset)
        else:
            # Load Dataset
            return_code = self.ml_model.read_dataset(self.file_path)
            self.load_dataset(self.ui,self.ml_model, return_code)

    def load_dataset(self,ui,ml_model,return_code):

        if return_code == 'sucess':

            self.populate_tablewidget_with_dataframe(ui, ui.dataset_tableWidget, ml_model.dataset)
            self.populate_tablewidget_with_dataframe(ui, ui.pre_process_dataset_tableWidget, ml_model.dataset)

            widgets_to_enable = [ui.plot_radioButton, ui.boxplot_radioButton, ui.histogram_radioButton,
                                 ui.remove_duplicates_pushButton, ui.remove_constant_variables_pushButton,
                                 ui.numeric_scaling_pushButton, ui.remove_outliers_pushButton,
                                 ui.addrule_filter_value_pushButton, ui.addrule_replace_value_pushButton,
                                 ui.add_input_columns_pushButton, ui.add_output_columns_pushButton]

            [item.setEnabled(True) for item in widgets_to_enable]

            # Here we update the columnSelection_comboBox
            if ui.columnSelection_comboBox.count() > 0:  # If the comboBox is not empty
                # Disconnecting
                ui.columnSelection_comboBox.setUpdatesEnabled(False)  # Disconnect the signal first, then clear
                ui.replace_columnSelection_comboBox.setUpdatesEnabled(False)  # Disconnect the signal first, then clear
                ui.filter_columnSelection_comboBox.setUpdatesEnabled(False)  # Disconnect the signal first, then clear
                # Clearing
                ui.columnSelection_comboBox.clear()  # Delete all values from comboBox, then re-connect the signal
                ui.replace_columnSelection_comboBox.clear()  # Delete all values from comboBox, then re-connect the signal
                ui.filter_columnSelection_comboBox.clear()  # Delete all values from comboBox, then re-connect the signal
                # Re-connecting
                ui.columnSelection_comboBox.setUpdatesEnabled(True)  # Disconnect the signal first, then clear
                ui.replace_columnSelection_comboBox.setUpdatesEnabled(True)  # Disconnect the signal first, then clear
                ui.filter_columnSelection_comboBox.setUpdatesEnabled(True)  # Disconnect the signal first, then clear

            self.signals.update_train_test_shape_label.emit(ui, ml_model)

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
                ui.columnSelection_comboBox.addItem(each_column)  # Fill columnSelection_comboBox from the Visualise Tab
                ui.replace_columnSelection_comboBox.addItem(each_column)  # from the Pre-process Tab
                ui.filter_columnSelection_comboBox.addItem(each_column)  # from the Pre-process Tab
                ui.available_columns_listWidget.addItem(each_column)

                if ml_model.column_types_pd_series[each_column].kind in 'iO':  # i = Integer , O = Object
                    ui.clas_output_colum_comboBox.addItem(each_column)
                    # Setting the index so the widget does not show empty
                    ui.clas_output_colum_comboBox.setCurrentIndex(0)

            # Setting the index so the widget does not show empty
            if len(ml_model.dataset.columns) > 0 :
                ui.columnSelection_comboBox.setCurrentIndex(0)
                ui.replace_columnSelection_comboBox.setCurrentIndex(0)
                ui.filter_columnSelection_comboBox.setCurrentIndex(0)

        elif return_code == 'invalid_file_extension':
            self.signals.display_message.emit(QtWidgets.QMessageBox.Warning, 'Error',
                                              'Invalid Input format. \nTry Excel or .csv format',
                                              'Error')

        elif return_code == 'exception_in_the_file':
            self.signals.display_message.emit(QtWidgets.QMessageBox.Warning, 'Error', 'Invalid Input File', 'Error')


    def populate_tablewidget_with_dataframe(self,ui,table_widget, filling_dataframe):

        table_widget.clear()

        # Fill dataset_tableWidget from the Dataset Load Tab with the head of the dataset
        number_of_rows_to_display = 20
        table_widget.setRowCount(len(filling_dataframe.head(number_of_rows_to_display)))
        table_widget.setColumnCount(len(filling_dataframe.columns))


        # Adding the labels at the top of the Table
        data = {'header_labels': filling_dataframe.columns}
        #Updating the table_widget in the GUI Thread
        # table_widget.setHorizonßtalHeaderLabels(filling_dataframe.columns)
        self.signals.update_table_widget.emit(ui,table_widget, 'update_header', data)

        # Filling the Table with the dataset
        for i in range(table_widget.rowCount()):
            for j in range(table_widget.columnCount()):
                dataset_value = filling_dataframe.iloc[i, j]  # Get the value from the dataset
                dataset_value_converted = dataset_value if (type(dataset_value) is str) else '{:}'.format(dataset_value)
                qt_item = QtWidgets.QTableWidgetItem(dataset_value_converted)  # Creates an qt item
                qt_item.setTextAlignment(QtCore.Qt.AlignHCenter)  # Aligns the item in the horizontal center
                # Updating the table_widget in the GUI Thread
                # table_widget.setItem(i, j, qt_item)
                data = {'i': i, 'j': j, 'qt_item': qt_item}
                self.signals.update_table_widget.emit(ui,table_widget, 'fill_table', data)
        # Stopping the loading spinner
        self.signals.update_table_widget.emit(ui, table_widget, 'stop_spinner', data)
