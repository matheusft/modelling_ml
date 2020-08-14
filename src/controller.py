import os
import random
import sys
import threads
import seaborn as sns
from os.path import join, abspath
from docutils.nodes import address
from view import QtCore, QtWidgets
import personalised_widgets

# Set seaborn aesthetic parameters
sns.set()


class ViewController:
    def __init__(self, ui, ml_model):
        self.ui = ui
        self.ml_model = ml_model

        self.configure_gui()

    def configure_gui(self):

        ui = self.ui
        _translate = QtCore.QCoreApplication.translate

        # Seting up the thread pool for multi-threading
        ui.threadpool = QtCore.QThreadPool()

        # Populating the example_dataset_comboBox
        current_working_directory = os.getcwd()
        datasets_path = current_working_directory + '/../data/'
        list_of_datasets = os.listdir(datasets_path)
        for dataset in list_of_datasets:
            if not dataset.startswith('.'):
                # Each Item receives the dataset name as text and the dataset path as data
                ui.example_dataset_comboBox.addItem(dataset.split('.')[0], datasets_path + dataset)

        self.connect_signals()

        ui.nn_classification_radioButton.click()
        ui.nn_regression_radioButton.click()
        ui.nn_regression_radioButton.click()

        ui.tabs_widget.setCurrentIndex(0)
        ui.pre_process_tabWidget.setCurrentIndex(0)
        ui.output_selection_stackedWidget.setCurrentIndex(0)

        widgets_to_disable = [ui.plot_radioButton, ui.boxplot_radioButton, ui.histogram_radioButton,
                              ui.remove_duplicates_pushButton, ui.remove_constant_variables_pushButton,
                              ui.numeric_scaling_pushButton, ui.remove_outliers_pushButton,
                              ui.addrule_filter_value_pushButton, ui.addrule_replace_value_pushButton,
                              ui.addrule_filter_value_pushButton, ui.export_model_pushButton,
                              ui.add_input_columns_pushButton, ui.add_output_columns_pushButton,
                              ui.train_model_pushButton]

        for widget in widgets_to_disable:
            widget.setEnabled(False)

        # Todo : Check whether all number_of_neuros of reg_nn_layers_tableWidget are int greater than 0

        # TODO : Disable all button and functions while Dataset is not chosen -
        # Todo : Check things that break when the dataset is not loaded yet

    def connect_signals(self):
        ui = self.ui
        ml_model = self.ml_model

        # Connecting load_file_pushButton - Dataset Load Tab
        ui.load_file_pushButton.clicked.connect(lambda: self.trigger_loading_dataset_thread(ui.load_file_pushButton))
        ui.example_dataset_comboBox.currentIndexChanged.connect(
            lambda: self.trigger_loading_dataset_thread(ui.example_dataset_comboBox))

        # Connecting columnSelection_comboBox - Visualise Tab
        ui.variable_to_plot_comboBox.currentIndexChanged.connect(lambda: self.update_visualisation_options())

        # Connecting radio_button_change - Visualise Tab
        ui.boxplot_radioButton.clicked.connect(lambda: self.update_visualisation_widgets())
        ui.plot_radioButton.clicked.connect(lambda: self.update_visualisation_widgets())
        ui.histogram_radioButton.clicked.connect(lambda: self.update_visualisation_widgets())

        ui.remove_duplicates_pushButton.clicked.connect(lambda: self.add_rm_duplicate_rows_rule())
        ui.remove_constant_variables_pushButton.clicked.connect(lambda: self.add_rm_constant_var_rule())
        ui.numeric_scaling_pushButton.clicked.connect(lambda: self.add_num_scaling_rule())
        ui.remove_outliers_pushButton.clicked.connect(lambda: self.add_rm_outliers_rule())
        ui.addrule_filter_value_pushButton.clicked.connect(lambda: self.generate_filtering_rule())
        ui.addrule_replace_value_pushButton.clicked.connect(lambda: self.generate_replacing_rule())

        ui.preprocess_sequence_listWidget.model().rowsInserted.connect(lambda: self.update_pre_process())
        ui.preprocess_sequence_listWidget.model().rowsRemoved.connect(lambda: self.update_pre_process())

        ui.outliers_treshold_horizontalSlider.valueChanged.connect(
            lambda: self.update_label_from_slider_change(ui.outliers_treshold_horizontalSlider.value(),
                                                         ui.outliers_treshold_label))

        ui.replace_columnSelection_comboBox.currentIndexChanged.connect(
            lambda: self.update_preprocess_replace())  # Update the pre_process_replacing_stackedWidget according to the replace_columnSelection_comboBox
        ui.filter_columnSelection_comboBox.currentIndexChanged.connect(
            lambda: self.update_preprocess_filtering())  # Update the pre_process_replacing_stackedWidget according to the replace_columnSelection_comboBox

        ui.add_input_columns_pushButton.clicked.connect(
            lambda: self.update_input_output_columns(ui.input_columns_listWidget))
        ui.add_output_columns_pushButton.clicked.connect(
            lambda: slef.update_input_output_columns(ui.output_columns_listWidget))

        ui.remove_input_columns_pushButton.clicked.connect(
            lambda: self.remove_item_from_listwidget(ui.input_columns_listWidget))
        ui.remove_output_columns_pushButton.clicked.connect(
            lambda: self.remove_item_from_listwidget(ui.output_columns_listWidget))

        ui.clear_input_columns_pushButton.clicked.connect(lambda: self.clear_listwidget(ui.input_columns_listWidget))
        ui.clear_output_columns_pushButton.clicked.connect(lambda: self.clear_listwidget(ui.output_columns_listWidget))

        model_selection_radio_buttons = [ui.regression_selection_radioButton, ui.classification_selection_radioButton,
                                         ui.gradientboosting_classification_radioButton,
                                         ui.knn_classification_radioButton,
                                         ui.nn_classification_radioButton, ui.randomforest_classification_radioButton,
                                         ui.svm_classification_radioButton, ui.svm_regression_radioButton,
                                         ui.randomforest_regression_radioButton, ui.nn_regression_radioButton,
                                         ui.gradientboosting_regression_radioButton]

        for model_option in model_selection_radio_buttons:
            model_option.clicked.connect(lambda: self.model_selection_tab_events())

        ui.reg_nn_layers_horizontalSlider.valueChanged.connect(
            lambda: self.update_label_from_slider_change(ui.reg_nn_layers_horizontalSlider.value(),
                                                         ui.reg_nn_layers_label))
        ui.clas_nn_val_percentage_horizontalSlider.valueChanged.connect(
            lambda: self.update_label_from_slider_change(ui.clas_nn_val_percentage_horizontalSlider.value(),
                                                         ui.reg_nn_val_percent_label))
        ui.reg_nn_max_iter_horizontalSlider.valueChanged.connect(
            lambda: self.update_label_from_slider_change(ui.reg_nn_max_iter_horizontalSlider.value(),
                                                         ui.reg_nn_max_iter_label))
        ui.reg_nn_alpha_horizontalSlider.valueChanged.connect(
            lambda: self.update_label_from_slider_change(ui.reg_nn_alpha_horizontalSlider.value(),
                                                         ui.reg_nn_alpha_label))
        ui.clas_nn_layers_horizontalSlider.valueChanged.connect(
            lambda: self.update_label_from_slider_change(ui.clas_nn_layers_horizontalSlider.value(),
                                                         ui.clas_nn_layers_label))
        ui.reg_nn_val_percentage_horizontalSlider.valueChanged.connect(
            lambda: self.update_label_from_slider_change(ui.reg_nn_val_percentage_horizontalSlider.value(),
                                                         ui.clas_nn_val_percent_label))
        ui.clas_nn_max_iter_horizontalSlider.valueChanged.connect(
            lambda: self.update_label_from_slider_change(ui.clas_nn_max_iter_horizontalSlider.value(),
                                                         ui.clas_nn_max_iter_label))
        ui.clas_nn_alpha_horizontalSlider.valueChanged.connect(
            lambda: self.update_label_from_slider_change(ui.clas_nn_alpha_horizontalSlider.value(),
                                                         ui.clas_nn_alpha_label))
        ui.train_percentage_horizontalSlider.valueChanged.connect(
            lambda: self.update_label_from_slider_change(ui.train_percentage_horizontalSlider.value(),
                                                         ui.train_percentage_label))
        ui.test_percentage_horizontalSlider.valueChanged.connect(
            lambda: self.update_label_from_slider_change(ui.test_percentage_horizontalSlider.value(),
                                                         ui.test_percentage_label))

        ui.train_model_pushButton.clicked.connect(lambda: self.trigger_train_model_thread())

    def trigger_loading_dataset_thread(self, data_source):
        ui = self.ui
        ml_model = self.ml_model

        ui.dataset_tableWidget.spinner.start()
        ui.pre_process_dataset_tableWidget.spinner.start()

        if data_source.objectName() == 'load_file_pushButton':
            fileDlg = QtWidgets.QFileDialog()
            file_address = fileDlg.getOpenFileName()[0]
            if file_address == '' or file_address == None:
                ui.dataset_tableWidget.spinner.stop()
                ui.pre_process_dataset_tableWidget.spinner.stop()
                return
        elif data_source.objectName() == 'example_dataset_comboBox':
            selected_index = ui.example_dataset_comboBox.currentIndex()
            file_address = ui.example_dataset_comboBox.itemData(selected_index)

        ui.load_file_pushButton.setDisabled(True)
        ui.example_dataset_comboBox.setDisabled(True)

        # Creating an object worker
        worker = threads.Load_Dataset_Thread(ui, ml_model, file_address)

        # Connecting the signals from the created worker to its functions
        worker.signals.display_message.connect(display_message)
        worker.signals.update_train_test_shape_label.connect(self.update_train_test_shape_label)
        worker.signals.populate_tablewidget_with_dataframe.connect(
            self.generate_qt_items_to_fill_tablewidget)

        # Starts the thread
        ui.threadpool.start(worker)  # Todo: Uncomment this when finished debugging
        # result = ml_model.read_dataset(file_address)

        # TODO: add checkbox Dataset with Column name or not
        # Todo: Check what needs to be reset/cleared when a new dataset is loaded

    def update_train_test_shape_label(self):
        ui = self.ui
        ml_model = self.ml_model

        dataset_shape = ml_model.pre_processed_dataset.shape

        number_of_rows_train = round(dataset_shape[0] * ui.train_percentage_horizontalSlider.value() / 100)
        number_of_columns_train = ui.input_columns_listWidget.count()

        number_of_rows_test = round(dataset_shape[0] * ui.test_percentage_horizontalSlider.value() / 100)
        number_of_columns_test = ui.input_columns_listWidget.count()

        ui.train_dataset_shape_label.setText('{} x {}'.format(number_of_rows_train, number_of_columns_train))
        ui.test_dataset_shape_label.setText('{} x {}'.format(number_of_rows_test, number_of_columns_test))

    def add_pre_process_rule(self):
        ui = self.ui
        ml_model = self.ml_model

    def clear_listwidget(self, target_listwidget):
        ui = self.ui
        ml_model = self.ml_model

        if target_listwidget == ui.preprocess_sequence_listWidget:
            ui.preprocess_sequence_listWidget.clear()

        elif target_listwidget == ui.input_columns_listWidget:

            for _ in range(ui.input_columns_listWidget.count()):
                item = ui.input_columns_listWidget.takeItem(0)
                ui.available_columns_listWidget.addItem(item)

                # Adding the variables back to the clas_output_colum_comboBox
                if item.text() in ml_model.categorical_variables:
                    ui.clas_output_colum_comboBox.addItem(item.text())

            ui.train_model_pushButton.setDisabled(True)
            update_train_test_shape_label(ui, ml_model)

        elif target_listwidget == ui.output_columns_listWidget:

            ui.train_model_pushButton.setDisabled(True)

            for _ in range(ui.output_columns_listWidget.count()):
                item = ui.output_columns_listWidget.takeItem(0)
                ui.available_columns_listWidget.addItem(item)

    def update_train_model_button_status(self, is_regression):
        ui = self.ui
        if is_regression:
            if ui.output_columns_listWidget.count() > 0 and ui.input_columns_listWidget.count() > 0:
                ui.train_model_pushButton.setDisabled(False)
            else:
                ui.train_model_pushButton.setDisabled(True)
        else:
            if ui.input_columns_listWidget.count() > 0 and ui.clas_output_colum_comboBox.count() > 0:
                ui.train_model_pushButton.setDisabled(False)
            else:
                ui.train_model_pushButton.setDisabled(True)

    def create_listwidgetitem(self, text, data):
        string_to_add = text
        my_qlist_item = QtWidgets.QListWidgetItem()  # Create a QListWidgetItem
        my_qlist_item.setText(string_to_add)  # Add the text to be displayed in the listWidget
        my_qlist_item.setData(QtCore.Qt.UserRole, data)  # Set data to the item
        return my_qlist_item

    def add_rm_duplicate_rows_rule(self):
        ui = self.ui

        rule_text = 'Remove Duplicate Rows'
        rule_data = {'pre_processing_action': 'rm_duplicate_rows'}
        item_to_add = self.create_listwidgetitem(rule_text, rule_data)
        self.add_pre_processing_rule_to_listWidget(item_to_add, ui.preprocess_sequence_listWidget)

    def add_rm_constant_var_rule(self):
        ui = self.ui

        rule_text = 'Remove Constant Variables (Columns)'
        rule_data = {'pre_processing_action': 'rm_constant_var'}
        item_to_add = self.create_listwidgetitem(rule_text, rule_data)
        self.add_pre_processing_rule_to_listWidget(item_to_add, ui.preprocess_sequence_listWidget)

    def add_num_scaling_rule(self):
        ui = self.ui

        rule_text = 'Apply Numeric Scaling'
        rule_data = {'pre_processing_action': 'apply_num_scaling'}
        item_to_add = self.create_listwidgetitem(rule_text, rule_data)
        self.add_pre_processing_rule_to_listWidget(item_to_add, ui.preprocess_sequence_listWidget)

    def add_rm_outliers_rule(self):
        ui = self.ui

        cut_off = ui.outliers_treshold_horizontalSlider.value() / 10
        text = 'Remove Outliers, Cut-off = {}σ'.format(cut_off)
        data = {'pre_processing_action': 'rm_outliers', 'cut_off': cut_off}
        item_to_add = self.create_listwidgetitem(text, data)
        self.add_pre_processing_rule_to_listWidget(item_to_add, ui.preprocess_sequence_listWidget)

    def add_numeric_filtering_rule(self):
        ui = self.ui
        filtering_value = ui.filtering_dataset_value_lineEdit.text()
        filtering_variable = ui.filter_columnSelection_comboBox.currentText()
        filtering_operator = ui.filter_operator_comboBox.currentText()
        if filtering_value != '':  # If input is not empty
            try:
                float(filtering_value)  # Check whether it is a valid input
            except:
                display_message(QtWidgets.QMessageBox.Critical, 'Invalid Input',
                                'Type a valid numeric input for column {}'.format(filtering_operator), 'Error')
                return
            rule_text = 'Exclude values from {} {} {}'.format(filtering_variable, filtering_operator, filtering_value)
            rule_data = {'pre_processing_action': 'apply_filtering', 'variable': filtering_variable, 'is_numeric': True,
                         'filtering_operator': filtering_operator, 'filtering_value': filtering_value}
            item_to_add = self.create_listwidgetitem(rule_text, rule_data)
            self.add_pre_processing_rule_to_listWidget(item_to_add, ui.preprocess_sequence_listWidget)
        else:  # Display a message if the inputs are empty and a rule is added
            display_message(QtWidgets.QMessageBox.Information, 'Empty Input',
                            'Type a valid rule', 'Error')

    def add_categorical_filtering_rule(self):
        ui = self.ui
        filtering_value = ui.filtering_dataset_value_comboBox.currentText()
        filtering_variable = ui.filter_columnSelection_comboBox.currentText()
        filtering_operator = ui.filter_operator_comboBox.currentText()
        rule_text = 'Exclude values from {} {} {}'.format(filtering_variable, filtering_operator, filtering_value)
        rule_data = {'pre_processing_action': 'apply_filtering', 'variable': filtering_variable, 'is_numeric': False,
                     'filtering_operator': filtering_operator, 'filtering_value': filtering_value}
        item_to_add = self.create_listwidgetitem(rule_text, rule_data)
        self.add_pre_processing_rule_to_listWidget(item_to_add, ui.preprocess_sequence_listWidget)

    def generate_filtering_rule(self):
        ui = self.ui
        ml_model = self.ml_model

        current_variable_name = ui.filter_columnSelection_comboBox.currentText()
        variable_type = ml_model.column_types_pd_series[current_variable_name].kind
        is_numeric_variable = variable_type in 'iuf'  # iuf = i int (signed), u unsigned int, f float

        if is_numeric_variable:  # If numeric
            self.add_numeric_filtering_rule()

        else:  # If not numeric
            self.add_categorical_filtering_rule()

    def add_numeric_replacing_rule(self):
        ui = self.ui
        replacing_variable = ui.replace_columnSelection_comboBox.currentText()
        old_values = ui.replaced_value_lineEdit.text()
        new_values = ui.replacing_value_lineEdit.text()

        if old_values != '' and new_values != '':  # If inputs are not empty
            try:
                float(new_values), float(old_values)  # Check whether it is a valid input
            except:
                display_message(QtWidgets.QMessageBox.Critical, 'Invalid Input',  # Display error message and return
                                'Type a valid numeric input for column {}'.format(replacing_variable), 'Error')
                return
            rule_text = 'Replace {} in {} with {}'.format(old_values, replacing_variable, new_values)
            rule_data = {'pre_processing_action': 'replace_values', 'variable': replacing_variable, 'is_numeric': True,
                         'old_values': old_values, 'new_values': new_values}
            item_to_add = self.create_listwidgetitem(rule_text, rule_data)
            self.add_pre_processing_rule_to_listWidget(item_to_add, ui.preprocess_sequence_listWidget)
        else:  # Display a message if the inputs are empty and a rule is added
            display_message(QtWidgets.QMessageBox.Information, 'Empty Input',
                            'Type a valid rule', 'Error')

    def add_categorical_replacing_rule(self):
        ui = self.ui
        replacing_variable = ui.replace_columnSelection_comboBox.currentText()
        old_values = ui.replaced_value_comboBox.currentText()
        new_values = ui.replacing_value_lineEdit.text()

        if new_values != '':  # If inputs are not empty
            rule_text = 'Replace {} in {} with {}'.format(old_values, replacing_variable, new_values)
            rule_data = {'pre_processing_action': 'replace_values', 'variable': replacing_variable, 'is_numeric': False,
                         'old_values': old_values, 'new_values': new_values}
            item_to_add = self.create_listwidgetitem(rule_text, rule_data)
            self.add_pre_processing_rule_to_listWidget(item_to_add, ui.preprocess_sequence_listWidget)
        else:  # Display a message if the inputs are empty and a rule is added
            display_message(QtWidgets.QMessageBox.Information, 'Empty Input',
                            'Type a valid rule', 'Error')

    def generate_replacing_rule(self):
        ui = self.ui

        is_numeric_variable = ui.pre_process_replacing_stackedWidget.currentIndex() == 0

        if is_numeric_variable:  # If numeric
            self.add_numeric_replacing_rule()

        else:  # If not numeric
            self.add_categorical_replacing_rule()

    def model_selection_tab_events(self):
        ui = self.ui

        is_regression = ui.regression_selection_radioButton.isChecked()

        if is_regression:
            ui.regression_and_classification_stackedWidget.setCurrentIndex(0)  # Change to Regression Tab
            ui.train_metrics_stackedWidget.setCurrentIndex(0)  # Change to Regression Tab
            ui.output_selection_stackedWidget.setCurrentIndex(0)
            self.update_train_model_button_status(is_regression)

            if ui.nn_regression_radioButton.isChecked():
                ui.regression_parameters_stackedWidget.setCurrentIndex(0)

            elif ui.svm_regression_radioButton.isChecked():
                ui.regression_parameters_stackedWidget.setCurrentIndex(1)

            elif ui.randomforest_regression_radioButton.isChecked():
                ui.regression_parameters_stackedWidget.setCurrentIndex(2)

            elif ui.gradientboosting_regression_radioButton.isChecked():
                ui.regression_parameters_stackedWidget.setCurrentIndex(3)

        elif ui.classification_selection_radioButton.isChecked():
            ui.regression_and_classification_stackedWidget.setCurrentIndex(1)  # Change to Classification Tab
            ui.train_metrics_stackedWidget.setCurrentIndex(1)  # Change to Regression Tab
            ui.output_selection_stackedWidget.setCurrentIndex(1)
            self.update_train_model_button_status(is_regression)

            if ui.nn_classification_radioButton.isChecked():
                ui.classification_parameters_stackedWidget.setCurrentIndex(0)

            elif ui.svm_classification_radioButton.isChecked():
                ui.classification_parameters_stackedWidget.setCurrentIndex(1)

            elif ui.randomforest_classification_radioButton.isChecked():
                ui.classification_parameters_stackedWidget.setCurrentIndex(2)

            elif ui.gradientboosting_classification_radioButton.isChecked():
                ui.classification_parameters_stackedWidget.setCurrentIndex(3)

            elif ui.knn_classification_radioButton.isChecked():
                ui.classification_parameters_stackedWidget.setCurrentIndex(4)

    def add_pre_processing_rule_to_listWidget(self, item, listWidget):

        number_of_rules = listWidget.count()
        item.setText('{} - '.format(number_of_rules + 1) + item.text())
        listWidget.addItem(item)

        # Todo: call update pre-process

    def generate_qt_items_to_fill_tablewidget(self, table_widget, filling_dataframe):

        table_widget.clear()

        # Fill dataset_tableWidget from the Dataset Load Tab with the head of the dataset
        number_of_rows_to_display = 20
        table_widget.setRowCount(len(filling_dataframe.head(number_of_rows_to_display)))
        table_widget.setColumnCount(len(filling_dataframe.columns))

        # Adding the labels at the top of the Table
        data = {'header_labels': filling_dataframe.columns}
        # Updating the table_widget in the GUI Thread
        self.update_table_widget(table_widget, 'update_header', data)

        # Filling the Table with the dataset
        for i in range(table_widget.rowCount()):
            for j in range(table_widget.columnCount()):
                dataset_value = filling_dataframe.iloc[i, j]  # Get the value from the dataset
                dataset_value_converted = dataset_value if (type(dataset_value) is str) else '{:}'.format(dataset_value)
                qt_item = QtWidgets.QTableWidgetItem(dataset_value_converted)  # Creates an qt item
                qt_item.setTextAlignment(QtCore.Qt.AlignHCenter)  # Aligns the item in the horizontal center
                # Updating the table_widget in the GUI Thread
                data = {'i': i, 'j': j, 'qt_item': qt_item}
                self.update_table_widget(table_widget, 'fill_table', data)
        # Stopping the loading spinner
        self.update_table_widget(table_widget, 'stop_spinner', data)

    def update_pre_process(self):

        ui = self.ui
        ml_model = self.ml_model

        ui.pre_process_dataset_tableWidget.spinner.start()

        worker = threads.Pre_Process_Dataset_Thread(ui, ml_model)

        worker.signals.update_pre_process_tableWidget.connect(self.generate_qt_items_to_fill_tablewidget)
        worker.signals.display_message.connect(display_message)

        ui.threadpool.start(worker)


    def update_visualisation_widgets(self):
        ui = self.ui
        ml_model = self.ml_model

        selected_column = ui.variable_to_plot_comboBox.currentText()  # Get the selected value in the comboBox

        ui.columnSummary_textBrowser.clear()
        description = ml_model.dataset[selected_column].describe()
        for i in range(len(description)):
            ui.columnSummary_textBrowser.append('{} = {}'.format(description.keys()[i].title(), description.values[i]))

        is_categorical = ml_model.column_types_pd_series[
                             selected_column].kind not in 'iuf'  # iuf = i int (signed), u unsigned int, f float

        self.trigger_plot_matplotlib_to_qt_widget_thread(target_widget=ui.dataVisualisePlot_widget,
                                                         content={'data': ml_model.dataset[selected_column],
                                                                  'is_categorical': is_categorical})

    def update_visualisation_options(self):
        ui = self.ui
        ml_model = self.ml_model

        """Update the available visualisation options for each column in the radio buttons.
    
        Args:
            :param ml_model: The Machine Learning Model.
            :param ui:  The ui to be updated
    
        """
        selected_column = ui.variable_to_plot_comboBox.currentText()  # Get the selected value in the comboBox
        # Create a list of all radioButton objects
        radio_buttons_list = [ui.plot_radioButton, ui.boxplot_radioButton, ui.histogram_radioButton]
        # Check if the selected value in the variable_to_plot_comboBox is a numeric column in the dataset

        # iuf = i int (signed), u unsigned int, f float
        if ml_model.column_types_pd_series[selected_column].kind in 'iuf':
            radio_buttons_list[0].setEnabled(True)
            radio_buttons_list[1].setEnabled(True)
            radio_buttons_list[2].setEnabled(True)

            checked_objects = list(map(lambda x: x.isChecked(), radio_buttons_list))
            if not any(checked_objects):  # Checks whether any radio button is checked
                radio_buttons_list[0].setChecked(True)  # If not, checks the first one.

        else:  # If not numeric, disable all visualisation options
            radio_buttons_list[0].setEnabled(False)
            radio_buttons_list[1].setEnabled(False)
            radio_buttons_list[2].setEnabled(False)

        self.update_visualisation_widgets()

        # TODO: Clear plot area

    def trigger_plot_matplotlib_to_qt_widget_thread(self, target_widget, content):

        # Creating an object worker
        worker = threads.Plotting_in_MplWidget_Thread(self.ui, target_widget, content)

        # Starts the thread
        self.ui.threadpool.start(worker)  # Todo: Uncomment this when finished debugging

    def update_table_widget(self, table_widget, function, data):
        ui = self.ui

        if function == 'update_header':
            table_widget.setHorizontalHeaderLabels(data['header_labels'])
            header = table_widget.horizontalHeader()
            header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)

        elif function == 'fill_table':
            i = data['i']
            j = data['j']
            qt_item = data['qt_item']
            table_widget.setItem(i, j, qt_item)

        elif function == 'stop_spinner':
            table_widget.spinner.stop()
            if table_widget.objectName() == 'dataset_tableWidget':
                ui.load_file_pushButton.setDisabled(False)
                ui.example_dataset_comboBox.setDisabled(False)
            elif table_widget.objectName() == 'pre_process_dataset_tableWidget':
                self.update_train_test_shape_label()

    def update_preprocess_replace(self):
        ui = self.ui
        ml_model = self.ml_model
        selected_value = ui.replace_columnSelection_comboBox.currentText()

        if ml_model.column_types_pd_series[
            selected_value].kind in 'iuf':  # iuf = i int (signed), u unsigned int, f float
            ui.pre_process_replacing_stackedWidget.setCurrentIndex(0)
        else:
            ui.pre_process_replacing_stackedWidget.setCurrentIndex(1)

            ui.replaced_value_comboBox.clear()
            unique_values = ml_model.dataset[selected_value].unique().tolist()

            # Filling the comboBoxes
            for each_value in unique_values:
                ui.replaced_value_comboBox.addItem(each_value)  # Fill comboBox

    def update_preprocess_filtering(self):
        ui = self.ui
        ml_model = self.ml_model

        selected_value = ui.filter_columnSelection_comboBox.currentText()

        if ml_model.column_types_pd_series[
            selected_value].kind in 'iuf':  # iuf = i int (signed), u unsigned int, f float
            ui.pre_process_filtering_stackedWidget.setCurrentIndex(0)

            if ui.filter_operator_comboBox.count() == 2:  # 2 items mean only == and !=
                ui.filter_operator_comboBox.insertItem(2, 'Greater than or equal to')  # The index is always 2
                ui.filter_operator_comboBox.insertItem(2, 'Greater than')  # The list will keep shifting
                ui.filter_operator_comboBox.insertItem(2, 'Less than or equal to')
                ui.filter_operator_comboBox.insertItem(2, 'Less than')

        else:
            ui.pre_process_filtering_stackedWidget.setCurrentIndex(1)

            ui.filtering_dataset_value_comboBox.clear()
            unique_values = ml_model.dataset[selected_value].unique().tolist()

            if ui.filter_operator_comboBox.count() == 6:
                ui.filter_operator_comboBox.removeItem(2)  # Removing Less than
                ui.filter_operator_comboBox.removeItem(2)  # Removing Less than or equal to
                ui.filter_operator_comboBox.removeItem(2)  # Removing Greater than
                ui.filter_operator_comboBox.removeItem(2)  # Removing Greater than or equal to

            # Filling the comboBoxes
            for each_value in unique_values:
                ui.filtering_dataset_value_comboBox.addItem(each_value)  # Fill comboBox

    def update_nn_layers_table(self, table, value):
        if value > table.rowCount():
            while value > table.rowCount():
                table.insertRow(table.rowCount())
                item = QtWidgets.QTableWidgetItem(str(10))
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                table.setItem(table.rowCount() - 1, 0, item)
                item = QtWidgets.QTableWidgetItem('Hidden Layer ' + str(table.rowCount()))
                table.setVerticalHeaderItem(table.rowCount() - 1, item)
        else:
            while value < table.rowCount():
                table.removeRow(table.rowCount() - 1)

    def display_training_results(self, result, model_parameters):
        ui = self.ui
        ui.train_results_loading_widget.stop()
        ui.train_model_pushButton.setDisabled(False)
        ui.export_model_pushButton.setDisabled(False)

        if model_parameters['is_regression']:

            ui.reg_mse_label.setText('{:.4f}'.format(result['mse']))
            ui.reg_rmse_label.setText('{:.4f}'.format(result['rmse']))
            ui.reg_r2_label.setText('{:.4f}'.format(result['r2_score']))
            ui.reg_mea_label.setText('{:.4f}'.format(result['mae']))

            self.trigger_plot_matplotlib_to_qt_widget_thread(target_widget=ui.model_train_widget,
                                                             content={'data': result['data_to_plot'],
                                                                      'output_variables': model_parameters[
                                                                          'output_variables'],
                                                                      'is_regression': model_parameters[
                                                                          'is_regression']})

        else:
            ui.clas_accuracy_label.setText('{:.4f}'.format(result['accuracy']))
            ui.clas_recall_label.setText('{:.4f}'.format(result['recall_score']))
            ui.clas_precision_label.setText('{:.4f}'.format(result['precision_score']))
            ui.clas_f1_score_label.setText('{:.4f}'.format(result['f1_score']))

            self.trigger_plot_matplotlib_to_qt_widget_thread(target_widget=ui.model_train_widget,
                                                             content={'data': result['data_to_plot'],
                                                                      'output_variables': model_parameters[
                                                                          'output_variables'],
                                                                      'is_regression': model_parameters[
                                                                          'is_regression']})

    def update_input_output_columns(self, target_object):
        ui = self.ui
        ml_model = self.ml_model
        is_regression = ui.regression_selection_radioButton.isChecked()

        for selected_item in ui.available_columns_listWidget.selectedItems():
            item = ui.available_columns_listWidget.takeItem(ui.available_columns_listWidget.row(selected_item))
            is_variable_categorical = selected_item.text() in ml_model.categorical_variables
            is_output_variable = target_object.objectName() == 'output_columns_listWidget'
            if is_regression and is_variable_categorical and is_output_variable:
                ui.available_columns_listWidget.addItem(item)
                display_message(QtWidgets.QMessageBox.Information, 'Invalid Input',
                                'Categorical variables should not be used as regression output', 'Error')
            else:
                target_object.addItem(item)
                combobox = ui.clas_output_colum_comboBox
                items_in_combobox = [combobox.itemText(i) for i in range(combobox.count())]
                if selected_item.text() in items_in_combobox:
                    item_index = items_in_combobox.index(selected_item.text())
                    combobox.removeItem(item_index)

        if target_object.objectName() == 'input_columns_listWidget':
            self.update_train_test_shape_label()

        self.update_train_model_button_status(is_regression)

    def update_label_from_slider_change(self, slider_value, label_object):
        ui = self.ui
        ml_model = self.ml_model
        # Todo - Too repetitive, make this a function

        if label_object.objectName() == 'reg_nn_layers_label':
            label_object.setText('{}'.format(slider_value))
            self.update_nn_layers_table(ui.reg_nn_layers_tableWidget, slider_value)
        elif label_object.objectName() == 'clas_nn_layers_label':
            label_object.setText('{}'.format(slider_value))
            self.update_nn_layers_table(ui.clas_nn_layers_tableWidget, slider_value)
        elif label_object.objectName() == 'outliers_treshold_label':
            label_object.setText('{:.1f}'.format(slider_value / 10))
            # https://moonbooks.org/Articles/How-to-fill-an-area-in-matplotlib-/
            # https://i.pinimg.com/originals/e1/d6/30/e1d630a1719b4444bbd08b7df92b7bf1.gif
            # Todo: Plot a normal distribution with the included and excluded area (plt.fill_between)
        elif label_object.objectName() == 'reg_nn_val_percent_label':
            label_object.setText('{}%'.format(slider_value))
        elif label_object.objectName() == 'reg_nn_max_iter_label':
            label_object.setText('{}'.format(slider_value))
        elif label_object.objectName() == 'reg_nn_alpha_label':
            label_object.setText('{}'.format(slider_value / 10000))
        elif label_object.objectName() == 'clas_nn_val_percent_label':
            label_object.setText('{}%'.format(slider_value))
        elif label_object.objectName() == 'clas_nn_max_iter_label':
            label_object.setText('{}'.format(slider_value))
        elif label_object.objectName() == 'clas_nn_alpha_label':
            label_object.setText('{}'.format(slider_value / 10000))
        elif label_object.objectName() == 'train_percentage_label':
            label_object.setText('{}%'.format(slider_value))
            ui.test_percentage_horizontalSlider.setValue(100 - slider_value)
            if ml_model.is_data_loaded:
                self.update_train_test_shape_label()
        elif label_object.objectName() == 'test_percentage_label':
            label_object.setText('{}%'.format(slider_value))
            ui.train_percentage_horizontalSlider.setValue(100 - slider_value)
            if ml_model.is_data_loaded:
                self.update_train_test_shape_label()

    def trigger_train_model_thread(self):
        ui = self.ui
        ml_model = self.ml_model
        # Todo : check for condition before continuing: 1) Output columns is not empty

        train_percentage = (ui.train_percentage_horizontalSlider.value() / 100)
        test_percentage = (ui.test_percentage_horizontalSlider.value() / 100)
        shuffle_samples = ui.shuffle_samples_checkBox.isChecked()

        model_parameters = {'train_percentage': train_percentage, 'test_percentage': test_percentage,
                            'shuffle_samples': shuffle_samples}

        is_regression = ui.regression_selection_radioButton.isChecked()

        input_variables = []
        for i in range(ui.input_columns_listWidget.count()):
            input_variables.append(ui.input_columns_listWidget.item(i).text())

        if is_regression:
            output_variables = []
            for i in range(ui.output_columns_listWidget.count()):
                output_variables.append(ui.output_columns_listWidget.item(i).text())
        else:
            output_variables = [ui.clas_output_colum_comboBox.currentText()]

        if is_regression:
            algorithm_index = [ui.nn_regression_radioButton.isChecked(), ui.svm_regression_radioButton.isChecked(),
                               ui.randomforest_regression_radioButton.isChecked(),
                               ui.gradientboosting_regression_radioButton.isChecked()].index(1)
            algorithm = ['nn', 'svm', 'random_forest', 'grad_boosting'][algorithm_index]

            if algorithm == 'nn':
                n_of_hidden_layers = ui.reg_nn_layers_horizontalSlider.value()
                n_of_neurons_each_layer = []
                for i in range(n_of_hidden_layers):
                    n_of_neurons_each_layer.append(int(ui.reg_nn_layers_tableWidget.item(i, 0).text()))
                activation_func = ui.reg_nn_actvfunc_comboBox.currentText()
                solver = ui.reg_nn_solver_comboBox.currentText()
                learning_rate = ui.reg_nn_learnrate_comboBox.currentText()
                max_iter = ui.reg_nn_max_iter_horizontalSlider.value()
                alpha = ui.reg_nn_alpha_horizontalSlider.value() / 10000
                validation_percentage = ui.reg_nn_val_percentage_horizontalSlider.value() / 100

                algorithm_parameters = {'n_of_hidden_layers': n_of_hidden_layers,
                                        'n_of_neurons_each_layer': n_of_neurons_each_layer,
                                        'activation_func': activation_func, 'solver': solver,
                                        'learning_rate': learning_rate,
                                        'max_iter': max_iter, 'alpha': alpha,
                                        'validation_percentage': validation_percentage}
            elif algorithm == 'svm':
                algorithm_parameters = []
            elif algorithm == 'random_forest':
                algorithm_parameters = []
            elif algorithm == 'grad_boosting':
                algorithm_parameters = []

        else:
            algorithm_index = [ui.nn_classification_radioButton.isChecked(),
                               ui.svm_classification_radioButton.isChecked(),
                               ui.randomforest_classification_radioButton.isChecked(),
                               ui.gradientboosting_classification_radioButton.isChecked(),
                               ui.knn_classification_radioButton.isChecked()].index(1)
            algorithm = ['nn', 'svm', 'random_forest', 'grad_boosting', 'knn'][algorithm_index]

            if algorithm == 'nn':
                n_of_hidden_layers = ui.clas_nn_layers_horizontalSlider.value()
                n_of_neurons_each_layer = []
                for i in range(n_of_hidden_layers):
                    n_of_neurons_each_layer.append(int(ui.clas_nn_layers_tableWidget.item(i, 0).text()))
                activation_func = ui.clas_nn_actvfunc_comboBox.currentText()
                solver = ui.clas_nn_solver_comboBox.currentText()
                learning_rate = ui.clas_nn_learnrate_comboBox.currentText()
                max_iter = ui.clas_nn_max_iter_horizontalSlider.value()
                alpha = ui.clas_nn_alpha_horizontalSlider.value() / 10000
                validation_percentage = ui.clas_nn_val_percentage_horizontalSlider.value() / 100

                algorithm_parameters = {'n_of_hidden_layers': n_of_hidden_layers,
                                        'n_of_neurons_each_layer': n_of_neurons_each_layer,
                                        'activation_func': activation_func, 'solver': solver,
                                        'learning_rate': learning_rate,
                                        'max_iter': max_iter, 'alpha': alpha,
                                        'validation_percentage': validation_percentage}
            elif algorithm == 'svm':
                algorithm_parameters = []
            elif algorithm == 'random_forest':
                algorithm_parameters = []
            elif algorithm == 'grad_boosting':
                algorithm_parameters = []
            elif algorithm == 'knn':
                algorithm_parameters = []

        model_parameters.update(
            {'is_regression': is_regression, 'algorithm': algorithm, 'input_variables': input_variables,
             'output_variables': output_variables})

        # Creating an object worker
        worker = threads.Train_Model_Thread(ml_model, model_parameters, algorithm_parameters, ui)

        # Connecting the signals from the created worker to its functions
        worker.signals.finished.connect(display_training_results)

        ui.train_model_pushButton.setDisabled(True)
        ui.train_results_loading_widget.start()

        # Todo: Clean the plot before running the thread
        # Running the traning in a separate thread from the GUI
        # ui.threadpool.start(worker) #Todo uncomment when this when finished testing!!
        # Todo : Delete this when finished testing
        result = ml_model.train(model_parameters, algorithm_parameters)
        self.display_training_results(result, model_parameters)

    def remove_item_from_listwidget(self, target_listwidget):
        ui = self.ui
        ml_model = self.ml_model
        if target_listwidget == ui.preprocess_filter_listWidget:
            for item in ui.preprocess_filter_listWidget.selectedItems():
                ui.preprocess_filter_listWidget.takeItem(ui.preprocess_filter_listWidget.row(item))

        elif target_listwidget == ui.preprocess_replace_listWidget:
            for item in ui.preprocess_replace_listWidget.selectedItems():
                ui.preprocess_replace_listWidget.takeItem(ui.preprocess_replace_listWidget.row(item))

        elif target_listwidget == ui.input_columns_listWidget:
            for selected_item in target_listwidget.selectedItems():
                item = target_listwidget.takeItem(target_listwidget.row(selected_item))
                ui.available_columns_listWidget.addItem(item)

                # Adding the variables back to the clas_output_colum_comboBox
                if selected_item.text() in ml_model.categorical_variables:
                    ui.clas_output_colum_comboBox.addItem(selected_item.text())

            self.update_train_model_button_status(ui.regression_selection_radioButton.isChecked())

        elif target_listwidget == ui.output_columns_listWidget:
            for selected_item in target_listwidget.selectedItems():
                item = target_listwidget.takeItem(target_listwidget.row(selected_item))
                ui.available_columns_listWidget.addItem(item)

            self.update_train_model_button_status(ui.regression_selection_radioButton.isChecked())


def transform_to_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller.

    Args:
        relative_path (str): path to the file.

    Returns:
        str: path that works for both debug and standalone app
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = abspath(".")

    return join(base_path, relative_path)


def display_message(icon, main_message, informative_message, window_title):
    msg = QtWidgets.QMessageBox()
    msg.setIcon(icon)
    msg.setText(main_message)
    msg.setInformativeText(informative_message)
    msg.setWindowTitle(window_title)
    msg.exec()

# try:
#     sys._MEIPASS
#     files_folder = transform_to_resource_path('resources/')
# except:
#     files_folder = transform_to_resource_path('../resources/')
