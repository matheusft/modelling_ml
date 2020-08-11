import os
import random
import sys
import threads
import seaborn as sns
from os.path import join, abspath
import pandas as pd
from docutils.nodes import address
from view import QtCore, QtWidgets
from personalised_widgets import MplWidget, QtWaitingSpinner

#Set seaborn aesthetic parameters
sns.set()


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


def configure_gui(ui, ml_model):
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

    connect_signals(ui, ml_model)

    ui.nn_classification_radioButton.click()
    ui.nn_regression_radioButton.click()
    # ui.regression_selection_radioButton.setChecked(True)
    ui.nn_regression_radioButton.click()

    ui.tabs_widget.setCurrentIndex(0)
    ui.pre_process_tabWidget.setCurrentIndex(0)
    ui.output_selection_stackedWidget.setCurrentIndex(0)

    widgets_to_disable = [ui.plot_radioButton, ui.boxplot_radioButton, ui.histogram_radioButton,
                          ui.remove_duplicates_pushButton, ui.remove_constant_variables_pushButton,
                          ui.numeric_scaling_pushButton, ui.remove_outliers_pushButton,
                          ui.addrule_filter_value_pushButton, ui.addrule_replace_value_pushButton,
                          ui.addrule_filter_value_pushButton, ui.export_model_pushButton,
                          ui.add_input_columns_pushButton, ui.add_output_columns_pushButton, ui.train_model_pushButton]

    for widget in widgets_to_disable:
        widget.setEnabled(False)

    #Crating QtWaitingSpinners dinamically and positioning it over the tableWidgets
    ui.wait_spinner_dataset_tableWidget = QtWaitingSpinner(ui.dataset_tableWidget)
    ui.wait_spinner_prepdataset_tableWidget = QtWaitingSpinner(ui.pre_process_dataset_tableWidget)
    ui.wait_spinner_dataset_tableWidget.setSizePolicy(ui.dataset_tableWidget.sizePolicy())
    ui.wait_spinner_prepdataset_tableWidget.setSizePolicy(ui.pre_process_dataset_tableWidget.sizePolicy())

    # Todo : Check whether all number_of_neuros of reg_nn_layers_tableWidget are int greater than 0

    # TODO : Disable all button and functions while Dataset is not chosen -
    # Todo : Check things that break when the dataset is not loaded yet


def connect_signals(ui, ml_model):

    # Connecting load_file_pushButton - Dataset Load Tab
    ui.load_file_pushButton.clicked.connect(
        lambda: trigger_loading_dataset_thread(ui, ml_model, ui.load_file_pushButton))
    ui.example_dataset_comboBox.currentIndexChanged.connect(
        lambda: trigger_loading_dataset_thread(ui, ml_model, ui.example_dataset_comboBox))

    # Connecting columnSelection_comboBox - Visualise Tab
    ui.columnSelection_comboBox.currentIndexChanged.connect(lambda: update_visualisation_options(ui, ml_model))

    # Connecting radio_button_change - Visualise Tab
    ui.boxplot_radioButton.clicked.connect(lambda: update_visualisation_widgets(ui, ml_model))
    ui.plot_radioButton.clicked.connect(lambda: update_visualisation_widgets(ui, ml_model))
    ui.histogram_radioButton.clicked.connect(lambda: update_visualisation_widgets(ui, ml_model))

    # pre_process_checkboxes = [ui.numeric_scaling_checkBox, ui.remove_duplicates_checkBox, ui.remove_outliers_checkBox,
    #                           ui.replace_values_checkBox, ui.filter_values_checkBox]
    #
    # for pre_process_option in pre_process_checkboxes:
    #     pre_process_option.clicked.connect(lambda: update_pre_process(ui, ml_model))

    ui.outliers_treshold_horizontalSlider.valueChanged.connect(
        lambda: update_label_from_slider_change(ui, ui.outliers_treshold_horizontalSlider.value(),
                                                ui.outliers_treshold_label, ml_model))

    ui.replace_columnSelection_comboBox.currentIndexChanged.connect(lambda: update_preprocess_replace(ui,
                                                                                                      ml_model))  # Update the pre_process_replacing_stackedWidget according to the replace_columnSelection_comboBox
    ui.filter_columnSelection_comboBox.currentIndexChanged.connect(lambda: update_preprocess_filtering(ui,
                                                                                                       ml_model))  # Update the pre_process_replacing_stackedWidget according to the replace_columnSelection_comboBox

    ui.addrule_replace_value_pushButton.clicked.connect(lambda: add_replacing_rule(ui, ml_model))
    ui.addrule_filter_value_pushButton.clicked.connect(lambda: add_filtering_rule(ui, ml_model))

    # ui.remove_replace_value_pushButton.clicked.connect(
    #     lambda: remove_item_from_listwidget(ui, ui.preprocess_replace_listWidget, ml_model))
    # ui.remove_filter_rule_pushButton.clicked.connect(
    #     lambda: remove_item_from_listwidget(ui, ui.preprocess_filter_listWidget, ml_model))

    ui.add_input_columns_pushButton.clicked.connect(
        lambda: update_input_output_columns(ui, ui.input_columns_listWidget, ml_model))
    ui.add_output_columns_pushButton.clicked.connect(
        lambda: update_input_output_columns(ui, ui.output_columns_listWidget, ml_model))

    ui.remove_input_columns_pushButton.clicked.connect(
        lambda: remove_item_from_listwidget(ui, ui.input_columns_listWidget, ml_model))
    ui.remove_output_columns_pushButton.clicked.connect(
        lambda: remove_item_from_listwidget(ui, ui.output_columns_listWidget, ml_model))

    # ui.clear_replace_value_pushButton.clicked.connect(
    #     lambda: clear_listwidget(ui, ui.preprocess_replace_listWidget, ml_model))
    # ui.clear_filter_rule_pushButton.clicked.connect(
    #     lambda: clear_listwidget(ui, ui.preprocess_filter_listWidget, ml_model))
    ui.clear_input_columns_pushButton.clicked.connect(
        lambda: clear_listwidget(ui, ui.input_columns_listWidget, ml_model))
    ui.clear_output_columns_pushButton.clicked.connect(
        lambda: clear_listwidget(ui, ui.output_columns_listWidget, ml_model))

    model_selection_radio_buttons = [ui.regression_selection_radioButton, ui.classification_selection_radioButton,
                                     ui.gradientboosting_classification_radioButton, ui.knn_classification_radioButton,
                                     ui.nn_classification_radioButton, ui.randomforest_classification_radioButton,
                                     ui.svm_classification_radioButton, ui.svm_regression_radioButton,
                                     ui.randomforest_regression_radioButton, ui.nn_regression_radioButton,
                                     ui.gradientboosting_regression_radioButton]

    for model_option in model_selection_radio_buttons:
        model_option.clicked.connect(lambda: model_selection_tab_events(ui))

    ui.reg_nn_layers_horizontalSlider.valueChanged.connect(
        lambda: update_label_from_slider_change(ui, ui.reg_nn_layers_horizontalSlider.value(), ui.reg_nn_layers_label,
                                                ml_model))
    ui.clas_nn_val_percentage_horizontalSlider.valueChanged.connect(
        lambda: update_label_from_slider_change(ui, ui.clas_nn_val_percentage_horizontalSlider.value(),
                                                ui.reg_nn_val_percent_label, ml_model))
    ui.reg_nn_max_iter_horizontalSlider.valueChanged.connect(
        lambda: update_label_from_slider_change(ui, ui.reg_nn_max_iter_horizontalSlider.value(),
                                                ui.reg_nn_max_iter_label, ml_model))
    ui.reg_nn_alpha_horizontalSlider.valueChanged.connect(
        lambda: update_label_from_slider_change(ui, ui.reg_nn_alpha_horizontalSlider.value(), ui.reg_nn_alpha_label,
                                                ml_model))
    ui.clas_nn_layers_horizontalSlider.valueChanged.connect(
        lambda: update_label_from_slider_change(ui, ui.clas_nn_layers_horizontalSlider.value(),
                                                ui.clas_nn_layers_label, ml_model))
    ui.reg_nn_val_percentage_horizontalSlider.valueChanged.connect(
        lambda: update_label_from_slider_change(ui, ui.reg_nn_val_percentage_horizontalSlider.value(),
                                                ui.clas_nn_val_percent_label, ml_model))
    ui.clas_nn_max_iter_horizontalSlider.valueChanged.connect(
        lambda: update_label_from_slider_change(ui, ui.clas_nn_max_iter_horizontalSlider.value(),
                                                ui.clas_nn_max_iter_label, ml_model))
    ui.clas_nn_alpha_horizontalSlider.valueChanged.connect(
        lambda: update_label_from_slider_change(ui, ui.clas_nn_alpha_horizontalSlider.value(), ui.clas_nn_alpha_label,
                                                ml_model))
    ui.train_percentage_horizontalSlider.valueChanged.connect(
        lambda: update_label_from_slider_change(ui, ui.train_percentage_horizontalSlider.value(),
                                                ui.train_percentage_label, ml_model))
    ui.test_percentage_horizontalSlider.valueChanged.connect(
        lambda: update_label_from_slider_change(ui, ui.test_percentage_horizontalSlider.value(),
                                                ui.test_percentage_label, ml_model))

    ui.train_model_pushButton.clicked.connect(lambda: train_model(ui, ml_model))


def trigger_loading_dataset_thread(ui, ml_model, data_source):
    """Prompts the user to select an input file and call ml_model.read_dataset.

    Args:
        :param ml_model: An empty Machine Learning Model.
        :param ui:  The ui to be updated

    """

    if data_source.objectName() == 'load_file_pushButton':
        fileDlg = QtWidgets.QFileDialog()
        file_address = fileDlg.getOpenFileName()[0]
        if file_address == '' or file_address == None:
            return
    elif data_source.objectName() == 'example_dataset_comboBox':
        selected_index = ui.example_dataset_comboBox.currentIndex()
        file_address = ui.example_dataset_comboBox.itemData(selected_index)

    # This spinner is started here, but finished by the thread
    ui.wait_spinner_dataset_tableWidget.start() # Todo: Stop this when the thread is done
    ui.load_file_pushButton.setDisabled(True)
    ui.example_dataset_comboBox.setDisabled(True)

    # Creating an object worker
    worker = threads.Load_Dataset_Thread(ui, ml_model, file_address)
    # Connecting the signals from the created worker to its functions
    worker.signals.update_train_test_shape_label.connect(update_train_test_shape_label)
    worker.signals.display_message.connect(display_message)
    worker.signals.update_table_widget.connect(update_table_widget)
    #Starts the thread
    ui.threadpool.start(worker) # Todo: Uncomment this when finished debugging
    # result = ml_model.read_dataset(file_address)

    # TODO: add checkbox Dataset with Column name or not
    # Todo: Check what needs to be reset/cleared when a new dataset is loaded


def update_table_widget(ui, table_widget, function, data):
    if function == 'update_header':
        table_widget.setHorizontalHeaderLabels(data['header_labels'])
    elif function == 'fill_table':
        i = data['i']
        j = data['j']
        qt_item = data['qt_item']
        table_widget.setItem(i, j, qt_item)
    elif function == 'stop_spinner':
        if table_widget.objectName() == 'dataset_tableWidget':
            ui.wait_spinner_dataset_tableWidget.stop()
            # Enabling back the widgets after finishing the data loading
            ui.load_file_pushButton.setDisabled(False)
            ui.example_dataset_comboBox.setDisabled(False)
        elif table_widget.objectName() == 'pre_process_dataset_tableWidget':
            ui.wait_spinner_prepdataset_tableWidget.stop()
            print('Call update_train_test_shape_label')
            #Todo : call update_train_test_shape_label


def update_visualisation_widgets(ui, ml_model):
    selected_column = ui.columnSelection_comboBox.currentText()  # Get the selected value in the comboBox

    ui.columnSummary_textBrowser.clear()
    description = ml_model.dataset[selected_column].describe()
    for i in range(len(description)):
        ui.columnSummary_textBrowser.append('{} = {}'.format(description.keys()[i].title(), description.values[i]))

    is_categorical = ml_model.column_types_pd_series[
                         selected_column].kind not in 'iuf'  # iuf = i int (signed), u unsigned int, f float

    plot_matplotlib_to_qt_widget(ui=ui, target_widget=ui.dataVisualisePlot_widget,
                                 content={'data': ml_model.dataset[selected_column],
                                          'is_categorical': is_categorical})


def add_replacing_rule(ui, ml_model):
    if ui.pre_process_replacing_stackedWidget.currentIndex() == 0:  # If numeric
        if ui.replaced_value_lineEdit.text() != '' and ui.replacing_value_lineEdit.text() != '':  # If inputs are not empty
            try:
                float(ui.replacing_value_lineEdit.text()), float(
                    ui.replaced_value_lineEdit.text())  # Check whether it is a valid input
            except:
                display_message(QtWidgets.QMessageBox.Critical, 'Invalid Input',  # Display error message and return
                                'Type a valid numeric input for column {}'.format(
                                    ui.replace_columnSelection_comboBox.currentText()), 'Error')
                return
            # Make String to be displayed in the rule
            string_to_add = '- Replace {} in {} with {}'.format(ui.replaced_value_lineEdit.text(),
                                                                ui.replace_columnSelection_comboBox.currentText(),
                                                                ui.replacing_value_lineEdit.text())
            item_to_add = QtWidgets.QListWidgetItem()  # Create a QListWidgetItem
            item_to_add.setText(string_to_add)  # Add the text to be displayed in the listWidget
            # Add the data embedded in each QListWidgetItem element from the listWidget
            item_to_add.setData(QtCore.Qt.UserRole, ['Numeric', ui.replaced_value_lineEdit.text(),
                                                     ui.replace_columnSelection_comboBox.currentText(),
                                                     ui.replacing_value_lineEdit.text()])
            # Todo: Make this loop a function. It happens 4x in the code
            for i in range(ui.preprocess_replace_listWidget.count()):
                if ui.preprocess_replace_listWidget.item(i).text() == item_to_add.text():
                    display_message(QtWidgets.QMessageBox.Information, 'Duplicate Rule',
                                    'Type a valid rule', 'Error')
                    return
            ui.preprocess_replace_listWidget.addItem(item_to_add)  # Add the new rule to the list widget

        else:  # Display a message if the inputs are empty and a rule is added
            display_message(QtWidgets.QMessageBox.Information, 'Empty Input',
                            'Type a valid rule', 'Error')

    else:  # If not numeric
        if ui.replacing_value_lineEdit.text() != '':  # If input is not empty
            string_to_add = '- Replace {} in {} with {}'.format(ui.replaced_value_comboBox.currentText(),
                                                                ui.replace_columnSelection_comboBox.currentText(),
                                                                ui.replacing_value_lineEdit.text())
            item_to_add = QtWidgets.QListWidgetItem()  # Create a QListWidgetItem
            item_to_add.setText(string_to_add)  # Add the text to be displayed in the listWidget
            # Add the data embedded in each QListWidgetItem element from the listWidget
            item_to_add.setData(QtCore.Qt.UserRole, ['Categorical', ui.replaced_value_comboBox.currentText(),
                                                     ui.replace_columnSelection_comboBox.currentText(),
                                                     ui.replacing_value_lineEdit.text()])
            for i in range(ui.preprocess_replace_listWidget.count()):
                if ui.preprocess_replace_listWidget.item(i).text() == item_to_add.text():
                    display_message(QtWidgets.QMessageBox.Information, 'Duplicate Rule',
                                    'Type a valid rule', 'Error')
                    return
            ui.preprocess_replace_listWidget.addItem(item_to_add)  # Add the new rule to the list widget
        else:  # Display a message if the inputs are empty and a rule is added
            display_message(QtWidgets.QMessageBox.Information, 'Empty Input',
                            'Type a valid rule', 'Error')
            return

    if ui.replace_values_checkBox.isChecked():
        update_pre_process(ui, ml_model)


def add_filtering_rule(ui, ml_model):
    is_numeric_column = ml_model.column_types_pd_series[
                            ui.filter_columnSelection_comboBox.currentText()].kind in 'iuf'  # iuf = i int (signed), u unsigned int, f float

    if is_numeric_column:  # If numeric
        if ui.filtering_dataset_value_lineEdit.text() != '':  # If input is not empty
            try:
                float(ui.filtering_dataset_value_lineEdit.text())  # Check whether it is a valid input
            except:
                display_message(QtWidgets.QMessageBox.Critical, 'Invalid Input',
                                'Type a valid numeric input for column {}'.format(
                                    ui.filter_columnSelection_comboBox.currentText()), 'Error')
                return
            string_to_add = '- Exclude {} values {} {}'.format(ui.filter_columnSelection_comboBox.currentText(),
                                                               ui.filter_operator_comboBox.currentText(),
                                                               ui.filtering_dataset_value_lineEdit.text())
            item_to_add = QtWidgets.QListWidgetItem()  # Create a QListWidgetItem
            item_to_add.setText(string_to_add)  # Add the text to be displayed in the listWidget
            # Add the data
            item_to_add.setData(QtCore.Qt.UserRole, ['Numeric', ui.filter_columnSelection_comboBox.currentText(),
                                                     ui.filter_operator_comboBox.currentText(),
                                                     ui.filtering_dataset_value_lineEdit.text()])
            # Todo Make This Loop a function
            for i in range(ui.preprocess_filter_listWidget.count()):
                if ui.preprocess_filter_listWidget.item(i).text() == item_to_add.text():
                    display_message(QtWidgets.QMessageBox.Information, 'Duplicate Rule',
                                    'Type a valid rule', 'Error')
                    return
            ui.preprocess_filter_listWidget.addItem(item_to_add)  # Add the new rule to the list widget


        else:  # Display a message if the inputs are empty and a rule is added
            display_message(QtWidgets.QMessageBox.Information, 'Empty Input',
                            'Type a valid rule', 'Error')

    else:  # If not numeric
        string_to_add = '- Exclude {} values {} to {}'.format(ui.filter_columnSelection_comboBox.currentText(),
                                                              ui.filter_operator_comboBox.currentText(),
                                                              ui.filtering_dataset_value_comboBox.currentText())
        item_to_add = QtWidgets.QListWidgetItem()  # Create a QListWidgetItem
        item_to_add.setText(string_to_add)  # Add the text to be displayed in the listWidget
        # Add the data
        item_to_add.setData(QtCore.Qt.UserRole, ['Categorical', ui.filter_columnSelection_comboBox.currentText(),
                                                 ui.filter_operator_comboBox.currentText(),
                                                 ui.filtering_dataset_value_comboBox.currentText()])
        for i in range(ui.preprocess_filter_listWidget.count()):
            if ui.preprocess_filter_listWidget.item(i).text() == item_to_add.text():
                display_message(QtWidgets.QMessageBox.Information, 'Duplicate Rule',
                                'Type a valid rule', 'Error')
                return
        ui.preprocess_filter_listWidget.addItem(item_to_add)  # Add the new rule to the list widget

    if ui.filter_values_checkBox.isChecked():
        update_pre_process(ui, ml_model)


def update_preprocess_replace(ui, ml_model):
    selected_value = ui.replace_columnSelection_comboBox.currentText()

    if ml_model.column_types_pd_series[selected_value].kind in 'iuf':  # iuf = i int (signed), u unsigned int, f float
        ui.pre_process_replacing_stackedWidget.setCurrentIndex(0)
    else:
        ui.pre_process_replacing_stackedWidget.setCurrentIndex(1)

        ui.replaced_value_comboBox.clear()
        unique_values = ml_model.dataset[selected_value].unique().tolist()

        # Filling the comboBoxes
        for each_value in unique_values:
            ui.replaced_value_comboBox.addItem(each_value)  # Fill comboBox


def update_preprocess_filtering(ui, ml_model):
    selected_value = ui.filter_columnSelection_comboBox.currentText()

    if ml_model.column_types_pd_series[selected_value].kind in 'iuf':  # iuf = i int (signed), u unsigned int, f float
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


def update_visualisation_options(ui, ml_model):
    """Update the available visualisation options for each column in the radio buttons.

    Args:
        :param ml_model: The Machine Learning Model.
        :param ui:  The ui to be updated

    """
    selected_column = ui.columnSelection_comboBox.currentText()  # Get the selected value in the comboBox
    # Create a list of all radioButton objects
    radio_buttons_list = [ui.plot_radioButton, ui.boxplot_radioButton, ui.histogram_radioButton]
    # Check if the selected value in the columnSelection_comboBox is a numeric column in the dataset

    if ml_model.column_types_pd_series[selected_column].kind in 'iuf':  # iuf = i int (signed), u unsigned int, f float
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

    update_visualisation_widgets(ui, ml_model)

    # TODO: Clear plot area


def plot_matplotlib_to_qt_widget(ui, target_widget, content):

    #Todo - Put this function in a thread!!
    print('Plotting something - DO THIS IN A THREAD!!!')

    target_widget.canvas.axes.clear()
    # If the figure has multiple axes - This happens when the Confusion Matrix is plotted
    if len(target_widget.canvas.figure.axes) > 1:
        target_widget.canvas.figure.clf() # Clear the figure
        target_widget.canvas.axes = target_widget.canvas.figure.add_subplot() # Add an Axes to the figure
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
            else:
                target_widget.canvas.axes.bar(content['data']['labels'], content['data']['values'])
                target_widget.canvas.axes.set_xticklabels(content['data']['labels'], rotation='vertical')
                target_widget.canvas.axes.axes.set_title('Percentage Error')
        else:
            # Print the number labels in the cells if n_of_classes < 10, otherwise just colours
            is_annot = False if len(content['data']) > 10 else True
            _ = sns.heatmap(content['data'], cmap="YlGnBu", annot=is_annot, ax=target_widget.canvas.axes)
            target_widget.canvas.axes.tick_params(axis='y', labelrotation=0)
            target_widget.canvas.axes.tick_params(axis='x', labelrotation=60)
            target_widget.canvas.axes.set_xlabel('Actual')
            target_widget.canvas.axes.set_ylabel('Predicted')

    target_widget.canvas.draw()


def update_label_from_slider_change(ui, slider_value, label_object, ml_model):
    # Todo - Too repetitive, make this a function

    if label_object.objectName() == 'reg_nn_layers_label':
        label_object.setText('{}'.format(slider_value))
        update_nn_layers_table(ui.reg_nn_layers_tableWidget, slider_value)
    elif label_object.objectName() == 'clas_nn_layers_label':
        label_object.setText('{}'.format(slider_value))
        update_nn_layers_table(ui.clas_nn_layers_tableWidget, slider_value)
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
            update_train_test_shape_label(ui, ml_model)
    elif label_object.objectName() == 'test_percentage_label':
        label_object.setText('{}%'.format(slider_value))
        ui.train_percentage_horizontalSlider.setValue(100 - slider_value)
        if ml_model.is_data_loaded:
            update_train_test_shape_label(ui, ml_model)


def update_nn_layers_table(table, value):
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


def remove_item_from_listwidget(ui, target_listwidget, ml_model):
    if target_listwidget == ui.preprocess_filter_listWidget:
        for item in ui.preprocess_filter_listWidget.selectedItems():
            ui.preprocess_filter_listWidget.takeItem(ui.preprocess_filter_listWidget.row(item))

        if ui.filter_values_checkBox.isChecked():
            update_pre_process(ui, ml_model)

    elif target_listwidget == ui.preprocess_replace_listWidget:
        for item in ui.preprocess_replace_listWidget.selectedItems():
            ui.preprocess_replace_listWidget.takeItem(ui.preprocess_replace_listWidget.row(item))

        if ui.replace_values_checkBox.isChecked():
            update_pre_process(utrain_percentage_horizontalSlideri, ml_model)

    elif target_listwidget == ui.input_columns_listWidget:
        for selected_item in target_listwidget.selectedItems():
            item = target_listwidget.takeItem(target_listwidget.row(selected_item))
            ui.available_columns_listWidget.addItem(item)

            # Adding the variables back to the clas_output_colum_comboBox
            if selected_item.text() in ml_model.categorical_variables:
                ui.clas_output_colum_comboBox.addItem(selected_item.text())

        update_train_model_button_status(ui, ui.regression_selection_radioButton.isChecked())

    elif target_listwidget == ui.output_columns_listWidget:
        for selected_item in target_listwidget.selectedItems():
            item = target_listwidget.takeItem(target_listwidget.row(selected_item))
            ui.available_columns_listWidget.addItem(item)

        update_train_model_button_status(ui, ui.regression_selection_radioButton.isChecked())


def clear_listwidget(ui, target_listwidget, ml_model):
    if target_listwidget == ui.preprocess_filter_listWidget:
        ui.preprocess_filter_listWidget.clear()
        if ui.filter_values_checkBox.isChecked():
            update_pre_process(ui, ml_model)

    elif target_listwidget == ui.preprocess_replace_listWidget:
        ui.preprocess_replace_listWidget.clear()
        if ui.replace_values_checkBox.isChecked():
            update_pre_process(ui, ml_model)

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


def display_message(icon, main_message, informative_message, window_title):
    msg = QtWidgets.QMessageBox()
    msg.setIcon(icon)
    msg.setText(main_message)
    msg.setInformativeText(informative_message)
    msg.setWindowTitle(window_title)
    msg.exec()


def update_pre_process(ui, ml_model):

    rm_outliers = [ui.remove_outliers_checkBox.isChecked(), ui.outliers_treshold_horizontalSlider.value() / 10]

    if ui.replace_values_checkBox.isChecked() and ui.preprocess_replace_listWidget.count() > 0:  # Just read the replacing Values if the box is checked
        replacing_rules = []
        for rule_index in range(ui.preprocess_replace_listWidget.count()):  # Looping through all rules
            item_data = ui.preprocess_replace_listWidget.item(rule_index).data(
                QtCore.Qt.UserRole)  # Getting the data embedded in each item from the listWidget
            if item_data[0] == 'Numeric':
                replacing_rules.append([(item_data[1]), item_data[2], (item_data[3])])
            elif item_data[0] == 'Categorical':
                replacing_rules.append([(item_data[1]), item_data[2], (item_data[3])])
        replace = [ui.replace_values_checkBox.isChecked(), replacing_rules]
    else:
        replace = [False]

    if ui.filter_values_checkBox.isChecked() and ui.preprocess_filter_listWidget.count() > 0:
        filtering_rules = []
        for rule_index in range(ui.preprocess_filter_listWidget.count()):  # Looping through all rules
            item_data = ui.preprocess_filter_listWidget.item(rule_index).data(
                QtCore.Qt.UserRole)  # Getting the data embedded in each item from the listWidget
            if item_data[0] == 'Numeric':
                filtering_rules.append([(item_data[1]), item_data[2], float(item_data[3])])
            elif item_data[0] == 'Categorical':
                filtering_rules.append([(item_data[1]), item_data[2], (item_data[3])])
        filter_dataset = [ui.filter_values_checkBox.isChecked(), filtering_rules]
    else:
        filter_dataset = [False]

    # Todo: Check input values before calling ml_model

    pre_processed_dataset = ml_model.pre_process_data(scaling, rm_duplicate, rm_outliers, replace, filter_dataset)

    if pre_processed_dataset.empty:  # Empty Dataframe
        ui.preprocess_filter_listWidget.takeItem(ui.preprocess_filter_listWidget.count() - 1)  # Drop last rule
        display_message(QtWidgets.QMessageBox.Critical, 'Invalid Pre-processing',
                        'These pre-processing rules are too restrictive and would return an empty dataset', 'Error')
    else:
        worker = threads.Load_Dataset_Thread(ui, ml_model, 'populate_tablewidget_only')
        worker.signals.update_train_test_shape_label.connect(update_train_test_shape_label)
        worker.signals.display_message.connect(display_message)
        worker.signals.update_table_widget.connect(update_table_widget)
        # Starts the thread
        ui.threadpool.start(worker)  # Todo: Uncomment this when finished debugging
        # populate_tablewidget_with_dataframe(ui.pre_process_dataset_tableWidget, pre_processed_dataset,
        #                                     ui.wait_spinner_prepdataset_tableWidget)
        update_train_test_shape_label(ui, ml_model)

    # Todo Update the values from the comboboxes after filtering/replacing according to the pro-process dataset


def update_input_output_columns(ui, target_object, ml_model):
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
        update_train_test_shape_label(ui, ml_model)

    update_train_model_button_status(ui, is_regression)


def model_selection_tab_events(ui):
    is_regression = ui.regression_selection_radioButton.isChecked()

    if is_regression:
        ui.regression_and_classification_stackedWidget.setCurrentIndex(0)  # Change to Regression Tab
        ui.train_metrics_stackedWidget.setCurrentIndex(0)  # Change to Regression Tab
        ui.output_selection_stackedWidget.setCurrentIndex(0)
        update_train_model_button_status(ui, is_regression)

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
        update_train_model_button_status(ui, is_regression)

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


def update_train_test_shape_label(ui, ml_model):
    dataset_shape = ml_model.pre_processed_dataset.shape

    number_of_rows_train = round(dataset_shape[0] * ui.train_percentage_horizontalSlider.value() / 100)
    number_of_columns_train = ui.input_columns_listWidget.count()

    number_of_rows_test = round(dataset_shape[0] * ui.test_percentage_horizontalSlider.value() / 100)
    number_of_columns_test = ui.input_columns_listWidget.count()

    ui.train_dataset_shape_label.setText('{} x {}'.format(number_of_rows_train, number_of_columns_train))
    ui.test_dataset_shape_label.setText('{} x {}'.format(number_of_rows_test, number_of_columns_test))


def update_train_model_button_status(ui, is_regression):
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


def train_model(ui, ml_model):
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
        algorithm_index = [ui.nn_classification_radioButton.isChecked(), ui.svm_classification_radioButton.isChecked(),
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

    model_parameters.update({'is_regression': is_regression, 'algorithm': algorithm, 'input_variables': input_variables,
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
    display_training_results(ui, result, model_parameters)


def display_training_results(ui, result, model_parameters):
    ui.train_results_loading_widget.stop()
    ui.train_model_pushButton.setDisabled(False)
    ui.export_model_pushButton.setDisabled(False)

    if model_parameters['is_regression']:

        ui.reg_mse_label.setText('{:.4f}'.format(result['mse']))
        ui.reg_rmse_label.setText('{:.4f}'.format(result['rmse']))
        ui.reg_r2_label.setText('{:.4f}'.format(result['r2_score']))
        ui.reg_mea_label.setText('{:.4f}'.format(result['mae']))

        plot_matplotlib_to_qt_widget(ui=ui, target_widget=ui.model_train_widget,
                                     content={'data': result['data_to_plot'],
                                              'output_variables': model_parameters['output_variables'],
                                              'is_regression': model_parameters['is_regression']})


    else:
        ui.clas_accuracy_label.setText('{:.4f}'.format(result['accuracy']))
        ui.clas_recall_label.setText('{:.4f}'.format(result['recall_score']))
        ui.clas_precision_label.setText('{:.4f}'.format(result['precision_score']))
        ui.clas_f1_score_label.setText('{:.4f}'.format(result['f1_score']))

        plot_matplotlib_to_qt_widget(ui=ui, target_widget=ui.model_train_widget,
                                     content={'data': result['data_to_plot'],
                                              'output_variables': model_parameters['output_variables'],
                                              'is_regression': model_parameters['is_regression']})

try:
    sys._MEIPASS
    files_folder = transform_to_resource_path('resources/')
except:
    pass
files_folder = transform_to_resource_path('../resources/')
