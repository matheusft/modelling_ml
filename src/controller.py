from view import QtCore, QtWidgets
from os.path import join, abspath
import os, random
import model as md
import sys
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvas


# This class extends QtWidgets.QWidget. It is needed for plotting with matplotlib in PyQt5
# A QtWidgets.QWidget was manually promoted to a MplWidget parent class in the Qt Creator
class MplWidget(QtWidgets.QWidget):

    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)

        self.canvas = FigureCanvas(Figure())

        vertical_layout = QtWidgets.QVBoxLayout()
        vertical_layout.addWidget(self.canvas)

        self.canvas.axes = self.canvas.figure.add_subplot(111)
        self.setLayout(vertical_layout)
        self.canvas.axes.axis('off')  # Turn off axis lines and labels. Show a white canvas in the initialisation


def resource_path(relative_path):
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

    # Connecting load_file_pushButton - Dataset Load Tab
    ui.load_file_pushButton.clicked.connect(lambda ml_mode: load_dataset(ui, ml_model))

    # Connecting columnSelection_comboBox - Visualise Tab
    ui.columnSelection_comboBox.currentIndexChanged.connect(lambda: update_visualisation_options(ui, ml_model))

    # Connecting radio_button_change - Visualise Tab
    ui.boxplot_radioButton.clicked.connect(lambda: update_visualisation_widgets(ui, ml_model))
    ui.plot_radioButton.clicked.connect(lambda: update_visualisation_widgets(ui, ml_model))
    ui.histogram_radioButton.clicked.connect(lambda: update_visualisation_widgets(ui, ml_model))

    pre_process_checkboxes = [ui.numeric_scaling_checkBox, ui.remove_duplicates_checkBox, ui.remove_outliers_checkBox,
                              ui.replace_values_checkBox, ui.filter_values_checkBox]

    for pre_process_option in pre_process_checkboxes:
        pre_process_option.clicked.connect(lambda *args, object=pre_process_option: update_pre_process(ui, ml_model))

    ui.outliers_treshold_horizontalSlider.valueChanged.connect(
        lambda: update_treshold_label(ui.outliers_treshold_horizontalSlider.value(), ui.outliers_treshold_label))

    # TODO : Disable all button and functions while Dataset is not chosen


def load_dataset(ui, ml_model):
    """Prompts the user to select an input file and call ml_model.read_dataset.

    Args:
        :param ml_model: An empty Machine Learning Model.
        :param ui:  The ui to be updated

    """
    # TODO: uncomment this when finish doing tests
    # fileDlg = QtWidgets.QFileDialog()
    # file_address = fileDlg.getOpenFileName()[0]

    # # TODO : Check whether the file_address is valid or empty (Is empty if the user cancel)
    file_address = ''
    while file_address == '' or file_address[0] == '.':
        file_address = random.choice(os.listdir('/Users/matheustorquato/Documents/GitHub/generic_ml/data'))
    file_address = '/Users/matheustorquato/Documents/GitHub/generic_ml/data/' + file_address
    return_code = ml_model.read_dataset(file_address)

    # TODO: add checkbox Dataset with Column name or not
    # Todo: Check what needs to be reset/cleared when a new dataset is loaded

    if return_code == 0:

        populate_tableWidget_with_dataset(ui.dataset_tableWidget, ml_model.dataset)
        populate_tableWidget_with_dataset(ui.pre_process_dataset_tableWidget, ml_model.dataset)

        # Here we update the columnSelection_comboBox
        if ui.columnSelection_comboBox.count() > 0: # If the comboBox is not empty
            ui.columnSelection_comboBox.currentIndexChanged.disconnect() # Disconnect the signal first, then clear
            ui.columnSelection_comboBox.clear() # Delete all values from comboBox, then re-connect the signal
            ui.columnSelection_comboBox.currentIndexChanged.connect(lambda: update_visualisation_options(ui, ml_model))

        # Filling the comboBoxes
        for each_column in ml_model.dataset.columns:
            ui.columnSelection_comboBox.addItem(each_column) # Fill columnSelection_comboBox from the Visualise Tab
            ui.replace_columnSelection_comboBox.addItem(each_column) # from the Pre-process Tab
            ui.filter_columnSelection_comboBox.addItem(each_column) # from the Pre-process Tab

    elif return_code == 1:  # Invalid file extension
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setText("Error")
        msg.setInformativeText('Invalid Input format. \nTry Excel or .csv formats')
        msg.setWindowTitle("Error")
        msg.exec()

    elif return_code == 2:  # Exception while reading the file
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setText("Error")
        msg.setInformativeText('Invalid Input File')
        msg.setWindowTitle("Error")
        msg.exec()


def populate_tableWidget_with_dataset(tableWidget, filling_dataset):
    # Fill dataset_tableWidget from the Dataset Load Tab with the head of the dataset
    tableWidget.setRowCount(len(filling_dataset.head(10)) + 1)  # +1 to add the Column Names in line 0
    tableWidget.setColumnCount(len(filling_dataset.columns))

    header = tableWidget.horizontalHeader()  # uses this header in order to adjust the column width

    # Adding the labels at the top of the Table
    for i in range(tableWidget.columnCount()):
        header.setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeToContents)  # Column width fits the content
        qt_item = QtWidgets.QTableWidgetItem(filling_dataset.columns[i])  # Creates an qt item
        qt_item.setTextAlignment(QtCore.Qt.AlignHCenter)  # Aligns the item in the horizontal center
        tableWidget.setItem(0, i, qt_item)

    # Filling the Table with the dataset
    for i in range(tableWidget.rowCount()):
        for j in range(tableWidget.columnCount()):
            dataset_value = filling_dataset.iloc[i, j]  # Get the value from the dataset
            # If the value is numeric, format it to two decimals
            dataset_value_converted = dataset_value if (type(dataset_value) is str) else '{:.2f}'.format(dataset_value)
            qt_item = QtWidgets.QTableWidgetItem(dataset_value_converted)  # Creates an qt item
            qt_item.setTextAlignment(QtCore.Qt.AlignHCenter)  # Aligns the item in the horizontal center
            tableWidget.setItem(i + 1, j, qt_item)  # i+1 to skip the top row (column names)


def update_visualisation_widgets(ui, ml_model):
    selected_column = ui.columnSelection_comboBox.currentText()  # Get the selected value in the comboBox

    ui.columnSummary_textBrowser.clear()
    ui.columnSummary_textBrowser.append(ml_model.dataset[selected_column].describe().
                                        to_string(float_format='{:.2f}'.format).title().replace('     ', '  = '))

    if ml_model.column_types_pd_series[selected_column].kind in 'iuf':  # iuf = i int (signed), u unsigned int, f float
        plot_matplotlib_to_qt_widget(ml_model.dataset[selected_column], ui)
    else:
        ui.dataVisualisePlot_widget.canvas.axes.clear()
        ui.dataVisualisePlot_widget.canvas.axes.axis('off')
        ui.dataVisualisePlot_widget.canvas.draw()


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
        if not any(checked_objects): # Checks whether any radio button is checked
            radio_buttons_list[0].setChecked(True) # If not, checks the first one.

    else:  # If not numeric, disable all visualisation options
        radio_buttons_list[0].setEnabled(False)
        radio_buttons_list[1].setEnabled(False)
        radio_buttons_list[2].setEnabled(False)

    update_visualisation_widgets(ui, ml_model)

    # TODO: Clear plot area


def plot_matplotlib_to_qt_widget(data, ui):
    target_widget = ui.dataVisualisePlot_widget
    target_widget.canvas.axes.clear()
    target_widget.canvas.axes.axis('on')

    if ui.plot_radioButton.isChecked() and ui.plot_radioButton.isEnabled():
        data.plot(ax=target_widget.canvas.axes, grid=False)

    elif ui.boxplot_radioButton.isChecked() and ui.boxplot_radioButton.isEnabled():
        pd.DataFrame(data).boxplot(ax=target_widget.canvas.axes, grid=False)

    elif ui.histogram_radioButton.isChecked() and ui.histogram_radioButton.isEnabled():
        pd.DataFrame(data).hist(ax=target_widget.canvas.axes, grid=False)

    else:
        target_widget.canvas.axes.axis('off')

    target_widget.canvas.draw()


def update_treshold_label(slider_value,label_object):

    label_object.setText('{:.1f}'.format(slider_value/10))


def update_pre_process(ui,ml_model):

    scaling = ui.numeric_scaling_checkBox.isChecked()
    rm_duplicate = ui.remove_duplicates_checkBox.isChecked()
    rm_outliers = [ui.remove_outliers_checkBox.isChecked(), ui.outliers_treshold_horizontalSlider.value()]
    replace = [ui.replace_values_checkBox.isChecked(), ui.replace_columnSelection_comboBox.currentText(),
               ui.replace_text_lineEdit.text(), ui.replacing_value_lineEdit.text()]
    filter_dataset = [ui.filter_values_checkBox.isChecked(), ui.filter_columnSelection_comboBox.currentText(),
                      ui.filter_operator_comboBox.currentText(),
                      ui.filtering_dataset_value_lineEdit.text()]

    print(scaling, rm_duplicate, rm_outliers, replace, filter_dataset)

    pre_processed_dataset = ml_model.pre_process_data(scaling, rm_duplicate, rm_outliers, replace, filter_dataset)

    populate_tableWidget_with_dataset(ui.pre_process_dataset_tableWidget,pre_processed_dataset)

try:
    sys._MEIPASS
    files_folder = resource_path('resources/')
except:
    pass
files_folder = resource_path('../resources/')
