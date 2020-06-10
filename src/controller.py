from view import QtCore, QtWidgets
from os.path import join, abspath
import model as md
import sys


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
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

    # Connecting radio_button_change - Visualise Tab
    ui.boxplot_radioButton.clicked.connect(lambda : update_visualisation(ui, ml_model,ui.boxplot_radioButton.text()))
    ui.summary_radioButton.clicked.connect(lambda : update_visualisation(ui, ml_model,ui.summary_radioButton.text()))
    ui.plot_radioButton.clicked.connect(lambda : update_visualisation(ui, ml_model,ui.plot_radioButton.text()))
    ui.histogram_radioButton.clicked.connect(lambda : update_visualisation(ui, ml_model,ui.histogram_radioButton.text()))


def load_dataset(ui, ml_model):
    """Prompts the user to select an input file and call ml_model.read_dataset.

    Args:
        :param ml_model: An empty Machine Learning Model.
        :param ui:  The ui to be updated

    """
    # TODO: uncomment this when finish doing tests
    # fileDlg = QtWidgets.QFileDialog()
    # file_address = fileDlg.getOpenFileName()[0]
    # return_code = ml_model.read_dataset(file_address)
    file_address = '/Users/matheustorquato/Desktop/RSC_Data_2.xlsx'
    return_code = ml_model.read_dataset(file_address)

    if return_code == 0:
        populateWithDatasetData(ui, ml_model.dataset)

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


def populateWithDatasetData(ui, dataset):
    # Fill dataset_tableWidget from the Dataset Load Tab with the head of the dataset
    ui.dataset_tableWidget.setRowCount(len(dataset.head(10)) + 1)  # +1 to add the Column Names in line 0
    ui.dataset_tableWidget.setColumnCount(len(dataset.columns))

    header = ui.dataset_tableWidget.horizontalHeader()  # uses this header in order to adjust the column width

    # Adding the labels at the top of the Table
    for i in range(ui.dataset_tableWidget.columnCount()):
        header.setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeToContents)  # Column width fits the content
        qt_item = QtWidgets.QTableWidgetItem(dataset.columns[i])  # Creates an qt item
        qt_item.setTextAlignment(QtCore.Qt.AlignHCenter)  # Aligns the item in the horizontal center
        ui.dataset_tableWidget.setItem(0, i, qt_item)

    # Filling the Table with the dataset
    for i in range(ui.dataset_tableWidget.rowCount()):
        for j in range(ui.dataset_tableWidget.columnCount()):
            qt_item = QtWidgets.QTableWidgetItem(str(dataset.iloc[i, j]))  # Creates an qt item
            qt_item.setTextAlignment(QtCore.Qt.AlignHCenter)  # Aligns the item in the horizontal center
            ui.dataset_tableWidget.setItem(i + 1, j, qt_item)  # i+1 to skip the top row (column names)

    # Fill columnSelection_comboBox from the Visualise Tab
    for each_column in dataset.columns:
        ui.columnSelection_comboBox.addItem(each_column)


def update_visualisation(ui, dataset, radio_name):
    print('I AM GOING TO UPDATE EVERYTHING {}'.format(radio_name))


try:
    sys._MEIPASS
    files_folder = resource_path('resources/')
except:
    pass
files_folder = resource_path('../resources/')
