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

def configure_GUI(ui,ml_model):

    _translate = QtCore.QCoreApplication.translate

    #Connecting load_file_pushButton
    # ui.load_file_pushButton.clicked.connect(lambda *args, ml_model = ml_model:load_dataset(ml_model))
    ui.load_file_pushButton.clicked.connect(lambda ml_mode: load_dataset(ui,ml_model))

def load_dataset(ui,ml_model):
    """Prompts the user to select an input file and call ml_model.read_dataset.

    Args:
        ml_model (ML_model): An empty Machine Learning Model.

    """
    #TODO
    # uncomment this when finish doing tests
    # fileDlg = QtWidgets.QFileDialog()
    # file_address = fileDlg.getOpenFileName()[0]
    # return_code = ml_model.read_dataset(file_address)
    file_address = '/Users/matheustorquato/Desktop/RSC_Data_2.xlsx'
    return_code = ml_model.read_dataset(file_address)

    if return_code == 0:
        update_dataset_table(ui,ml_model.dataset)

    elif return_code == 1:  #Invalid file extension
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setText("Error")
        msg.setInformativeText('Invalid Input format. \nTry Excel or .csv formats')
        msg.setWindowTitle("Error")
        msg.exec()

    elif return_code == 2: #Exception while reading the file
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setText("Error")
        msg.setInformativeText('Invalid Input File')
        msg.setWindowTitle("Error")
        msg.exec()

def update_dataset_table(ui,dataset):

    ui.dataset_tableWidget.setRowCount(len(dataset.head(10))+1) # +1 to add the Column Name
    ui.dataset_tableWidget.setColumnCount(len(dataset.columns))

    for i in range(ui.dataset_tableWidget.rowCount()):
        for j in range(ui.dataset_tableWidget.columnCount()):
            if i == 0:
                qt_item = QtWidgets.QTableWidgetItem(dataset.columns[j])
            else:
                qt_item = QtWidgets.QTableWidgetItem(str(dataset.iloc[i, j]))
            qt_item.setTextAlignment(QtCore.Qt.AlignHCenter)
            ui.dataset_tableWidget.setItem(i, j, qt_item)


try:
    sys._MEIPASS
    files_folder = resource_path('resources/')
except:
    files_folder = resource_path('../resources/')

