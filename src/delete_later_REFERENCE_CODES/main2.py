
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from gui import QtCore, QtWidgets, Ui_Dialog
from os.path import join, abspath
import sys
import pickle
import pandas


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = abspath("..")

    return join(base_path, relative_path)

def configure_GUI(ui):

    _translate = QtCore.QCoreApplication.translate

    #Connecting reset button

    ui.load_file_pushButton.clicked.connect(lambda *args, objects=[]:run_prediction(ui,objects))

def run_prediction(ui,fdfd):

    fileDlg = QtWidgets.QFileDialog()
    file_address = fileDlg.getOpenFileName()[0]



    ui.file_adress_label.setText(file_address)

try:
    sys._MEIPASS
    files_folder = resource_path('resources/')
except:
    files_folder = resource_path('../../resources/')


app = QtCore.QCoreApplication.instance()
if app is None:
    app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
ui = Ui_Dialog()
ui.setupUi(MainWindow)
configure_GUI(ui)
MainWindow.show()
sys.exit(app.exec_())
