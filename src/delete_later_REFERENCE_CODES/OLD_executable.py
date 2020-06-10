
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from GUI import QtCore, QtWidgets, Ui_Dialog
from os.path import join, abspath
import sys
import pickle
import pandas

class WorkerSignals(QtCore.QObject):
    """
    https://www.learnpyqt.com/courses/concurrent-execution/multithreading-pyqt-applications-qthreadpool/
    Defines the signals available from a running worker thread.

    """
    finished = QtCore.pyqtSignal(float,object)
    error_invalid_input = QtCore.pyqtSignal()

class Worker(QtCore.QRunnable):
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
        super(Worker, self).__init__()

        self.ui = args[0]
        self.value = args[1]

        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @QtCore.pyqtSlot()
    def run(self):

        try:
            ##Style is not used yet since all examples have the same value
            # style = ui.Style_comboBox.currentText()
            [grammage,shape] = grade_string_to_grammage_and_shape(ui.Grade_comboBox.currentText())
            encoded_shape = label_encoder.transform([shape])
            thickness = float(ui.Thickness_lineEdit.text())
            length = float(ui.Length_lineEdit.text())
            width = float(ui.Width_lineEdit.text())
            depth = float(ui.Depth_lineEdit.text())
            ect = float(ui.ECT_lineEdit.text())
        except:
            #Delete Input Values from GUI
            reset_ui_values([ui.Width_lineEdit,ui.Depth_lineEdit,ui.Length_lineEdit,ui.Thickness_lineEdit,ui.ECT_lineEdit])
            #Display Error Message
            self.signals.error_invalid_input.emit()

        #Input Array order = 'Grammage', 'Shape', 'Thickness', 'Length', 'Width', 'Depth', 'ECT'
        #Encoded_shape is added to the array just after the scaler transformation
        regression_input_array = [grammage,thickness,length,width,depth,ect]

        #Scaling the input values
        standardised_input = regression_input_scaler.transform([regression_input_array]).tolist()

        #Inserting the non-scaled encoded_shape into the input array
        standardised_input[0].insert(1,encoded_shape[0])

        #Predicting the output used the loaded model
        model_prediction = regression_model.predict(standardised_input)

        regression_output = model_prediction[0]

        self.signals.finished.emit(regression_output,self.ui)

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = abspath("..")

    return join(base_path, relative_path)

def grade_string_to_grammage_and_shape(grade_string):

    for i in range(3, len(grade_string) + 1):
        if not grade_string[:i].isdigit():
            return [int(grade_string[:i - 1]), grade_string[i - 1:]]

def configure_GUI(ui,unique_styles,unique_grades):

    _translate = QtCore.QCoreApplication.translate

    #Seting up the thread pool
    ui.threadpool = QtCore.QThreadPool()

    for index in range(len(unique_styles)):
        ui.Style_comboBox.addItem("")
        ui.Style_comboBox.setItemText(index, _translate("Dialog", unique_styles[index]))

    for index in range(len(unique_grades)):
        ui.Grade_comboBox.addItem("")
        ui.Grade_comboBox.setItemText(index, _translate("Dialog", unique_grades[index]))

    ui.buttonBox.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(False)
    ui.Tri_Wall_Logo.setText(_translate("Dialog",'<html><head/><body><p><img src=\"{0}"/></p></body></html>'.format(
                                                   files_folder+'images/Tri-Wall-Logo_NO_BG_small.png')))

    input_line_edit = [ui.Width_lineEdit, ui.Depth_lineEdit, ui.Length_lineEdit, ui.Thickness_lineEdit, ui.ECT_lineEdit]
    input_comboBox = [ui.Style_comboBox, ui.Grade_comboBox]
    output_object = [ui.CST_Result]

    #Connecting the change in the value of line_edits to functions
    for index, each_object in enumerate(input_line_edit):
        each_object.textChanged.connect(lambda *args, object_value=each_object,
                                                      object_index = index:
                                                      validate_values(ui,object_value,object_index))

    #Connecting reset button
    ui.buttonBox.button(QtWidgets.QDialogButtonBox.Reset).clicked.connect(lambda *args, objects=input_line_edit+output_object:
                                                                       reset_ui_values(objects))

    #Connecting OK button
    ui.buttonBox.button(QtWidgets.QDialogButtonBox.Ok).clicked.connect(lambda *args, objects=input_comboBox+input_line_edit:
                                                                       run_prediction(ui,objects))

def validate_values(ui,input_string,index):

    global valid_input

    try:
        float_input = float(input_string.text())
        valid_input[index] = True
        input_string.setStyleSheet('color: black')

    except:
        valid_input[index] = False
        if input_string.text() != '':
            input_string.setStyleSheet('color: red')
        else:
            input_string.setStyleSheet('color: black')

    if all(valid_input):
        ui.buttonBox.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(True)
    else:
        ui.buttonBox.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(False)

def reset_ui_values(objects):
    for index, each_object in enumerate(objects):
        each_object.setText("")
        each_object.setStyleSheet('color: black')

def run_prediction(ui,objects):

    len(objects)

    #Creating an object worker
    worker = Worker(ui, ui.Width_lineEdit.text())

    #Connecting the signals from the created worker to its functions
    worker.signals.finished.connect(update_prediction_result)
    worker.signals.error_invalid_input.connect(display_error_message)

    #Running the prediction in a separate thread from the GUI
    ui.threadpool.start(worker)

def update_prediction_result(result,ui):

    ui.CST_Result.setText('{:.2f}'.format((result)))

def display_error_message():

    msg = QtWidgets.QMessageBox()
    msg.setIcon(QtWidgets.QMessageBox.Critical)
    msg.setText("Error")
    msg.setInformativeText('Invalid Input Values')
    msg.setWindowTitle("Error")
    msg.exec()

try:
    sys._MEIPASS
    files_folder = resource_path('resources/')
except:
    files_folder = resource_path('../../resources/')

f = open(resource_path('{}unique_styles.pckl'.format(files_folder+'pickle/')), 'rb')
unique_styles = pickle.load(f)
f.close()

f = open(resource_path('{}unique_grades.pckl'.format(files_folder+'pickle/')), 'rb')
unique_grades = pickle.load(f)
f.close()

f = open(resource_path('{}{}.pckl'.format(files_folder+'pickle/','regression_model')) , 'rb')
regression_model = pickle.load(f)
f.close()

f = open(resource_path('{}{}.pckl'.format(files_folder+'pickle/','label_encoder')) , 'rb')
label_encoder = pickle.load(f)
f.close()

f = open(resource_path('{}{}.pckl'.format(files_folder+'pickle/','regression_input_scaler')) , 'rb')
regression_input_scaler = pickle.load(f)
f.close()

valid_input = [False]*5

app = QtCore.QCoreApplication.instance()
if app is None:
    app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
ui = Ui_Dialog()
ui.setupUi(MainWindow)
configure_GUI(ui,unique_styles,unique_grades)
MainWindow.show()
sys.exit(app.exec_())
