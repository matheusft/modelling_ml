import model as md
from view import QtCore, QtWidgets, Ui_Dialog
import controller
import sys

app = QtCore.QCoreApplication.instance()
if app is None:
    app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
ui = Ui_Dialog()
ui.setupUi(MainWindow)
user_ML_model = md.MlModel()
controller.configure_GUI(ui,user_ML_model)
MainWindow.show()
sys.exit(app.exec_())