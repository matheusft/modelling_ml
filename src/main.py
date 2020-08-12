import model as md
from view import QtCore, QtWidgets, Ui_MainWindow
from controller import View_Controller
import controller
import sys

app = QtCore.QCoreApplication.instance()
if app is None:
    app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(MainWindow)
user_ML_model = md.MlModel()
view_controller = View_Controller(ui,user_ML_model)
# controller.configure_gui(ui, user_ML_model)
MainWindow.show()
sys.exit(app.exec_())