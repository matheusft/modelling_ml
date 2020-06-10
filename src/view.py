# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '../resources/ui/ml_gui.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1040, 634)
        self.train_tab = QtWidgets.QTabWidget(Dialog)
        self.train_tab.setGeometry(QtCore.QRect(20, 40, 951, 521))
        self.train_tab.setObjectName("train_tab")
        self.dataset_load_tab = QtWidgets.QWidget()
        self.dataset_load_tab.setObjectName("dataset_load_tab")
        self.load_file_pushButton = QtWidgets.QPushButton(self.dataset_load_tab)
        self.load_file_pushButton.setGeometry(QtCore.QRect(280, 90, 113, 32))
        self.load_file_pushButton.setObjectName("load_file_pushButton")
        self.dataset_tableWidget = QtWidgets.QTableWidget(self.dataset_load_tab)
        self.dataset_tableWidget.setGeometry(QtCore.QRect(25, 240, 901, 211))
        self.dataset_tableWidget.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.dataset_tableWidget.setObjectName("dataset_tableWidget")
        self.dataset_tableWidget.setColumnCount(0)
        self.dataset_tableWidget.setRowCount(0)
        self.train_tab.addTab(self.dataset_load_tab, "")
        self.visualise_tab = QtWidgets.QWidget()
        self.visualise_tab.setObjectName("visualise_tab")
        self.train_tab.addTab(self.visualise_tab, "")
        self.inputoutput_tab = QtWidgets.QWidget()
        self.inputoutput_tab.setObjectName("inputoutput_tab")
        self.train_tab.addTab(self.inputoutput_tab, "")
        self.pre_process_tab = QtWidgets.QWidget()
        self.pre_process_tab.setObjectName("pre_process_tab")
        self.train_tab.addTab(self.pre_process_tab, "")
        self.model_selection_tab = QtWidgets.QWidget()
        self.model_selection_tab.setObjectName("model_selection_tab")
        self.train_tab.addTab(self.model_selection_tab, "")
        self.train_tab1 = QtWidgets.QWidget()
        self.train_tab1.setObjectName("train_tab1")
        self.train_tab.addTab(self.train_tab1, "")
        self.test_tab = QtWidgets.QWidget()
        self.test_tab.setObjectName("test_tab")
        self.train_tab.addTab(self.test_tab, "")

        self.retranslateUi(Dialog)
        self.train_tab.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Machine Learning Generic Modeller"))
        self.load_file_pushButton.setText(_translate("Dialog", "Load File"))
        self.train_tab.setTabText(self.train_tab.indexOf(self.dataset_load_tab), _translate("Dialog", "Dataset Load"))
        self.train_tab.setTabText(self.train_tab.indexOf(self.visualise_tab), _translate("Dialog", "Visualise"))
        self.train_tab.setTabText(self.train_tab.indexOf(self.inputoutput_tab), _translate("Dialog", "Input/Output"))
        self.train_tab.setTabText(self.train_tab.indexOf(self.pre_process_tab), _translate("Dialog", "Pre-Process"))
        self.train_tab.setTabText(self.train_tab.indexOf(self.model_selection_tab), _translate("Dialog", "Model Selction"))
        self.train_tab.setTabText(self.train_tab.indexOf(self.train_tab1), _translate("Dialog", "Train"))
        self.train_tab.setTabText(self.train_tab.indexOf(self.test_tab), _translate("Dialog", "Test"))

